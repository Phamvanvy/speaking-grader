"""Reference metadata, skip resolution, deletion scoring + per-word detail/timing.

Các helper "không-DTW": map âm→từ, đánh dấu onset/coda/reducible, chấm 1 âm bị nuốt
(deletion), và dựng WordPronunciation (IPA + accuracy + cửa sổ phát lại) cho UI.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Mapping

from ..diagnostics import DRIFT_WINDOW_PAD_SEC, is_within_word_window
from ..ipa import (
    deletion_penalty,
    deletion_severity,
    is_vowel,
    normalize_ipa,
)
from ..l1_vietnamese import PenaltyReason, match_l1_final_deletion
from ..models import (
    PhonemePoint,
    PhonemeSegment,
    WordPronunciation,
    WordSpan,
)
from ..reliability import SkipDecision
from .constants import (
    MAX_WORDS_RETURNED,
    PHONEME_COVERAGE_GATE_CAP,
    PHONEME_COVERAGE_GATE_MAX_LEN,
    PHONEME_COVERAGE_GATE_MIN_ASR_PROB,
    PHONEME_G2P_UNCERTAIN_CAP,
)

logger = logging.getLogger("toeic.phoneme.scoring")


def _severity_from_penalty(penalty: float) -> str:
    """Map penalty liên tục → nhãn severity (cùng nguồn với điểm → không lệch nhau)."""
    if penalty >= 0.6:
        return "high"
    if penalty >= 0.3:
        return "medium"
    return "low"


# ──────────────────────────────────────────────────────────────────────────────
# Word mapping: gắn từ vào lỗi phoneme theo reference position
# ──────────────────────────────────────────────────────────────────────────────

def _word_at(position: int, spans: list[WordSpan], starts: list[int]) -> str | None:
    """Tìm từ chứa reference-index `position` bằng binary search.

    `spans` sắp xếp tăng dần, không chồng lấn; `starts` là [s.start_idx] đã tính
    sẵn (truyền vào để không tạo lại mỗi lần gọi). bisect_right + idx-1 cho span
    có start_idx lớn nhất mà ≤ position. Chỉ trả word nếu position thực sự nằm
    trong [start_idx, end_idx); nếu rơi vào gap (từ bị drop) hoặc position ==
    end_idx của span trước thì trả None (KHÔNG mượn từ liền trước).
    """
    idx = bisect.bisect_right(starts, position) - 1
    if idx < 0:
        return None
    span = spans[idx]
    if span.start_idx <= position < span.end_idx:
        return span.word
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Reference metadata: từ chứa mỗi âm + đánh dấu onset (đầu từ/cụm phụ âm đầu)
# ──────────────────────────────────────────────────────────────────────────────

def _ref_metadata(
    reference: list[str],
    spans: list[WordSpan] | None,
    stress: list[str | None] | None,
) -> tuple[
    list[str | None], list[bool], list[str | None], list[bool], list[bool], list[bool],
    list[bool],
]:
    """Trả (ref_word, ref_is_onset, ref_stress, ref_reducible, ref_is_coda,
    ref_g2p_uncertain, ref_r_droppable) song song reference.

    - ref_word[i]: từ chứa âm i (None nếu không có spans / âm rơi ngoài span).
    - ref_is_onset[i]: True nếu âm i là phụ âm thuộc cụm ĐẦU TỪ (từ start tới
      trước nguyên âm đầu tiên) — dùng để chấm nặng khi nuốt onset (think θ→∅,
      school sk→s) so với coda.
    - ref_stress[i]: nhấn âm (đệm None nếu thiếu / lệch độ dài).
    - ref_reducible[i]: True nếu âm i là NGUYÊN ÂM được phép rút gọn (KHÔNG phải nhân
      chính của từ) HOẶC nằm trong function word. Nhân chính = nguyên âm có nhấn
      primary; nếu từ không có âm nhấn (đơn âm tiết) thì nhân chính = nguyên âm DUY
      NHẤT/đầu tiên → bird /bɜːd/ ɜ KHÔNG reducible (giữ lỗi thật), water -er ɜ thì có.
    - ref_is_coda[i]: True nếu âm i là phụ âm CUỐI TỪ (sau nguyên âm cuối cùng tới hết
      từ) — bổ sung của onset; dùng cho L1 final-consonant tolerance (hand n,d; school l).
    - ref_g2p_uncertain[i]: True nếu IPA của từ chứa âm i lấy từ eSpeak G2P (OOV/tên
      riêng, WordSpan.source == "espeak") → reference tự nó kém tin cậy, sub/del bị
      cap penalty về "low" (PHONEME_G2P_UNCERTAIN_CAP).
    - ref_r_droppable[i]: True nếu âm i là /r/ KHÔNG đứng trước nguyên âm trong cùng
      từ (coda âm tiết: cuối từ HOẶC trước phụ âm — mo(r)ning, Satu(r)day). Giọng
      non-rhotic (Anh-Anh) nuốt /r/ ở MỌI vị trí này chứ không chỉ cuối từ; dùng cho
      accent_variant khi accept_accent_variants. /r/ trước nguyên âm (red, very)
      KHÔNG droppable — nuốt là lỗi thật.
    """
    from ..ipa import FUNCTION_WORDS

    n = len(reference)
    ref_word: list[str | None] = [None] * n
    ref_is_onset: list[bool] = [False] * n
    ref_reducible: list[bool] = [False] * n
    ref_is_coda: list[bool] = [False] * n
    ref_g2p_uncertain: list[bool] = [False] * n
    ref_r_droppable: list[bool] = [False] * n
    ref_stress: list[str | None] = list(stress) if stress else [None] * n
    if len(ref_stress) < n:
        ref_stress += [None] * (n - len(ref_stress))

    if spans:
        for span in spans:
            lo, hi = span.start_idx, min(span.end_idx, n)
            in_func = span.word.lower().strip(".,;:!?\"'()[]{}") in FUNCTION_WORDS
            g2p_uncertain = getattr(span, "source", "cmudict") == "espeak"
            for i in range(lo, hi):
                ref_word[i] = span.word
                if g2p_uncertain:
                    ref_g2p_uncertain[i] = True
                # /r/ không đứng trước nguyên âm TRONG CÙNG TỪ (cuối từ / trước phụ
                # âm) = coda âm tiết → non-rhotic được nuốt (xem docstring).
                if normalize_ipa(reference[i]) == "r" and (
                    i == hi - 1 or not is_vowel(reference[i + 1])
                ):
                    ref_r_droppable[i] = True
            # Onset = các phụ âm liên tiếp từ đầu từ cho tới nguyên âm đầu tiên.
            for i in range(lo, hi):
                if is_vowel(reference[i]):
                    break
                ref_is_onset[i] = True
            # Nhân chính của từ: nguyên âm có nhấn primary, nếu không có thì nguyên
            # âm đầu tiên. Các nguyên âm còn lại (+ mọi âm trong function word) reducible.
            vowels = [i for i in range(lo, hi) if is_vowel(reference[i])]
            primary = next((i for i in vowels if ref_stress[i] == "primary"), None)
            main = primary if primary is not None else (vowels[0] if vowels else None)
            for i in range(lo, hi):
                if in_func or (is_vowel(reference[i]) and i != main):
                    ref_reducible[i] = True
            # Coda = phụ âm sau nguyên âm CUỐI CÙNG của từ tới hết từ (bổ sung onset).
            if vowels:
                for i in range(vowels[-1] + 1, hi):
                    ref_is_coda[i] = True

    return (
        ref_word, ref_is_onset, ref_stress, ref_reducible, ref_is_coda,
        ref_g2p_uncertain, ref_r_droppable,
    )


def _resolve_skips(
    reference: list[str],
    spans: list[WordSpan] | None,
    skips: Mapping[int, SkipDecision] | None,
) -> tuple[list[bool], dict[int, str]]:
    """Trải quyết định skip (keyed theo CHỈ SỐ SPAN chuẩn) thành:
      - ref_skipped[i]: âm i thuộc 1 từ bị skip hay không (song song reference).
      - span_skip_reason[k]: lý do skip cho span thứ k (cho WordPronunciation.skip_reason).

    Scorer KHÔNG quyết định reliability — chỉ tiêu thụ `skips` do tầng trên đưa xuống.
    """
    n = len(reference)
    ref_skipped = [False] * n
    span_skip_reason: dict[int, str] = {}
    if spans and skips:
        for k, span in enumerate(spans):
            decision = skips.get(k)
            if decision is None:
                continue
            span_skip_reason[k] = decision.reason.value
            for i in range(span.start_idx, min(span.end_idx, n)):
                ref_skipped[i] = True
    return ref_skipped, span_skip_reason


# ──────────────────────────────────────────────────────────────────────────────
# Per-word phoneme detail: IPA full từng từ + trạng thái từng âm (UI kiểu ELSA)
# ──────────────────────────────────────────────────────────────────────────────

def _word_segment_times(
    path: list[tuple[int, int]],
    segments: list[PhonemeSegment],
    spans: list[WordSpan] | None,
) -> dict[int, tuple[float, float]]:
    """Cửa sổ thời gian PHÁT LẠI của từng từ, suy từ wav2vec phoneme segment.

    Vì sao: wav2vec chạy 1 forward pass trên toàn waveform nên mỗi `PhonemeSegment.
    start/end` là timestamp TUYỆT ĐỐI (~20ms) — chính xác và đúng "cái wav2vec nghe"
    hơn hẳn Whisper word window (±100–300ms, đôi khi rỗng → playback bị bíp/lẹm từ kế).

    Cách gán: CHỈ lấy cặp ĐƯỜNG CHÉO của DTW path (`pred_idx>=0 AND ref_idx>=0`) — mỗi
    segment ghép tối đa MỘT ref index (xem backtrace `_dtw_align`) nên thuộc đúng MỘT từ;
    bỏ insertion (`ref_idx<0`) để không nới biên sang từ kế. Cửa sổ từ = (min start, max
    end) các segment của nó. Path đơn điệu + segment xếp theo thời gian ⇒ cửa sổ các từ
    liên tiếp không chồng lấn.

    Trả {span_index: (start, end)} — CHỈ chứa key của từ có ≥1 segment (không bao giờ gọi
    min/max trên rỗng). Từ toàn deletion ⇒ không có key ⇒ caller fallback Whisper window.
    """
    if not spans or not segments or not path:
        return {}
    n = max((s.end_idx for s in spans), default=0)
    if n <= 0:
        return {}
    ref_to_span = [-1] * n
    for k, s in enumerate(spans):
        for i in range(s.start_idx, min(s.end_idx, n)):
            ref_to_span[i] = k

    acc: dict[int, list[float]] = {}  # span k → [min_start, max_end]
    for pred_idx, ref_idx in path:
        # Chỉ cặp đường chéo có segment + ref hợp lệ. Bound-check pred_idx phòng vệ
        # (pipeline hiện tại luôn hợp lệ vì path dựng từ [s.phoneme for s in segments]).
        if ref_idx < 0 or ref_idx >= n or not (0 <= pred_idx < len(segments)):
            continue
        k = ref_to_span[ref_idx]
        if k < 0:
            continue
        seg = segments[pred_idx]
        if k not in acc:
            acc[k] = [seg.start, seg.end]
        else:
            if seg.start < acc[k][0]:
                acc[k][0] = seg.start
            if seg.end > acc[k][1]:
                acc[k][1] = seg.end
    return {k: (v[0], v[1]) for k, v in acc.items()}


# Đệm phát lại (giây). Lead lớn hơn để bắt trọn onset (wav2vec hay cắt sát mép âm đầu);
# trail nhỏ vì từ kế thường bắt đầu ngay sau coda. CẢ HAI đều bị CLAMP theo từ liền kề
# nên không bao giờ lẹm sang từ khác — đây là chỗ DUY NHẤT tinh chỉnh đệm (không phải FE).
_WORD_PLAY_LEAD: float = 0.10
_WORD_PLAY_TRAIL: float = 0.04


def _pad_and_clamp_windows(
    times: dict[int, tuple[float, float]],
    lead: float = _WORD_PLAY_LEAD,
    trail: float = _WORD_PLAY_TRAIL,
) -> dict[int, tuple[float, float]]:
    """Đệm cửa sổ phát lại từng từ rồi CLAMP theo từ liền kề → không từ nào lẹm sang từ khác.

    `times`: {span_index: (start, end)} thô (từ segment/Whisper). Trả dict cùng key, mỗi
    cửa sổ đã `start-lead .. end+trail` nhưng:
      - start KHÔNG lùi qua `end` THÔ của từ k-1 (không nuốt coda từ trước);
      - end   KHÔNG vượt `start` THÔ của từ k+1 (không lấn onset từ kế — chặn "in→order").
    Đọc hàng xóm từ `times` GỐC (chưa đệm) để clamp đối xứng, không lan truyền sai số. Từ
    không có hàng xóm trong dict (đầu/cuối hoặc hàng xóm thiếu timing) → chỉ đệm tự do
    (start clamp về ≥0; end để playback tự dừng ở cuối file). Backend phát ra cửa sổ ĐÃ
    sẵn sàng phát → frontend chỉ việc phát verbatim [start, end].
    """
    if not times:
        return times
    out: dict[int, tuple[float, float]] = {}
    for k, (s, e) in times.items():
        ns = max(0.0, s - lead)
        prev = times.get(k - 1)
        if prev is not None:
            ns = max(ns, prev[1])      # không lùi qua coda từ trước
        ne = e + trail
        nxt = times.get(k + 1)
        if nxt is not None:
            ne = min(ne, nxt[0])       # không lấn onset từ kế
        if ne < e:                     # hàng xóm chồng (Whisper fallback) → ưu tiên ranh giới
            ne = nxt[0] if nxt is not None else e
        out[k] = (round(ns, 3), round(ne, 3))
    return out


def _build_word_details(
    point_by_ref: dict[int, PhonemePoint],
    reference: list[str],
    spans: list[WordSpan] | None,
    max_words: int = MAX_WORDS_RETURNED,
    span_skip_reason: dict[int, str] | None = None,
    word_times: Mapping[int, tuple[float, float]] | None = None,
) -> tuple[list[WordPronunciation], bool, int]:
    """Từ point_by_ref (đã align đủ cho MỌI reference index) → chi tiết từng từ.

    `point_by_ref` chứa đúng 1 PhonemePoint cho mỗi index trong reference (ok/sub/
    del/skipped), do compute_phoneme_score dựng sẵn → ở đây chỉ cắt theo ranh giới
    từ (không bao giờ cắt giữa từ). accuracy của từ = ok / (số âm KHÔNG skip);
    từ toàn skip → accuracy 1.0 (không tính là sai). `span_skip_reason` (theo CHỈ SỐ
    span chuẩn) gắn `skip_reason` cho từ bị Recognition Reliability bỏ qua.

    `word_times` (optional): cửa sổ thời gian PHÁT LẠI (start, end giây) theo CHỈ SỐ TỪ
    chuẩn (khớp `spans`). Caller dựng sẵn từ wav2vec segment (chính xác ~20ms), fallback
    Whisper window — xem `compute_phoneme_score`. Có → gắn start/end cho WordPronunciation
    để UI phát lại đoạn audio của riêng từ đó. Thiếu cho 1 từ → start/end = None.

    Returns (words, truncated, total_words).
    """
    if not spans:
        return [], False, 0

    span_skip_reason = span_skip_reason or {}
    total = len(spans)
    kept = spans[:max_words]
    truncated = total > len(kept)

    words: list[WordPronunciation] = []
    for k, span in enumerate(kept):  # k = chỉ số span chuẩn (kept cắt từ đầu list)
        points = [point_by_ref[i] for i in range(span.start_idx, span.end_idx)
                  if i in point_by_ref]
        if not points:
            continue
        scored = [p for p in points if p.status != "skipped"]
        ok = sum(1 for p in scored if p.status == "ok")
        win = word_times.get(k) if word_times else None
        words.append(WordPronunciation(
            word=span.word,
            ipa="".join(p.symbol for p in points),
            phonemes=points,
            accuracy=(ok / len(scored)) if scored else 1.0,
            skip_reason=span_skip_reason.get(k),
            start=win[0] if win else None,
            end=win[1] if win else None,
        ))

    if truncated:
        logger.info(
            "Word details truncated: kept %d / %d words", len(words), total
        )
    return words, truncated, total


# ──────────────────────────────────────────────────────────────────────────────
# Deletion scoring: 1 âm reference bị THIẾU
# ──────────────────────────────────────────────────────────────────────────────

def _score_deletion(
    ref_ph: str,
    *,
    is_onset: bool,
    is_coda: bool,
    stress: str | None,
    l1_enabled: bool,
    display_stress: str | None = None,
    accept_accent_variants: bool = False,
    g2p_uncertain: bool = False,
    r_droppable: bool = False,
) -> tuple[PhonemePoint, float, float]:
    """1 âm reference bị THIẾU → (PhonemePoint, penalty đã điều chỉnh, penalty gốc).

    Pipeline stage 1→3: base (deletion_penalty theo severity) → L1 (chỉ khi l1_enabled &
    âm ở coda & khớp rule final-deletion). DELETION KHÔNG đi qua confidence (stage 4) —
    không có predicted segment nên không có confidence. Dùng chung cho _align_points và
    vòng bổ sung deletion trong compute_phoneme_score.

    `accept_accent_variants` (chế độ accent "default"): /r/ non-prevocalic bị THIẾU =
    giọng Anh-Anh non-rhotic → KHÔNG phải lỗi → trả "ok" (penalty 0), tag ACCENT_VARIANT.
    Áp cho `r_droppable` (coda ÂM TIẾT: cuối từ HOẶC trước phụ âm trong cùng từ —
    car /kɑr/→/kɑ/, mo(r)ning, Satu(r)day); giữ thêm điều kiện cũ (is_coda & /r/) làm
    fallback cho caller chưa truyền r_droppable. LƯU Ý: đây CHƯA phải "union" đầy đủ
    GB/US. Các khác biệt hệ thống GB/US còn lại (oʊ↔əʊ, ɒ/ɑ↔ɔ, ɚ/ɝ↔ə, ɛ↔e) đã được
    normalize_ipa() gộp sẵn nên tự khớp. BATH split (æ↔ɑ) CỐ Ý không gộp (sau normalize
    thành æ↔ɔ, lẫn với lỗi thật).

    `g2p_uncertain`: IPA của từ lấy từ eSpeak (OOV/tên riêng) → reference tự nó là đoán →
    cap penalty về PHONEME_G2P_UNCERTAIN_CAP (severity "low", vào nhóm hidden-noise).
    """
    if accept_accent_variants and (
        r_droppable or (is_coda and normalize_ipa(ref_ph) == "r")
    ):
        point = PhonemePoint(
            symbol=ref_ph, status="ok", stress=stress, display_stress=display_stress,
            penalty_reason=PenaltyReason.ACCENT_VARIANT.value, penalty_adjustment=0.0,
        )
        return point, 0.0, 0.0
    raw = deletion_penalty(ref_ph, is_onset=is_onset, stress=stress)
    penalty = raw
    reason: str | None = None
    adjustment = 1.0
    severity = deletion_severity(ref_ph, is_onset=is_onset, stress=stress)
    if l1_enabled:
        reason = PenaltyReason.HARD_ERROR.value
        match = match_l1_final_deletion(ref_ph) if is_coda else None
        if match is not None:
            penalty = raw * match.multiplier
            adjustment = match.multiplier
            reason = PenaltyReason.L1_FINAL_DELETION.value
            severity = _severity_from_penalty(penalty)  # severity khớp penalty đã giảm
    if g2p_uncertain and penalty > PHONEME_G2P_UNCERTAIN_CAP:
        penalty = PHONEME_G2P_UNCERTAIN_CAP
        adjustment = (penalty / raw) if raw > 0 else 0.0
        reason = PenaltyReason.G2P_UNCERTAIN.value
        severity = _severity_from_penalty(penalty)  # < 0.3 → "low"
    point = PhonemePoint(
        symbol=ref_ph, status="del", severity=severity, stress=stress,
        display_stress=display_stress,
        penalty_reason=reason, penalty_adjustment=round(adjustment, 4),
    )
    return point, penalty, raw


# ──────────────────────────────────────────────────────────────────────────────
# Coverage gate (Track A): từ bị "del" toàn bộ + wav2vec im lặng + Whisper tự tin
# ──────────────────────────────────────────────────────────────────────────────

def _apply_coverage_gate(
    result: dict[int, tuple[PhonemePoint, float | None]],
    raw_by_ref: dict[int, float],
    reference: list[str],
    spans: list[WordSpan],
    ref_skipped: list[bool],
    predicted_times: list[tuple[float, float]],
    word_windows: Mapping[int, tuple[float, float]] | None,
    word_probs: Mapping[int, float] | None,
    claimed_preds: frozenset[int] | set[int] | None = None,
    *,
    max_len: int = PHONEME_COVERAGE_GATE_MAX_LEN,
    cap: float = PHONEME_COVERAGE_GATE_CAP,
    min_asr_prob: float = PHONEME_COVERAGE_GATE_MIN_ASR_PROB,
    pad: float = DRIFT_WINDOW_PAD_SEC,
) -> None:
    """Cap penalty các từ "coverage collapse" — wav2vec không nhả ÂM NÀO cho từ dù
    Whisper (nguồn độc lập) tự tin từ đó có trong audio. Mutate `result` in-place.

    CẢ 4 điều kiện phải cùng đúng (định nghĩa chặt — không chỉ "100% del"):
      1. Từ không bị Reliability skip; 100% âm không-skip là "del"; số âm ≤ max_len.
      2. KHÔNG có acoustic evidence CHƯA-CÓ-CHỦ: không có segment UNCLAIMED (insertion
         — DTW không gán cho ref nào) nằm trong window (±pad). Segment trong window
         nhưng ĐÃ bị từ hàng xóm nhận (diagonal pair, `claimed_preds`) KHÔNG tính —
         nói liền mạch + window Whisper lệch ±100-300ms nên âm từ kế lem vào window
         là chuyện thường; đòi "window trống tuyệt đối" làm gate gần như không bao
         giờ bắn (đo 9.0.mp4: 2/1503 từ). Âm unclaimed trong window mới là bằng chứng
         "wav2vec CÓ nghe thấy gì đó ở đây mà không từ nào nhận" → có thể lỗi thật,
         không cap. `claimed_preds=None` (caller cũ) → mọi segment đều tính (chặt như cũ).
      3. Whisper word prob ≥ min_asr_prob (transcript không phải ground truth tuyệt
         đối; prob thiếu/≤0 = "không có số liệu" — cùng convention assess_asr_confidence
         → không cap). LƯU Ý: whisperx dùng thang alignment score thấp hơn — caller
         truyền ngưỡng riêng (coverage_gate_min_asr_prob_whisperx).
      4. Từ phải CÓ window (không window → không kiểm chứng được (2) → không cap).

    An toàn của (2) nới lỏng: free-speech thì reference dựng từ CHÍNH transcript nên
    từ nào cũng được Whisper nghe thấy (không có chuyện "bỏ hẳn từ" lọt vào đây);
    scripted thì từ bỏ hẳn đã bị assess_reliability skip (delete opcode → không window).

    Precedence: chỉ HẠ penalty (`penalty > cap` mới ghi) — point đã 0 (connected_speech
    /recognizer_noise/...) hoặc đã cap 0.2 (g2p_uncertain) giữ nguyên reason cũ.
    KHÔNG đụng raw_by_ref (đây là cap, không phải elimination — l1_adjustment_ratio
    vẫn phản ánh mức giảm, giống pattern g2p_uncertain).
    """
    if not spans or word_windows is None or word_probs is None:
        return
    n = len(reference)
    for k, span in enumerate(spans):
        win = word_windows.get(k)
        if win is None:
            continue
        prob = word_probs.get(k, 0.0)
        if prob < min_asr_prob or prob <= 0:
            continue
        idxs = [i for i in range(span.start_idx, min(span.end_idx, n))
                if not ref_skipped[i]]
        if not idxs or len(idxs) > max_len:
            continue
        points = [result[i] for i in idxs if i in result]
        if len(points) != len(idxs) or any(p.status != "del" for p, _pen in points):
            continue
        # (2) acoustic evidence: chỉ segment UNCLAIMED (insertion) trong window mới
        # chặn cap; segment đã bị từ khác nhận = "stolen" / lem biên → bỏ qua.
        if any(
            is_within_word_window(ps, pe, win, pad)
            for pi, (ps, pe) in enumerate(predicted_times)
            if claimed_preds is None or pi not in claimed_preds
        ):
            continue
        for i in idxs:
            point, penalty = result[i]
            if penalty is None or penalty <= cap:
                continue
            raw = raw_by_ref.get(i, penalty)
            result[i] = (
                PhonemePoint(
                    symbol=point.symbol, status="del",
                    severity=_severity_from_penalty(cap),
                    stress=point.stress, display_stress=point.display_stress,
                    penalty_reason=PenaltyReason.COVERAGE_COLLAPSE.value,
                    penalty_adjustment=round(cap / raw, 4) if raw > 0 else 0.0,
                ),
                cap,
            )

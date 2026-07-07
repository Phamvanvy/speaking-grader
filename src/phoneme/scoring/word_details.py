"""Reference metadata, skip resolution, deletion scoring + per-word detail/timing.

Các helper "không-DTW": map âm→từ, đánh dấu onset/coda/reducible, chấm 1 âm bị nuốt
(deletion), và dựng WordPronunciation (IPA + accuracy + cửa sổ phát lại) cho UI.
"""

from __future__ import annotations

import bisect
import dataclasses
import logging
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

from ..diagnostics import DRIFT_WINDOW_PAD_SEC, is_within_word_window
from ..ipa import (
    deletion_penalty,
    deletion_severity,
    is_vowel,
    normalize_ipa,
    phoneme_similarity,
)
from ..l1_vietnamese import PenaltyReason, match_l1_final_deletion
from ..models import (
    EVIDENCE_VERSION,
    PhonemePoint,
    PhonemeSegment,
    WordPronunciation,
    WordSpan,
)

if TYPE_CHECKING:
    from ..wav2vec_backend import FramePosteriors
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
    """Cửa sổ thời gian theo wav2vec phoneme segment của từng từ.

    Hai người dùng: (a) deletion-evidence probe (cửa sổ nền — xem
    _attach_deletion_evidence), (b) FALLBACK playback cho từ KHÔNG có Whisper word
    window. Playback ưu tiên Whisper word timestamp (xem compute_phoneme_score):
    cửa sổ segment phụ thuộc DTW attribution — khi DTW "mượn" âm từ từ kế, min/max
    phình ra cả cụm (bug "discount" phát thành "20 percent discount") nên KHÔNG
    dùng làm nguồn chính cho playback.

    Cách gán: CHỈ lấy cặp ĐƯỜNG CHÉO của DTW path (`pred_idx>=0 AND ref_idx>=0`) — mỗi
    segment ghép tối đa MỘT ref index (xem backtrace `_dtw_align`) nên thuộc đúng MỘT từ;
    bỏ insertion (`ref_idx<0`) để không nới biên sang từ kế. Cửa sổ từ = (min start, max
    end) các segment của nó.

    Trả {span_index: (start, end)} — CHỈ chứa key của từ có ≥1 segment (không bao giờ gọi
    min/max trên rỗng). Từ toàn deletion ⇒ không có key.
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


# Đệm phát lại (giây) — "rất nhỏ, 50–100ms mỗi phía": cửa sổ nguồn giờ là Whisper
# WORD timestamp (ranh giới từ, không phải segment) nên chỉ cần đệm bù sai số biên
# ±100–300ms của Whisper một phần; đệm to hơn sẽ nghe lẹm từ kế. CẢ HAI đều bị CLAMP
# theo từ liền kề nên không bao giờ lẹm sang từ khác — đây là chỗ DUY NHẤT tinh chỉnh
# đệm (không phải FE).
_WORD_PLAY_LEAD: float = 0.08
_WORD_PLAY_TRAIL: float = 0.08
# Sàn thời lượng cho cửa sổ ĐÃ SIẾT trong _merge_playback_windows: giao Whisper∩segment
# ngắn hơn mức này coi như seg_times gán THIẾU (âm bị DTW "mượn" sang từ kế / deletion giả
# của wav2vec) → không tin intersection, giữ nguyên Whisper. ~2 âm vị (60–120ms/âm).
_MERGE_MIN_DUR: float = 0.2


def _merge_playback_windows(
    seg_times: dict[int, tuple[float, float]],
    word_windows: dict[int, tuple[float, float]],
    locked: Collection[int] | None = None,
) -> dict[int, tuple[float, float]]:
    """Gộp cửa sổ phát lại: Whisper WORD window (ranh giới TỪ) SIẾT theo cửa sổ wav2vec
    segment (âm vị THỰC của từ) khi cả hai chồng nhau.

    Whisper word timestamp ổn định về VỊ TRÍ từ nhưng có thể "lem" sang từ kế khi nối âm
    hoặc gộp token (vd "helps me" → "helps" phát cả "me"): cửa sổ Whisper thô nuốt luôn
    từ hàng xóm. Riêng token alphanumeric ("9am" → ref "am") đã được CẮT từ upstream
    (diagnostics.subtoken_window) trước khi tới đây — merge chỉ còn là lớp siết thêm. Cửa sổ segment (min/max các
    segment DTW gán đúng âm vị của từ — xem _word_segment_times) bám sát âm vị thực nên
    dùng để SIẾT biên: `start = max(whisper.start, seg.start)`, `end = min(whisper.end,
    seg.end)`. Chỉ siết khi giao đủ dài (`>= _MERGE_MIN_DUR`); giao rỗng/quá ngắn (mapping
    lệch / seg_times gán thiếu vì âm bị DTW "mượn" sang từ kế hoặc deletion giả) → GIỮ NGUYÊN
    Whisper — siết theo cửa sổ thiếu sẽ cắt mất tiếng thật, tệ nhất co về một mẩu sub-word.
    Limitation còn lại: mất MỘT âm biên khi seg thiếu đúng âm đó vẫn có thể xảy ra (đệm
    _WORD_PLAY_TRAIL bù một phần) — chấp nhận, không chặn được rẻ hơn.
    Từ chỉ có MỘT nguồn (Whisper-only: từ toàn deletion; seg-only: không map transcript) →
    dùng nguyên nguồn đó. `locked`: chỉ số từ có cửa sổ ĐÃ CẮT sub-token từ upstream
    (token alphanumeric "9am" → ref "am") — DTW attribution của các từ này KHÔNG đáng tin
    (âm phần số bị rơi không có trong reference, bị "hút" vào từ, seg_times trỏ nhầm chỗ)
    → BỎ QUA siết, giữ nguyên cửa sổ đã cắt (đảm bảo chứa từ thật).
    Trả dict theo CHỈ SỐ TỪ chuẩn, đưa thẳng vào _pad_and_clamp_windows.
    """
    if not word_windows:
        return dict(seg_times)
    out: dict[int, tuple[float, float]] = dict(seg_times)
    locked_set = frozenset(locked or ())
    for k, (ws, we) in word_windows.items():
        seg = seg_times.get(k)
        if seg is None or k in locked_set:  # toàn deletion / cửa sổ đã cắt sub-token
            out[k] = (ws, we)               # → chỉ dùng Whisper
            continue
        ns, ne = max(ws, seg[0]), min(we, seg[1])
        out[k] = (ns, ne) if ne - ns >= _MERGE_MIN_DUR else (ws, we)  # giao rỗng/sliver → giữ Whisper
    return out


# Gap cost cho fitting alignment re-anchor — CÙNG thang với homograph._GAP_COST
# (sub cost tối đa = 1 − similarity ∈ [0,1]) để indel không rẻ hơn sub tệ nhất.
_REANCHOR_GAP_COST: float = 1.0
# Trần cost TRUNG BÌNH mỗi âm reference để chấp nhận vùng khớp: vượt trần nghĩa là
# trong cửa sổ không có đoạn nào giống từ (học viên không đọc / wav2vec không nghe
# ra) → đừng neo vào rác, giữ cửa sổ Whisper đã cắt (fallback an toàn).
_REANCHOR_MAX_AVG_COST: float = 0.6


def _fit_word_segments(
    predicted: list[str], reference: list[str]
) -> tuple[int, int, float] | None:
    """FITTING alignment (reference khớp TRỌN VẸN vào 1 đoạn predicted) + traceback.

    Cùng công thức với homograph._alignment_cost (sub = 1 − phoneme_similarity,
    gap = _REANCHOR_GAP_COST, prefix/suffix predicted bỏ TỰ DO) nhưng trả thêm VỊ TRÍ:
    (i0, i1, cost) — [i0, i1) là đoạn predicted mà reference khớp vào, cost là tổng
    cost vùng khớp. None nếu reference khớp toàn gap (không có bằng chứng segment nào).
    Nhỏ (từ ~7 âm × window ~10 segment) nên O(n·m) bảng đầy đủ không đáng kể.
    """
    n, m = len(predicted), len(reference)
    if not n or not m:
        return None
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    dp[0] = [j * _REANCHOR_GAP_COST for j in range(m + 1)]
    for i in range(1, n + 1):
        # dp[i][0] = 0: bỏ predicted prefix miễn phí (fitting)
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + (
                1.0 - phoneme_similarity(predicted[i - 1], reference[j - 1])
            )
            dp[i][j] = min(sub, dp[i - 1][j] + _REANCHOR_GAP_COST,
                           dp[i][j - 1] + _REANCHOR_GAP_COST)
    # Kết thúc tại i bất kỳ (bỏ suffix miễn phí): lấy i nhỏ nhất đạt cost min —
    # deterministic, ưu tiên occurrence đầu.
    best_i = min(range(n + 1), key=lambda i: dp[i][m])
    cost = dp[best_i][m]
    # Traceback về j=0, ghi các predicted index đi qua nhánh sub (diagonal). So sánh
    # float bằng == an toàn: tính lại đúng biểu thức đã điền bảng.
    matched: list[int] = []
    i, j = best_i, m
    while j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            1.0 - phoneme_similarity(predicted[i - 1], reference[j - 1])
        ):
            matched.append(i - 1)
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + _REANCHOR_GAP_COST:
            i -= 1
        else:
            j -= 1
    if not matched:
        return None
    return (min(matched), max(matched) + 1, cost)


def _reanchor_locked_windows(
    times: dict[int, tuple[float, float]],
    locked: Collection[int],
    segments: list[PhonemeSegment],
    reference_phonemes: list[str],
    reference_spans: list[WordSpan],
    pad: float = DRIFT_WINDOW_PAD_SEC,
) -> None:
    """Neo lại cửa sổ phát lại của từ LOCKED theo acoustic (sửa `times` in-place).

    Từ locked = ref chỉ là PHẦN CHỮ của token alphanumeric ("9am" → "am"): cửa sổ
    Whisper đã cắt theo tỉ lệ ký tự chỉ là XẤP XỈ — học viên ngập ngừng/kéo dài phần
    số thì phần cắt vẫn dính "nine". DTW attribution cũng không tin được (âm phần số
    bị "hút" vào từ). Nguồn đáng tin duy nhất: fitting-align âm vị THAM CHIẾU của từ
    vào các segment trong cửa sổ (±pad, cùng phép overlap với các gate) rồi lấy đúng
    khoảng thời gian đoạn khớp — vd ref /æ m/ giữa [ə n aɪ aɪ ɛ m] khớp /ɛ m/ →
    phát đúng "A-M", bỏ "nine". Khớp quá tệ (> _REANCHOR_MAX_AVG_COST/âm) hoặc không
    có segment → giữ cửa sổ đã cắt. CHỈ playback — scoring/telemetry không đổi.
    """
    for k in locked:
        win = times.get(k)
        if win is None or not (0 <= k < len(reference_spans)):
            continue
        span = reference_spans[k]
        ref = reference_phonemes[span.start_idx:span.end_idx]
        cands = [
            s for s in segments
            if is_within_word_window(s.start, s.end, win, pad)
        ]
        if not ref or not cands:
            continue
        fit = _fit_word_segments([s.phoneme for s in cands], ref)
        if fit is None:
            continue
        i0, i1, cost = fit
        if cost / len(ref) > _REANCHOR_MAX_AVG_COST:
            continue
        ns = min(s.start for s in cands[i0:i1])
        ne = max(s.end for s in cands[i0:i1])
        if ne > ns:
            times[k] = (ns, ne)


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
    (start clamp về ≥0; end để playback tự dừng ở cuối file). Hàng xóm chồng ĐÈ cả cửa sổ
    làm clamp đảo chiều (start ≥ end) → bỏ clamp hàng xóm, giữ cửa sổ đệm của chính từ —
    KHÔNG BAO GIỜ trả cửa sổ rỗng/âm (frontend sẽ phát 0ms). Backend phát ra cửa sổ ĐÃ
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
        if ns >= ne:                   # hàng xóm ĐÈ cả cửa sổ (vd prev Whisper thô phủ qua
            # từ đã siết theo segment) → clamp đảo chiều (start ≥ end), frontend phát 0ms.
            # Ranh giới hàng xóm vô nghĩa ở đây → bỏ clamp, giữ cửa sổ đệm của chính từ.
            ns, ne = max(0.0, s - lead), e + trail
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
    chuẩn (khớp `spans`). Caller dựng sẵn từ Whisper WORD timestamp (ranh giới từ ổn
    định), fallback wav2vec segment cho từ không map được transcript — xem
    `compute_phoneme_score`. Có → gắn start/end cho WordPronunciation
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
# Deletion-evidence probe (SHADOW): soi posterior của chính âm bị thiếu
# ──────────────────────────────────────────────────────────────────────────────

def _attach_deletion_evidence(
    result: dict[int, tuple[PhonemePoint, float | None]],
    reference: list[str],
    spans: list[WordSpan],
    ref_skipped: list[bool],
    path: list[tuple[int, int]],
    segments: list[PhonemeSegment],
    seg_times: Mapping[int, tuple[float, float]],
    word_windows: Mapping[int, tuple[float, float]] | None,
    posteriors: FramePosteriors,
) -> None:
    """Gắn EvidenceStats vào mọi PhonemePoint "del" — SHADOW ONLY, mutate `result`
    in-place nhưng KHÔNG chạm penalty/status/severity (điểm bất biến bit-for-bit).

    Trả lời "âm bị thiếu có bằng chứng âm học không?": trong cửa sổ thời gian của
    từ, mass posterior của nhóm token khớp âm đó cao ⇒ âm CÓ trong audio nhưng thua
    argmax (recognizer hallucinate deletion); mass ~0 ⇒ thiếu âm thật (lỗi L1).
    Ngưỡng/quyết định để sprint sau — ở đây chỉ đo + log (dữ liệu gán nhãn cho
    Word Reliability Gate).

    Cửa sổ probe cho âm i thuộc từ k:
      - Nền: seg_times[k] (wav2vec, ~20ms) → "wav2vec_window"; thiếu → word_windows[k]
        (Whisper) → "whisper_window"; thiếu nốt → evidence=None, source="none".
      - Tinh chỉnh trong-từ: âm khớp (diagonal) LIỀN KỀ hai bên i trong CÙNG từ bound
        cửa sổ lại (tránh nhiễm âm trùng — "little" có 2 /l/). GUARD: refinement cho
        start >= end (segment hai bên chạm nhau) → fallback cửa sổ nền, không bao
        giờ probe trên cửa sổ rỗng do refinement.
    """
    n = len(reference)
    ref_to_span = [-1] * n
    for k, s in enumerate(spans):
        for i in range(s.start_idx, min(s.end_idx, n)):
            ref_to_span[i] = k
    # ref_idx → pred_idx của cặp diagonal (âm đã có segment "nhận") — cùng cách
    # dựng với build_word_diagnostics để hai bên không lệch attribution.
    pred_for_ref: dict[int, int] = {
        r: p for p, r in path if p >= 0 and 0 <= r < n and p < len(segments)
    }

    for i in range(n):
        if ref_skipped[i] or i not in result:
            continue
        point, pen = result[i]
        if point.status != "del":
            continue
        k = ref_to_span[i]
        span = spans[k] if k >= 0 else None
        base = seg_times.get(k) if k >= 0 else None
        source = "wav2vec_window" if base is not None else None
        if base is None and word_windows is not None and k >= 0:
            base = word_windows.get(k)
            source = "whisper_window" if base is not None else None
        stats = None
        if base is not None and span is not None:
            t0, t1 = base
            # Tinh chỉnh theo âm khớp liền kề trong cùng từ.
            lo = next(
                (segments[pred_for_ref[j]].end
                 for j in range(i - 1, span.start_idx - 1, -1)
                 if j in pred_for_ref),
                t0,
            )
            hi = next(
                (segments[pred_for_ref[j]].start
                 for j in range(i + 1, min(span.end_idx, n))
                 if j in pred_for_ref),
                t1,
            )
            if lo >= hi:  # guard cửa sổ đảo → fallback cửa sổ nền
                lo, hi = t0, t1
            stats = posteriors.evidence_stats(reference[i], lo, hi)
            logger.info(
                "Deletion evidence[%s]: word=%r ph=%r src=%s win=[%.3f,%.3f] %s",
                EVIDENCE_VERSION, span.word, reference[i], source, lo, hi,
                (
                    "max=%.4f topk=%.4f p90=%.4f n=%d argmax=%s(%.3f)" % (
                        stats.max_mass, stats.top_k_mean, stats.p90,
                        stats.n_frames, stats.argmax_token, stats.argmax_prob,
                    )
                    if stats is not None
                    else "no_token_group"
                ),
            )
        else:
            source = "none"
            logger.info(
                "Deletion evidence[%s]: word=%r ph=%r src=none (không có cửa sổ)",
                EVIDENCE_VERSION, span.word if span else None, reference[i],
            )
        result[i] = (
            dataclasses.replace(
                point, evidence=stats, evidence_source=source,
                evidence_version=EVIDENCE_VERSION,
            ),
            pen,
        )


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

"""Phoneme scoring — so sánh predicted phonemes với reference IPA sequence.

Dùng Dynamic Time Warping (DTW) để align 2 sequences (predicted vs reference),
từ đó detect substitution / deletion / insertion + tính overall accuracy.

Architecture:
  - compute_phoneme_score(): main entry point → PhonemeScore
  - _dtw_align(): align 2 phoneme sequences bằng DTW
  - _align_points(): 1 lượt qua path → point+penalty cho mỗi reference index
    (qua phonemes_match / phoneme_similarity / confidence + skip_words)
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Callable, Mapping
from typing import Final

from .diagnostics import WordDiagnostic, build_word_diagnostics
from .ipa import (
    deletion_penalty,
    deletion_severity,
    is_real_error_substitution,
    is_vowel,
    normalize_ipa,
    phoneme_similarity,
    phonemes_match,
)
from .l1_vietnamese import PenaltyReason, match_l1_final_deletion
from .reliability import SkipDecision
from .models import (
    PhonemeError,
    PhonemeErrorType,
    PhonemePoint,
    PhonemeScore,
    PhonemeSegment,
    WordPronunciation,
    WordSpan,
)

logger = logging.getLogger("toeic.phoneme.scoring")

# Số lỗi tối đa trả về trong results (tránh payload quá lớn)
MAX_ERRORS_RETURNED: Final[int] = 30

# Số từ tối đa trả về trong word details (cắt theo ranh giới từ, không giữa từ)
MAX_WORDS_RETURNED: Final[int] = 80

# Knee của confidence weighting: predicted phoneme có confidence < knee thì penalty
# của lỗi sub bị hạ tỉ lệ (recognizer không chắc → ít khả năng là lỗi người đọc).
# Mặc định 0.5; override qua Config.phoneme_confidence_knee (env TOEIC_PHONEME_CONFIDENCE_KNEE).
PHONEME_CONFIDENCE_KNEE: Final[float] = 0.5

# L1-aware scoring layer (default OFF — bật qua Config.phoneme_l1_*; xem l1_vietnamese.py).
# l1_min_confidence: ngưỡng confidence để áp L1 *substitution* tolerance (chưa dùng ở v1).
# low_conf_floor: sub có confidence < ngưỡng này → penalty bị TRUNG HOÀ về 0 (soften,
# KHÔNG skip). Chỉ áp cho sub (âm được nhận diện); deletion KHÔNG đi qua confidence.
PHONEME_L1_MIN_CONFIDENCE: Final[float] = 0.70
PHONEME_LOW_CONF_FLOOR: Final[float] = 0.40

# Recognizer-noise gate (ĐỘC LẬP với L1): 1 sub bị coi là wav2vec hallucinate (KHÔNG
# phải lỗi học viên) khi cặp (ref→pred) BẤT KHẢ THI về âm học (sim < SIM, và không nằm
# trong _REAL_ERROR_SUBS) VÀ recognizer KHÔNG chắc (conf < CONF). Khi đó penalty về 0 +
# severity "low" → rơi vào nhóm "Hidden recognizer noise" (không tô đỏ), giống cơ chế
# LOW_CONFIDENCE_NEUTRALIZED nhưng có điều kiện bất-khả-thi nên KHÔNG giấu lỗi near-pair.
#
# Ngưỡng conf THEO LOẠI ÂM (hiệu chỉnh từ telemetry tel3.jsonl): nguyên âm wav2vec/espeak
# vốn confidence thấp hơn nhiều dù ĐÚNG (median ~0.67 vs phụ âm ~0.91) → 1 ngưỡng chung
# sẽ gate oan nguyên âm. CONF=0 → tắt gate (conf < 0 không bao giờ đúng → bit-for-bit như cũ).
#
# GIỚI HẠN ĐÃ BIẾT (sprint này, có chủ đích): gate này CHỈ bắt sub bất khả thi + CONFIDENCE
# THẤP. Nó KHÔNG xử lý "whole-word hallucination" CONFIDENCE CAO — khi wav2vec tự tin nhả
# sai cả từ (vd famous /feɪməs/→/leɪmz/ f→l @0.98) hoặc Whisper chép nhầm từ (blood→"floods"
# nên reference IPA sai). Confidence KHÔNG bắt được các ca này (đang cao), và word-accuracy
# KHÔNG tách được chúng khỏi lỗi phát âm THẬT (vd Vietnam v→b, nuốt cụm cuối first/most) → ẩn
# theo accuracy sẽ giấu lỗi thật. Hướng tương lai: "Word Reliability Gate" thiết kế TỪ DỮ
# LIỆU telemetry per-word (diagnostics.py đã ghi đủ: ref/pred IPA + alignment + per-phone
# confidence), KHÔNG bake heuristic conf/sim ở production.
PHONEME_RECOGNIZER_NOISE_SIM: Final[float] = 0.2
PHONEME_RECOGNIZER_NOISE_CONF: Final[float] = 0.6        # phụ âm
PHONEME_RECOGNIZER_NOISE_CONF_VOWEL: Final[float] = 0.45  # nguyên âm (confidence nền thấp hơn)

# Xếp hạng để chọn status "tốt hơn" khi 1 reference index bị chạm nhiều lần.
# "skipped" (từ ASR nghe nhầm) coi như tốt nhất — không phải lỗi, loại khỏi điểm.
_STATUS_RANK: Final[dict[str, int]] = {"ok": 0, "skipped": 0, "sub": 1, "del": 2}
_SEVERITY_RANK: Final[dict[str, int]] = {"low": 0, "medium": 1, "high": 2}


def _severity_from_penalty(penalty: float) -> str:
    """Map penalty liên tục → nhãn severity (cùng nguồn với điểm → không lệch nhau)."""
    if penalty >= 0.6:
        return "high"
    if penalty >= 0.3:
        return "medium"
    return "low"


# Phụ âm yếu wav2vec hay NUỐT (ð trong "the/this", h trong "his/heard"). Khi bị nuốt,
# DTW thường không xếp thành deletion sạch mà lệch align: phụ âm này "thay" bằng
# NGUYÊN ÂM kế bên (vd the /ðə/ → /ə/ thành ð→ə). Đó là lỗi NHẬN DẠNG, không phải
# người đọc → hạ về low. Vẫn giữ lỗi thật th-stopping (ð→d/z) vì predicted là PHỤ ÂM.
_RECOGNIZER_DROP_CONS: Final[frozenset[str]] = frozenset({"ð", "h"})


# ──────────────────────────────────────────────────────────────────────────────
# DTW alignment (simplified, pure Python — không cần numpy dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _dtw_align(
    predicted: list[str],
    reference: list[str],
) -> list[tuple[int, int]]:
    """Dynamic Time Warping: align predicted → reference phoneme sequences.

    Returns list of (pred_idx, ref_idx) pairs theo optimal path.
    """
    if not predicted or not reference:
        return []

    n, m = len(predicted), len(reference)

    # Cost matrix: 0 = match, 1-m similarity = mismatch cost
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        cost[i][0] = float("inf")
    for j in range(1, m + 1):
        cost[0][j] = float("inf")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = phoneme_similarity(predicted[i - 1], reference[j - 1])
            match_cost = 1.0 - sim  # 0.0 = match, 1.0 = hoàn toàn khác
            cost[i][j] = match_cost + min(
                cost[i - 1][j - 1],   # match/substitution (diagonal)
                cost[i - 1][j],       # insertion in predicted (vertical)
                cost[i][j - 1],       # deletion from reference (horizontal)
            )

    # Backtrace
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            path.append((-1, j))  # deletion
        elif j == 0:
            i -= 1
            path.append((i, -1))  # insertion
        else:
            # Find which predecessor gave the min cost (excluding match_cost)
            cum_diag = cost[i - 1][j - 1]
            cum_up = cost[i - 1][j]
            cum_left = cost[i][j - 1]
            minimum = min(cum_diag, cum_up, cum_left)
            if cum_diag == minimum:
                i -= 1
                j -= 1
                path.append((i, j))  # match or substitution
            elif cum_up == minimum:
                i -= 1
                path.append((i, -1))  # insertion (predicted i, không có reference)
            else:
                j -= 1
                path.append((-1, j))  # deletion

    path.reverse()
    return path


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
) -> tuple[list[str | None], list[bool], list[str | None], list[bool], list[bool]]:
    """Trả (ref_word, ref_is_onset, ref_stress, ref_reducible, ref_is_coda) song song reference.

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
    """
    from .ipa import FUNCTION_WORDS

    n = len(reference)
    ref_word: list[str | None] = [None] * n
    ref_is_onset: list[bool] = [False] * n
    ref_reducible: list[bool] = [False] * n
    ref_is_coda: list[bool] = [False] * n
    ref_stress: list[str | None] = list(stress) if stress else [None] * n
    if len(ref_stress) < n:
        ref_stress += [None] * (n - len(ref_stress))

    if spans:
        for span in spans:
            lo, hi = span.start_idx, min(span.end_idx, n)
            in_func = span.word.lower().strip(".,;:!?\"'()[]{}") in FUNCTION_WORDS
            for i in range(lo, hi):
                ref_word[i] = span.word
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

    return ref_word, ref_is_onset, ref_stress, ref_reducible, ref_is_coda


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

def _better_point(a: PhonemePoint, b: PhonemePoint) -> PhonemePoint:
    """Chọn point 'tốt hơn' khi 1 reference index bị chạm nhiều lần trong path.

    Ưu tiên status ok > sub > del; nếu cùng status (vd 2 sub) thì ưu tiên
    severity nhẹ hơn (similarity cao hơn).
    """
    ra, rb = _STATUS_RANK.get(a.status, 9), _STATUS_RANK.get(b.status, 9)
    if ra != rb:
        return a if ra < rb else b
    sa = _SEVERITY_RANK.get(a.severity or "", -1)
    sb = _SEVERITY_RANK.get(b.severity or "", -1)
    return a if sa <= sb else b


def _build_word_details(
    point_by_ref: dict[int, PhonemePoint],
    reference: list[str],
    spans: list[WordSpan] | None,
    max_words: int = MAX_WORDS_RETURNED,
    span_skip_reason: dict[int, str] | None = None,
) -> tuple[list[WordPronunciation], bool, int]:
    """Từ point_by_ref (đã align đủ cho MỌI reference index) → chi tiết từng từ.

    `point_by_ref` chứa đúng 1 PhonemePoint cho mỗi index trong reference (ok/sub/
    del/skipped), do compute_phoneme_score dựng sẵn → ở đây chỉ cắt theo ranh giới
    từ (không bao giờ cắt giữa từ). accuracy của từ = ok / (số âm KHÔNG skip);
    từ toàn skip → accuracy 1.0 (không tính là sai). `span_skip_reason` (theo CHỈ SỐ
    span chuẩn) gắn `skip_reason` cho từ bị Recognition Reliability bỏ qua.

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
        words.append(WordPronunciation(
            word=span.word,
            ipa="".join(p.symbol for p in points),
            phonemes=points,
            accuracy=(ok / len(scored)) if scored else 1.0,
            skip_reason=span_skip_reason.get(k),
        ))

    if truncated:
        logger.info(
            "Word details truncated: kept %d / %d words", len(words), total
        )
    return words, truncated, total


# ──────────────────────────────────────────────────────────────────────────────
# Main scoring function
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
) -> tuple[PhonemePoint, float, float]:
    """1 âm reference bị THIẾU → (PhonemePoint, penalty đã điều chỉnh, penalty gốc).

    Pipeline stage 1→3: base (deletion_penalty theo severity) → L1 (chỉ khi l1_enabled &
    âm ở coda & khớp rule final-deletion). DELETION KHÔNG đi qua confidence (stage 4) —
    không có predicted segment nên không có confidence. Dùng chung cho _align_points và
    vòng bổ sung deletion trong compute_phoneme_score.

    `accept_accent_variants` (chế độ accent "default"): coda /r/ bị THIẾU = giọng Anh-Anh
    non-rhotic (car /kɑr/→/kɑ/) → KHÔNG phải lỗi → trả "ok" (penalty 0), tag ACCENT_VARIANT.
    LƯU Ý: đây CHƯA phải "union" đầy đủ GB/US — chỉ chấp nhận nuốt coda /r/. Các khác biệt
    hệ thống GB/US còn lại (oʊ↔əʊ, ɒ/ɑ↔ɔ, ɚ/ɝ↔ə, ɛ↔e) đã được normalize_ipa() gộp sẵn nên
    tự khớp. BATH split (æ↔ɑ) CỐ Ý không gộp (sau normalize thành æ↔ɔ, lẫn với lỗi thật).
    """
    if accept_accent_variants and is_coda and normalize_ipa(ref_ph) == "r":
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
    point = PhonemePoint(
        symbol=ref_ph, status="del", severity=severity, stress=stress,
        display_stress=display_stress,
        penalty_reason=reason, penalty_adjustment=round(adjustment, 4),
    )
    return point, penalty, raw


def _align_points(
    path: list[tuple[int, int]],
    predicted: list[str],
    predicted_conf: list[float],
    reference: list[str],
    ref_word: list[str | None],
    ref_is_onset: list[bool],
    ref_stress: list[str | None],
    ref_reducible: list[bool],
    ref_skipped: list[bool],
    ref_is_coda: list[bool],
    knee: float,
    ref_display_stress: list[str | None],
    *,
    l1_enabled: bool = False,
    low_conf_floor: float = PHONEME_LOW_CONF_FLOOR,
    recognizer_noise_sim: float = PHONEME_RECOGNIZER_NOISE_SIM,
    recognizer_noise_conf: float = PHONEME_RECOGNIZER_NOISE_CONF,
    recognizer_noise_conf_vowel: float = PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    accept_accent_variants: bool = False,
) -> tuple[dict[int, tuple[PhonemePoint, float | None]], int, dict[int, float]]:
    """Một lượt qua DTW path → (point+penalty mỗi ref index, số insertion, penalty gốc/ref).

    Penalty pipeline (THỨ TỰ CỐ ĐỊNH): base → recognizer cap → L1 (deletion) → confidence
    (substitution). L1 layer chỉ tác động khi `l1_enabled`; khi tắt, hàm chạy y hệt trước
    (bit-for-bit). Mỗi ref index ghi `penalty_reason` (why) + `penalty_adjustment` (how-much)
    trên PhonemePoint, và penalty GỐC (trước L1/neutralization) vào raw_by_ref để tính metadata.

    - Từ bị skip (Recognition Reliability) → "skipped", penalty None (kiểm tra TRƯỚC mọi thứ).
    - phonemes_match → "ok", penalty 0.
    - Substitution: base=(1-sim)*conf_factor [+ cap ð/h]; nếu l1_enabled & conf<low_conf_floor
      → TRUNG HOÀ penalty về 0 (low_confidence_neutralized). Deletion: xem _score_deletion.
    - Mỗi ref index giữ point TỐT NHẤT nếu path chạm nhiều lần (_better_point).
    """
    result: dict[int, tuple[PhonemePoint, float | None]] = {}
    raw_by_ref: dict[int, float] = {}
    insertion_count = 0
    for pred_idx, ref_idx in path:
        if ref_idx < 0:
            if pred_idx >= 0:
                insertion_count += 1
            continue
        ref_ph = reference[ref_idx]
        word = ref_word[ref_idx]
        stress = ref_stress[ref_idx]
        disp = ref_display_stress[ref_idx]  # nhấn-hiển-thị (onset) — chỉ gắn lên point
        iter_raw: float | None = None
        if ref_skipped[ref_idx]:
            # Cả từ bị skip → mọi âm "skipped" bất kể có khớp hay không.
            point = PhonemePoint(symbol=ref_ph, status="skipped", stress=stress,
                                 display_stress=disp)
            penalty: float | None = None
        elif pred_idx >= 0:
            pred_ph = predicted[pred_idx]
            conf = predicted_conf[pred_idx] if pred_idx < len(predicted_conf) else 1.0
            if phonemes_match(
                ref_ph, pred_ph, word=word, reducible=ref_reducible[ref_idx]
            ):
                point = PhonemePoint(symbol=ref_ph, status="ok", stress=stress,
                                     display_stress=disp)
                penalty = 0.0
            elif (
                accept_accent_variants
                and ref_is_coda[ref_idx]
                and normalize_ipa(ref_ph) == "r"
                and is_vowel(pred_ph)
            ):
                # Accent "default": coda /r/ non-rhotic (Anh-Anh) — predicted là NGUYÊN ÂM
                # (r-coloring residue / schwa lệch align lên /r/), KHÔNG phải lỗi. CHỈ khoan
                # dung khi predicted là vowel; phụ âm thay /r/ (l/w/j/n...) VẪN là lỗi thật.
                point = PhonemePoint(
                    symbol=ref_ph, status="ok", stress=stress, display_stress=disp,
                    penalty_reason=PenaltyReason.ACCENT_VARIANT.value,
                    penalty_adjustment=0.0,
                )
                penalty = 0.0
            else:
                sim = phoneme_similarity(pred_ph, ref_ph)
                conf_factor = min(1.0, conf / knee) if knee > 0 else 1.0
                base_sub = (1.0 - sim) * conf_factor
                # Stage 2 recognizer cap: ð/h "thay" bằng nguyên âm = recognizer nuốt phụ
                # âm rồi lệch align (không phải lỗi người đọc) → hạ về low.
                if normalize_ipa(ref_ph) in _RECOGNIZER_DROP_CONS and is_vowel(pred_ph):
                    base_sub = min(base_sub, 0.1)
                penalty = base_sub
                reason: str | None = None
                adjustment = 1.0
                # Stage 2.5 recognizer-noise gate (ĐỘC LẬP với L1): sub bất khả thi về
                # âm học + recognizer không chắc → wav2vec hallucinate, KHÔNG phải lỗi
                # người đọc. Bảo vệ near-pair (sim cao) và lỗi VN thật (_REAL_ERROR_SUBS).
                noise_conf = (
                    recognizer_noise_conf_vowel if is_vowel(ref_ph)
                    else recognizer_noise_conf
                )
                if conf < noise_conf and not is_real_error_substitution(
                    ref_ph, pred_ph, sim_floor=recognizer_noise_sim
                ):
                    penalty = 0.0
                    reason = PenaltyReason.RECOGNIZER_NOISE.value
                    adjustment = 0.0
                elif l1_enabled:
                    reason = PenaltyReason.HARD_ERROR.value
                    # Stage 4 confidence (chỉ substitution): conf rất thấp → trung hoà.
                    if conf < low_conf_floor:
                        penalty = 0.0
                        reason = PenaltyReason.LOW_CONFIDENCE_NEUTRALIZED.value
                        adjustment = 0.0
                point = PhonemePoint(
                    symbol=ref_ph, status="sub", heard=pred_ph,
                    severity=_severity_from_penalty(penalty), stress=stress,
                    display_stress=disp,
                    penalty_reason=reason, penalty_adjustment=round(adjustment, 4),
                )
                iter_raw = base_sub
        else:
            point, penalty, iter_raw = _score_deletion(
                ref_ph, is_onset=ref_is_onset[ref_idx], is_coda=ref_is_coda[ref_idx],
                stress=stress, display_stress=disp, l1_enabled=l1_enabled,
                accept_accent_variants=accept_accent_variants,
            )

        existing = result.get(ref_idx)
        if existing is None or _better_point(existing[0], point) is point:
            result[ref_idx] = (point, penalty)
            if iter_raw is not None:
                raw_by_ref[ref_idx] = iter_raw
            elif ref_idx in raw_by_ref:
                del raw_by_ref[ref_idx]  # point được thay = ok/skipped → bỏ raw cũ
    return result, insertion_count, raw_by_ref


def compute_phoneme_score(
    segments: list[PhonemeSegment],
    reference_phonemes: list[str],
    reference_spans: list[WordSpan] | None = None,
    reference_stress: list[str | None] | None = None,
    reference_display_stress: list[str | None] | None = None,
    max_words: int = MAX_WORDS_RETURNED,
    skips: Mapping[int, SkipDecision] | None = None,
    confidence_knee: float = PHONEME_CONFIDENCE_KNEE,
    diagnostics_sink: Callable[[list[WordDiagnostic]], None] | None = None,
    word_windows: Mapping[int, tuple[float, float]] | None = None,
    l1_enabled: bool = False,
    l1_min_confidence: float = PHONEME_L1_MIN_CONFIDENCE,
    low_conf_floor: float = PHONEME_LOW_CONF_FLOOR,
    recognizer_noise_sim: float = PHONEME_RECOGNIZER_NOISE_SIM,
    recognizer_noise_conf: float = PHONEME_RECOGNIZER_NOISE_CONF,
    recognizer_noise_conf_vowel: float = PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    accept_accent_variants: bool = False,
) -> PhonemeScore | None:
    """Tính phoneme accuracy score từ predicted segments + reference.

    Pipeline de-noise: normalize tối thiểu → phoneme_similarity (liên tục) →
    phonemes_match (allophone/reduction) → confidence weighting → điểm có trọng số.

    Args:
        reference_spans: map reference index → từ (WordSpan) để gắn `word` cho lỗi
            và xác định onset; None thì lỗi giữ word=None (tương thích ngược).
        reference_stress: nhấn âm song song 1-1 với reference_phonemes (TRÊN nguyên âm —
            scoring đọc để xác định nhân chính/severity).
        reference_display_stress: nhấn âm ĐÃ dời về onset (chỉ HIỂN THỊ) song song 1-1 với
            reference_phonemes. CHỈ gắn vào PhonemePoint.display_stress cho UI — KHÔNG tham gia
            alignment/severity/điểm. None = không có (PhonemePoint.display_stress = None).
        skips: quyết định bỏ qua từ Recognition Reliability (tầng TRÊN), keyed theo
            CHỈ SỐ SPAN chuẩn (vị trí trong reference_spans). Scorer CHỈ tiêu thụ —
            KHÔNG tự quyết định reliability (không suy từ match-ratio/similarity/penalty).
            Từ bị skip: mọi âm thành "skipped", loại khỏi cả tử số lẫn mẫu số accuracy.
        confidence_knee: ngưỡng confidence để hạ penalty lỗi sub (xem PHONEME_CONFIDENCE_KNEE).
        diagnostics_sink: optional — nhận list[WordDiagnostic] để ghi telemetry (PR2).
            CHỈ để quan sát; KHÔNG ảnh hưởng điểm. None = không tính telemetry (zero overhead).
        word_windows: optional (PR3-0) — cửa sổ thời gian Whisper theo CHỈ SỐ TỪ chuẩn
            (khớp reference_spans). CHỈ dùng cho telemetry drift-vs-hallucination; KHÔNG
            ảnh hưởng điểm. Bỏ qua nếu không có diagnostics_sink.
        l1_enabled: bật L1-aware layer (Vietnamese). False (mặc định) = hành vi y hệt trước
            (bit-for-bit). True = giảm penalty cho nuốt phụ âm cuối kiểu L1 + trung hoà sub
            confidence rất thấp; vẫn hiển thị (accent note). KHÔNG skip (Reliability mới được skip).
        l1_min_confidence: ngưỡng confidence để áp L1 *substitution* tolerance (chưa dùng ở v1).
        low_conf_floor: sub có confidence < ngưỡng → penalty trung hoà về 0 (chỉ khi l1_enabled).
        recognizer_noise_sim: ngưỡng similarity BẢO VỆ — sub có sim ≥ ngưỡng (hoặc nằm trong
            _REAL_ERROR_SUBS) KHÔNG bao giờ bị gate noise (xem PHONEME_RECOGNIZER_NOISE_SIM).
        recognizer_noise_conf / recognizer_noise_conf_vowel: ngưỡng confidence (phụ âm / nguyên âm)
            của recognizer-noise gate. Sub bất khả thi + conf dưới ngưỡng → recognizer hallucinate →
            penalty 0 + severity low. ĐỘC LẬP với l1_enabled. Đặt 0 để tắt gate (bit-for-bit như cũ).
        accept_accent_variants: chế độ accent "default" (caller map từ accent=="default"). Chấp nhận
            coda /r/ non-rhotic (Anh-Anh) — nuốt /r/ hoặc nguyên âm align lên /r/ → "ok" (ACCENT_VARIANT),
            KHÔNG trừ điểm. False (mặc định) = bit-for-bit như cũ. CHƯA phải union GB/US đầy đủ: các khác
            biệt còn lại đã được normalize_ipa() gộp; BATH (æ↔ɑ) cố ý không gộp (xem _score_deletion).

    Returns None nếu reference_phonemes rỗng.
    """
    if not reference_phonemes:
        return None

    predicted_phonemes = [s.phoneme for s in segments]
    predicted_conf = [s.confidence for s in segments]
    avg_confidence = (
        sum(predicted_conf) / len(predicted_conf) if predicted_conf else 0.0
    )
    ref_word, ref_is_onset, ref_stress, ref_reducible, ref_is_coda = _ref_metadata(
        reference_phonemes, reference_spans, reference_stress
    )
    # Nhấn-hiển-thị (dời về onset) song song reference — CHỈ để gắn PhonemePoint.display_stress.
    # Pad/cắt về đúng len(reference) như ref_stress; không tham gia bất kỳ tính toán điểm nào.
    n_ref = len(reference_phonemes)
    ref_display_stress: list[str | None] = (
        list(reference_display_stress) if reference_display_stress else [None] * n_ref
    )
    if len(ref_display_stress) < n_ref:
        ref_display_stress += [None] * (n_ref - len(ref_display_stress))
    # Map quyết định skip (theo chỉ số span chuẩn) → cờ per-phoneme + lý do per-span.
    ref_skipped, span_skip_reason = _resolve_skips(
        reference_phonemes, reference_spans, skips
    )

    # "Said nothing" (không có phoneme nào nhận diện được) → 0% (path rỗng).
    if not predicted_phonemes:
        path: list[tuple[int, int]] = []
        result: dict[int, tuple[PhonemePoint, float | None]] = {}
        insertion_count = 0
        raw_by_ref: dict[int, float] = {}
        empty_prediction = True
    else:
        path = _dtw_align(predicted_phonemes, reference_phonemes)
        result, insertion_count, raw_by_ref = _align_points(
            path, predicted_phonemes, predicted_conf, reference_phonemes,
            ref_word, ref_is_onset, ref_stress, ref_reducible, ref_skipped,
            ref_is_coda, confidence_knee, ref_display_stress,
            l1_enabled=l1_enabled, low_conf_floor=low_conf_floor,
            recognizer_noise_sim=recognizer_noise_sim,
            recognizer_noise_conf=recognizer_noise_conf,
            recognizer_noise_conf_vowel=recognizer_noise_conf_vowel,
            accept_accent_variants=accept_accent_variants,
        )
        empty_prediction = False

    # Bổ sung mọi reference index chưa được path chạm → deletion (hoặc skipped).
    for i in range(len(reference_phonemes)):
        if i in result:
            continue
        stress = ref_stress[i]
        disp = ref_display_stress[i]
        if ref_skipped[i]:
            result[i] = (PhonemePoint(symbol=reference_phonemes[i], status="skipped",
                                      stress=stress, display_stress=disp), None)
        else:
            point, pen, raw = _score_deletion(
                reference_phonemes[i], is_onset=ref_is_onset[i],
                is_coda=ref_is_coda[i], stress=stress, display_stress=disp,
                l1_enabled=l1_enabled,
                accept_accent_variants=accept_accent_variants,
            )
            result[i] = (point, pen)
            raw_by_ref[i] = raw

    # Tổng hợp: errors + counts + penalty (một nguồn → không lệch nhau).
    errors: list[PhonemeError] = []
    matches = substitutions = deletions = 0
    total_penalty = 0.0
    raw_penalty_sum = 0.0
    l1_adjusted_count = low_conf_neutralized_count = recognizer_noise_count = 0
    scored = 0
    point_by_ref: dict[int, PhonemePoint] = {}
    for i in range(len(reference_phonemes)):
        point, penalty = result[i]
        point_by_ref[i] = point
        if point.status == "ok":
            matches += 1
        elif point.status == "sub":
            substitutions += 1
            errors.append(PhonemeError(
                error_type=PhonemeErrorType.SUBSTITUTION,
                expected=reference_phonemes[i], predicted=point.heard,
                position=i, severity=point.severity or "medium", word=ref_word[i],
            ))
        elif point.status == "del":
            deletions += 1
            errors.append(PhonemeError(
                error_type=PhonemeErrorType.DELETION,
                expected=reference_phonemes[i], predicted=None,
                position=i, severity=point.severity or "medium", word=ref_word[i],
            ))
        # skipped → không phải lỗi
        if penalty is not None:
            total_penalty += penalty
            scored += 1
            # L1 metadata: penalty GỐC (trước L1/neutralization) để tính tỉ lệ giảm.
            raw_penalty_sum += raw_by_ref.get(i, penalty)
        if point.penalty_reason == PenaltyReason.L1_FINAL_DELETION.value:
            l1_adjusted_count += 1
        elif point.penalty_reason == PenaltyReason.LOW_CONFIDENCE_NEUTRALIZED.value:
            low_conf_neutralized_count += 1
        elif point.penalty_reason == PenaltyReason.RECOGNIZER_NOISE.value:
            recognizer_noise_count += 1

    if empty_prediction:
        accuracy = 0.0
    elif scored > 0:
        accuracy = max(0.0, 1.0 - total_penalty / scored)
    else:
        accuracy = 1.0  # mọi âm đều skip (ASR nghe nhầm cả) → không có gì để chấm

    words, words_truncated, words_total = _build_word_details(
        point_by_ref, reference_phonemes, reference_spans, max_words, span_skip_reason
    )

    # Telemetry (PR2/PR3-0) — DIAGNOSTIC ONLY, chỉ tính khi có sink (zero overhead khi tắt).
    if diagnostics_sink is not None:
        predicted_times = [(s.start, s.end) for s in segments]
        diagnostics_sink(build_word_diagnostics(
            path, predicted_phonemes, predicted_conf, reference_phonemes,
            reference_spans, result, span_skip_reason,
            predicted_times=predicted_times, word_windows=word_windows,
        ))

    # Sort errors by severity (high → medium → low) rồi cap.
    severity_order = {"high": 0, "medium": 1, "low": 2}
    errors.sort(key=lambda e: severity_order.get(e.severity, 3))

    l1_adjustment_ratio = (
        (raw_penalty_sum - total_penalty) / raw_penalty_sum
        if raw_penalty_sum > 0 else 0.0
    )

    logger.info(
        "Phoneme score: accuracy=%.2f | matches=%d | subs=%d | del=%d | ins=%d | "
        "skipped_words=%d | ref=%d | pred=%d | l1_adj=%d | low_conf_neut=%d | recog_noise=%d",
        accuracy, matches, substitutions, deletions, insertion_count,
        len(span_skip_reason), len(reference_phonemes), len(predicted_phonemes),
        l1_adjusted_count, low_conf_neutralized_count, recognizer_noise_count,
    )

    return PhonemeScore(
        overall_accuracy=round(accuracy, 4),
        substitution_count=substitutions,
        deletion_count=deletions,
        insertion_count=insertion_count,
        reference_count=len(reference_phonemes),
        predicted_count=len(predicted_phonemes),
        avg_confidence=round(avg_confidence, 4),
        errors=errors[:MAX_ERRORS_RETURNED],
        words=words,
        words_truncated=words_truncated,
        words_total=words_total,
        raw_penalty=raw_penalty_sum,
        adjusted_penalty=total_penalty,
        l1_adjusted_count=l1_adjusted_count,
        low_conf_neutralized_count=low_conf_neutralized_count,
        recognizer_noise_count=recognizer_noise_count,
        l1_adjustment_ratio=l1_adjustment_ratio,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: weighted accuracy (có tính severity)
# ──────────────────────────────────────────────────────────────────────────────

def weighted_accuracy(score: PhonemeScore) -> float:
    """Tính weighted accuracy: mỗi lỗi penalty theo severity.

    - high severity: penalty = 1.0
    - medium severity: penalty = 0.5
    - low severity: penalty = 0.25

    Formula: weighted = 1.0 - (weighted_penalty_sum / reference_count)
    """
    if score.reference_count == 0:
        return 0.0

    penalty_weights = {"high": 1.0, "medium": 0.5, "low": 0.25}
    total_penalty = sum(
        penalty_weights.get(e.severity, 0.5)
        for e in score.errors
    )
    weighted = 1.0 - (total_penalty / score.reference_count)
    return round(max(0.0, min(1.0, weighted)), 4)
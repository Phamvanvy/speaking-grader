"""DTW alignment + one-pass scoring của path (predicted ↔ reference phonemes).

_dtw_align dựng optimal path; _align_points đi 1 lượt qua path → PhonemePoint + penalty
cho mỗi reference index (qua phonemes_match / phoneme_similarity / confidence + skip_words).
"""

from __future__ import annotations

from typing import Final

from ..ipa import (
    FUNCTION_WORDS,
    is_nasal_coda_linking,
    is_real_error_substitution,
    is_vowel,
    normalize_ipa,
    phoneme_similarity,
    phonemes_match,
)
from ..l1_vietnamese import PenaltyReason
from ..models import PhonemePoint
from .constants import (
    PHONEME_G2P_UNCERTAIN_CAP,
    PHONEME_LOW_CONF_FLOOR,
    PHONEME_RECOGNIZER_NOISE_CONF,
    PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    PHONEME_RECOGNIZER_NOISE_SIM,
)
from .word_details import _score_deletion, _severity_from_penalty

# Xếp hạng để chọn status "tốt hơn" khi 1 reference index bị chạm nhiều lần.
# "skipped" (từ ASR nghe nhầm) coi như tốt nhất — không phải lỗi, loại khỏi điểm.
_STATUS_RANK: Final[dict[str, int]] = {"ok": 0, "skipped": 0, "sub": 1, "del": 2}
_SEVERITY_RANK: Final[dict[str, int]] = {"low": 0, "medium": 1, "high": 2}

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


def _links_into_vowel(
    reference: list[str], ref_word: list[str | None], ref_idx: int, word: str | None
) -> bool:
    """True nếu âm tại ref_idx nối sang nguyên âm ĐẦU của TỪ KẾ TIẾP (linking).

    Điều kiện nối âm liên-từ: phoneme reference ngay sau thuộc MỘT TỪ KHÁC và là nguyên âm
    (vd coda /n/ của "in" + onset /ɔ/ của "order"). Cùng từ → không tính là nối âm.
    """
    nxt = ref_idx + 1
    if nxt >= len(reference) or ref_word[nxt] == word:
        return False
    return is_vowel(reference[nxt])


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
    ref_g2p_uncertain: list[bool] | None = None,
    ref_r_droppable: list[bool] | None = None,
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
    if ref_g2p_uncertain is None:
        ref_g2p_uncertain = [False] * len(reference)
    if ref_r_droppable is None:
        # Fallback cho caller cũ: chỉ coda /r/ cuối từ (hành vi trước khi mở rộng).
        ref_r_droppable = [
            ref_is_coda[i] and normalize_ipa(reference[i]) == "r"
            for i in range(len(reference))
        ]
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
                and ref_r_droppable[ref_idx]
                and (is_vowel(pred_ph) or normalize_ipa(pred_ph) == "w")
            ):
                # Accent "default": /r/ non-prevocalic (coda âm tiết — cuối từ hoặc trước
                # phụ âm: car, mo(r)ning) của giọng non-rhotic — predicted là NGUYÊN ÂM
                # (r-coloring residue / schwa lệch align lên /r/) hoặc GLIDE /w/ (offglide
                # của ʊə/aʊə: "our" /aʊər/→/ɑːw/), KHÔNG phải lỗi. Phụ âm khác thay /r/
                # (l/j/n...) VẪN là lỗi thật; /r/ trước nguyên âm (red, very) không droppable.
                point = PhonemePoint(
                    symbol=ref_ph, status="ok", stress=stress, display_stress=disp,
                    penalty_reason=PenaltyReason.ACCENT_VARIANT.value,
                    penalty_adjustment=0.0,
                )
                penalty = 0.0
            elif (
                ref_is_coda[ref_idx]
                and (word or "").lower() in FUNCTION_WORDS
                and is_nasal_coda_linking(ref_ph, pred_ph)
                and _links_into_vowel(reference, ref_word, ref_idx, word)
            ):
                # Nối âm: coda MŨI của function word ("in", "on", "an"...) nối sang nguyên
                # âm đầu từ kế ("in order" /ɪn/+/ɔ/) hay bị wav2vec gán thành stop homorganic
                # (n→t/d). KHÔNG phải nuốt âm cũng KHÔNG phải lỗi người đọc → ok. Ràng buộc
                # function word + nguyên âm theo sau giữ "in" vs "it" vẫn là lỗi mọi ngữ cảnh khác.
                point = PhonemePoint(
                    symbol=ref_ph, status="ok", stress=stress, display_stress=disp,
                    penalty_reason=PenaltyReason.LINKING_VARIANT.value,
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
                # Stage 5 g2p-uncertain cap: IPA chuẩn của từ lấy từ eSpeak (OOV/tên
                # riêng) → reference tự nó là đoán → cap penalty về "low" (hidden-noise).
                if (
                    ref_g2p_uncertain[ref_idx]
                    and penalty > PHONEME_G2P_UNCERTAIN_CAP
                ):
                    penalty = PHONEME_G2P_UNCERTAIN_CAP
                    adjustment = penalty / base_sub if base_sub > 0 else 0.0
                    reason = PenaltyReason.G2P_UNCERTAIN.value
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
                g2p_uncertain=ref_g2p_uncertain[ref_idx],
                r_droppable=ref_r_droppable[ref_idx],
            )

        existing = result.get(ref_idx)
        if existing is None or _better_point(existing[0], point) is point:
            result[ref_idx] = (point, penalty)
            if iter_raw is not None:
                raw_by_ref[ref_idx] = iter_raw
            elif ref_idx in raw_by_ref:
                del raw_by_ref[ref_idx]  # point được thay = ok/skipped → bỏ raw cũ
    return result, insertion_count, raw_by_ref

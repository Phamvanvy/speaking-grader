"""DTW alignment + one-pass scoring của path (predicted ↔ reference phonemes).

_dtw_align dựng optimal path; _align_points đi 1 lượt qua path → PhonemePoint + penalty
cho mỗi reference index (qua phonemes_match / phoneme_similarity / confidence + skip_words).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Final

from ..diagnostics import DRIFT_WINDOW_PAD_SEC, is_within_word_window
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
from ..models import PhonemePoint, WordSpan
from .constants import (
    PHONEME_DRIFT_SUB_CAP,
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

# Gap cost cho indel (insertion/deletion) trong DTW. = chi phí substitution TỆ NHẤT
# (1 − sim ∈ [0,1]) nên del+ins (2·GAP = 2.0) KHÔNG BAO GIỜ rẻ hơn 1 substitution
# thật (≤1.0) → sub thật không bao giờ vỡ thành del+ins. Nhưng khi 1 âm reference bị
# NUỐT và âm của từ kế lấp vào chỗ đó, del(âm nuốt)+match(âm từ kế) THẮNG sub+del méo
# — sửa mis-attribution biên từ (vd non-rhotic "are you": /r/ nuốt, /j/ của "you" lấp
# vào; trước đây r→j sub oan + /j/ của "you" thành deletion).
_DTW_GAP_COST: Final[float] = 1.0


def _dtw_align(
    predicted: list[str],
    reference: list[str],
) -> list[tuple[int, int]]:
    """Global alignment (Needleman–Wunsch) predicted → reference phoneme sequences.

    Returns list of (pred_idx, ref_idx) pairs theo optimal path (-1 = gap).

    Sub cost = 1 − phoneme_similarity (cùng nguồn chân lý với scoring); indel = _DTW_GAP_COST
    CỐ ĐỊNH (KHÔNG phải local match_cost — công thức cũ cộng match_cost vào cả nước indel làm
    aligner ưu tiên substitution hơn deletion+match dù del+match đúng và rẻ hơn thật).
    Biên inf (góc neo hai đầu) giữ endpoint constraint như DTW cũ — không mở gap đầu/cuối.
    Backpointer lưu trong forward pass (thay vì suy lại từ cumulative cost) → tie-break rõ:
    ưu tiên diagonal (sub/match) khi hoà, giữ hành vi cũ.
    """
    if not predicted or not reference:
        return []

    n, m = len(predicted), len(reference)

    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    # move[i][j]: hướng đi tới ô này — 'D' diagonal (sub/match), 'U' insertion, 'L' deletion.
    move: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = cost[i - 1][j - 1] + (
                1.0 - phoneme_similarity(predicted[i - 1], reference[j - 1])
            )
            up = cost[i - 1][j] + _DTW_GAP_COST      # insertion (predicted i, no reference)
            left = cost[i][j - 1] + _DTW_GAP_COST    # deletion (reference j, no predicted)
            best = min(sub, up, left)
            cost[i][j] = best
            # Hoà → ưu tiên diagonal trước, rồi insertion (khớp thứ tự cũ).
            move[i][j] = "D" if best == sub else ("U" if best == up else "L")

    # Backtrace theo backpointer.
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            path.append((-1, j))  # deletion (biên trên)
        elif j == 0:
            i -= 1
            path.append((i, -1))  # insertion (biên trái)
        elif move[i][j] == "D":
            i -= 1
            j -= 1
            path.append((i, j))   # match or substitution
        elif move[i][j] == "U":
            i -= 1
            path.append((i, -1))  # insertion (predicted i, không có reference)
        else:
            j -= 1
            path.append((-1, j))  # deletion

    path.reverse()
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Boundary refinement (flag TOEIC_PHONEME_BOUNDARY_REFINE): sửa bleed cục bộ
# quanh ranh giới từ SAU DTW, TRƯỚC _align_points — không đụng cost DTW.
# ──────────────────────────────────────────────────────────────────────────────

# Số ref slot / pred segment xét mỗi bên quanh ranh giới từ.
_BOUNDARY_REFINE_SPAN: Final[int] = 2
_BOUNDARY_REFINE_EPS: Final[float] = 1e-9


def _refine_boundary_bleed(
    path: list[tuple[int, int]],
    predicted: list[str],
    reference: list[str],
    spans: list[WordSpan],
    ref_word: list[str | None],
    ref_is_onset: list[bool],
    ref_is_coda: list[bool],
    ref_stress: list[str | None],
    ref_reducible: list[bool],
    ref_skipped: list[bool],
    ref_r_droppable: list[bool],
    ref_g2p_uncertain: list[bool],
    *,
    accept_accent_variants: bool = False,
    l1_enabled: bool = False,
    predicted_times: list[tuple[float, float]] | None = None,
    word_windows: Mapping[int, tuple[float, float]] | None = None,
    word_windows_locked: Collection[int] | None = None,
    pad: float = DRIFT_WINDOW_PAD_SEC,
) -> tuple[list[tuple[int, int]], list[dict]]:
    """Sửa mis-attribution biên từ trên DTW path (case "our eyes" 2026-07-05/08).

    Bleed: pred `aʊ aɪ z z` vs ref our=`aʊ r` eyes=`aɪ z` → DTW gán pred aɪ vào
    slot /r/ của "our" (coda-r acceptance làm cặp này "ok") → "eyes" hiển thị
    /z z/ + sub aɪ→z oan. QUAN TRỌNG: path bleed và path đúng HOÀ cost DTW thô
    (sim(aɪ,r)=0, sim(z,aɪ)=0 → 1+1 = del(1)+match(0)+ins(1) = 2.0); bleed thắng
    chỉ vì tie-break ưu tiên diagonal trong _dtw_align. Vì vậy tiêu chí accept ở
    đây CỐ Ý dùng thang SCORING (insertion = 0, deletion droppable-/r/ accent =
    0 qua _score_deletion) chứ KHÔNG phải thang _DTW_GAP_COST — đổi về cost DTW
    thô sẽ tắt pass này âm thầm (pin bởi test flagship our/eyes).

    Mỗi cặp từ kề xét ±_BOUNDARY_REFINE_SPAN slot/segment quanh biên, hai hướng
    (trái ăn của phải, rồi mirror), tối đa 1 move/biên, pred đã move không move
    lại (chống oscillation). Move = re-pair (p,r)→(p,r') tại chỗ; cặp cũ của
    đích thành insertion; slot nguồn rơi khỏi path → vòng bổ sung deletion của
    compute_phoneme_score chấm bằng _score_deletion (statuses/penalty do
    _align_points tính lại trên path đã sửa — một nguồn chân lý duy nhất).

    Guards (theo thứ tự): đích phải phonemes_match THẬT với segment (chặn false
    positive mạnh nhất); không bao giờ đè cặp cost-0 ("ok"); mọi slot giữa nguồn
    và đích phải đang unpaired (multi-segment bleed ngoài scope); time-veto phụ —
    từ đích có cửa sổ Whisper đáng tin (không locked) thì segment phải nằm trong
    (không có window → cost tự quyết, degrade cho engine thiếu Whisper); accept
    khi tổng scoring cost cục bộ giảm CHẶT. Giới hạn đã biết: pair cost ở đây
    không nhân confidence/noise-gate như _align_points (không có sẵn tại path
    level) — bench 0-regression là backstop.

    Trả (path mới, list move đã áp để caller log). Không move → trả path gốc.
    """
    if not path or not spans or len(spans) < 2:
        return path, []

    n_ref = len(reference)
    # Trong NW path mỗi ref index xuất hiện ĐÚNG 1 lần (diagonal hoặc deletion),
    # mỗi pred index đúng 1 lần (diagonal hoặc insertion).
    diag: dict[int, int] = {}
    pos_of_ref: dict[int, int] = {}
    for idx, (p, r) in enumerate(path):
        if r >= 0:
            pos_of_ref[r] = idx
            if p >= 0:
                diag[r] = p

    def pair_cost(pred_idx: int, ref_idx: int) -> float:
        # Mirror thang chấp nhận của _align_points: match/coda-r/nasal-linking → 0.
        pred_ph = predicted[pred_idx]
        ref_ph = reference[ref_idx]
        word = ref_word[ref_idx]
        if phonemes_match(
            ref_ph, pred_ph, word=word, reducible=ref_reducible[ref_idx]
        ):
            return 0.0
        if (
            accept_accent_variants
            and ref_r_droppable[ref_idx]
            and (is_vowel(pred_ph) or normalize_ipa(pred_ph) == "w")
        ):
            return 0.0
        if (
            ref_is_coda[ref_idx]
            and (word or "").lower() in FUNCTION_WORDS
            and is_nasal_coda_linking(ref_ph, pred_ph)
            and _links_into_vowel(reference, ref_word, ref_idx, word)
        ):
            return 0.0
        return 1.0 - phoneme_similarity(pred_ph, ref_ph)

    def deletion_cost(ref_idx: int) -> float:
        _, pen, _ = _score_deletion(
            reference[ref_idx],
            is_onset=ref_is_onset[ref_idx], is_coda=ref_is_coda[ref_idx],
            stress=ref_stress[ref_idx], l1_enabled=l1_enabled,
            accept_accent_variants=accept_accent_variants,
            g2p_uncertain=ref_g2p_uncertain[ref_idx],
            r_droppable=ref_r_droppable[ref_idx],
        )
        return pen

    entries: list[tuple[int, int] | None] = list(path)
    moves: list[dict] = []
    frozen: set[int] = set()
    locked = frozenset(word_windows_locked or ())

    for k in range(len(spans) - 1):
        left, right = spans[k], spans[k + 1]
        l_lo, l_hi = left.start_idx, min(left.end_idx, n_ref)
        r_lo, r_hi = right.start_idx, min(right.end_idx, n_ref)
        if l_lo >= l_hi or r_lo >= r_hi:
            continue
        if any(ref_skipped[i] for i in range(l_lo, l_hi)) or any(
            ref_skipped[i] for i in range(r_lo, r_hi)
        ):
            continue
        # Pred index "biên": pred đầu tiên gán vào từ phải; fallback pred cuối
        # của từ trái + 1; cả hai từ đều trống → không có gì để xét.
        boundary_pred: int | None = None
        for i in range(r_lo, r_hi):
            if i in diag:
                boundary_pred = diag[i]
                break
        if boundary_pred is None:
            left_preds = [diag[i] for i in range(l_lo, l_hi) if i in diag]
            if not left_preds:
                continue
            boundary_pred = max(left_preds) + 1

        # Hướng A: từ TRÁI ăn segment của từ phải (our/eyes) — nguồn ở slot cuối
        # từ trái (gần biên trước), đích ở slot đầu từ phải. Hướng B: mirror.
        directions = (
            (range(l_hi - 1, max(l_hi - _BOUNDARY_REFINE_SPAN, l_lo) - 1, -1),
             range(r_lo, min(r_lo + _BOUNDARY_REFINE_SPAN, r_hi)),
             k + 1),
            (range(r_lo, min(r_lo + _BOUNDARY_REFINE_SPAN, r_hi)),
             range(l_hi - 1, max(l_hi - _BOUNDARY_REFINE_SPAN, l_lo) - 1, -1),
             k),
        )
        applied = False
        for sources, dests, dest_span in directions:
            if applied:
                break
            for r in sources:
                if applied:
                    break
                p = diag.get(r)
                if (
                    p is None or p in frozen
                    or abs(p - boundary_pred) > _BOUNDARY_REFINE_SPAN
                ):
                    continue
                src_cost = pair_cost(p, r)
                for rp in dests:
                    if rp == r:
                        continue
                    # Guard cứng: đích phải khớp THẬT với segment.
                    if not phonemes_match(
                        reference[rp], predicted[p],
                        word=ref_word[rp], reducible=ref_reducible[rp],
                    ):
                        continue
                    q = diag.get(rp)
                    if q is not None and pair_cost(q, rp) <= 0.0:
                        continue  # không bao giờ đè cặp "ok"
                    lo, hi = (r, rp) if r < rp else (rp, r)
                    if any(x in diag for x in range(lo + 1, hi)):
                        continue  # multi-segment bleed ngoài scope
                    # Time-veto phụ: từ đích có window đáng tin → segment phải
                    # nằm trong; locked/thiếu window → cost tự quyết.
                    window_ok: bool | None = None
                    if (
                        word_windows
                        and predicted_times is not None
                        and p < len(predicted_times)
                        and dest_span not in locked
                    ):
                        win = word_windows.get(dest_span)
                        if win is not None:
                            ps, pe = predicted_times[p]
                            window_ok = is_within_word_window(ps, pe, win, pad)
                            if not window_ok:
                                continue
                    # Accept = giảm CHẶT tổng scoring cost cục bộ. after chỉ còn
                    # deletion slot nguồn: cặp đích mới = 0 (match theo guard),
                    # pred cũ của đích thành insertion = 0 (scoring không phạt).
                    dest_before = (
                        pair_cost(q, rp) if q is not None else deletion_cost(rp)
                    )
                    before = src_cost + dest_before
                    after = deletion_cost(r)
                    if after + _BOUNDARY_REFINE_EPS >= before:
                        continue
                    # Áp move: (p,r)→(p,rp) TẠI CHỖ (giữ nguyên thứ tự pred);
                    # cặp cũ của đích thành insertion; entry deletion (-1,rp)
                    # (đích đang unmatched) bị bỏ; slot r rơi khỏi path.
                    entries[pos_of_ref[r]] = (p, rp)
                    if q is not None:
                        entries[pos_of_ref[rp]] = (q, -1)
                    else:
                        entries[pos_of_ref[rp]] = None
                    pos_of_ref[rp] = pos_of_ref.pop(r)
                    del diag[r]
                    diag[rp] = p
                    frozen.add(p)
                    moves.append({
                        "left_word": left.word, "right_word": right.word,
                        "pred_idx": p, "pred_ph": predicted[p],
                        "pred_time": (
                            predicted_times[p]
                            if predicted_times is not None
                            and p < len(predicted_times) else None
                        ),
                        "from_ref": r, "from_ph": reference[r],
                        "to_ref": rp, "to_ph": reference[rp],
                        "displaced_pred_idx": q,
                        "cost_before": round(before, 4),
                        "cost_after": round(after, 4),
                        "window_ok": window_ok,
                    })
                    applied = True
                    break

    if not moves:
        return path, []
    return [e for e in entries if e is not None], moves


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


# ──────────────────────────────────────────────────────────────────────────────
# Drift cap (Track B): sub có predicted segment NGOÀI window Whisper của từ
# ──────────────────────────────────────────────────────────────────────────────

def _apply_drift_cap(
    result: dict[int, tuple[PhonemePoint, float | None]],
    raw_by_ref: dict[int, float],
    path: list[tuple[int, int]],
    reference: list[str],
    spans: list[WordSpan],
    predicted_times: list[tuple[float, float]],
    word_windows: Mapping[int, tuple[float, float]] | None,
    *,
    pad: float = DRIFT_WINDOW_PAD_SEC,
    cap: float = PHONEME_DRIFT_SUB_CAP,
) -> None:
    """Cap penalty các sub NGHI DRIFT — predicted segment nằm ngoài cửa sổ Whisper của
    chính từ đó (±pad) → khả năng DTW "mượn" âm từ kế bên chứ không phải lỗi phát âm.
    Cùng evidence với telemetry `sub_outside_window` (is_within_word_window dùng chung),
    nay promote vào scoring (opt-in qua drift_cap_enabled). Mutate `result` in-place.

    Safe no-op khi word_windows None/rỗng (giữ contract hôm nay). Precedence: chỉ HẠ
    penalty (`penalty > cap`) — sub đã penalty 0 (connected_speech/recognizer_noise/…)
    hoặc đã cap 0.2 (g2p_uncertain) giữ nguyên reason cũ. Chỉ đụng status "sub" —
    disjoint với coverage gate (chỉ "del").
    """
    if not word_windows or not spans or not path:
        return
    n = len(reference)
    ref_to_span = [-1] * n
    for k, s in enumerate(spans):
        for i in range(s.start_idx, min(s.end_idx, n)):
            ref_to_span[i] = k
    # Cặp diagonal cuối cùng cho mỗi ref_idx (khớp pred_for_ref của diagnostics).
    pred_for_ref: dict[int, int] = {}
    for pred_idx, ref_idx in path:
        if ref_idx >= 0 and pred_idx >= 0:
            pred_for_ref[ref_idx] = pred_idx

    for ref_idx, (point, penalty) in result.items():
        if point.status != "sub" or penalty is None or penalty <= cap:
            continue
        if not (0 <= ref_idx < n):
            continue
        k = ref_to_span[ref_idx]
        if k < 0:
            continue
        win = word_windows.get(k)
        pi = pred_for_ref.get(ref_idx)
        if win is None or pi is None or pi >= len(predicted_times):
            continue
        ps, pe = predicted_times[pi]
        if is_within_word_window(ps, pe, win, pad):
            continue  # segment trong từ → hallucination/lỗi thật, KHÔNG cap
        raw = raw_by_ref.get(ref_idx, penalty)
        result[ref_idx] = (
            PhonemePoint(
                symbol=point.symbol, status="sub", heard=point.heard,
                severity=_severity_from_penalty(cap),
                stress=point.stress, display_stress=point.display_stress,
                penalty_reason=PenaltyReason.DRIFT_SUSPECTED.value,
                penalty_adjustment=round(cap / raw, 4) if raw > 0 else 0.0,
            ),
            cap,
        )

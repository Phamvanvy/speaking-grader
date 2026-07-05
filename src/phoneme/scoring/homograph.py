"""Multi-reference homograph selection — chọn entry CMUdict khớp acoustic nhất.

Vấn đề (case "project" 2026-07-05): `_rank_cmudict_entries` là context-free nên mỗi
từ đa-entry bị khóa cứng vào 1 phát âm — "project" luôn so với dạng ĐỘNG TỪ
/prədʒekt/, người đọc danh từ /prɑːdʒekt/ bị 2 false sub. Thống kê
(scripts/analyze_homographs.py): 8,447 từ đa-entry, 6,068 từ lệch thật sau
normalize (chọn sai = false sub/del), 509 mang chữ ký noun/verb stress-shift.

Cách sửa Ở ĐÂY (thay vì override từng từ hoặc POS tagger): với mỗi từ tham chiếu
đa-entry CÓ cửa sổ Whisper, align các phoneme wav2vec nghe được trong cửa sổ với
TỪNG entry CMUdict (Needleman–Wunsch, cost = 1 − phoneme_similarity — cùng nguồn
chân lý với DTW scoring) và chọn entry cost thấp nhất. Triết lý: người nói khớp
BẤT KỲ phát âm từ điển nào là đúng — bao phủ cả homograph không stress-shift
(read/live/close) mà không cần POS.

Bất biến an toàn:
  - CHỈ đổi lát reference của chính từ đó; số spans/chỉ số span KHÔNG đổi →
    skips/word_windows/word_probs (keyed theo span index) vẫn đúng.
  - Chỉ swap khi cost entry khác THẤP HƠN HẲN entry mặc định (strict, epsilon) —
    hòa giữ mặc định (ranked entry) → deterministic, không fuzzy.
  - Từ không có window / không có segment trong window / source != "cmudict" /
    bị skip → giữ nguyên.
  - Flag default OFF (TOEIC_PHONEME_MULTIREF) → bit-for-bit như cũ.

Lỗi thật KHÔNG bị nuốt: học viên đọc sai thì mọi entry đều lệch, chọn entry gần
nhất chỉ đổi cách mô tả lỗi (so với biến thể gần nhất) chứ không xoá lỗi.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from ..diagnostics import is_within_word_window
from ..ipa import phoneme_similarity
from ..ipa.g2p import (
    _arpabet_tokens_to_ipa_stress,
    _finalize_stress,
    _get_cmudict,
    place_stress_at_onset,
)
from ..models import PhonemeSegment, WordSpan
from ..reliability import SkipDecision

logger = logging.getLogger("toeic.phoneme.homograph")

# Gap cost cho Needleman–Wunsch (thêm/bớt 1 âm). Cùng thang với sub cost tối đa
# (1 − similarity ∈ [0,1]) để indel không rẻ hơn sub tệ nhất.
_GAP_COST: float = 1.0
# Swap chỉ khi cost thấp hơn HẲN mức này — chặn dao động float, hòa → giữ default.
_MIN_IMPROVEMENT: float = 1e-6
# Ký tự bao quanh từ cần strip — KHỚP word_to_ipa_with_stress_source (g2p key).
_WORD_STRIP_CHARS = ".,;:!?\"'()[]{}"


def _alignment_cost(predicted: list[str], reference: list[str]) -> float:
    """FITTING alignment cost: reference khớp TRỌN VẸN vào 1 đoạn của predicted.

    sub = 1 − phoneme_similarity (cùng nguồn chân lý với DTW scoring), gap =
    _GAP_COST. KHÁC global NW ở chỗ predicted được bỏ TỰ DO ở hai mép (prefix/
    suffix) — window Whisper hay lem âm của từ kề (±pad), global alignment sẽ để
    entry DÀI hơn "hấp thụ" âm lem thành sub rẻ thay vì trả gap → thắng oan
    (bench 2026-07-05: what→/hwʌt/ vì /ð/ lem khớp /h/). Với fitting, âm lem ở
    mép miễn phí cho MỌI candidate như nhau; entry dài vẫn phải trả đủ cho âm
    reference không có bằng chứng. Gap GIỮA predicted vẫn tính (âm thừa giữa từ
    là bằng chứng thật). Nhỏ (từ ~7 âm × window ~10 segment) nên O(n·m) không
    đáng kể.
    """
    n, m = len(predicted), len(reference)
    prev = [j * _GAP_COST for j in range(m + 1)]  # i=0: thiếu reference phải trả
    best = prev[m]
    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)  # dp[i][0] = 0: bỏ predicted prefix miễn phí
        for j in range(1, m + 1):
            sub = prev[j - 1] + (
                1.0 - phoneme_similarity(predicted[i - 1], reference[j - 1])
            )
            cur[j] = min(sub, prev[j] + _GAP_COST, cur[j - 1] + _GAP_COST)
        prev = cur
        best = min(best, cur[m])  # kết thúc tại i bất kỳ: bỏ suffix miễn phí
    return best


def _candidate_entries(word: str) -> list[list[str]]:
    """Mọi entry CMUdict của từ (đã strip như g2p key); [] nếu <2 entry."""
    key = word.lower().strip(_WORD_STRIP_CHARS)
    entries = _get_cmudict().get(key) if key else None
    return entries if entries and len(entries) >= 2 else []


def select_homograph_references(
    reference_phonemes: list[str],
    reference_spans: list[WordSpan],
    reference_stress: list[str | None] | None,
    reference_display_stress: list[str | None] | None,
    segments: list[PhonemeSegment],
    word_windows: Mapping[int, tuple[float, float]],
    skips: Mapping[int, SkipDecision] | None = None,
) -> tuple[
    list[str], list[WordSpan], list[str | None] | None, list[str | None] | None
]:
    """Chọn lại entry phát âm cho từ đa-entry theo acoustic best-match.

    Trả về (phonemes, spans, stress, display_stress) MỚI — bản gốc không bị sửa.
    Không có swap nào → trả về đúng 4 object đầu vào (caller so sánh identity được).

    Chỉ số span giữ nguyên 1-1 với đầu vào (chỉ start_idx/end_idx dịch theo độ dài
    lát mới) nên skips/word_windows/word_probs keyed theo span index vẫn đúng.
    """
    # {span index: (symbols, stresses)} — entry thay thế thắng entry mặc định.
    swaps: dict[int, tuple[list[str], list[str | None]]] = {}
    for k, span in enumerate(reference_spans):
        if span.source != "cmudict" or (skips and k in skips):
            continue
        window = word_windows.get(k)
        if window is None:
            continue
        entries = _candidate_entries(span.word)
        if not entries:
            continue
        predicted = [
            s.phoneme for s in segments
            if is_within_word_window(s.start, s.end, window)
        ]
        if not predicted:
            continue
        current = reference_phonemes[span.start_idx:span.end_idx]
        best_cost = _alignment_cost(predicted, current)
        default_cost = best_cost
        best: tuple[list[str], list[str | None]] | None = None
        for entry in entries:
            symbols, stresses, syllables = _arpabet_tokens_to_ipa_stress(entry)
            symbols, stresses = _finalize_stress(symbols, stresses, syllables)
            if not symbols or symbols == current:
                continue
            cost = _alignment_cost(predicted, symbols)
            # Strict: thắng HẲN mức đang giữ mới swap; hòa → giữ (deterministic).
            if cost < best_cost - _MIN_IMPROVEMENT:
                best_cost = cost
                best = (symbols, stresses)
        if best is not None:
            swaps[k] = best
            logger.info(
                "Homograph swap | word=%r span=%d /%s/ → /%s/ (cost %.3f → %.3f)",
                span.word, k, "".join(current), "".join(best[0]),
                default_cost, best_cost,
            )

    if not swaps:
        return (reference_phonemes, reference_spans,
                reference_stress, reference_display_stress)

    # ── Rebuild các list phẳng, dịch start/end_idx — thứ tự span KHÔNG đổi ──────
    has_stress = reference_stress is not None
    has_disp = reference_display_stress is not None
    new_phonemes: list[str] = []
    new_spans: list[WordSpan] = []
    new_stress: list[str | None] = []
    new_disp: list[str | None] = []
    for k, span in enumerate(reference_spans):
        if k in swaps:
            symbols, stresses = swaps[k]
            disp = place_stress_at_onset(symbols, stresses)
        else:
            symbols = reference_phonemes[span.start_idx:span.end_idx]
            stresses = (
                reference_stress[span.start_idx:span.end_idx] if has_stress
                else [None] * len(symbols)
            )
            disp = (
                reference_display_stress[span.start_idx:span.end_idx] if has_disp
                else [None] * len(symbols)
            )
        start = len(new_phonemes)
        new_phonemes.extend(symbols)
        new_stress.extend(stresses)
        new_disp.extend(disp)
        new_spans.append(
            span._replace(start_idx=start, end_idx=len(new_phonemes))
        )
    return (
        new_phonemes, new_spans,
        new_stress if has_stress else None,
        new_disp if has_disp else None,
    )

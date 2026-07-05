"""Tests cho multi-reference homograph selection (scoring/homograph.py).

Ba lớp:
  - _alignment_cost: identical = 0, sub tính theo 1 − similarity, gap = 1.0/âm.
  - select_homograph_references: case "project" (nói danh từ, ranker chọn động từ)
    → swap sang entry danh từ; giữ nguyên khi khớp default / không window / skip /
    source espeak; bất biến span (số span không đổi, offset liền mạch, từ không
    swap giữ nguyên lát phonemes/stress); determinism.
  - compute_phoneme_score end-to-end: flag OFF = bit-for-bit như cũ; flag ON sửa
    false sub của "project" mà KHÔNG nuốt lỗi thật ở từ khác.
"""

from src.phoneme.ipa import text_to_ipa_sequence_with_spans
from src.phoneme.models import PhonemeSegment
from src.phoneme.scoring import compute_phoneme_score
from src.phoneme.scoring.homograph import (
    _alignment_cost,
    select_homograph_references,
)


def _segments(symbols, start=0.0, dur=0.05, gap=0.01, conf=0.9):
    """Dãy PhonemeSegment liên tiếp từ list symbol (thời gian tổng hợp)."""
    out, t = [], start
    for p in symbols:
        out.append(PhonemeSegment(phoneme=p, start=t, end=t + dur, confidence=conf))
        t += dur + gap
    return out


def _reference(text):
    phonemes, spans, stress, disp = text_to_ipa_sequence_with_spans(text)
    return phonemes, spans, stress, disp


def _window_for(segs):
    return (segs[0].start, segs[-1].end)


# ── _alignment_cost ───────────────────────────────────────────────────────────

def test_alignment_cost_identical_is_zero():
    assert _alignment_cost(["p", "r", "ə"], ["p", "r", "ə"]) == 0.0


def test_alignment_cost_orders_candidates():
    # Nghe /prɑːdʒɪkt/: entry danh từ /prɑːdʒekt/ phải rẻ hơn động từ /prədʒekt/.
    heard = ["p", "r", "ɑː", "dʒ", "ɪ", "k", "t"]
    noun = ["p", "r", "ɑː", "dʒ", "e", "k", "t"]
    verb = ["p", "r", "ə", "dʒ", "e", "k", "t"]
    assert _alignment_cost(heard, noun) < _alignment_cost(heard, verb)


def test_alignment_cost_gap():
    # Reference thiếu bằng chứng = 1 gap/âm — entry dài không được "miễn phí".
    assert _alignment_cost(["p", "r"], ["p", "r", "t"]) == 1.0
    # Predicted thừa ở MÉP (bleed từ kề) = miễn phí (fitting alignment)...
    assert _alignment_cost(["p", "r", "t"], ["p", "r"]) == 0.0
    assert _alignment_cost(["k", "p", "r"], ["p", "r"]) == 0.0
    # ...nhưng âm thừa GIỮA từ vẫn trả gap (bằng chứng thật).
    assert _alignment_cost(["p", "x", "r"], ["p", "r"]) == 1.0


def test_alignment_cost_bleed_does_not_favor_longer_entry():
    """Bench 2026-07-05: /ð/ lem trước "what" làm global NW cho /hwʌt/ thắng oan
    (ð hấp thụ thành sub với h). Fitting alignment: bleed ở mép miễn phí cho cả
    hai, /hwʌt/ vẫn phải trả cho /h/ không bằng chứng → /wʌt/ thắng."""
    heard = ["ð", "w", "ʌ", "t"]  # window lem /ð/ của từ trước
    what_plain = ["w", "ʌ", "t"]
    what_aspirated = ["h", "w", "ʌ", "t"]
    assert _alignment_cost(heard, what_plain) < _alignment_cost(
        heard, what_aspirated)


# ── select_homograph_references ───────────────────────────────────────────────

def test_project_swaps_to_noun_entry():
    """Case gốc: nói danh từ /prɑːdʒɪkt/, ranker chọn động từ /prədʒekt/ → swap."""
    phonemes, spans, stress, disp = _reference("project")
    assert phonemes == ["p", "r", "ə", "dʒ", "e", "k", "t"]  # guard: ranker = verb
    segs = _segments(["p", "ɹ", "ɑː", "dʒ", "ɪ", "k", "t"])
    new_ph, new_spans, new_stress, new_disp = select_homograph_references(
        phonemes, spans, stress, disp, segs, {0: _window_for(segs)},
    )
    assert new_ph == ["p", "r", "ɑː", "dʒ", "e", "k", "t"]  # entry danh từ
    assert len(new_spans) == len(spans)
    assert (new_spans[0].start_idx, new_spans[0].end_idx) == (0, 7)
    assert len(new_stress) == len(new_ph) == len(new_disp)
    # Entry danh từ AA1: primary stress phải nằm ở âm tiết ĐẦU (trên/quanh ɑː).
    assert new_stress[2] == "primary"


def test_no_swap_when_speaker_matches_default():
    """Nói đúng dạng động từ /prədʒekt/ → giữ nguyên reference (identity)."""
    phonemes, spans, stress, disp = _reference("project")
    segs = _segments(["p", "ɹ", "ə", "dʒ", "e", "k", "t"])
    result = select_homograph_references(
        phonemes, spans, stress, disp, segs, {0: _window_for(segs)},
    )
    assert result[0] is phonemes  # không swap → trả đúng object đầu vào


def test_no_swap_without_window_or_segments():
    phonemes, spans, stress, disp = _reference("project")
    segs = _segments(["p", "ɹ", "ɑː", "dʒ", "ɪ", "k", "t"])
    # Không có window cho từ.
    assert select_homograph_references(
        phonemes, spans, stress, disp, segs, {},
    )[0] is phonemes
    # Window nằm ngoài mọi segment (wav2vec im lặng trong từ).
    assert select_homograph_references(
        phonemes, spans, stress, disp, segs, {0: (99.0, 99.5)},
    )[0] is phonemes


def test_no_swap_for_skipped_word():
    from src.phoneme.reliability import SkipDecision, SkipReason

    phonemes, spans, stress, disp = _reference("project")
    segs = _segments(["p", "ɹ", "ɑː", "dʒ", "ɪ", "k", "t"])
    result = select_homograph_references(
        phonemes, spans, stress, disp, segs, {0: _window_for(segs)},
        skips={0: SkipDecision(0, SkipReason.ASR_LOW_CONFIDENCE)},
    )
    assert result[0] is phonemes


def test_neighbor_words_untouched_and_offsets_contiguous():
    """Swap 1 từ giữa câu: từ khác giữ nguyên lát, offset span liền mạch."""
    text = "the project failed"
    phonemes, spans, stress, disp = _reference(text)
    k = next(i for i, s in enumerate(spans) if s.word == "project")
    segs = _segments(["p", "ɹ", "ɑː", "dʒ", "ɪ", "k", "t"], start=1.0)
    new_ph, new_spans, new_stress, new_disp = select_homograph_references(
        phonemes, spans, stress, disp, segs, {k: _window_for(segs)},
    )
    assert new_ph[new_spans[k].start_idx:new_spans[k].end_idx] == \
        ["p", "r", "ɑː", "dʒ", "e", "k", "t"]
    # Offset liền mạch + từ không swap giữ nguyên phonemes/stress.
    pos = 0
    for old, new in zip(spans, new_spans):
        assert new.start_idx == pos
        pos = new.end_idx
        assert new.word == old.word and new.source == old.source
        if old.word != "project":
            assert new_ph[new.start_idx:new.end_idx] == \
                phonemes[old.start_idx:old.end_idx]
            assert new_stress[new.start_idx:new.end_idx] == \
                stress[old.start_idx:old.end_idx]
    assert pos == len(new_ph)


def test_deterministic():
    phonemes, spans, stress, disp = _reference("the project failed")
    k = next(i for i, s in enumerate(spans) if s.word == "project")
    segs = _segments(["p", "ɹ", "ɑː", "dʒ", "ɪ", "k", "t"], start=1.0)
    runs = [
        select_homograph_references(
            phonemes, spans, stress, disp, segs, {k: _window_for(segs)},
        )[0]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# ── compute_phoneme_score end-to-end ──────────────────────────────────────────

def _score(text, segs, windows, enabled):
    phonemes, spans, stress, disp = _reference(text)
    return compute_phoneme_score(
        segs, phonemes, spans, stress,
        reference_display_stress=disp,
        word_windows=windows,
        homograph_selection_enabled=enabled,
    )


def test_flag_off_keeps_false_subs_flag_on_fixes_them():
    """Case "project": OFF giữ 2 false sub (ə→ɑː, e→ɪ...); ON hết sub nguyên âm 1.

    LƯU Ý: e→ɪ vẫn còn cả khi ON (CMUdict không có biến thể /-dʒɪkt/) — đúng
    scope: multiref sửa CHỌN ENTRY, không sửa pronunciation variants (task sau).
    """
    segs = _segments(["p", "ɹ", "ɑː", "dʒ", "ɪ", "k", "t"])
    windows = {0: _window_for(segs)}

    off = _score("project", segs, windows, enabled=False)
    on = _score("project", segs, windows, enabled=True)

    off_pairs = {(e.expected, e.predicted) for e in off.errors}
    on_pairs = {(e.expected, e.predicted) for e in on.errors}
    assert ("ə", "ɑː") in off_pairs          # false sub do entry động từ
    assert ("ə", "ɑː") not in on_pairs       # multiref chọn entry danh từ
    assert on.substitution_count < off.substitution_count
    assert on.overall_accuracy > off.overall_accuracy


def test_flag_on_does_not_hide_real_errors():
    """Đọc sai thật (project → /pɹɑːʒæk/, thiếu âm + sai nguyên âm): mọi entry đều
    lệch → vẫn phải có lỗi, không entry nào "nuốt" được."""
    segs = _segments(["p", "ɹ", "ɑː", "ʒ", "æ", "k"])
    windows = {0: _window_for(segs)}
    on = _score("project", segs, windows, enabled=True)
    assert on.substitution_count + on.deletion_count >= 2
    assert on.overall_accuracy < 0.85


def test_flag_off_bit_for_bit_on_non_homograph_text():
    """Text không có từ đa-entry hưởng lợi: ON và OFF cho cùng kết quả."""
    text = "hello"
    phonemes, *_ = _reference(text)
    segs = _segments(phonemes)
    windows = {0: _window_for(segs)}
    off = _score(text, segs, windows, enabled=False)
    on = _score(text, segs, windows, enabled=True)
    assert off.overall_accuracy == on.overall_accuracy
    assert off.substitution_count == on.substitution_count

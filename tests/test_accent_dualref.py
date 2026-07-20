"""Unit test cho accent dual-reference (BATH-split UK/US).

Kiểm tra (1) biến đổi uk_variant thuần logic, (2) cơ chế fitting trong
select_homograph_references thực sự swap sang reference UK khi audio khớp UK và GIỮ
NGUYÊN khi tắt cờ / từ ngoài BATH set (learner guard) — không cần model/audio thật."""
from __future__ import annotations

from src.phoneme.models import PhonemeSegment, WordSpan
from src.phoneme.scoring.accent_variant import uk_variant
from src.phoneme.scoring.homograph import select_homograph_references


# ── uk_variant (thuần logic) ──────────────────────────────────────────────────

def test_bath_word_swaps_trap_to_bath_vowel():
    got = uk_variant(["d", "æ", "n", "s"], [None, "primary", None, None], "dance")
    assert got is not None
    assert got[0] == ["d", "ɑː", "n", "s"]
    assert got[1] == [None, "primary", None, None]  # trọng âm giữ nguyên


def test_non_bath_word_returns_none():
    # "cat" là TRAP (không thuộc BATH) → không đổi (guard: không nới oan æ ở từ khác).
    assert uk_variant(["k", "æ", "t"], [None, "primary", None], "cat") is None


def test_bath_word_without_trap_returns_none():
    # từ trong set nhưng lát reference không có /æ/ → None (không tạo candidate thừa).
    assert uk_variant(["p", "ɑː", "θ"], [None, "primary", None], "path") is None


def test_word_strip_and_case_insensitive():
    assert uk_variant(["k", "l", "æ", "s"], [None, None, "primary", None], "Class,") is not None


# ── fitting trong select_homograph_references ────────────────────────────────

def _dance_span():
    ref = ["d", "æ", "n", "s"]
    spans = [WordSpan(word="dance", start_idx=0, end_idx=4, source="cmudict")]
    stress = [None, "primary", None, None]
    return ref, spans, stress


def _segs(vowel: str):
    return [
        PhonemeSegment("d", 0.0, 0.1, 0.9),
        PhonemeSegment(vowel, 0.1, 0.3, 0.9),
        PhonemeSegment("n", 0.3, 0.4, 0.9),
        PhonemeSegment("s", 0.4, 0.5, 0.9),
    ]


def test_dualref_swaps_to_uk_when_audio_is_uk():
    ref, spans, stress = _dance_span()
    out_ph, *_ = select_homograph_references(
        ref, spans, stress, stress, _segs("ɑː"), {0: (0.0, 0.5)},
        homograph_enabled=False, accent_dualref=True,
    )
    assert out_ph == ["d", "ɑː", "n", "s"]  # đã swap sang UK


def test_dualref_off_keeps_us_reference():
    ref, spans, stress = _dance_span()
    out_ph, *_ = select_homograph_references(
        ref, spans, stress, stress, _segs("ɑː"), {0: (0.0, 0.5)},
        homograph_enabled=False, accent_dualref=False,
    )
    assert out_ph == ["d", "æ", "n", "s"]  # cờ tắt = bit-for-bit như cũ


def test_dualref_keeps_us_reference_when_audio_is_us():
    # Học viên đọc giọng Mỹ /dæns/ → US khớp hoàn hảo, KHÔNG swap sang UK.
    ref, spans, stress = _dance_span()
    out_ph, *_ = select_homograph_references(
        ref, spans, stress, stress, _segs("æ"), {0: (0.0, 0.5)},
        homograph_enabled=False, accent_dualref=True,
    )
    assert out_ph == ["d", "æ", "n", "s"]


def test_dualref_no_swap_for_control_word():
    # "cat" ngoài BATH set → không có UK candidate → không swap dù nghe /kɑːt/
    # (lỗi thật của học viên phải giữ nguyên, không được nới).
    ref = ["k", "æ", "t"]
    spans = [WordSpan(word="cat", start_idx=0, end_idx=3, source="cmudict")]
    stress = [None, "primary", None]
    segs = [PhonemeSegment("k", 0, 0.1, 0.9), PhonemeSegment("ɑː", 0.1, 0.3, 0.9),
            PhonemeSegment("t", 0.3, 0.4, 0.9)]
    out_ph, *_ = select_homograph_references(
        ref, spans, stress, stress, segs, {0: (0.0, 0.4)},
        homograph_enabled=False, accent_dualref=True,
    )
    assert out_ph == ["k", "æ", "t"]

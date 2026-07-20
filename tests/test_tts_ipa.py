"""Unit test cho nhánh đọc-IPA của TTS (src/tts.py:_ipa_to_phoneme_tokens).

Kiểm tra lớp chuẩn hoá IPA-hiển-thị → token phoneme espeak (thuần logic, dùng bảng
phoneme GIẢ nên KHÔNG cần nạp Piper/voice). Các quyết định ở đây được bench âm học
scripts/bench_tts_ipa.py chốt (xem CACHE_VERSION v6 trong src/tts.py)."""
from __future__ import annotations

from src.tts import _ipa_to_phoneme_tokens

# Bảng phoneme giả: mọi ký hiệu espeak (1 codepoint) + dấu nhấn + '.' đều hợp lệ.
_FAKE_MAP = {
    ch: [i]
    for i, ch in enumerate(
        list("abdefghijklmnopstuvwzæðŋɐɑɒɔəɘɛɜɡɪɹʃʊʌʒθ") + ["ˈ", "ˌ", "ː", "."]
    )
}


def _toks(ipa: str) -> list[str]:
    kept, _dropped = _ipa_to_phoneme_tokens(ipa, _FAKE_MAP)
    return kept


def test_r_maps_to_approximant():
    # 'r' hiển thị (ARPABET R) → 'ɹ' (espeak 'r' là rung lưỡi, sai âm tiếng Anh).
    assert _toks("stɔːr") == ["s", "t", "ɔ", "ː", "ɹ", "."]


def test_lone_e_becomes_dress_but_ei_kept():
    # 'e' đơn = DRESS /ɛ/; 'eɪ' = FACE giữ 'e'.
    assert _toks("red") == ["ɹ", "ɛ", "d", "."]          # read (quá khứ)
    assert _toks("beɪk")[:3] == ["b", "e", "ɪ"]          # bake: eɪ giữ nguyên


def test_stress_moves_to_nucleus():
    # ˈ ở onset ('ˈdʒeriː') dời về ngay TRƯỚC nguyên âm 'ɛ'.
    assert _toks("ˈdʒeriː") == ["d", "ʒ", "ˈ", "ɛ", "ɹ", "i", "ː", "."]


def test_secondary_stress_repositioned():
    # company 'ˈkʌmpəˌniː': ˈ→trước ʌ, ˌ→trước i.
    assert _toks("ˈkʌmpəˌniː") == [
        "k", "ˈ", "ʌ", "m", "p", "ə", "n", "ˌ", "i", "ː", "."
    ]


def test_terminal_anchor_appended_once():
    toks = _toks("ðə")
    assert toks[-1] == "."
    assert toks.count(".") == 1


def test_slashes_and_spaces_dropped():
    assert _toks("/stɔːr/") == ["s", "t", "ɔ", "ː", "ɹ", "."]


def test_empty_or_unmappable_yields_no_phonemes():
    # Rỗng / chỉ ký tự ngăn cách → không token thật (guard synthesize dựa vào đây).
    assert _ipa_to_phoneme_tokens("", _FAKE_MAP) == ([], [])
    assert _ipa_to_phoneme_tokens("///", _FAKE_MAP) == ([], [])
    # Dấu nhấn KHÔNG có nguyên âm theo sau → không tính là đọc được gì (không có '.').
    kept, _ = _ipa_to_phoneme_tokens("ˈ", _FAKE_MAP)
    assert kept == []


def test_unknown_symbol_reported_dropped():
    # Ký hiệu ngoài bảng (vd 'ʔ' glottal) → bị bỏ + báo trong dropped, không crash.
    kept, dropped = _ipa_to_phoneme_tokens("ʔə", _FAKE_MAP)
    assert "ə" in kept and kept[-1] == "."
    assert "ʔ" in dropped

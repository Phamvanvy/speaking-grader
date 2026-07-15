"""Tách jamo âm tiết Hangul — thuần số học Unicode, không dependency.

Âm tiết precomposed U+AC00..U+D7A3 = AC00 + (choseong×21 + jungseong)×28 + jongseong.
Trả jamo dạng COMPATIBILITY (ㄱ ㅏ ...) để bảng phonology/IPA đọc được bằng ký tự
quen mắt. Deterministic tuyệt đối: cùng input → cùng output, không phụ thuộc locale.
"""

from __future__ import annotations

from typing import Final

HANGUL_BASE: Final[int] = 0xAC00
HANGUL_LAST: Final[int] = 0xD7A3

# 19 phụ âm đầu (choseong) theo thứ tự Unicode.
CHOSEONG: Final[str] = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
# 21 nguyên âm (jungseong) theo thứ tự Unicode.
JUNGSEONG: Final[str] = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
# 28 phụ âm cuối (jongseong); index 0 = không có coda ("").
JONGSEONG: Final[tuple[str, ...]] = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
    "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)


def is_hangul_syllable(ch: str) -> bool:
    """True nếu `ch` là 1 âm tiết Hangul precomposed (가..힣)."""
    return len(ch) == 1 and HANGUL_BASE <= ord(ch) <= HANGUL_LAST


def decompose_syllable(ch: str) -> tuple[str, str, str]:
    """Tách 1 âm tiết → (choseong, jungseong, jongseong); jongseong "" nếu không có.

    Raise ValueError nếu không phải âm tiết Hangul — caller (tokenizer đã lọc
    [가-힣]) không bao giờ đưa ký tự khác vào; raise để lỗi lộ sớm thay vì chấm sai.
    """
    if not is_hangul_syllable(ch):
        raise ValueError(f"Không phải âm tiết Hangul: {ch!r}")
    idx = ord(ch) - HANGUL_BASE
    cho, rem = divmod(idx, 21 * 28)
    jung, jong = divmod(rem, 28)
    return CHOSEONG[cho], JUNGSEONG[jung], JONGSEONG[jong]


def decompose_word(word: str) -> list[list[str]]:
    """Tách 1 eojeol (chuỗi âm tiết Hangul) → list [cho, jung, jong] MUTABLE.

    Trả list-of-list (không tuple) vì rule engine phonology sửa jamo tại chỗ.
    """
    return [list(decompose_syllable(ch)) for ch in word]

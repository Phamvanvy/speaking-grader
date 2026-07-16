"""Bộ âm vị IPA tiếng Hàn + bảng jamo→IPA + normalize_ipa_ko / is_vowel_ko.

Quy ước symbol (chọn để GIAO tối đa với vocab của wav2vec2-xlsr-53-espeak-cv-ft —
bench M2 sẽ kiểm chứng, đổi bảng ở đây nếu model không có tense series):
  - Lenis (âm thường):    p t k tɕ s      (ㅂ ㄷ ㄱ ㅈ ㅅ)
  - Aspirated (bật hơi):  pʰ tʰ kʰ tɕʰ    (ㅍ ㅌ ㅋ ㅊ)
  - Tense (âm căng):      p͈ t͈ k͈ t͈ɕ s͈     (ㅃ ㄸ ㄲ ㅉ ㅆ)
  - Sonorant:             m n ŋ l ɾ h     (ㄹ: onset [ɾ], coda/geminate [l])
  - Nguyên âm đơn:        a ʌ o u ɯ i e ɛ (ㅐ/ㅔ merge trong normalize)
  - Glide: j w (tách riêng khỏi nguyên âm — khớp cách model espeak emit); ㅢ → ɯ+i
    (dạng cẩn trọng [ɰi]; ɰ gần như chắc chắn không có trong vocab model).

Coda stop KHÔNG mang dấu unreleased (k̚ → k): vocab model không có diacritic; tính
"không nổ" xử lý ở deletion_severity_ko (coda stop = low).
"""

from __future__ import annotations

import os
from typing import Final

# Tense-fold: vocab của model mặc định (xlsr-53-espeak-cv-ft, kiểm 2026-07-16)
# KHÔNG có token tense (p͈ t͈ k͈ s͈ t͈ɕ) → model không bao giờ emit được → âm căng
# trong reference chỉ sinh substitution OAN trên audio đọc đúng. Fold tense→plain
# Ở NGUỒN G2P (reference-side; predicted-side không bao giờ có tense với model
# này). Default ON theo bằng chứng vocab; đặt TOEIC_PHONEME_KO_TENSE_FOLD=0 khi
# đổi TOEIC_PHONEME_WAV2VEC_MODEL_KO sang model Korean-phone có tense (bench M2).
# Tradeoff đã chấp nhận trong plan: lỗi học viên tense→plain tạm không chấm được.
KO_TENSE_FOLD: bool = (
    os.getenv("TOEIC_PHONEME_KO_TENSE_FOLD", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}

_TENSE_TO_PLAIN: Final[dict[str, str]] = {
    "p͈": "p", "t͈": "t", "k͈": "k", "s͈": "s", "t͈ɕ": "tɕ",
}

# ── jamo → IPA ────────────────────────────────────────────────────────────────

# Phụ âm đầu (choseong). ㅇ đầu âm tiết = không có âm → chuỗi rỗng.
CHO_TO_IPA: Final[dict[str, tuple[str, ...]]] = {
    "ㄱ": ("k",), "ㄲ": ("k͈",), "ㄴ": ("n",), "ㄷ": ("t",), "ㄸ": ("t͈",),
    "ㄹ": ("ɾ",), "ㅁ": ("m",), "ㅂ": ("p",), "ㅃ": ("p͈",), "ㅅ": ("s",),
    "ㅆ": ("s͈",), "ㅇ": (), "ㅈ": ("tɕ",), "ㅉ": ("t͈ɕ",), "ㅊ": ("tɕʰ",),
    "ㅋ": ("kʰ",), "ㅌ": ("tʰ",), "ㅍ": ("pʰ",), "ㅎ": ("h",),
}

# Nguyên âm (jungseong) — glide tách riêng.
JUNG_TO_IPA: Final[dict[str, tuple[str, ...]]] = {
    "ㅏ": ("a",), "ㅐ": ("ɛ",), "ㅑ": ("j", "a"), "ㅒ": ("j", "ɛ"),
    "ㅓ": ("ʌ",), "ㅔ": ("e",), "ㅕ": ("j", "ʌ"), "ㅖ": ("j", "e"),
    "ㅗ": ("o",), "ㅘ": ("w", "a"), "ㅙ": ("w", "ɛ"), "ㅚ": ("w", "e"),
    "ㅛ": ("j", "o"), "ㅜ": ("u",), "ㅝ": ("w", "ʌ"), "ㅞ": ("w", "e"),
    "ㅟ": ("w", "i"), "ㅠ": ("j", "u"), "ㅡ": ("ɯ",), "ㅢ": ("ɯ", "i"),
    "ㅣ": ("i",),
}

# Phụ âm cuối (jongseong) SAU coda neutralization — chỉ còn 7 âm chuẩn.
JONG_TO_IPA: Final[dict[str, tuple[str, ...]]] = {
    "ㄱ": ("k",), "ㄴ": ("n",), "ㄷ": ("t",), "ㄹ": ("l",),
    "ㅁ": ("m",), "ㅂ": ("p",), "ㅇ": ("ŋ",),
}

# ── inventory ────────────────────────────────────────────────────────────────

_VOWELS_KO: Final[frozenset[str]] = frozenset({"a", "ʌ", "o", "u", "ɯ", "i", "e", "ɛ"})

KOREAN_IPA_PHONEMES: Final[frozenset[str]] = frozenset(
    {
        "p", "t", "k", "tɕ", "s",
        "pʰ", "tʰ", "kʰ", "tɕʰ",
        "p͈", "t͈", "k͈", "t͈ɕ", "s͈",
        "m", "n", "ŋ", "l", "ɾ", "h",
        "j", "w",
    }
    | _VOWELS_KO
)

# ── normalize ────────────────────────────────────────────────────────────────

# Gộp biến thể symbol về dạng chuẩn — CHỈ các merger đúng trong tiếng Seoul hiện
# đại (ㅐ=ㅔ) + nhãn tương đương mà recognizer espeak có thể emit. Voiced lenis
# (k↔ɡ...) KHÔNG gộp ở đây — đó là allophone theo VỊ TRÍ, xử lý ở phonemes_match_ko.
_IPA_EQUIV_KO: Final[dict[str, str]] = {
    "ɛ": "e",       # ㅐ/ㅔ merger (người bản xứ trẻ không phân biệt)
    "tʃ": "tɕ",     # nhãn espeak/ARPA cho affricate
    "tʃʰ": "tɕʰ",
    # xlsr-espeak KHÔNG có token tɕ — emit "ts" cho ㅈ (smoke 2026-07-16: 저→tsɔ,
    # 하고→...; audio TTS bản xứ). Alias nhãn thuần, không phải tuning.
    "ts": "tɕ",
    "tsʰ": "tɕʰ",
    # Bench M2 (48 clip native, 2026-07-16): model emit ɔ cho ㅓ một cách hệ thống
    # (6 sub oan med/high + nhiều low) → nhãn của model cho ʌ. Hệ quả đã chấp
    # nhận: contrast ㅓ/ㅗ KHÔNG chấm được với model này (ghi trong bench report).
    "ɔ": "ʌ",
    "ɕ": "s",       # ㅅ trước /i/ — allophone, không phải lỗi
    "ɭ": "l",
    "ɹ": "ɾ",
    "r": "ɾ",
    "ʌ̹": "ʌ",
    "ɐ": "a",
    "ʊ": "u",       # model emit ʊ cho ㅜ ngắn
    "ɨ": "ɯ",       # espeak hay dùng ɨ cho ㅡ
    "ɘ": "ʌ",
}


def normalize_ipa_ko(symbol: str) -> str:
    """Chuẩn hoá 1 symbol IPA tiếng Hàn (strip stress/length/tone marks + fold equiv).

    Deterministic thuần bảng — KHÔNG đụng tense/aspirated (phân biệt có nghĩa).
    Chữ số + dấu chấm là tone/stress marker kiểu espeak dính vào token ("u5",
    "i.5" — bench M2 2026-07-16) → strip.
    """
    s = symbol.strip().replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    s = s.translate(_TONE_MARK_STRIP)
    return _IPA_EQUIV_KO.get(s, s)


_TONE_MARK_STRIP: Final[dict[int, None]] = str.maketrans("", "", "0123456789.")


def is_vowel_ko(symbol: str) -> bool:
    """True nếu symbol (sau normalize) là nguyên âm đơn tiếng Hàn (glide j/w không tính)."""
    return normalize_ipa_ko(symbol) in _VOWELS_KO

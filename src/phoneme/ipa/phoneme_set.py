"""Tập phonemes IPA tiếng Anh + chuẩn hoá ký hiệu (nền tảng, không phụ thuộc submodule khác).

Cung cấp:
  - ENGLISH_IPA_PHONEMES, ARPABET_TO_IPA, IPA_TO_ARPABET: bảng phoneme/ánh xạ.
  - _WORD_IPA_OVERRIDES: per-word corrections (validate fail-fast lúc import).
  - normalize_ipa(): quy 1 ký hiệu IPA về dạng chuẩn (gộp eSpeak ↔ ARPAbet).
  - is_vowel(): nhận diện nguyên âm sau khi chuẩn hoá.
  - FUNCTION_WORDS: tập từ chức năng (dùng cho tolerance + chọn entry CMUdict).
"""

from __future__ import annotations

import re
from typing import Final, Literal

# Nhãn nhấn âm (word stress) song song với IPA symbol. Khai báo kiểu rõ ràng để
# tránh typo ("primay"/"Primary") khi giá trị này đi qua nhiều layer tới UI.
StressType = Literal["primary", "secondary"]

# ──────────────────────────────────────────────────────────────────────────────
# Tập phonemes IPA tiếng Anh (RP/General American unified)
# ──────────────────────────────────────────────────────────────────────────────

ENGLISH_IPA_PHONEMES: Final[list[str]] = [
    # 20 vowels (monophthong + diphthong)
    "iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʌ", "ʊ", "uː",
    "ə", "ɜː",
    "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "ɪə", "eə", "ʊə",
    # 24 consonants
    "p", "b",
    "t", "d",
    "k", "ɡ",
    "tʃ", "dʒ",
    "s", "z",
    "ʃ", "ʒ",
    "f", "v",
    "θ", "ð",
    "h",
    "m", "n", "ŋ",
    "r", "l",
    "w", "j",
]

# ARPAbet → IPA mapping (tiếng Anh, dựa trên CMUdict phoneme set)
ARPABET_TO_IPA: Final[dict[str, str]] = {
    # Vowels
    "AA": "ɑː", "AE": "æ", "AH": "ə", "AO": "ɔː", "AW": "aʊ",
    "AY": "aɪ", "EH": "e", "ER": "ɜː", "EY": "eɪ", "IH": "ɪ",
    "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː",
    # Consonants
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
    "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "r",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Reverse mapping: IPA → ARPAbet (cho alignment ngược)
IPA_TO_ARPABET: Final[dict[str, str]] = {v: k for k, v in ARPABET_TO_IPA.items()}

# Valid ARPAbet bases for defensive validation.
_VALID_ARPABET_BASES: Final[frozenset[str]] = frozenset(ARPABET_TO_IPA.keys())

# Per-word IPA corrections (ARPAbet KÈM stress digit) — HARD PRIORITY: bypasses ALL
# validation and fallback. Use only for proper nouns not in CMUdict and genuine dialect
# exceptions where CMUdict gives the wrong pronunciation for our target accent.
# Each token must have a valid base in ARPABET_TO_IPA (validated at import → fail-fast).
_WORD_IPA_OVERRIDES: Final[dict[str, list[str]]] = {
    # CMUdict all entries start with AH0 (ə) not IH0 (ɪ) — genuine accent exception.
    "especially": ["IH0", "S", "P", "EH1", "SH", "AH0", "L", "IY0"],
    # CMUdict R IH0 Z IH1 L Y AH0 N T: Y→j ("l j ə") instead of IY0→iː ("l iː ə").
    "resilient": ["R", "IH0", "Z", "IH1", "L", "IY0", "AH0", "N", "T"],
    # CMUdict R IY0 L EY1 SH AH0 N SH IH2 P: IY0→iː (not ɪ) at start; IH2 adds
    # secondary stress on last syllable. Override pins IH0 → /rɪˈleɪʃənʃɪp/.
    "relationship": ["R", "IH0", "L", "EY1", "SH", "AH0", "N", "SH", "IH0", "P"],
    # CMUdict F EY1 V ER0 AH0 T: ER0→ɜː wrong for unstressed -vor-; AH0+R = ə+r → /ˈfeɪvərɪt/.
    "favorite": ["F", "EY1", "V", "AH0", "R", "IH0", "T"],
    # CMUdict transcribes "extreme(ly)" inconsistently with EH0 (→e) while the rest of the
    # ex- family uses IH0 (→ɪ). EH0→e gives wrong /ekˈstriːm.../; pin IH0 → /ɪkˈstriːm.../.
    "extreme": ["IH0", "K", "S", "T", "R", "IY1", "M"],
    "extremely": ["IH0", "K", "S", "T", "R", "IY1", "M", "L", "IY0"],
    # Proper nouns not in CMUdict — must stay permanently.
    "vietnamese": ["V", "IY2", "EH0", "T", "N", "AH0", "M", "IY1", "Z"],
    "vietnam": ["V", "IY2", "EH0", "T", "N", "AA1", "M"],
}


def _validate_word_ipa_overrides() -> None:
    """Fail-fast lúc import: mọi token override phải strip về base có trong ARPABET_TO_IPA."""
    for word, tokens in _WORD_IPA_OVERRIDES.items():
        for tok in tokens:
            base = re.sub(r"\d", "", tok)
            if base not in ARPABET_TO_IPA:
                raise ValueError(
                    f"_WORD_IPA_OVERRIDES[{word!r}]: token {tok!r} (base {base!r}) "
                    f"không có trong ARPABET_TO_IPA."
                )


_validate_word_ipa_overrides()

# ──────────────────────────────────────────────────────────────────────────────
# Phân loại phonemes theo manner/place cho similarity scoring
# ──────────────────────────────────────────────────────────────────────────────

_VOWELS = {"iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʌ", "ʊ", "uː", "ə", "ɜː",
           "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "ɪə", "eə", "ʊə"}
_PLOSIVES = {"p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ"}
_FRICATIVES = {"s", "z", "ʃ", "ʒ", "f", "v", "θ", "ð", "h"}
_NASALS = {"m", "n", "ŋ"}
_APPROXIMANTS = {"r", "l", "w", "j"}

# ──────────────────────────────────────────────────────────────────────────────
# Chuẩn hóa IPA — gộp khác biệt hệ thống giữa eSpeak (output wav2vec) và
# ARPAbet→IPA / g2p_en (reference) để không tính nhầm phát âm đúng thành lỗi.
# ──────────────────────────────────────────────────────────────────────────────

# Các cặp tương đương (sau khi đã bỏ dấu trường ː). CHỈ giữ những cặp gần như CHẮC
# CHẮN tương đương giữa eSpeak (wav2vec) và ARPAbet→IPA (g2p) — mọi tolerance "có
# thể chấp nhận" (flap, vowel reduction, near-vowels) KHÔNG nằm ở đây mà ở
# phonemes_match() / phoneme_similarity(), để normalize không vô tình nuốt lỗi thật.
#
# Cố ý KHÔNG có:
#   - `ɾ → t`: flap chỉ là allophone — xử lý riêng trong phonemes_match (_ALLOPHONE_PAIRS),
#     không erase ở đây (nếu erase thì writer/water mất khả năng phân biệt).
#   - `ɜ → ə`: giữ ɜː khác ə để lỗi thật như bird /bɜːd/ vs /bəd/ không bị bỏ sót
#     (chỉ vùng KHÔNG nhấn mới khoan dung ɜ↔ə, do phonemes_match quyết định theo stress).
_IPA_EQUIV: Final[dict[str, str]] = {
    "ɹ": "r",                 # eSpeak r ↔ CMU r
    "g": "ɡ",                 # ascii g ↔ IPA ɡ
    "ɚ": "ə", "ɝ": "ə",       # r-colored schwa → schwa (giữ ɜ riêng)
    "ʌ": "ə", "ɐ": "ə",       # schwa nhấn/không nhấn
    "ɛ": "e",                 # EH
    "ɒ": "ɔ", "ɑ": "ɔ", "o": "ɔ",  # back vowels gộp 1 nhóm (cot-caught)
    "oʊ": "əʊ",               # OW
}


def normalize_ipa(phoneme: str) -> str:
    """Quy 1 ký hiệu IPA về dạng chuẩn để so khớp giữa eSpeak và ARPAbet.

    Bỏ dấu trường (ː) rồi áp bảng tương đương. Diphthong là 1 token nên xử lý
    nguyên khối (vd 'oʊ' → 'əʊ').
    """
    p = phoneme.strip().replace("ː", "")
    return _IPA_EQUIV.get(p, p)


# Tập nguyên âm ở dạng đã chuẩn hoá (sau normalize_ipa) — dùng để nhận diện
# nguyên âm sau khi đã bỏ ː / gộp tương đương (vd "iː"→"i", "ɒ"→"ɔ").
_VOWELS_NORM: Final[frozenset[str]] = frozenset(normalize_ipa(v) for v in _VOWELS)

# Full IPA inventory normalized — used to validate eSpeak output (post-normalize_ipa).
_IPA_PHONEMES_NORM: Final[frozenset[str]] = frozenset(
    normalize_ipa(p) for p in ENGLISH_IPA_PHONEMES
)


def is_vowel(phoneme: str) -> bool:
    """True nếu phoneme là nguyên âm (so sau khi chuẩn hoá: bỏ ː, gộp tương đương)."""
    return normalize_ipa(phoneme) in _VOWELS_NORM


# Function words: thường đọc dạng yếu, cho phép reduction kể cả khi (đơn âm tiết)
# stress=None không phân biệt được. Tập TĨNH — không NLP. Thêm contraction sau nếu cần.
FUNCTION_WORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "to", "for", "from", "in", "on",
    "at", "by", "with", "as", "is", "am", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "that", "than", "them", "then",
    "their", "there", "he", "she", "we", "you", "his", "her", "him", "us",
    "our", "your", "it", "its", "my", "me", "i",
})

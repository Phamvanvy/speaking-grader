"""IPA phoneme set và ánh xạ word → IPA cho tiếng Anh.

Cung cấp:
  - ENGLISH_IPA_PHONEMES: tập 44 phonemes tiếng Anh (20 nguyên âm + 24 phụ âm + /hm/)
  - word_to_ipa(): ánh xạ từ → IPA sequence (dựa trên ARPAbet → IPA)
  - text_to_ipa_sequence(): chuyển đoạn text → danh sách phonemes tham chiếu
  - phoneme_similarity(): tính khoảng cách giữa 2 phonemes (cho severity scoring)
"""

from __future__ import annotations

import re
from typing import Final

# ──────────────────────────────────────────────────────────────────────────────
# Tập phonemes IPA tiếng Anh (RP/General American unified)
# ──────────────────────────────────────────────────────────────────────────────

ENGLISH_IPA_PHONEMES: Final[list[str]] = [
    # 20 vowels (monophthong + diphthong)
    "iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʌ", "ʊ", "uː",
    "ə", "ɜː",
    "eɪ", "aɪ", "ɔɪ", "əʊ", "aʊ", "ɪə", "eə", "ʊə",
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
    "AA": "ɑː", "AE": "æ", "AH": "ə", "AO": "ɒ", "AW": "aʊ",
    "AY": "aɪ", "EH": "e", "ER": "ɜː", "EY": "eɪ", "IH": "ɪ",
    "IY": "iː", "OW": "əʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː",
    # Consonants
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
    "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "r",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

# Reverse mapping: IPA → ARPAbet (cho alignment ngược)
IPA_TO_ARPABET: Final[dict[str, str]] = {v: k for k, v in ARPABET_TO_IPA.items()}

# ──────────────────────────────────────────────────────────────────────────────
# Simple English word → ARPAbet → IPA  (built-in fallback dictionary)
# ──────────────────────────────────────────────────────────────────────────────

# Mini dictionary của các từ phổ biến trong TOEIC Speaking
# Format: word_lowercase → [ARPAbet phonemes]
_COMMON_WORD_PRONUNCIATIONS: Final[dict[str, list[str]]] = {
    # Articles, pronouns
    "the": ["DH", "AH"], "a": "EY".split()[0:1] or ["EH"], "an": ["AE", "N"],
    "i": ["AH"], "me": ["M", "IY"], "my": ["M", "AY"],
    "he": ["HH", "IY"], "him": ["HH", "IH", "M"], "his": ["HH", "IH", "Z"],
    "she": ["SH", "IY"], "her": ["HH", "ER"],
    "we": ["W", "IY"], "us": ["AH", "S"], "our": ["AH", "R"],
    "you": ["Y", "UW"], "your": ["Y", "ER"],
    "it": ["IH", "T"], "its": ["IH", "T", "S"],
    "they": ["DH", "EY"], "them": ["DH", "EH", "M"], "their": ["DH", "ER"],
    "what": ["W", "AH", "T"], "where": ["W", "ER"], "when": ["W", "EH", "N"],
    "which": ["W", "CH", "IH"], "who": ["HH", "UW"], "why": ["W", "AY"],
    "how": ["HH", "AW"],
    # Common verbs
    "is": ["IH", "Z"], "are": ["AH", "R"], "was": ["W", "AH", "Z"],
    "were": ["W", "ER"], "be": ["B", "IY"], "been": ["B", "IH", "N"],
    "being": ["B", "IH", "NG"], "do": ["D", "UW"], "does": ["D", "AH", "Z"],
    "did": ["D", "IH", "D"], "have": ["HH", "AE", "V"], "has": ["HH", "AE", "Z"],
    "had": ["HH", "AE", "D"], "will": ["W", "IH", "L"], "would": ["W", "UH", "L"],
    "could": ["K", "UH", "L"], "should": ["SH", "UH", "L"],
    "can": ["K", "AE", "N"], "may": ["M", "EY"], "might": ["M", "AY", "T"],
    "must": ["M", "AH", "ST"], "shall": ["SH", "AE", "L"],
    "say": ["S", "EY"], "said": ["S", "EH", "D"],
    "go": ["G", "UW"], "went": ["W", "EH", "NT"], "gone": ["G", "AO", "N"],
    "come": ["K", "AH", "M"], "came": ["K", "EY", "M"],
    "get": ["G", "EH", "T"], "got": ["G", "AH", "T"],
    "make": ["M", "EY", "K"], "made": ["M", "EY", "D"],
    "take": ["T", "EY", "K"], "took": ["T", "UH", "K"],
    "give": ["G", "IH", "V"], "gave": ["G", "AE", "V"],
    "know": ["N", "UW"], "knew": ["N", "UW"],
    "think": ["TH", "IH", "K"], "thought": ["TH", "AO", "T"],
    "see": ["S", "IY"], "saw": ["S", "AO"],
    "look": ["L", "UH", "K"], "like": ["L", "AY", "K"],
    "find": ["F", "AY", "ND"], "feel": ["F", "IY", "L"],
    "want": ["W", "AH", "NT"], "need": ["N", "IY", "D"],
    "use": ["Y", "UW", "Z"], "used": ["Y", "UW", "Z", "D"],
    "work": ["W", "ER", "K"],
    "try": ["T", "R", "AY"], "show": ["SH", "UW"],
    "tell": ["T", "EH", "L"], "ask": ["AE", "SK"],
    "tell": ["T", "EH", "L"],
    "move": ["M", "UW", "V"], "live": ["L", "IH", "V"],
    "run": ["R", "AH", "N"], "help": ["HH", "EH", "L", "P"],
    "talk": ["T", "AO", "L", "K"], "start": ["ST", "AH", "R", "T"],
    "play": ["P", "LE", "Y"],
    "pay": ["P", "EY"],
    # Common nouns (TOEIC context)
    "time": ["T", "AY", "M"], "day": ["D", "EY"], "year": ["Y", "ER"],
    "way": ["W", "AY"], "thing": ["TH", "IH", "NG"],
    "man": ["M", "AE", "N"], "men": ["M", "EH", "N"],
    "woman": ["W", "AH", "M", "AE", "N"], "people": ["P", "IH", "P", "AH", "L"],
    "world": ["W", "ER", "L", "D"], "life": ["L", "AY", "F"],
    "hand": ["HH", "AE", "ND"], "part": ["P", "AH", "R", "T"],
    "child": ["CH", "AY", "L", "D"], "children": ["CH", "IH", "L", "D", "R", "EH", "N"],
    "eye": ["AY"], "place": ["P", "LE", "S"], "work": ["W", "ER", "K"],
    "week": ["W", "IH", "K"], "company": ["K", "AH", "M", "P", "AH", "N", "IY"],
    "number": ["N", "AH", "M", "B", "ER"], "state": ["S", "EY", "T"],
    "family": ["F", "AE", "M", "IY"], "student": ["ST", "UW", "DH", "EH", "N", "T"],
    "group": ["G", "R", "UH", "P"], "country": ["K", "AH", "N", "T", "R", "IY"],
    # Common adjectives
    "good": ["G", "UH", "D"], "new": ["N", "UW"], "first": ["F", "ER", "ST"],
    "last": ["L", "AE", "ST"], "long": ["L", "AH", "NG"], "great": ["G", "RE", "T"],
    "little": ["L", "IH", "T", "AH", "L"], "own": ["OW", "N"],
    "other": ["AH", "TH", "ER"], "old": ["OW", "L", "D"], "right": ["R", "AY", "T"],
    "big": ["B", "IH", "G"], "high": ["HH", "AY"], "small": ["SM", "AO", "L"],
    "different": ["D", "IH", "F", "EH", "R", "EH", "NT"],
    "important": ["IH", "M", "P", "AO", "R", "EH", "NT"],
    # Prepositions
    "in": ["IH", "N"], "on": ["AH", "N"], "at": ["AE", "T"],
    "to": ["T", "UW"], "of": ["AH", "F"], "for": ["F", "ER"],
    "with": ["W", "IH", "TH"], "about": ["AE", "B", "AH", "T"],
    "between": ["B", "IH", "T", "W", "IH", "N"], "after": ["AE", "F", "ER"],
    "before": ["B", "IH", "F", "ER"], "under": ["AH", "ND", "ER"],
    "over": ["OW", "V", "ER"], "through": ["TH", "R", "UW"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Phoneme similarity — tính khoảng cách âm vị giữa 2 phonemes
# ──────────────────────────────────────────────────────────────────────────────

# Phân loại phonemes theo manner/place cho similarity scoring
_VOWELS = {"iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʌ", "ʊ", "uː", "ə", "ɜː",
           "eɪ", "aɪ", "ɔɪ", "əʊ", "aʊ", "ɪə", "eə", "ʊə"}
_PLOSIVES = {"p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ"}
_FRICATIVES = {"s", "z", "ʃ", "ʒ", "f", "v", "θ", "ð", "h"}
_NASALS = {"m", "n", "ŋ"}
_APPROXIMANTS = {"r", "l", "w", "j"}


def _same_class(p1: str, p2: str) -> bool:
    """Kiểm tra 2 phonemes có cùng phonological class không."""
    for cls in (_PLOSIVES, _FRICATIVES, _NASALS, _APPROXIMANTS, _VOWELS):
        if p1 in cls and p2 in cls:
            return True
    return False


def _same_place_of_articulation(p1: str, p2: str) -> bool:
    """Kiểm tra 2 phụ âm có cùng place of articulation không."""
    place_groups = [
        {"p", "b", "m"},                         # bilabial
        {"t", "d", "n"},                          # alveolar
        {"k", "ɡ", "ŋ"},                          # velar
        {"tʃ", "dʒ"},                             # post-alveolar
        {"ʃ", "ʒ", "r"},                          # post-alveolar
        {"s", "z", "l"},                          # alveolar fricative/lateral
        {"f", "v"},                               # labiodental
        {"θ", "ð"},                               # dental
        {"w", "j"},                               # semi-vowels
    ]
    for group in place_groups:
        if p1 in group and p2 in group:
            return True
    return False


def phoneme_similarity(p1: str, p2: str) -> float:
    """Tính độ tương đồng giữa 2 phonemes (0.0 = hoàn toàn khác, 1.0 = giống hệt).

    Algorithm:
      - Giống hệt: 1.0
      - Cùng class + cùng place: 0.7
      - Cùng class hoặc cùng place: 0.4
      - Khác hoàn toàn: 0.0
    """
    if p1 == p2:
        return 1.0
    same_cls = _same_class(p1, p2)
    same_place = _same_place_of_articulation(p1, p2)
    if same_cls and same_place:
        return 0.7
    if same_cls or same_place:
        return 0.4
    return 0.0


def error_severity(similarity: float) -> str:
    """Chuyển similarity score thành severity label."""
    if similarity >= 0.7:
        return "low"
    if similarity >= 0.4:
        return "medium"
    return "high"


# ──────────────────────────────────────────────────────────────────────────────
# Word → IPA sequence
# ──────────────────────────────────────────────────────────────────────────────

def word_to_ipa(word: str) -> list[str]:
    """Chuyển 1 từ tiếng Anh thành danh sách IPA phonemes.

    Priority:
      1. Built-in dictionary (_COMMON_WORD_PRONUNCIATIONS)
      2. Try g2p module (grapheme-to-phoneme)
      3. Fallback: empty list (caller sẽ handle missing words)
    """
    key = word.lower().strip(".,;:!?\"'()[]{}")
    if not key:
        return []

    # Built-in dictionary first
    if key in _COMMON_WORD_PRONUNCIATIONS:
        arpabet = _COMMON_WORD_PRONUNCIATIONS[key]
        return [ARPABET_TO_IPA.get(a, a) for a in arpabet]

    # Try g2p if available
    try:
        import g2p_en
        transcriber = g2p_en.G2p()
        # g2p_en trả về list của (word, [(token, arpabet), ...])
        result = transcriber(key)
        if result and len(result) > 0:
            arpabet_seq = [a for _, a in result[0]]
            return [ARPABET_TO_IPA.get(a, a) for a in arpabet_seq]
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: return the word itself as placeholder
    # (caller should detect empty results and handle gracefully)
    return []


def text_to_ipa_sequence(text: str) -> list[str]:
    """Chuyển đoạn text thành danh sách phonemes tham chiếu.

    Input: "The quick brown fox"
    Output: ["DH", "AH", "K", "W", "IH", "K", ...] → IPA
    """
    if not text:
        return []

    words = re.findall(r"[a-zA-Z'-]+", text)
    phonemes: list[str] = []
    for word in words:
        word_phones = word_to_ipa(word)
        if word_phones:
            phonemes.extend(word_phones)
        else:
            # Word not in dictionary — skip but don't fail
            pass

    return phonemes
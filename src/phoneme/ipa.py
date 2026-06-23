"""IPA phoneme set và ánh xạ word → IPA cho tiếng Anh.

Cung cấp:
  - ENGLISH_IPA_PHONEMES: tập 44 phonemes tiếng Anh (20 nguyên âm + 24 phụ âm + /hm/)
  - word_to_ipa(): ánh xạ từ → IPA sequence (dựa trên ARPAbet → IPA)
  - text_to_ipa_sequence(): chuyển đoạn text → danh sách phonemes tham chiếu
  - phoneme_similarity(): tính khoảng cách giữa 2 phonemes (cho severity scoring)
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Final, Literal

from .models import WordSpan

logger = logging.getLogger("toeic.phoneme.ipa")

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

# Mini dictionary của các từ phổ biến trong TOEIC Speaking.
# Format: word_lowercase → [ARPAbet phonemes] (CMUdict, đã bỏ stress digit).
# QUY TẮC: mỗi phần tử phải là MỘT ký hiệu ARPAbet hợp lệ có trong ARPABET_TO_IPA
# (vd "N","D" — KHÔNG ghép "ND"; "S","T" — KHÔNG ghép "ST"). Token ghép sẽ không
# map được sang IPA và lọt vào chuỗi tham chiếu thành "phoneme" rác → báo lỗi oan.
# Built-in path còn lọc lại theo ARPABET_TO_IPA nên token lạ bị bỏ, nhưng giữ dict
# đúng ngay từ đầu để reference không bị thiếu âm.
_COMMON_WORD_PRONUNCIATIONS: Final[dict[str, list[str]]] = {
    # Articles, pronouns
    "the": ["DH", "AH"], "a": ["AH"], "an": ["AE", "N"],
    "i": ["AY"], "me": ["M", "IY"], "my": ["M", "AY"],
    "he": ["HH", "IY"], "him": ["HH", "IH", "M"], "his": ["HH", "IH", "Z"],
    "she": ["SH", "IY"], "her": ["HH", "ER"],
    "we": ["W", "IY"], "us": ["AH", "S"], "our": ["AW", "ER"],
    "you": ["Y", "UW"], "your": ["Y", "AO", "R"],
    "it": ["IH", "T"], "its": ["IH", "T", "S"],
    "they": ["DH", "EY"], "them": ["DH", "EH", "M"], "their": ["DH", "EH", "R"],
    "what": ["W", "AH", "T"], "where": ["W", "EH", "R"], "when": ["W", "EH", "N"],
    "which": ["W", "IH", "CH"], "who": ["HH", "UW"], "why": ["W", "AY"],
    "how": ["HH", "AW"],
    # Common verbs
    "is": ["IH", "Z"], "are": ["AA", "R"], "was": ["W", "AH", "Z"],
    "were": ["W", "ER"], "be": ["B", "IY"], "been": ["B", "IH", "N"],
    "being": ["B", "IY", "IH", "NG"], "do": ["D", "UW"], "does": ["D", "AH", "Z"],
    "did": ["D", "IH", "D"], "have": ["HH", "AE", "V"], "has": ["HH", "AE", "Z"],
    "had": ["HH", "AE", "D"], "will": ["W", "IH", "L"], "would": ["W", "UH", "D"],
    "could": ["K", "UH", "D"], "should": ["SH", "UH", "D"],
    "can": ["K", "AE", "N"], "may": ["M", "EY"], "might": ["M", "AY", "T"],
    "must": ["M", "AH", "S", "T"], "shall": ["SH", "AE", "L"],
    "say": ["S", "EY"], "said": ["S", "EH", "D"],
    "go": ["G", "OW"], "went": ["W", "EH", "N", "T"], "gone": ["G", "AO", "N"],
    "come": ["K", "AH", "M"], "came": ["K", "EY", "M"],
    "get": ["G", "EH", "T"], "got": ["G", "AA", "T"],
    "make": ["M", "EY", "K"], "made": ["M", "EY", "D"],
    "take": ["T", "EY", "K"], "took": ["T", "UH", "K"],
    "give": ["G", "IH", "V"], "gave": ["G", "EY", "V"],
    "know": ["N", "OW"], "knew": ["N", "UW"],
    "think": ["TH", "IH", "NG", "K"], "thought": ["TH", "AO", "T"],
    "see": ["S", "IY"], "saw": ["S", "AO"],
    "look": ["L", "UH", "K"], "like": ["L", "AY", "K"],
    "find": ["F", "AY", "N", "D"], "feel": ["F", "IY", "L"],
    "want": ["W", "AA", "N", "T"], "need": ["N", "IY", "D"],
    "use": ["Y", "UW", "Z"], "used": ["Y", "UW", "Z", "D"],
    "work": ["W", "ER", "K"],
    "try": ["T", "R", "AY"], "show": ["SH", "OW"],
    "tell": ["T", "EH", "L"], "ask": ["AE", "S", "K"],
    "move": ["M", "UW", "V"], "live": ["L", "IH", "V"],
    "run": ["R", "AH", "N"], "help": ["HH", "EH", "L", "P"],
    "talk": ["T", "AO", "K"], "start": ["S", "T", "AA", "R", "T"],
    "play": ["P", "L", "EY"],
    "pay": ["P", "EY"],
    # Common nouns (TOEIC context)
    "time": ["T", "AY", "M"], "day": ["D", "EY"], "year": ["Y", "IH", "R"],
    "way": ["W", "EY"], "thing": ["TH", "IH", "NG"],
    "man": ["M", "AE", "N"], "men": ["M", "EH", "N"],
    "woman": ["W", "UH", "M", "AH", "N"], "people": ["P", "IY", "P", "AH", "L"],
    "world": ["W", "ER", "L", "D"], "life": ["L", "AY", "F"],
    "hand": ["HH", "AE", "N", "D"], "part": ["P", "AA", "R", "T"],
    "child": ["CH", "AY", "L", "D"], "children": ["CH", "IH", "L", "D", "R", "AH", "N"],
    "eye": ["AY"], "place": ["P", "L", "EY", "S"],
    "week": ["W", "IY", "K"], "company": ["K", "AH", "M", "P", "AH", "N", "IY"],
    "number": ["N", "AH", "M", "B", "ER"], "state": ["S", "T", "EY", "T"],
    "family": ["F", "AE", "M", "AH", "L", "IY"],
    "student": ["S", "T", "UW", "D", "AH", "N", "T"],
    "group": ["G", "R", "UW", "P"], "country": ["K", "AH", "N", "T", "R", "IY"],
    # Common adjectives
    "good": ["G", "UH", "D"], "new": ["N", "UW"], "first": ["F", "ER", "S", "T"],
    "last": ["L", "AE", "S", "T"], "long": ["L", "AO", "NG"],
    "great": ["G", "R", "EY", "T"],
    "little": ["L", "IH", "T", "AH", "L"], "own": ["OW", "N"],
    "other": ["AH", "DH", "ER"], "old": ["OW", "L", "D"], "right": ["R", "AY", "T"],
    "big": ["B", "IH", "G"], "high": ["HH", "AY"], "small": ["S", "M", "AO", "L"],
    "different": ["D", "IH", "F", "R", "AH", "N", "T"],
    "important": ["IH", "M", "P", "AO", "R", "T", "AH", "N", "T"],
    # Prepositions
    "in": ["IH", "N"], "on": ["AA", "N"], "at": ["AE", "T"],
    "to": ["T", "UW"], "of": ["AH", "V"], "for": ["F", "ER"],
    "with": ["W", "IH", "DH"], "about": ["AH", "B", "AW", "T"],
    "between": ["B", "IH", "T", "W", "IY", "N"], "after": ["AE", "F", "T", "ER"],
    "before": ["B", "IH", "F", "AO", "R"], "under": ["AH", "N", "D", "ER"],
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


def is_vowel(phoneme: str) -> bool:
    """True nếu phoneme là nguyên âm (so sau khi chuẩn hoá: bỏ ː, gộp tương đương)."""
    return normalize_ipa(phoneme) in _VOWELS_NORM


def _near(a: str, b: str, score: float) -> tuple[frozenset[str], float]:
    """Build 1 entry của bảng near-pair với key đã chuẩn hoá (symbol pair)."""
    return frozenset({normalize_ipa(a), normalize_ipa(b)}), score


# Bảng "cặp âm gần nhau" được tuyển CHỌN TAY (không xây hệ toạ độ IPA đầy đủ) —
# dễ bảo trì/tuning/giải thích hơn. Score cao = penalty thấp = severity nhẹ.
# Cặp KHÔNG có trong bảng rơi về heuristic class/place bên dưới.
# LƯU Ý: key đã qua normalize nên "iː"→"i", "ʊ"/"uː"→"ʊ"/"u" v.v.
_NEAR_PAIRS: Final[dict[frozenset[str], float]] = dict([
    _near("ɪ", "iː", 0.85),   # bit ↔ beat (rút gọn nguyên âm)
    _near("ʊ", "uː", 0.85),   # book ↔ boot
    _near("e", "eɪ", 0.80),   # bed ↔ bay
    _near("æ", "e", 0.80),    # bat ↔ bet (lỗi ESL phổ biến nhưng gần)
    _near("ɜː", "ə", 0.70),   # NURSE ↔ schwa: gần (đuôi -er KHÔNG nhấn còn được
                              #   phonemes_match cho khớp hẳn); nhấn thì vẫn là lỗi nhẹ
    _near("ə", "ɪ", 0.80),    # reduced vowels
    _near("ə", "ʊ", 0.70),
    _near("ə", "uː", 0.70),
    _near("iː", "eɪ", 0.60),
    _near("ɔ", "əʊ", 0.55),   # thought ↔ go (back, hơi gần)
])


@functools.lru_cache(maxsize=8192)
def phoneme_similarity(p1: str, p2: str) -> float:
    """Tính độ tương đồng giữa 2 phonemes (0.0 = hoàn toàn khác, 1.0 = giống hệt).

    Continuous (không chỉ 4 mức) để penalty = 1 - similarity mượt — vd ɪ↔iː rất
    gần (0.85, penalty 0.15) trong khi θ↔k khác hẳn (0.0). Đây là NGUỒN CHÂN LÝ
    điều khiển cả penalty lẫn severity của substitution.

    Algorithm:
      1. Giống hệt (sau chuẩn hóa eSpeak↔ARPAbet): 1.0
      2. Cặp trong bảng near-pair tuyển tay (chủ yếu nguyên âm): score của bảng
      3. Heuristic class/place: cùng class + cùng place 0.7; chỉ 1 trong 2: 0.4
      4. Khác hoàn toàn: 0.0

    Cached (lru_cache) vì DTW gọi O(n·m) lần mà inventory IPA chỉ vài chục âm.
    """
    n1, n2 = normalize_ipa(p1), normalize_ipa(p2)
    if n1 == n2:
        return 1.0
    near = _NEAR_PAIRS.get(frozenset({n1, n2}))
    if near is not None:
        return near
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
# Tolerance: phonemes_match — quyết định 2 âm có coi là KHỚP không (≠ identical)
# ──────────────────────────────────────────────────────────────────────────────

# Allophone — biến thể phát âm KHÔNG phải lỗi (sau normalize). Giữ NHỎ + tường minh.
#  - {t,ɾ}, {d,ɾ}: alveolar flap kiểu Mỹ (water/writer/abilities) — vẫn đúng.
#  - {r,ɜ}: g2p hay biến ER/r-âm-tiết KHÔNG nhấn thành ɜː trong khi người đọc phát
#    [ɹ] (every /evɜːiː/→/evɹiː/, arrives /ɜːaɪvz/→/ɹaɪvz/). Coi là khớp.
_ALLOPHONE_PAIRS: Final[frozenset[frozenset[str]]] = frozenset({
    frozenset({"t", "ɾ"}),
    frozenset({"d", "ɾ"}),
    frozenset({"r", "ɜ"}),
})

# Nguyên âm rút gọn (dạng đã normalize) — ở vị trí KHÔNG nhấn / trong function word,
# các nguyên âm này hoán đổi cho nhau là chấp nhận được (weak/strong form).
_REDUCED_VOWELS: Final[frozenset[str]] = frozenset({"ə", "ʊ", "ɪ", "i", "u", "ɜ"})

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


def phonemes_match(
    expected: str,
    predicted: str,
    *,
    stress: str | None = None,
    word: str | None = None,
    reducible: bool | None = None,
) -> bool:
    """True nếu `predicted` coi là phát âm ĐÚNG của `expected` (rộng hơn identical).

    Tầng tolerance (theo thứ tự, dừng ở True đầu tiên):
      1. Bằng nhau sau normalize tối thiểu.
      2. Cặp allophone (flap t/d↔ɾ, r↔ɜ) — biến thể đúng.
      3. Vowel reduction ở vị trí được phép rút gọn: các nguyên âm rút gọn hoán đổi
         cho nhau (đuôi -er ɜ↔ə, to /tuː/→/tʊ/...). `æ↔ə` chỉ mở cho function word.

    `reducible`: cho phép rút gọn nguyên âm tại VỊ TRÍ NÀY hay không. Caller (scoring)
    tính sẵn = "nguyên âm KHÔNG phải nhân chính của từ" HOẶC function word — vì stress
    đơn lẻ không phân biệt được nhân chính của từ ĐƠN âm tiết (bird /bɜːd/: stress=None
    nhưng ɜ là nhân chính → KHÔNG được rút gọn thành ə, giữ là lỗi thật). Nếu None thì
    suy từ stress/word (stress is None hoặc function word) — chỉ dùng khi gọi độc lập.
    """
    e, p = normalize_ipa(expected), normalize_ipa(predicted)
    if e == p:
        return True
    pair = frozenset({e, p})
    if pair in _ALLOPHONE_PAIRS:
        return True
    in_func = bool(word) and word.lower().strip(".,;:!?\"'()[]{}") in FUNCTION_WORDS
    if reducible is None:
        reducible = stress is None or in_func
    if reducible and e in _REDUCED_VOWELS and p in _REDUCED_VOWELS:
        return True
    # æ↔ə (strong/weak "and", "a", "an", "at"...) chỉ cho function word.
    if in_func and pair == frozenset({"æ", "ə"}):
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Deletion severity/penalty — KHÔNG còn luôn "high"; phụ thuộc loại âm + vị trí
# ──────────────────────────────────────────────────────────────────────────────

# Âm wav2vec hay "nuốt" (yếu/ngắn) — mất các âm này thường là lỗi NHẬN DẠNG, không
# phải lỗi người đọc (dạng đã normalize: ð, h, schwa, ɪ, ʊ).
_RECOGNIZER_PRONE: Final[frozenset[str]] = frozenset({"ð", "h", "ə", "ɪ", "ʊ"})

_SEVERITY_PENALTY: Final[dict[str, float]] = {"low": 0.1, "medium": 0.5, "high": 0.9}


def deletion_severity(
    phoneme: str, *, is_onset: bool = False, stress: str | None = None
) -> str:
    """Severity cho 1 âm reference bị THIẾU, theo loại âm + vị trí.

    - Âm recognizer-prone (ð, h, ə, ɪ, ʊ) → low (nhiều khả năng do recognizer nuốt).
    - Nguyên âm: high nếu được nhấn (primary/secondary), còn lại low (reduction).
    - Phụ âm: high nếu là onset (đầu từ / trong cụm đầu — vd think θ→∅, school sk→s),
      ngược lại (coda) medium.
    """
    p = normalize_ipa(phoneme)
    if p in _RECOGNIZER_PRONE:
        return "low"
    if p in _VOWELS_NORM:
        return "high" if stress in ("primary", "secondary") else "low"
    return "high" if is_onset else "medium"


def deletion_penalty(
    phoneme: str, *, is_onset: bool = False, stress: str | None = None
) -> float:
    """Penalty liên tục (qua bucket severity) cho 1 âm bị thiếu — xem deletion_severity."""
    return _SEVERITY_PENALTY[deletion_severity(phoneme, is_onset=is_onset, stress=stress)]


# ──────────────────────────────────────────────────────────────────────────────
# Word → IPA sequence
# ──────────────────────────────────────────────────────────────────────────────

# g2p_en.G2p() nặng (nạp CMUdict + POS tagger) → khởi tạo 1 lần, cache lại.
# None = chưa thử; False = không khả dụng (thiếu package / lỗi nạp).
_g2p_instance: object | None | bool = None


def _ensure_nltk_data() -> None:
    """Tải dữ liệu NLTK g2p_en cần (nếu thiếu). nltk mới đổi tên tagger thành
    '..._eng' mà g2p_en không tự tải → tự xử lý để chạy ngay lần đầu."""
    import nltk

    resources = [
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/cmudict", "cmudict"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:  # noqa: BLE001 - mạng/permission; g2p sẽ tự báo lỗi
                pass


def _get_g2p() -> object | None:
    """Lazy-init + cache g2p_en.G2p(). Trả None nếu không khả dụng."""
    global _g2p_instance
    if _g2p_instance is None:
        try:
            import g2p_en

            _ensure_nltk_data()
            _g2p_instance = g2p_en.G2p()
        except Exception:  # noqa: BLE001 - thiếu package / lỗi nạp model
            _g2p_instance = False
    return _g2p_instance or None


def word_to_ipa(word: str) -> list[str]:
    """Chuyển 1 từ tiếng Anh thành danh sách IPA phonemes.

    Thin wrapper của word_to_ipa_with_stress() — giữ chữ ký cũ cho các caller
    chỉ cần danh sách symbol (không cần nhấn âm).

    Priority:
      1. Built-in dictionary (_COMMON_WORD_PRONUNCIATIONS)
      2. g2p module (grapheme-to-phoneme), instance được cache
      3. Fallback: empty list (caller sẽ handle missing words)
    """
    symbols, _stress = word_to_ipa_with_stress(word)
    return symbols


def word_to_ipa_with_stress(
    word: str,
) -> tuple[list[str], list[StressType | None]]:
    """Như word_to_ipa() nhưng kèm danh sách nhấn âm song song 1-1 với symbols.

    Nhấn âm CHỈ dùng để hiển thị — KHÔNG bao giờ chèn vào chuỗi phoneme dùng cho
    DTW alignment (predicted từ wav2vec không có dấu nhấn → sẽ lệch).

    - Built-in dictionary: ARPAbet không kèm stress digit → stress toàn None.
    - g2p_en: ARPAbet vowel token kèm stress digit (AH0/AH1/AH2). Map
      1→"primary", 2→"secondary", 0/không có→None. Digit chỉ nằm trên nguyên âm
      nên số token có digit = số âm tiết.
    - Từ ĐƠN ÂM TIẾT (≤1 âm tiết): set toàn bộ stress về None — từ điển chuẩn
      không ký hiệu nhấn cho từ 1 âm tiết (cat /kæt/ chứ không /ˈkæt/).

    Returns (symbols, stresses) với len(symbols) == len(stresses).
    """
    key = word.lower().strip(".,;:!?\"'()[]{}")
    if not key:
        return [], []

    # Built-in dictionary first. Lọc theo ARPABET_TO_IPA (giống nhánh g2p) để
    # token không hợp lệ không lọt vào chuỗi IPA tham chiếu thành "phoneme" rác.
    # Dictionary lưu ARPAbet KHÔNG kèm stress digit → stress toàn None.
    if key in _COMMON_WORD_PRONUNCIATIONS:
        symbols = [ARPABET_TO_IPA[a] for a in _COMMON_WORD_PRONUNCIATIONS[key]
                   if a in ARPABET_TO_IPA]
        return symbols, [None] * len(symbols)

    # Try g2p if available (cached instance — KHÔNG khởi tạo lại mỗi từ)
    transcriber = _get_g2p()
    if transcriber is not None:
        try:
            symbols: list[str] = []
            stresses: list[StressType | None] = []
            syllables = 0
            # g2p_en trả về flat list các ARPAbet token (kèm stress digit + space).
            # Parse digit TRƯỚC khi strip để giữ nhấn âm; build symbol+stress lockstep
            # theo đúng filter `a in ARPABET_TO_IPA` → index khớp 1-1.
            for raw in transcriber(key):
                if not raw.strip():
                    continue
                digit = re.search(r"\d", raw)
                base = re.sub(r"\d", "", raw)  # AH0 → AH
                if base not in ARPABET_TO_IPA:
                    continue
                symbols.append(ARPABET_TO_IPA[base])
                if digit is not None:
                    syllables += 1  # digit chỉ trên nguyên âm → đếm âm tiết
                stresses.append(
                    "primary" if (digit and digit.group() == "1")
                    else "secondary" if (digit and digit.group() == "2")
                    else None
                )
            if symbols:
                # Từ đơn âm tiết: không ký hiệu nhấn.
                if syllables <= 1:
                    stresses = [None] * len(symbols)
                if len(symbols) != len(stresses):
                    raise ValueError(
                        "IPA symbols and stress list length mismatch"
                    )
                return symbols, stresses
        except ValueError:
            raise  # guard alignment — không nuốt
        except Exception:  # noqa: BLE001 - lỗi runtime g2p
            pass

    # Fallback: empty list (caller detect và handle missing words)
    return [], []


def text_to_ipa_sequence_with_spans(
    text: str,
) -> tuple[list[str], list[WordSpan], list[StressType | None]]:
    """Chuyển text → (phonemes tham chiếu, WordSpan, nhấn âm song song).

    Input:  "The quick brown fox"
    Output: (["ð", "ə", "k", "w", "ɪ", "k", ...],
             [WordSpan("The", 0, 2), WordSpan("quick", 2, 6), ...],
             [None, None, None, None, "primary", None, ...])

    Phonemes, spans và stress được build trong CÙNG vòng lặp tokenize/
    word_to_ipa_with_stress, nên luôn khớp 1-1 theo index: từ nào không tra được
    IPA (dropped) sẽ KHÔNG sinh span và KHÔNG đẩy phoneme/stress nào → index của
    các từ sau không bị lệch. `stress` song song 1-1 với `phonemes`.

    Span dùng để map ngược lỗi phoneme (theo position trong reference sequence)
    về đúng từ. `word` giữ nguyên dạng token (re.findall đã loại dấu câu) và giữ
    nguyên hoa/thường để hiển thị; word_to_ipa tự lower() khi tra từ điển.
    """
    if not text:
        return [], [], []

    words = re.findall(r"[a-zA-Z'-]+", text)
    phonemes: list[str] = []
    spans: list[WordSpan] = []
    stress: list[StressType | None] = []
    dropped: list[str] = []
    for word in words:
        word_phones, word_stress = word_to_ipa_with_stress(word)
        if word_phones:
            start = len(phonemes)
            phonemes.extend(word_phones)
            stress.extend(word_stress)
            spans.append(WordSpan(word, start, len(phonemes)))
        else:
            # Word không tra được IPA (không có trong dict & g2p) — bỏ qua, ghi log.
            # Không thêm span → indices của các từ sau vẫn khớp với phoneme list.
            dropped.append(word)

    if dropped:
        logger.warning(
            "text_to_ipa_sequence: bỏ %d/%d từ không tra được IPA "
            "(reference sẽ thiếu → điểm phoneme kém tin cậy): %s%s",
            len(dropped),
            len(words),
            ", ".join(dropped[:10]),
            " ..." if len(dropped) > 10 else "",
        )

    return phonemes, spans, stress


def text_to_ipa_sequence(text: str) -> list[str]:
    """Chuyển đoạn text thành danh sách phonemes tham chiếu.

    Input: "The quick brown fox"
    Output: ["DH", "AH", "K", "W", "IH", "K", ...] → IPA

    Thin wrapper của text_to_ipa_sequence_with_spans() — giữ chữ ký cũ cho các
    caller chỉ cần phoneme list (không cần word mapping).
    """
    phonemes, _spans, _stress = text_to_ipa_sequence_with_spans(text)
    return phonemes
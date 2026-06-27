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
# Phoneme similarity — tính khoảng cách âm vị giữa 2 phonemes
# ──────────────────────────────────────────────────────────────────────────────

# Phân loại phonemes theo manner/place cho similarity scoring
_VOWELS = {"iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʌ", "ʊ", "uː", "ə", "ɜː",
           "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "ɪə", "eə", "ʊə"}
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

# Full IPA inventory normalized — used to validate eSpeak output (post-normalize_ipa).
_IPA_PHONEMES_NORM: Final[frozenset[str]] = frozenset(
    normalize_ipa(p) for p in ENGLISH_IPA_PHONEMES
)


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


# ──────────────────────────────────────────────────────────────────────────────
# Recognizer-noise gate: cặp substitution là LỖI HỌC VIÊN THẬT (được bảo vệ) hay
# ARTIFACT của recognizer (wav2vec hallucinate)? — xem scoring._align_points.
# ──────────────────────────────────────────────────────────────────────────────

# Các cặp substitution CÙNG LOẠI nhưng similarity < ngưỡng bảo vệ (sim<0.2) mà VẪN là
# lỗi phát âm THẬT của người Việt → KHÔNG được gate thành "recognizer noise". Bảng này
# là CHÍNH SÁCH dựa-trên-bằng-chứng (không phải danh sách ngoại lệ phình mãi): chỉ thêm
# cặp khi telemetry cho thấy NHẤT QUÁN là lỗi học viên thật (xem gated-candidate log).
# Các cặp similarity ≥ 0.2 (θ↔s, ð↔z, v↔f, z↔s, ʃ↔s, r↔l, ŋ↔n, æ↔e, k↔p...) đã được
# ngưỡng sim bảo vệ sẵn → KHÔNG cần liệt kê ở đây.
#   th-stopping ð→d, θ→t (high-freq trong tel3) · deaffrication tʃ→ʃ · dʒ→z/j ·
#   l↔n (lẫn l/n) · v→b/w (tiếng Việt thiếu /v/) · f→p (fricative→stop).
_REAL_ERROR_SUBS: Final[frozenset[frozenset[str]]] = frozenset(
    frozenset({normalize_ipa(a), normalize_ipa(b)})
    for a, b in [
        ("ð", "d"), ("θ", "t"), ("tʃ", "ʃ"), ("dʒ", "z"), ("dʒ", "j"),
        ("l", "n"), ("v", "b"), ("v", "w"), ("f", "p"),
    ]
)


# Nối âm (linking): coda MŨI (n/m/ŋ) của một từ đứng trước nguyên âm đầu từ kế tiếp hay
# bị recognizer nghe thành âm TẮC CÙNG VỊ TRÍ — "in order" /ɪn/→/ɪt/, "some apples" /m/→/p/,
# "long ago" /ŋ/→/k/. Đây là artifact giải phóng/nối coda khi linking, KHÔNG phải nuốt âm
# (deletion) cũng KHÔNG phải lỗi người đọc. KHÁC với _FINAL_DELETION (l1_vietnamese.py) vốn
# CỐ Ý không khoan dung nuốt nasal cuối: ở đây nasal VẪN được phát, chỉ bị gán nhãn sai thành
# stop homorganic. Caller (_align_points) còn ràng buộc thêm: chỉ áp cho FUNCTION_WORDS + có
# nguyên âm nối ngay sau → giữ phân biệt thật như "in" vs "it" ở mọi ngữ cảnh khác.
_NASAL_CODA_STOP_LINKS: Final[frozenset[frozenset[str]]] = frozenset(
    frozenset({normalize_ipa(a), normalize_ipa(b)})
    for a, b in [
        ("n", "t"), ("n", "d"),
        ("m", "p"), ("m", "b"),
        ("ŋ", "k"), ("ŋ", "ɡ"),
    ]
)


def is_nasal_coda_linking(expected: str, predicted: str) -> bool:
    """True nếu (expected, predicted) là cặp coda-mũi↔stop-cùng-vị-trí của nối âm.

    expected là nasal /n m ŋ/, predicted là stop homorganic (n↔t/d, m↔p/b, ŋ↔k/ɡ).
    Ngữ cảnh nối âm (function word + nguyên âm theo sau) do caller kiểm tra riêng.
    """
    return (
        frozenset({normalize_ipa(expected), normalize_ipa(predicted)})
        in _NASAL_CODA_STOP_LINKS
    )


def is_real_error_substitution(expected: str, predicted: str, *, sim_floor: float) -> bool:
    """True nếu cặp sub (expected→predicted) là lỗi học viên THẬT → KHÔNG gate thành noise.

    Được bảo vệ khi: (a) similarity ≥ sim_floor (đủ gần → near-pair/cùng class+place, vd
    θ→s, æ→e), HOẶC (b) nằm trong `_REAL_ERROR_SUBS` (lỗi VN cùng-loại nhưng sim thấp,
    vd th-stopping ð→d). Mọi cặp khác (sim thấp & không trong bảng — vd f→l, b→f, phụ-âm→
    nguyên-âm) là ứng viên recognizer-noise.
    """
    if phoneme_similarity(predicted, expected) >= sim_floor:
        return True
    return frozenset({normalize_ipa(expected), normalize_ipa(predicted)}) in _REAL_ERROR_SUBS


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

# ── Layer 2: CMUdict (~134k words, validated ARPAbet with stress digits) ──────

_cmudict_data: dict | None = None


def _get_cmudict() -> dict:
    """Lazy-load CMU Pronouncing Dictionary. Singleton — loaded once on first call."""
    global _cmudict_data
    if _cmudict_data is None:
        import cmudict as _pkg  # noqa: PLC0415
        _cmudict_data = _pkg.dict()
    return _cmudict_data


# Reduced (unstressed) ARPAbet vowels — natural weak-form markers. Their presence
# nudges selection toward conversational reduced pronunciations.
_REDUCED_ARPABET: Final[frozenset[str]] = frozenset({"AH0", "IH0", "UH0"})


def _primary_stress_count(entry: list[str]) -> int:
    """Số nguyên âm mang nhấn chính (token ARPAbet kết thúc bằng '1')."""
    return sum(1 for t in entry if t.endswith("1"))


def _reduction_count(entry: list[str]) -> int:
    """Số nguyên âm rút gọn (AH0/IH0/UH0) — dấu hiệu dạng yếu tự nhiên."""
    return sum(1 for t in entry if t in _REDUCED_ARPABET)


def _entry_score(entry: list[str], *, is_function_word: bool) -> float:
    """Điểm chọn entry (THẤP hơn = tốt hơn).

    score = 3·|primary − target| + 0.1·len − 0.05·reduction
      - stress (w=3.0) CHI PHỐI: function word → ưu tiên 0 nhấn chính (dạng yếu);
        content word → ưu tiên đúng 1 nhấn chính.
      - len (w=0.1) là tie-break CHÍNH, giữ hành vi gần với min(len) cũ.
      - reduction (w=−0.05, thưởng) là nudge dưới mức len → thiên về dạng rút gọn
        tự nhiên khi mọi thứ khác bằng nhau. Dấu ÂM (thưởng) là chủ ý: reduction
        nhiều hơn ⇒ score thấp hơn ⇒ được chọn.
    """
    target = 0 if is_function_word else 1
    stress = abs(_primary_stress_count(entry) - target)
    return 3.0 * stress + 0.1 * len(entry) - 0.05 * _reduction_count(entry)


def _rank_cmudict_entries(
    entries: list[list[str]], *, is_function_word: bool
) -> list[str]:
    """Chọn phát âm tốt nhất trong nhiều entry CMUdict bằng điểm ngôn ngữ học.

    Thay heuristic min(len) cũ (không có cơ sở ngôn ngữ, phụ thuộc thứ tự CMUdict)
    bằng _entry_score: cấu trúc nhấn (function vs content word) + rút gọn nguyên âm,
    với độ dài làm tie-break. Deterministic: min() ổn định → entry SỚM NHẤT khi điểm
    bằng nhau.

    Examples:
      "usually" entries[0]=8 tokens (spurious W), entries[1]=6 tokens, cùng 1 nhấn →
        len tie-break chọn entries[1]: Y UW1 ZH AH0 L IY0 → /juːʒəliː/. ✓
      "the"     function word → ưu tiên 0 nhấn chính: DH AH0 (/ðə/) thắng DH AH1/DH IY0. ✓
    """
    return min(entries, key=lambda e: _entry_score(e, is_function_word=is_function_word))


def _lookup_cmudict(word: str) -> list[str] | None:
    """Return ranked CMUdict pronunciation for word, or None if OOV."""
    entries = _get_cmudict().get(word.lower())
    if not entries:
        return None
    return _rank_cmudict_entries(
        entries, is_function_word=word.lower() in FUNCTION_WORDS
    )


def _cmudict_entry_is_valid(tokens: list[str]) -> bool:
    """Defensive all-or-nothing check: every token base must be a known ARPAbet phoneme.

    CMUdict is a trusted source so this should never fail in practice.
    Returns False for the entire entry if *any* token is invalid — no partial stripping.
    """
    return all(t.rstrip("012") in _VALID_ARPABET_BASES for t in tokens)


# ── Layer 3: eSpeak NG — deterministic rule-based G2P for OOV words ──────────
# Requires: pip install phonemizer  +  apt-get install espeak-ng espeak-ng-data
# Falls through to hard failure ([], []) if espeak-ng binary is unavailable.

_espeak_backend: object | None | bool = None  # None=not tried; False=unavailable


def _get_espeak() -> object | None:
    """Lazy-init eSpeak NG backend (singleton). Returns None if unavailable."""
    global _espeak_backend
    if _espeak_backend is None:
        try:
            from phonemizer.backend import EspeakBackend  # noqa: PLC0415
            _espeak_backend = EspeakBackend(
                "en-us",
                with_stress=True,
                preserve_punctuation=False,
                language_switch="remove-flags",
            )
        except Exception:  # noqa: BLE001 - missing package or espeak-ng binary
            _espeak_backend = False
    return _espeak_backend or None


def _espeak_token_to_canonical(tok: str) -> tuple[str, StressType | None]:
    """Strip stress prefix and normalize one eSpeak phone token.

    Single call to normalize_ipa() — no custom IPA parser.
    Returns (canonical_ipa_symbol, stress_type | None).
    eSpeak places ˈ/ˌ before the vowel nucleus of the stressed syllable.
    """
    stress: StressType | None = None
    if tok.startswith("ˈ"):
        stress = "primary"
        tok = tok[1:]
    elif tok.startswith("ˌ"):
        stress = "secondary"
        tok = tok[1:]
    return normalize_ipa(tok), stress


def _espeak_word_to_symbols_stress(
    word: str,
) -> tuple[list[str], list[StressType | None]] | None:
    """Deterministic eSpeak G2P for OOV words.

    Returns (canonical_symbols, stresses) or None if output is empty or backend unavailable.
    Validation: per-symbol against _IPA_PHONEMES_NORM after normalize_ipa().
    Invalid symbols are strictly dropped (not the whole prediction).
    Empty result after drop → None.
    """
    backend = _get_espeak()
    if backend is None:
        return None

    from phonemizer.separator import Separator  # noqa: PLC0415

    try:
        raw: str = backend.phonemize(
            [word],
            separator=Separator(phone=" ", word="", syllable=""),
            strip=True,
        )[0]
    except Exception:  # noqa: BLE001
        logger.warning("ipa_resolve word=%r source=espeak ERROR", word)
        return None

    if not raw.strip():
        return None

    symbols: list[str] = []
    stresses: list[StressType | None] = []
    syllables = 0
    for tok in raw.split():
        canonical, stress = _espeak_token_to_canonical(tok)
        if canonical in _IPA_PHONEMES_NORM:
            symbols.append(canonical)
            stresses.append(stress)
            if canonical in _VOWELS_NORM:
                syllables += 1
        else:
            logger.debug("ipa_resolve word=%r espeak_drop_symbol=%r (from %r)", word, canonical, tok)

    if not symbols:
        return None
    return _finalize_stress(symbols, stresses, syllables)


def _arpabet_tokens_to_ipa_stress(
    tokens: list[str],
) -> tuple[list[str], list[StressType | None], int]:
    """ARPAbet token (kèm/không kèm stress digit) → (IPA symbols, stresses, syllables).

    THUẦN parsing/conversion — KHÔNG áp luật đơn-âm-tiết / length guard (để ở caller).
    - AH split theo nhấn: AH1/AH2 → ʌ; AH0 hoặc thiếu/không hợp lệ digit → ə (fail-soft
      CHỈ cho stress digit của AH, KHÔNG cho base lạ).
    - Base không có trong ARPABET_TO_IPA → BỎ QUA (skip), KHÔNG bịa IPA.
    """
    symbols: list[str] = []
    stresses: list[StressType | None] = []
    syllables = 0
    for raw in tokens:
        if not raw.strip():
            continue
        digit = re.search(r"\d", raw)
        base = re.sub(r"\d", "", raw)  # AH0 → AH
        if base not in ARPABET_TO_IPA:
            continue  # base lạ → không bịa (override validate lúc import; g2p tự lọc)
        d = digit.group() if digit is not None else None
        # AH split: chỉ nhấn rõ (1/2) mới là ʌ; còn lại (gồm thiếu digit) → ə.
        symbol = ("ʌ" if d in ("1", "2") else "ə") if base == "AH" else ARPABET_TO_IPA[base]
        symbols.append(symbol)
        if digit is not None:
            syllables += 1  # digit chỉ trên nguyên âm → đếm âm tiết
        stresses.append("primary" if d == "1" else "secondary" if d == "2" else None)
    return symbols, stresses, syllables


def _finalize_stress(
    symbols: list[str], stresses: list[StressType | None], syllables: int
) -> tuple[list[str], list[StressType | None]]:
    """Áp luật hiển thị nhấn: từ ≤1 âm tiết → bỏ ký hiệu nhấn; guard độ dài khớp."""
    if syllables <= 1:
        stresses = [None] * len(symbols)
    if len(symbols) != len(stresses):
        raise ValueError("IPA symbols and stress list length mismatch")
    return symbols, stresses


# ──────────────────────────────────────────────────────────────────────────────
# Dời ký hiệu nhấn về ĐẦU âm tiết (onset) — CHỈ để HIỂN THỊ
# ──────────────────────────────────────────────────────────────────────────────

# Onset clusters tiếng Anh hợp lệ (2–3 phụ âm), ký hiệu IPA hiển thị. Đây là BẢNG
# HEURISTIC cho việc đặt dấu nhấn — KHÔNG phải syllabifier sonority đầy đủ; có thể
# chia sai vài cụm hiếm/ngoại lai. Hệ quả DUY NHẤT của chia sai là vị trí dấu nhấn
# lệch về mặt hiển thị — KHÔNG bao giờ đổi chuỗi phoneme, alignment hay điểm số.
# (Affricate tʃ/dʒ là 1 token → tự động là onset đơn hợp lệ; đừng dùng bảng này để
# syllabify cho mục đích chấm điểm.)
_LEGAL_ONSETS: Final[frozenset[tuple[str, ...]]] = frozenset({
    # stop + approximant
    ("p", "l"), ("p", "r"), ("p", "j"),
    ("b", "l"), ("b", "r"), ("b", "j"),
    ("t", "r"), ("t", "w"), ("t", "j"),
    ("d", "r"), ("d", "w"), ("d", "j"),
    ("k", "l"), ("k", "r"), ("k", "w"), ("k", "j"),
    ("ɡ", "l"), ("ɡ", "r"), ("ɡ", "w"), ("ɡ", "j"),
    # fricative + approximant
    ("f", "l"), ("f", "r"), ("f", "j"),
    ("v", "j"),
    ("θ", "r"), ("θ", "w"), ("θ", "j"),
    ("ʃ", "r"),
    ("h", "j"),
    # s + consonant
    ("s", "p"), ("s", "t"), ("s", "k"), ("s", "m"), ("s", "n"),
    ("s", "l"), ("s", "w"), ("s", "f"), ("s", "j"),
    # nasal/lateral + j
    ("m", "j"), ("n", "j"), ("l", "j"),
    # s + stop + approximant (3 phụ âm)
    ("s", "p", "l"), ("s", "p", "r"), ("s", "p", "j"),
    ("s", "t", "r"), ("s", "t", "j"),
    ("s", "k", "l"), ("s", "k", "r"), ("s", "k", "w"), ("s", "k", "j"),
})


def place_stress_at_onset(
    symbols: list[str], stresses: list[StressType | None]
) -> list[StressType | None]:
    """Dời mỗi dấu nhấn từ NGUYÊN ÂM sang ĐẦU âm tiết (onset) — trả list song song MỚI.

    CHỈ để HIỂN THỊ. Từ điển đặt dấu nhấn trước cả âm tiết (`/ˈledʒənd/`) chứ không
    trước nguyên âm (`/lˈedʒənd/`). CMU/g2p gắn nhấn trên nguyên âm; hàm này dời dấu
    về phụ âm đầu của onset hợp lệ DÀI NHẤT của âm tiết đó.

    KHÔNG được dùng cho chấm điểm: danh sách `stresses` gốc (trên nguyên âm) mới là
    nguồn cho scoring (severity/reducible/nhân chính). Hàm này tạo kênh hiển thị TÁCH
    BIỆT, không đụng tới list gốc.

    Thuật toán (cho mỗi i có nhấn): gom phụ âm liền trước i lùi tới nguyên âm trước /
    đầu từ → run R; chọn k∈{3,2,1} LỚN NHẤT sao cho k phụ âm cuối của R là onset hợp lệ
    (đơn phụ âm luôn hợp lệ trừ ŋ); đặt dấu trước phụ âm đó (phụ âm trước nó là coda âm
    tiết trước). R rỗng (âm tiết mở đầu nguyên âm) hoặc không có onset hợp lệ → giữ trên
    nguyên âm. `_LEGAL_ONSETS` là heuristic hiển thị, không phải syllabifier đầy đủ.
    """
    display: list[StressType | None] = [None] * len(symbols)
    for i, st in enumerate(stresses):
        if st is None:
            continue
        j = i
        while j > 0 and not is_vowel(symbols[j - 1]):
            j -= 1
        run = symbols[j:i]  # phụ âm liền trước nguyên âm nhấn
        target = i  # mặc định: giữ trên nguyên âm (âm tiết mở đầu nguyên âm / không có onset)
        for k in range(min(3, len(run)), 0, -1):
            cluster = tuple(run[len(run) - k:])
            if (k == 1 and cluster[0] != "ŋ") or (k >= 2 and cluster in _LEGAL_ONSETS):
                target = i - k
                break
        display[target] = st
    return display


def word_to_ipa(word: str) -> list[str]:
    """Chuyển 1 từ tiếng Anh thành danh sách IPA phonemes.

    Thin wrapper của word_to_ipa_with_stress() — giữ chữ ký cũ cho các caller
    chỉ cần danh sách symbol (không cần nhấn âm).
    """
    symbols, _stress = word_to_ipa_with_stress(word)
    return symbols


def word_to_ipa_with_stress(
    word: str,
) -> tuple[list[str], list[StressType | None]]:
    """Convert one English word to IPA symbols + parallel stress list.

    4-layer deterministic pipeline (first match wins):
      Layer 1 — _WORD_IPA_OVERRIDES: HARD PRIORITY. Bypasses all validation and
        fallback. For proper nouns not in CMUdict and genuine dialect exceptions.
      Layer 2 — CMUdict (~134k words, validated ARPAbet + stress digits).
        Accepted all-or-nothing: if any token base is invalid the entry is rejected
        and falls through. Should never happen with a well-formed CMUdict package.
      Layer 3 — eSpeak NG (deterministic rule-based G2P for OOV words).
        Per-symbol validation against _IPA_PHONEMES_NORM after normalize_ipa().
        Invalid symbols strictly dropped; empty result → falls through.
      Layer 4 — HARD FAILURE: returns ([], []) + warning log.
        Callers must not treat this as a valid (silent) pronunciation.

    Stress marks CHỈ dùng để hiển thị — KHÔNG chèn vào chuỗi DTW (wav2vec không
    có dấu nhấn → sẽ lệch). Monosyllabic words (≤1 syllable) get all-None stress
    per IPA convention. Returns (symbols, stresses) with len(symbols)==len(stresses).

    Determinism: same input → identical output under fixed versions of CMUdict package,
    espeak-ng binary, and phonemizer library.
    """
    key = word.lower().strip(".,;:!?\"'()[]{}")
    if not key:
        return [], []

    # Layer 1: HARD PRIORITY — bypasses all validation and fallback logic.
    if key in _WORD_IPA_OVERRIDES:
        logger.debug("ipa_resolve word=%r source=override", key)
        return _finalize_stress(*_arpabet_tokens_to_ipa_stress(_WORD_IPA_OVERRIDES[key]))

    # Layer 2: CMUdict primary lexicon — all-or-nothing acceptance.
    cmu = _lookup_cmudict(key)
    if cmu is not None:
        if _cmudict_entry_is_valid(cmu):
            logger.debug("ipa_resolve word=%r source=cmudict", key)
            return _finalize_stress(*_arpabet_tokens_to_ipa_stress(cmu))
        else:
            logger.warning(
                "ipa_resolve word=%r source=cmudict INVALID_TOKENS=%r — falling through",
                key, cmu,
            )

    # Layer 3: eSpeak NG — deterministic rule-based G2P for OOV words.
    espeak = _espeak_word_to_symbols_stress(key)
    if espeak is not None:
        logger.debug("ipa_resolve word=%r source=espeak", key)
        return espeak

    # Layer 4: hard failure — callers must not treat [] as a valid pronunciation.
    logger.warning("ipa_resolve word=%r source=failed", key)
    return [], []


def text_to_ipa_sequence_with_spans(
    text: str,
) -> tuple[list[str], list[WordSpan], list[StressType | None], list[StressType | None]]:
    """Chuyển text → (phonemes tham chiếu, WordSpan, nhấn âm, nhấn-hiển-thị) song song.

    Input:  "The quick brown fox"
    Output: (["ð", "ə", "k", "w", "ɪ", "k", ...],
             [WordSpan("The", 0, 2), WordSpan("quick", 2, 6), ...],
             [None, None, None, None, "primary", None, ...],   # nhấn TRÊN nguyên âm (scoring)
             [None, None, None, "primary", None, None, ...])    # nhấn dời về ONSET (hiển thị)

    Phonemes, spans và stress được build trong CÙNG vòng lặp tokenize/
    word_to_ipa_with_stress, nên luôn khớp 1-1 theo index: từ nào không tra được
    IPA (dropped) sẽ KHÔNG sinh span và KHÔNG đẩy phoneme/stress nào → index của
    các từ sau không bị lệch. `stress` song song 1-1 với `phonemes`.

    `display_stress`: dấu nhấn ĐÃ dời về đầu âm tiết (onset) qua place_stress_at_onset,
    tính theo TỪNG TỪ → CHỈ để render `/ˈledʒənd/`. Scoring vẫn dùng `stress` (trên
    nguyên âm). Hai list cùng độ dài, song song 1-1 với `phonemes`.

    Span dùng để map ngược lỗi phoneme (theo position trong reference sequence)
    về đúng từ. `word` giữ nguyên dạng token (re.findall đã loại dấu câu) và giữ
    nguyên hoa/thường để hiển thị; word_to_ipa tự lower() khi tra từ điển.
    """
    if not text:
        return [], [], [], []

    words = re.findall(r"[a-zA-Z'-]+", text)
    phonemes: list[str] = []
    spans: list[WordSpan] = []
    stress: list[StressType | None] = []
    display_stress: list[StressType | None] = []
    dropped: list[str] = []
    for word in words:
        word_phones, word_stress = word_to_ipa_with_stress(word)
        if word_phones:
            start = len(phonemes)
            phonemes.extend(word_phones)
            stress.extend(word_stress)
            # Nhấn-hiển-thị dời về onset, tính theo PHẠM VI TỪ (onset không vượt biên từ).
            display_stress.extend(place_stress_at_onset(word_phones, word_stress))
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

    return phonemes, spans, stress, display_stress


def text_to_ipa_sequence(text: str) -> list[str]:
    """Chuyển đoạn text thành danh sách phonemes tham chiếu.

    Input: "The quick brown fox"
    Output: ["DH", "AH", "K", "W", "IH", "K", ...] → IPA

    Thin wrapper của text_to_ipa_sequence_with_spans() — giữ chữ ký cũ cho các
    caller chỉ cần phoneme list (không cần word mapping).
    """
    phonemes, _spans, _stress, _disp = text_to_ipa_sequence_with_spans(text)
    return phonemes
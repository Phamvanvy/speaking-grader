"""IPA phoneme set và ánh xạ word → IPA cho tiếng Anh.

Cung cấp:
  - ENGLISH_IPA_PHONEMES: tập 44 phonemes tiếng Anh (20 nguyên âm + 24 phụ âm + /hm/)
  - word_to_ipa(): ánh xạ từ → IPA sequence (dựa trên ARPAbet → IPA)
  - text_to_ipa_sequence(): chuyển đoạn text → danh sách phonemes tham chiếu
  - phoneme_similarity(): tính khoảng cách giữa 2 phonemes (cho severity scoring)
"""

from __future__ import annotations

import logging
import re
from typing import Final

from .models import WordSpan

logger = logging.getLogger("toeic.phoneme.ipa")

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

# Các cặp tương đương (sau khi đã bỏ dấu trường ː).
# Gộp back-vowels (cot-caught merger), r-âm, schwa nhấn/không nhấn... — chấp nhận
# được cho chấm phát âm ESL; thà bỏ sót lỗi còn hơn báo sai phát âm đúng.
_IPA_EQUIV: Final[dict[str, str]] = {
    "ɹ": "r",                 # eSpeak r ↔ CMU r
    "ɾ": "t",                 # flap (water/store kiểu Mỹ) ↔ t (allophone, không phải lỗi)
    "g": "ɡ",                 # ascii g ↔ IPA ɡ
    "ɚ": "ɜ", "ɝ": "ɜ",       # r-colored schwa ↔ ER
    "ʌ": "ə", "ɐ": "ə",       # schwa nhấn/không nhấn
    "ɛ": "e",                 # EH
    "ɒ": "ɔ", "ɑ": "ɔ", "o": "ɔ",  # back vowels gộp 1 nhóm
    "oʊ": "əʊ",               # OW
}


def normalize_ipa(phoneme: str) -> str:
    """Quy 1 ký hiệu IPA về dạng chuẩn để so khớp giữa eSpeak và ARPAbet.

    Bỏ dấu trường (ː) rồi áp bảng tương đương. Diphthong là 1 token nên xử lý
    nguyên khối (vd 'oʊ' → 'əʊ').
    """
    p = phoneme.strip().replace("ː", "")
    return _IPA_EQUIV.get(p, p)


def phoneme_similarity(p1: str, p2: str) -> float:
    """Tính độ tương đồng giữa 2 phonemes (0.0 = hoàn toàn khác, 1.0 = giống hệt).

    Algorithm:
      - Giống hệt (sau chuẩn hóa eSpeak↔ARPAbet): 1.0
      - Cùng class + cùng place: 0.7
      - Cùng class hoặc cùng place: 0.4
      - Khác hoàn toàn: 0.0
    """
    if p1 == p2 or normalize_ipa(p1) == normalize_ipa(p2):
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

    Priority:
      1. Built-in dictionary (_COMMON_WORD_PRONUNCIATIONS)
      2. g2p module (grapheme-to-phoneme), instance được cache
      3. Fallback: empty list (caller sẽ handle missing words)
    """
    key = word.lower().strip(".,;:!?\"'()[]{}")
    if not key:
        return []

    # Built-in dictionary first. Lọc theo ARPABET_TO_IPA (giống nhánh g2p) để
    # token không hợp lệ không lọt vào chuỗi IPA tham chiếu thành "phoneme" rác.
    if key in _COMMON_WORD_PRONUNCIATIONS:
        arpabet = _COMMON_WORD_PRONUNCIATIONS[key]
        return [ARPABET_TO_IPA[a] for a in arpabet if a in ARPABET_TO_IPA]

    # Try g2p if available (cached instance — KHÔNG khởi tạo lại mỗi từ)
    transcriber = _get_g2p()
    if transcriber is not None:
        try:
            # g2p_en trả về flat list các ARPAbet token (kèm stress digit + space)
            arpabet_seq = [
                re.sub(r"\d", "", p)  # bỏ stress marker (AH0 → AH)
                for p in transcriber(key)
                if p.strip()
            ]
            ipa = [ARPABET_TO_IPA.get(a, a) for a in arpabet_seq if a in ARPABET_TO_IPA]
            if ipa:
                return ipa
        except Exception:  # noqa: BLE001 - lỗi runtime g2p
            pass

    # Fallback: empty list (caller detect và handle missing words)
    return []


def text_to_ipa_sequence_with_spans(
    text: str,
) -> tuple[list[str], list[WordSpan]]:
    """Chuyển text → (danh sách phonemes tham chiếu, danh sách WordSpan).

    Input:  "The quick brown fox"
    Output: (["ð", "ə", "k", "w", "ɪ", "k", ...],
             [WordSpan("The", 0, 2), WordSpan("quick", 2, 6), ...])

    Phonemes và spans được build trong CÙNG vòng lặp tokenize/word_to_ipa, nên
    luôn khớp 1-1 theo index: từ nào không tra được IPA (dropped) sẽ KHÔNG sinh
    span và KHÔNG đẩy phoneme nào → index của các từ sau không bị lệch.

    Span dùng để map ngược lỗi phoneme (theo position trong reference sequence)
    về đúng từ. `word` giữ nguyên dạng token (re.findall đã loại dấu câu) và giữ
    nguyên hoa/thường để hiển thị; word_to_ipa tự lower() khi tra từ điển.
    """
    if not text:
        return [], []

    words = re.findall(r"[a-zA-Z'-]+", text)
    phonemes: list[str] = []
    spans: list[WordSpan] = []
    dropped: list[str] = []
    for word in words:
        word_phones = word_to_ipa(word)
        if word_phones:
            start = len(phonemes)
            phonemes.extend(word_phones)
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

    return phonemes, spans


def text_to_ipa_sequence(text: str) -> list[str]:
    """Chuyển đoạn text thành danh sách phonemes tham chiếu.

    Input: "The quick brown fox"
    Output: ["DH", "AH", "K", "W", "IH", "K", ...] → IPA

    Thin wrapper của text_to_ipa_sequence_with_spans() — giữ chữ ký cũ cho các
    caller chỉ cần phoneme list (không cần word mapping).
    """
    phonemes, _spans = text_to_ipa_sequence_with_spans(text)
    return phonemes
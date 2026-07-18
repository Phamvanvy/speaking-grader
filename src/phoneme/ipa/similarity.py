"""Phoneme similarity, tolerance (phonemes_match) và deletion severity/penalty.

NGUỒN CHÂN LÝ điều khiển penalty/severity của substitution + deletion. Thuần hàm,
không monkeypatch — phụ thuộc phoneme_set (normalize_ipa, is_vowel, class sets).
"""

from __future__ import annotations

import functools
from typing import Final

from .phoneme_set import (
    BACK_VOWEL_SPLIT_ENABLED,
    FUNCTION_WORDS,
    _APPROXIMANTS,
    _FRICATIVES,
    _NASALS,
    _PLOSIVES,
    _VOWELS,
    _VOWELS_NORM,
    is_vowel,
    normalize_ipa,
)


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
    # BACK_VOWEL_SPLIT: ɑ tách khỏi ɔ trong normalize → cặp thành sub thật ở đây.
    # 0.60 → severity "medium": HIỆN với người học (popup strict bắt được star/store)
    # nhưng penalty vừa phải (0.4·conf) cho giọng Mỹ cot-caught-merged ở câu dài.
    # Flag OFF: KHÔNG thêm — hai symbol cùng normalize về ɔ, key suy biến 1 phần tử.
    *([_near("ɑ", "ɔ", 0.60)] if BACK_VOWEL_SPLIT_ENABLED else []),
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


# Stop cuối từ có thể NUỐT khi nối từ (connected speech elision): "test preparation"
# → /tes-prep/ là phát âm bản xứ ĐÚNG khi từ kế bắt đầu bằng phụ âm. CHỈ stop — nuốt
# fricative/liquid/nasal cuối (/s z l n/...) là lỗi L1 VN kinh điển, KHÔNG được tha.
_ELIDABLE_FINAL_STOPS: Final[frozenset[str]] = frozenset({"t", "d", "p", "b", "k", "ɡ"})


def is_elidable_stop(phoneme: str) -> bool:
    """True nếu phoneme (sau normalize) là stop được phép nuốt ở cuối từ khi nối từ."""
    return normalize_ipa(phoneme) in _ELIDABLE_FINAL_STOPS


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

# Biến thể phát âm hợp lệ THEO TỪ (cả hai dạng đều chuẩn trong từ điển) — vd "with"
# /wɪθ/ ↔ /wɪð/. Key = từ đã lower/strip; value = các cặp hoán đổi được chấp nhận.
# Giữ NHỎ + tường minh: chỉ thêm khi cả hai biến thể có trong từ điển phát âm chuẩn.
_WORD_VARIANT_PAIRS: Final[dict[str, frozenset[frozenset[str]]]] = {
    "with": frozenset({frozenset({"θ", "ð"})}),
}

# Biến thể theo từ CHỈ ở vị trí rút gọn được (nguyên âm KHÔNG phải nhân chính) — vd
# "advantage" /ədˈvæn-/ ↔ /ædˈvæn-/ (Cambridge liệt kê cả hai; CMUdict chỉ có AE0).
# Guard reducible để KHÔNG tha ə cho nguyên âm nhấn (/væn/ → /vən/ vẫn là lỗi thật).
_WORD_REDUCIBLE_VARIANT_PAIRS: Final[dict[str, frozenset[frozenset[str]]]] = {
    w: frozenset({frozenset({"æ", "ə"})})
    for w in ("advantage", "advantages", "advantaged")
}


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
    word_key = word.lower().strip(".,;:!?\"'()[]{}") if word else ""
    # Biến thể theo từ: cả hai dạng đều là phát âm chuẩn ("with" θ↔ð) → khớp.
    if word_key and pair in _WORD_VARIANT_PAIRS.get(word_key, frozenset()):
        return True
    in_func = word_key in FUNCTION_WORDS
    if reducible is None:
        reducible = stress is None or in_func
    # Biến thể theo từ Ở VỊ TRÍ rút gọn được ("advantage" æ↔ə âm đầu) → khớp.
    variants = _WORD_REDUCIBLE_VARIANT_PAIRS.get(word_key)
    if reducible and variants and pair in variants:
        return True
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
# phải lỗi người đọc (dạng đã normalize: ð, h, schwa, ɪ, ʊ, w). /w/ là glide, wav2vec
# hay gộp vào nguyên âm kế ("website" → /ebsaɪt/ trên cả giọng máy đọc chuẩn); người
# học VN hầu như không nuốt /w/ onset nên rủi ro giấu lỗi thật rất thấp.
_RECOGNIZER_PRONE: Final[frozenset[str]] = frozenset({"ð", "h", "ə", "ɪ", "ʊ", "w"})

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

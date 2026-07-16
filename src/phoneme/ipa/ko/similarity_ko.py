"""Similarity / tolerance / deletion severity cho tiếng Hàn — mirror similarity.py (EN).

Chữ ký từng hàm GIỐNG HỆT bản EN (scoring gọi qua LangProfile, không phân biệt
ngôn ngữ). Thang giá trị giữ cùng scale với EN (exact 1.0 / near-pair theo bảng /
class-heuristic 0.7-0.4 / khác hẳn 0.0) để penalty và severity bucket so sánh được
giữa hai ngôn ngữ.

Điểm khác biệt CÓ CHỦ ĐÍCH so với EN:
  - Voiced lenis (k↔ɡ, t↔d, p↔b, tɕ↔dʑ/dʒ) là ALLOPHONE theo vị trí (giữa 2 âm
    hữu thanh) — recognizer emit voiced cho lenis intervocalic là nhãn TRUNG THỰC
    của phát âm đúng → phonemes_match, không phải lỗi.
  - Batchim: coda stop bị thiếu → severity "low" (coda tiếng Hàn là unreleased
    [p̚ t̚ k̚], CTC recognizer thường không thấy) — có nguyên tắc, tương đương
    _RECOGNIZER_PRONE của EN.
  - _REAL_ERROR_SUBS_KO khởi đầu RỖNG (đúng policy: chỉ thêm khi telemetry cho
    thấy nhất quán là lỗi học viên thật).
"""

from __future__ import annotations

import functools
from typing import Final

from .phoneme_set_ko import _VOWELS_KO, is_vowel_ko, normalize_ipa_ko

# ── nhóm âm (cho heuristic class/place) ──────────────────────────────────────

_PLOSIVES_KO: Final[frozenset[str]] = frozenset({
    "p", "t", "k", "pʰ", "tʰ", "kʰ", "p͈", "t͈", "k͈", "b", "d", "ɡ",
})
_AFFRICATES_KO: Final[frozenset[str]] = frozenset({"tɕ", "tɕʰ", "t͈ɕ", "dʑ", "dʒ"})
_FRICATIVES_KO: Final[frozenset[str]] = frozenset({"s", "s͈", "h"})
_NASALS_KO: Final[frozenset[str]] = frozenset({"m", "n", "ŋ"})
_LIQUIDS_KO: Final[frozenset[str]] = frozenset({"l", "ɾ"})
_GLIDES_KO: Final[frozenset[str]] = frozenset({"j", "w"})

_PLACE_GROUPS_KO: Final[list[frozenset[str]]] = [
    frozenset({"p", "pʰ", "p͈", "b", "m"}),                    # bilabial
    frozenset({"t", "tʰ", "t͈", "d", "n", "s", "s͈", "l", "ɾ"}),  # alveolar
    frozenset({"tɕ", "tɕʰ", "t͈ɕ", "dʑ", "dʒ", "j"}),          # palatal
    frozenset({"k", "kʰ", "k͈", "ɡ", "ŋ", "w"}),               # velar(+labiovelar)
]


def _same_class_ko(p1: str, p2: str) -> bool:
    for cls in (
        _PLOSIVES_KO, _AFFRICATES_KO, _FRICATIVES_KO, _NASALS_KO,
        _LIQUIDS_KO, _GLIDES_KO, _VOWELS_KO,
    ):
        if p1 in cls and p2 in cls:
            return True
    return False


def _same_place_ko(p1: str, p2: str) -> bool:
    for group in _PLACE_GROUPS_KO:
        if p1 in group and p2 in group:
            return True
    return False


def _near(a: str, b: str, score: float) -> tuple[frozenset[str], float]:
    return frozenset({normalize_ipa_ko(a), normalize_ipa_ko(b)}), score


# Cặp âm gần — tuyển tay theo cặp nhầm lẫn kinh điển của recognizer + học viên.
# Triad laryngeal (lenis/aspirated/tense) là NHẦM LẪN đặc trưng tiếng Hàn: giữ
# hiện diện như lỗi nhẹ/trung bình, không xoá trắng (contrast có nghĩa).
_NEAR_PAIRS_KO: Final[dict[frozenset[str], float]] = dict([
    # laryngeal triads — lenis↔tense / lenis↔aspirated gần hơn aspirated↔tense
    _near("k", "k͈", 0.75), _near("k", "kʰ", 0.75), _near("kʰ", "k͈", 0.65),
    _near("t", "t͈", 0.75), _near("t", "tʰ", 0.75), _near("tʰ", "t͈", 0.65),
    _near("p", "p͈", 0.75), _near("p", "pʰ", 0.75), _near("pʰ", "p͈", 0.65),
    _near("tɕ", "t͈ɕ", 0.75), _near("tɕ", "tɕʰ", 0.75), _near("tɕʰ", "t͈ɕ", 0.65),
    _near("s", "s͈", 0.75),
    # nguyên âm — cặp nhầm phổ biến (học viên VN + recognizer)
    _near("ɯ", "u", 0.80),
    _near("ʌ", "o", 0.75),
    _near("ʌ", "a", 0.60),
    _near("ɯ", "i", 0.60),
    _near("e", "i", 0.70),
    _near("o", "u", 0.75),
    # Bench M2 (native TTS, 2026-07-16) — artifact recognizer trên audio ĐÚNG:
    # i nghe thành glide j ở hiatus (있어요 → j iː...); nasal đầu từ nghe thành
    # stop cùng chỗ (물→bul, 니→di) — denasalization đầu từ là hiện tượng phát âm
    # THẬT của tiếng Hàn, không phải lỗi học viên.
    _near("i", "j", 0.80),
    _near("m", "b", 0.75),
    _near("n", "d", 0.75),
])

# Lỗi học viên THẬT có similarity thấp cần bảo vệ khỏi noise gate — RỖNG ở v1,
# chỉ thêm từ bằng chứng telemetry (policy giống _REAL_ERROR_SUBS bản EN).
_REAL_ERROR_SUBS_KO: Final[frozenset[frozenset[str]]] = frozenset()

# Allophone — biến thể phát âm ĐÚNG (không phải lỗi):
#  - lenis voiced giữa 2 âm hữu thanh (부부 [pubu]: recognizer emit b cho ㅂ thứ 2)
#  - ㄹ: ɾ (onset) ↔ l (coda/geminate) — cùng phoneme
_ALLOPHONE_PAIRS_KO: Final[frozenset[frozenset[str]]] = frozenset({
    frozenset({"k", "ɡ"}),
    frozenset({"t", "d"}),
    frozenset({"p", "b"}),
    frozenset({"tɕ", "dʑ"}),
    frozenset({"tɕ", "dʒ"}),
    frozenset({"ɾ", "l"}),
})


@functools.lru_cache(maxsize=8192)
def phoneme_similarity_ko(p1: str, p2: str) -> float:
    """Độ tương đồng 2 âm tiếng Hàn (0.0..1.0) — cùng scale với bản EN.

    1. Bằng nhau sau normalize (ㅐ/ㅔ merger...) → 1.0
    2. Allophone (lenis voiced, ɾ↔l) → 0.9 (gần tuyệt đối; phonemes_match đã cho
       khớp hẳn, giá trị này chỉ dùng khi DTW so chéo âm của từ khác)
    3. Bảng near-pair (triad laryngeal, cặp nguyên âm) → score bảng
    4. Heuristic: cùng class + cùng place 0.7; một trong hai 0.4
    5. Khác hẳn → 0.0
    """
    n1, n2 = normalize_ipa_ko(p1), normalize_ipa_ko(p2)
    if n1 == n2:
        return 1.0
    pair = frozenset({n1, n2})
    if pair in _ALLOPHONE_PAIRS_KO:
        return 0.9
    near = _NEAR_PAIRS_KO.get(pair)
    if near is not None:
        return near
    same_cls = _same_class_ko(n1, n2)
    same_place = _same_place_ko(n1, n2)
    if same_cls and same_place:
        return 0.7
    if same_cls or same_place:
        return 0.4
    return 0.0


def phonemes_match_ko(
    expected: str,
    predicted: str,
    *,
    stress: str | None = None,
    word: str | None = None,
    reducible: bool | None = None,
) -> bool:
    """True nếu predicted là phát âm ĐÚNG của expected (rộng hơn identical).

    Tiếng Hàn không có stress/vowel-reduction kiểu EN → stress/reducible nhận để
    giữ chữ ký LangProfile nhưng không dùng. Khớp khi: bằng nhau sau normalize
    HOẶC là allophone (lenis voiced theo vị trí, ɾ↔l).
    """
    e, p = normalize_ipa_ko(expected), normalize_ipa_ko(predicted)
    if e == p:
        return True
    return frozenset({e, p}) in _ALLOPHONE_PAIRS_KO


def is_real_error_substitution_ko(
    expected: str, predicted: str, *, sim_floor: float
) -> bool:
    """True nếu cặp sub là lỗi học viên THẬT → KHÔNG gate thành recognizer noise."""
    if phoneme_similarity_ko(predicted, expected) >= sim_floor:
        return True
    return (
        frozenset({normalize_ipa_ko(expected), normalize_ipa_ko(predicted)})
        in _REAL_ERROR_SUBS_KO
    )


# Coda stop tiếng Hàn là unreleased [p̚ t̚ k̚] — CTC recognizer thường không emit
# token nào cho chúng; thiếu coda stop nhiều khả năng là recognizer, không phải
# học viên (tương đương _RECOGNIZER_PRONE của EN nhưng theo NGUYÊN TẮC vị trí).
_CODA_STOPS_KO: Final[frozenset[str]] = frozenset({"p", "t", "k"})

_SEVERITY_PENALTY_KO: Final[dict[str, float]] = {"low": 0.1, "medium": 0.5, "high": 0.9}


def deletion_severity_ko(
    phoneme: str, *, is_onset: bool = False, stress: str | None = None
) -> str:
    """Severity cho 1 âm reference bị THIẾU.

    - Glide j/w: low (recognizer hay gộp vào nguyên âm kế — giống /w/ bản EN).
    - h: low (ㅎ giữa 2 âm hữu thanh yếu/tan trong khẩu ngữ — 전화 [저놔]).
    - Nguyên âm: low — bench M2 (48 clip native, 2026-07-16): model nuốt nguyên
      âm hàng loạt trên chính audio bản xứ (ɯ/ʌ del ×10 mỗi loại) → deletion
      nguyên âm là recognizer-prone với model này (cùng policy _RECOGNIZER_PRONE
      bản EN). Lỗi nguyên âm THẬT của học viên vẫn hiện qua SUBSTITUTION.
    - Phụ âm onset: high (như EN — nuốt onset là lỗi nặng).
    - Phụ âm coda: stop (p t k) low (unreleased); sonorant (n m ŋ l) medium
      (thiếu batchim mũi là lỗi thật đáng nhắc).
    """
    p = normalize_ipa_ko(phoneme)
    if p in ("j", "w", "h"):
        return "low"
    if p in _VOWELS_KO:
        return "low"
    if is_onset:
        return "high"
    if p in _CODA_STOPS_KO:
        return "low"
    return "medium"


def deletion_penalty_ko(
    phoneme: str, *, is_onset: bool = False, stress: str | None = None
) -> float:
    """Penalty liên tục qua bucket severity — cùng bảng giá trị với EN."""
    return _SEVERITY_PENALTY_KO[
        deletion_severity_ko(phoneme, is_onset=is_onset, stress=stress)
    ]


def is_elidable_stop_ko(phoneme: str) -> bool:
    """Connected-speech elision là rule TIẾNG ANH — tiếng Hàn không áp → luôn False.

    (Core ép connected_speech_enabled=False khi lang=ko; hàm này chỉ để LangProfile
    đủ chữ ký + an toàn nếu flag bị bật nhầm.)
    """
    return False


def is_nasal_coda_linking_ko(expected: str, predicted: str) -> bool:
    """Nasal-coda linking là rule TIẾNG ANH — luôn False (xem is_elidable_stop_ko)."""
    return False

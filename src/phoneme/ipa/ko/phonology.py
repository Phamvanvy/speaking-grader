"""Rule engine 표준 발음법 (chuẩn phát âm tiếng Hàn) — áp PER EOJEOL, thứ tự cố định.

Input: list [cho, jung, jong] mutable (từ hangul.decompose_word). Output: cùng list
đã biến đổi, jong chỉ còn 1 trong 7 coda chuẩn (hoặc "").

Thứ tự áp rule (CỐ ĐỊNH — đổi thứ tự là đổi kết quả):
  1. Palatalization 구개음화 (điều 17):  같이→가치, 굳이→구지
  2. ㅎ rules 격음화/ㅎ탈락 (điều 12):    좋다→조타, 좋아→조아, 입학→이팍, 많아→마나
  3. Liaison 연음 (điều 13/14):          옷이→오시, 앉아→안자, 읽어→일거, 있어→이써
  4. Coda neutralization 음절의 끝소리 (điều 9/10/11): 앞→압, 밖→박, 넋→넉
  5. Liquid rules 유음화/ㄹ비음화 (điều 19/20): 신라→실라, 칼날→칼랄, 침략→침냑, 독립→(동닙 sau 6)
  6. Nasalization 비음화 (điều 18):      국물→궁물, 입니다→임니다, 닫는→단는
  7. Tensification 경음화 (điều 23):     학교→학꾜, 국밥→국빱, 옆집→엽찝

Ngoài scope v1 (cần morphology/từ điển — dùng _WORD_KO_OVERRIDES trong ko/__init__):
관형사형 -ㄹ tensification (할 것을), 사이시옷 (콧등), ㄴ-insertion (솜이불, điều 29),
ngoại lệ 밟- (밟다→밥따), liaison từ ghép nguyên âm thật (겉옷→거돋, điều 15).
Sandhi LIÊN từ (cross-eojeol) cố ý KHÔNG áp — xem docstring ko/__init__.
"""

from __future__ import annotations

from typing import Final

Syllable = list[str]  # [cho, jung, jong] — mutable

# ── bảng tra ─────────────────────────────────────────────────────────────────

# Coda neutralization (điều 9): mọi coda đơn → 1 trong 7 âm chuẩn.
_NEUTRAL_SINGLE: Final[dict[str, str]] = {
    "ㄱ": "ㄱ", "ㄲ": "ㄱ", "ㅋ": "ㄱ",
    "ㄴ": "ㄴ",
    "ㄷ": "ㄷ", "ㅅ": "ㄷ", "ㅆ": "ㄷ", "ㅈ": "ㄷ", "ㅊ": "ㄷ", "ㅌ": "ㄷ", "ㅎ": "ㄷ",
    "ㄹ": "ㄹ",
    "ㅁ": "ㅁ",
    "ㅂ": "ㅂ", "ㅍ": "ㅂ",
    "ㅇ": "ㅇ",
}

# Coda đôi → coda còn lại khi ĐỨNG CUỐI/trước phụ âm (điều 10/11). Ngoại lệ
# morphology (밟다, 넓죽하다, 맑게...) → override theo từ.
_DOUBLE_CODA_KEEP: Final[dict[str, str]] = {
    "ㄳ": "ㄱ", "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄺ": "ㄱ", "ㄻ": "ㅁ", "ㄼ": "ㄹ",
    "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㄿ": "ㅂ", "ㅀ": "ㄹ", "ㅄ": "ㅂ",
}

# Coda đôi → (coda ở lại, phụ âm CHUYỂN sang onset kế) khi liaison (điều 14).
# Phần chuyển là chữ cái thứ hai; ㅅ chuyển thành ㅆ theo đúng điều 14 (값이→갑씨).
_DOUBLE_CODA_SPLIT: Final[dict[str, tuple[str, str]]] = {
    "ㄳ": ("ㄱ", "ㅆ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅆ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"),
    "ㅄ": ("ㅂ", "ㅆ"),
    # ㄶ/ㅀ + nguyên âm: ㅎ rơi (điều 12-4), phần còn lại nối — xử lý ở _h_rules
    # trước khi tới liaison, nhưng giữ entry để an toàn nếu còn sót.
    "ㄶ": ("", "ㄴ"), "ㅀ": ("", "ㄹ"),
}

# ㅎ hoá bật hơi (điều 12-1): lenis + ㅎ (2 chiều) → aspirated.
_ASPIRATE: Final[dict[str, str]] = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}

# Tensification sau coda tắc (điều 23): lenis onset → tense.
_TENSE: Final[dict[str, str]] = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}

# Nasalization coda tắc trước mũi (điều 18): ㄱ→ㅇ, ㄷ→ㄴ, ㅂ→ㅁ.
_NASALIZE_CODA: Final[dict[str, str]] = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}

_OBSTRUENT_CODA: Final[frozenset[str]] = frozenset({"ㄱ", "ㄷ", "ㅂ"})
_I_VOWELS: Final[frozenset[str]] = frozenset({"ㅣ"})  # 구개음화 chỉ trước ㅣ (v1)


# ── các pass (mỗi pass quét trái→phải 1 lượt) ────────────────────────────────

def _palatalize(sylls: list[Syllable]) -> None:
    """Điều 17: jong ㄷ/ㅌ + syllable kế "이" → ㅈ/ㅊ chuyển sang onset kế.

    PHẢI chạy trước liaison (không thì 같이 thành 가티). ㄾ+이 (핥이다) v1 bỏ qua.
    """
    for cur, nxt in zip(sylls, sylls[1:]):
        if nxt[0] == "ㅇ" and nxt[1] in _I_VOWELS:
            if cur[2] == "ㄷ":
                cur[2] = ""
                nxt[0] = "ㅈ"
            elif cur[2] == "ㅌ":
                cur[2] = ""
                nxt[0] = "ㅊ"


def _h_rules(sylls: list[Syllable]) -> None:
    """Điều 12: ㅎ hoá bật hơi (2 chiều) + ㅎ tan trước nguyên âm.

    - jong (ㅎ|ㄶ|ㅀ) + onset ㄱ/ㄷ/ㅈ → onset ㅋ/ㅌ/ㅊ; phần ㄴ/ㄹ ở lại (많다→만타).
    - jong (ㅎ|ㄶ|ㅀ) + onset ㅅ → ㅆ (닿소→다쏘).
    - jong ㅎ + onset ㄴ → jong ㄴ (놓는→논는); ㄶ→ㄴ, ㅀ→ㄹ.
    - jong (ㅎ|ㄶ|ㅀ) + nguyên âm → ㅎ rơi; ㄴ/ㄹ để lại cho liaison (많아→마나).
    - jong tắc (kể cả chưa neutralize: ㄱㄷㅂㅈ + ㅅㅆㅊㅌㅋㅍ qua neutralize cục bộ)
      + onset ㅎ → onset bật hơi, coda rơi (입학→이팍, 못하다→모타다).
    """
    for cur, nxt in zip(sylls, sylls[1:]):
        jong, onset = cur[2], nxt[0]
        if jong in ("ㅎ", "ㄶ", "ㅀ"):
            keep = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[jong]
            if onset in _ASPIRATE and onset != "ㅂ":  # ㅎ+ㅂ không hoá ㅍ (điều 12 chỉ ㄱㄷㅈ)
                nxt[0] = _ASPIRATE[onset]
                cur[2] = keep
            elif onset == "ㅅ":
                nxt[0] = "ㅆ"
                cur[2] = keep
            elif onset == "ㄴ":
                cur[2] = keep or "ㄴ"
            elif onset == "ㅇ":
                cur[2] = keep  # ㅎ tan; ㄴ/ㄹ còn lại sẽ liaison ở pass sau
        elif onset == "ㅎ":
            neutral = _NEUTRAL_SINGLE.get(jong, "")
            if jong in _DOUBLE_CODA_KEEP:
                neutral = _NEUTRAL_SINGLE.get(_DOUBLE_CODA_KEEP[jong], "")
            if neutral in _OBSTRUENT_CODA or neutral == "ㄷ":
                mapped = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ"}[neutral]
                nxt[0] = mapped
                cur[2] = ""


def _liaison(sylls: list[Syllable]) -> None:
    """Điều 13/14: coda + âm tiết kế bắt đầu nguyên âm (onset ㅇ) → coda nối sang.

    Coda đơn chuyển nguyên trạng (옷이→오시, ㅆ giữ ㅆ: 있어→이써); coda ㅇ không
    chuyển (강아지). Coda đôi: phụ âm đầu ở lại, phụ âm sau chuyển (ㅅ→ㅆ điều 14).
    """
    for cur, nxt in zip(sylls, sylls[1:]):
        jong = cur[2]
        if not jong or nxt[0] != "ㅇ":
            continue
        if jong == "ㅇ":
            continue
        if jong in _DOUBLE_CODA_SPLIT:
            stay, move = _DOUBLE_CODA_SPLIT[jong]
            cur[2] = stay
            nxt[0] = move
        else:
            cur[2] = ""
            nxt[0] = jong


def _neutralize(sylls: list[Syllable]) -> None:
    """Điều 9/10/11: coda còn lại → 7 âm chuẩn; coda đôi → 1 phụ âm."""
    for syl in sylls:
        jong = syl[2]
        if not jong:
            continue
        if jong in _DOUBLE_CODA_KEEP:
            jong = _DOUBLE_CODA_KEEP[jong]
        syl[2] = _NEUTRAL_SINGLE.get(jong, jong)


def _liquid_rules(sylls: list[Syllable]) -> None:
    """Điều 19/20: ㄹ hoá (ㄴ+ㄹ / ㄹ+ㄴ → ㄹㄹ) + ㄹ→ㄴ sau coda không phải ㄴ/ㄹ."""
    for cur, nxt in zip(sylls, sylls[1:]):
        jong, onset = cur[2], nxt[0]
        if onset == "ㄹ":
            if jong == "ㄴ":
                cur[2] = "ㄹ"        # 신라→실라
            elif jong and jong != "ㄹ":
                nxt[0] = "ㄴ"        # 침략→침냑, 독립→독닙 (→동닙 ở nasalize)
        elif onset == "ㄴ" and jong == "ㄹ":
            nxt[0] = "ㄹ"            # 칼날→칼랄


def _nasalize(sylls: list[Syllable]) -> None:
    """Điều 18: coda tắc ㄱ/ㄷ/ㅂ + onset mũi ㄴ/ㅁ → coda mũi hoá."""
    for cur, nxt in zip(sylls, sylls[1:]):
        if cur[2] in _NASALIZE_CODA and nxt[0] in ("ㄴ", "ㅁ"):
            cur[2] = _NASALIZE_CODA[cur[2]]


def _tensify(sylls: list[Syllable]) -> None:
    """Điều 23: coda tắc ㄱ/ㄷ/ㅂ + onset lenis ㄱㄷㅂㅅㅈ → onset tense."""
    for cur, nxt in zip(sylls, sylls[1:]):
        if cur[2] in _OBSTRUENT_CODA and nxt[0] in _TENSE:
            nxt[0] = _TENSE[nxt[0]]


def apply_phonology(sylls: list[Syllable]) -> list[Syllable]:
    """Áp toàn bộ rule 표준발음법 theo thứ tự cố định — mutate + trả lại sylls."""
    _palatalize(sylls)
    _h_rules(sylls)
    _liaison(sylls)
    _neutralize(sylls)
    _liquid_rules(sylls)
    _nasalize(sylls)
    _tensify(sylls)
    return sylls

"""Grapheme→phoneme: CMUdict + eSpeak NG + ARPAbet→IPA conversion, stress finalize.

Các helper được word_to_ipa_with_stress (trong __init__) gọi qua namespace package
nên monkeypatch ipa._lookup_cmudict / ipa._espeak_word_to_symbols_stress vẫn tác động.
"""

from __future__ import annotations

import logging
import re
from typing import Final

from .phoneme_set import (
    ARPABET_TO_IPA,
    FUNCTION_WORDS,
    StressType,
    _IPA_PHONEMES_NORM,
    _VALID_ARPABET_BASES,
    _VOWELS_NORM,
    is_vowel,
    normalize_ipa,
)

logger = logging.getLogger("toeic.phoneme.ipa")

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

"""G2P tiếng Hàn: Hangul → IPA sequence + WordSpan — thuần Python, deterministic.

Pipeline per eojeol (từ cách nhau bằng space):
  decompose (hangul.py) → 표준발음법 rule engine (phonology.py) → jamo→IPA
  (phoneme_set_ko.py) → flat phoneme list + WordSpan(source="ko_g2p").

KHÔNG dùng espeak-ng (không có trên dev machine, sandhi kém) / g2pk (kéo mecab,
output lệch theo dict version — vi phạm deterministic score-path). g2pk chỉ làm
shadow comparator trong bench (scripts/, không import vào đây).

Sandhi LIÊN từ (cross-eojeol) cố ý KHÔNG áp: trong-eojeol là bắt buộc theo chuẩn
phát âm, liên-eojeol tuỳ tốc độ/người nói — áp cứng sẽ phạt oan người đọc tách
từ. Khoan dung 2 chiều cho biên từ để ở phonemes_match_ko/bench M2 khi có data.

Tiếng Hàn không có lexical stress → stress list all-None (render sẵn xử lý None).
"""

from __future__ import annotations

import logging
import re
from typing import Final

from ...models import WordSpan
from .hangul import decompose_word
from .phoneme_set_ko import (
    CHO_TO_IPA,
    JONG_TO_IPA,
    JUNG_TO_IPA,
    KOREAN_IPA_PHONEMES,
    is_vowel_ko,
    normalize_ipa_ko,
)
from .phonology import apply_phonology
from .similarity_ko import (
    deletion_penalty_ko,
    deletion_severity_ko,
    is_elidable_stop_ko,
    is_nasal_coda_linking_ko,
    is_real_error_substitution_ko,
    phoneme_similarity_ko,
    phonemes_match_ko,
)

logger = logging.getLogger("toeic.phoneme.ipa.ko")

# Tokenizer: CHỈ âm tiết Hangul precomposed. Chữ số/Latin trong đề tiếng Hàn v1
# bị bỏ (cùng chính sách với tokenizer EN bỏ chữ số) — đọc số là backlog chung.
WORD_TOKEN_RE_KO: Final[re.Pattern[str]] = re.compile(r"[가-힣]+")

# Tokenizer TRANSCRIPT (Whisper ko): giữ Hangul; transcript có thể lẫn số/Latin
# nhưng reference không có → dùng cùng lớp ký tự để mapping cấu trúc 1-1.
TRANSCRIPT_TOKEN_RE_KO: Final[re.Pattern[str]] = WORD_TOKEN_RE_KO

# Ngoại lệ phát âm THEO TỪ (morphology-dependent, ngoài scope rule engine v1):
# eojeol → dạng Hangul ĐÃ PHÁT ÂM (surface form). Giá trị đi thẳng decompose→IPA,
# KHÔNG qua rule engine nữa (đã là dạng mặt). Cùng pattern _WORD_IPA_OVERRIDES EN:
# giữ NHỎ + tường minh, seed từ bench failure.
_WORD_KO_OVERRIDES: Final[dict[str, str]] = {
    "밟다": "밥따",      # điều 10 ngoại lệ: ㄼ trước phụ âm đọc ㅂ
    "밟고": "밥꼬",
    "밟지": "밥찌",
    "넓죽하다": "넙쭈카다",
    "맑게": "말께",      # điều 11 ngoại lệ: ㄺ trước ㄱ giữ ㄹ
    "읽고": "일꼬",
}


def _validate_word_ko_overrides() -> None:
    """Fail-fast lúc import (pattern _validate_word_ipa_overrides bản EN): surface
    form phải decompose được VÀ chỉ chứa coda đã neutralize (tra được JONG_TO_IPA)
    — không thì override lỗi chỉ lộ lúc runtime dưới dạng từ bị drop âm thầm."""
    for word, surface in _WORD_KO_OVERRIDES.items():
        for cho, jung, jong in decompose_word(surface):
            if cho not in CHO_TO_IPA or jung not in JUNG_TO_IPA or (
                jong and jong not in JONG_TO_IPA
            ):
                raise ValueError(
                    f"_WORD_KO_OVERRIDES[{word!r}] = {surface!r}: âm tiết có jamo "
                    f"không hợp lệ ở dạng mặt ({cho!r}/{jung!r}/{jong!r}) — surface "
                    "form phải là dạng ĐÃ PHÁT ÂM (coda thuộc 7 âm chuẩn)."
                )


_validate_word_ko_overrides()


def _syllables_to_ipa(sylls: list[list[str]]) -> list[str]:
    """jamo (đã qua phonology) → flat IPA list. Coda lạ (chưa neutralize hết) sẽ
    KeyError — fail lộ sớm thay vì lặng lẽ sinh reference sai.

    Tense-fold (xem phoneme_set_ko.KO_TENSE_FOLD): đọc module attribute TẠI CALL
    TIME (không bind lúc import) để test/bench monkeypatch được.
    """
    from . import phoneme_set_ko as _ps

    fold = _ps.KO_TENSE_FOLD
    out: list[str] = []
    for cho, jung, jong in sylls:
        for sym in CHO_TO_IPA[cho]:
            out.append(_ps._TENSE_TO_PLAIN.get(sym, sym) if fold else sym)
        out.extend(JUNG_TO_IPA[jung])
        if jong:
            out.extend(JONG_TO_IPA[jong])
    return out


def word_to_ipa_ko(word: str) -> list[str]:
    """1 eojeol Hangul → IPA list (override → rule engine)."""
    surface = _WORD_KO_OVERRIDES.get(word)
    if surface is not None:
        return _syllables_to_ipa(decompose_word(surface))
    return _syllables_to_ipa(apply_phonology(decompose_word(word)))


def text_to_ipa_sequence_with_spans_ko(
    text: str,
) -> tuple[list[str], list[WordSpan], list[str | None], list[str | None]]:
    """Text tiếng Hàn → (phonemes, spans, stress, display_stress) — chữ ký GIỐNG
    text_to_ipa_sequence_with_spans (EN) để LangProfile hoán đổi được.

    stress/display_stress: all-None (tiếng Hàn không có lexical stress).
    Eojeol lỗi decompose/tra bảng → drop + warning (giống chính sách EN drop từ
    failed) — KHÔNG chèn placeholder làm lệch span index.
    """
    phonemes: list[str] = []
    spans: list[WordSpan] = []
    for match in WORD_TOKEN_RE_KO.finditer(text or ""):
        word = match.group()
        try:
            symbols = word_to_ipa_ko(word)
        except (KeyError, ValueError):
            logger.warning("ko_g2p failed word=%r — drop khỏi reference", word)
            continue
        if not symbols:
            continue
        start = len(phonemes)
        phonemes.extend(symbols)
        spans.append(WordSpan(word, start, len(phonemes), source="ko_g2p"))
    stress: list[str | None] = [None] * len(phonemes)
    return phonemes, spans, stress, list(stress)


__all__ = [
    "KOREAN_IPA_PHONEMES",
    "WORD_TOKEN_RE_KO",
    "TRANSCRIPT_TOKEN_RE_KO",
    "deletion_penalty_ko",
    "deletion_severity_ko",
    "is_elidable_stop_ko",
    "is_nasal_coda_linking_ko",
    "is_real_error_substitution_ko",
    "is_vowel_ko",
    "normalize_ipa_ko",
    "phoneme_similarity_ko",
    "phonemes_match_ko",
    "text_to_ipa_sequence_with_spans_ko",
    "word_to_ipa_ko",
]

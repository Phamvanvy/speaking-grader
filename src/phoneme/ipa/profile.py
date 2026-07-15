"""LangProfile — bundle per-language cho scoring pipeline (dispatch seam duy nhất).

Scoring (alignment/word_details/compute_phoneme_score) KHÔNG import trực tiếp
hàm similarity/G2P theo ngôn ngữ nữa mà nhận 1 LangProfile. EN_PROFILE wrap đúng
các hàm hiện có (cùng function object) → mọi caller cũ (default EN) chạy y hệt
trước, bit-for-bit.

Nguyên tắc:
  - frozen dataclass: profile là hằng, không mutate theo request.
  - `english_rules_enabled` gate các rule ĐẶC THÙ tiếng Anh nằm ngoài các flag
    config sẵn có (nasal-coda linking, ð/h recognizer-drop). Các rule đã có flag
    riêng (s_cluster, homograph, connected_speech, accent) do core.py ép tắt
    khi lang != "en" — không gate ở đây để tránh 2 nguồn chân lý.
  - Ngôn ngữ mới = thêm 1 get_profile() branch + bộ hàm riêng (xem ipa/ko/ — M1).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..models import WordSpan
from .phoneme_set import FUNCTION_WORDS, StressType, is_vowel, normalize_ipa
from .similarity import (
    deletion_penalty,
    deletion_severity,
    is_elidable_stop,
    is_nasal_coda_linking,
    is_real_error_substitution,
    phoneme_similarity,
    phonemes_match,
)

# Tokenizer từ — PHẢI khớp bản trong text_to_ipa_sequence_with_spans (EN) để
# reliability đối chiếu transcript ↔ reference cùng một cách tách từ.
_EN_WORD_TOKEN_RE = re.compile(r"[a-zA-Z'-]+")


@dataclass(frozen=True)
class LangProfile:
    """Bộ hàm/bảng theo ngôn ngữ mà scoring pipeline tiêu thụ.

    Mỗi field là callable/hằng đã import sẵn — profile chỉ là chỗ GOM, không
    thêm logic. Chữ ký từng callable giữ nguyên như bản EN gốc (nguồn chân lý
    là docstring các hàm trong ipa/similarity.py + ipa/__init__.py).
    """

    lang: str
    # Tách từ trong text tham chiếu/transcript (reliability dùng chung).
    word_token_re: re.Pattern[str]
    # text → (phonemes, spans, stress, display_stress). Gán lazily (EN import
    # từ ipa/__init__ sẽ vòng — set qua get_profile()).
    text_to_ipa_with_spans: Callable[
        [str],
        tuple[
            list[str],
            list[WordSpan],
            list[StressType | None],
            list[StressType | None],
        ],
    ]
    phoneme_similarity: Callable[[str, str], float]
    phonemes_match: Callable[..., bool]
    normalize_ipa: Callable[[str], str]
    is_vowel: Callable[[str], bool]
    deletion_severity: Callable[..., str]
    deletion_penalty: Callable[..., float]
    is_elidable_stop: Callable[[str], bool]
    is_nasal_coda_linking: Callable[[str, str], bool]
    is_real_error_substitution: Callable[..., bool]
    function_words: frozenset[str] = field(default_factory=frozenset)
    # Gate các rule tiếng Anh KHÔNG có flag config riêng (xem docstring module).
    english_rules_enabled: bool = True
    # Tokenizer TRANSCRIPT cho RecognizerEvidence (khác word_token_re: transcript
    # Whisper có chữ số). None = để reliability dùng default EN của nó.
    transcript_token_re: re.Pattern[str] | None = None


_EN_PROFILE: LangProfile | None = None
_KO_PROFILE: LangProfile | None = None


def get_profile(lang: str = "en") -> LangProfile:
    """Trả LangProfile theo mã ngôn ngữ ("en" | "ko").

    Raise ValueError với mã lạ — caller (core/api) phải validate trước, không
    lặng lẽ fallback (một bài tiếng Hàn chấm bằng bảng tiếng Anh là sai câm).
    """
    global _EN_PROFILE, _KO_PROFILE
    if lang == "ko":
        if _KO_PROFILE is None:
            from .ko import (
                TRANSCRIPT_TOKEN_RE_KO,
                WORD_TOKEN_RE_KO,
                deletion_penalty_ko,
                deletion_severity_ko,
                is_elidable_stop_ko,
                is_nasal_coda_linking_ko,
                is_real_error_substitution_ko,
                is_vowel_ko,
                normalize_ipa_ko,
                phoneme_similarity_ko,
                phonemes_match_ko,
                text_to_ipa_sequence_with_spans_ko,
            )

            _KO_PROFILE = LangProfile(
                lang="ko",
                word_token_re=WORD_TOKEN_RE_KO,
                text_to_ipa_with_spans=text_to_ipa_sequence_with_spans_ko,
                phoneme_similarity=phoneme_similarity_ko,
                phonemes_match=phonemes_match_ko,
                normalize_ipa=normalize_ipa_ko,
                is_vowel=is_vowel_ko,
                deletion_severity=deletion_severity_ko,
                deletion_penalty=deletion_penalty_ko,
                is_elidable_stop=is_elidable_stop_ko,
                is_nasal_coda_linking=is_nasal_coda_linking_ko,
                is_real_error_substitution=is_real_error_substitution_ko,
                # Tiếng Hàn không có weak-form function words kiểu EN — rỗng
                # tắt luôn nhánh nasal-linking/reducible theo function word.
                function_words=frozenset(),
                english_rules_enabled=False,
                transcript_token_re=TRANSCRIPT_TOKEN_RE_KO,
            )
        return _KO_PROFILE
    if lang == "en":
        if _EN_PROFILE is None:
            # Import trễ để tránh vòng ipa/__init__ ↔ profile.
            from . import text_to_ipa_sequence_with_spans

            _EN_PROFILE = LangProfile(
                lang="en",
                word_token_re=_EN_WORD_TOKEN_RE,
                text_to_ipa_with_spans=text_to_ipa_sequence_with_spans,
                phoneme_similarity=phoneme_similarity,
                phonemes_match=phonemes_match,
                normalize_ipa=normalize_ipa,
                is_vowel=is_vowel,
                deletion_severity=deletion_severity,
                deletion_penalty=deletion_penalty,
                is_elidable_stop=is_elidable_stop,
                is_nasal_coda_linking=is_nasal_coda_linking,
                is_real_error_substitution=is_real_error_substitution,
                function_words=frozenset(FUNCTION_WORDS),
                english_rules_enabled=True,
            )
        return _EN_PROFILE
    raise ValueError(f"Không có LangProfile cho ngôn ngữ {lang!r}. Hợp lệ: en, ko")

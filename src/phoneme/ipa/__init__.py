"""IPA phoneme set và ánh xạ word → IPA cho tiếng Anh.

Cung cấp:
  - ENGLISH_IPA_PHONEMES: tập 44 phonemes tiếng Anh (20 nguyên âm + 24 phụ âm + /hm/)
  - word_to_ipa(): ánh xạ từ → IPA sequence (dựa trên ARPAbet → IPA)
  - text_to_ipa_sequence(): chuyển đoạn text → danh sách phonemes tham chiếu
  - phoneme_similarity(): tính khoảng cách giữa 2 phonemes (cho severity scoring)

Tổ chức (package):
  - phoneme_set.py: bảng phoneme/ánh xạ, normalize_ipa, is_vowel, FUNCTION_WORDS
  - similarity.py: phoneme_similarity, phonemes_match, deletion severity/penalty
  - g2p.py: CMUdict + eSpeak NG + ARPAbet→IPA, stress, place_stress_at_onset
  - __init__.py (đây): pipeline word/text + re-export công khai

LƯU Ý monkeypatch: word_to_ipa_with_stress / text_to_ipa_sequence_with_spans được
ĐỊNH NGHĨA ở module này, còn _lookup_cmudict / _espeak_word_to_symbols_stress được
import vào đây từ g2p — nên test patch `ipa._lookup_cmudict`, `ipa.word_to_ipa_with_stress`
vẫn tác động đúng (caller tra tên trong namespace package lúc gọi).
"""

from __future__ import annotations

import logging
import re

from ..models import WordSpan
from .g2p import (
    _LEGAL_ONSETS,
    _REDUCED_ARPABET,
    _arpabet_tokens_to_ipa_stress,
    _cmudict_data,
    _cmudict_entry_is_valid,
    _entry_score,
    _espeak_backend,
    _espeak_token_to_canonical,
    _espeak_word_to_symbols_stress,
    _finalize_stress,
    _get_cmudict,
    _get_espeak,
    _lookup_cmudict,
    _primary_stress_count,
    _rank_cmudict_entries,
    _reduction_count,
    place_stress_at_onset,
)
from .phoneme_set import (
    ARPABET_TO_IPA,
    ENGLISH_IPA_PHONEMES,
    FUNCTION_WORDS,
    IPA_TO_ARPABET,
    StressType,
    _APPROXIMANTS,
    _FRICATIVES,
    _IPA_EQUIV,
    _IPA_PHONEMES_NORM,
    _NASALS,
    _PLOSIVES,
    _VALID_ARPABET_BASES,
    _VOWELS,
    _VOWELS_NORM,
    _WORD_IPA_OVERRIDES,
    _validate_word_ipa_overrides,
    is_vowel,
    normalize_ipa,
)
from .similarity import (
    _ALLOPHONE_PAIRS,
    _NASAL_CODA_STOP_LINKS,
    _NEAR_PAIRS,
    _REAL_ERROR_SUBS,
    _RECOGNIZER_PRONE,
    _REDUCED_VOWELS,
    _SEVERITY_PENALTY,
    _near,
    _same_class,
    _same_place_of_articulation,
    deletion_penalty,
    deletion_severity,
    error_severity,
    is_nasal_coda_linking,
    is_real_error_substitution,
    phoneme_similarity,
    phonemes_match,
)

logger = logging.getLogger("toeic.phoneme.ipa")


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


__all__ = [
    # phoneme set / mapping
    "StressType",
    "ENGLISH_IPA_PHONEMES",
    "ARPABET_TO_IPA",
    "IPA_TO_ARPABET",
    "FUNCTION_WORDS",
    "normalize_ipa",
    "is_vowel",
    # similarity / tolerance / deletion
    "phoneme_similarity",
    "error_severity",
    "phonemes_match",
    "is_nasal_coda_linking",
    "is_real_error_substitution",
    "deletion_severity",
    "deletion_penalty",
    # g2p
    "place_stress_at_onset",
    "word_to_ipa",
    "word_to_ipa_with_stress",
    "text_to_ipa_sequence",
    "text_to_ipa_sequence_with_spans",
]

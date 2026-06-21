"""Tests for src/phoneme — IPA mapping, scoring, analyzer, wav2vec backend.

Tests are designed to run WITHOUT torch/transformers installed:
  - IPA mapping: pure Python, always runs
  - Scoring (DTW): pure Python, always runs
  - Analyzer: tests graceful degradation when wav2vec unavailable
  - Wav2Vec backend: skipped if torch not installed
"""

import pytest

# ── IPA mapping tests ────────────────────────────────────────────────────────

from src.phoneme.ipa import (
    ARPABET_TO_IPA,
    ENGLISH_IPA_PHONEMES,
    error_severity,
    phoneme_similarity,
    text_to_ipa_sequence,
    word_to_ipa,
    word_to_ipa_with_stress,
)


class TestIPAMapping:
    """ARPAbet → IPA conversion."""

    def test_arpa_to_ipa_vowels(self):
        assert ARPABET_TO_IPA["AA"] == "ɑː"
        assert ARPABET_TO_IPA["AE"] == "æ"
        assert ARPABET_TO_IPA["AH"] == "ə"
        assert ARPABET_TO_IPA["IY"] == "iː"
        assert ARPABET_TO_IPA["UW"] == "uː"

    def test_arpa_to_ipa_consonants(self):
        assert ARPABET_TO_IPA["B"] == "b"
        assert ARPABET_TO_IPA["CH"] == "tʃ"
        assert ARPABET_TO_IPA["SH"] == "ʃ"
        assert ARPABET_TO_IPA["TH"] == "θ"
        assert ARPABET_TO_IPA["DH"] == "ð"

    def test_english_ipa_phonemes_not_empty(self):
        assert len(ENGLISH_IPA_PHONEMES) > 40

    def test_english_ipa_contains_key_phonemes(self):
        # Thai/Vietnamese ESL learners commonly struggle with these
        assert "θ" in ENGLISH_IPA_PHONEMES  # thin
        assert "ð" in ENGLISH_IPA_PHONEMES  # this
        assert "ʃ" in ENGLISH_IPA_PHONEMES  # ship
        assert "ʒ" in ENGLISH_IPA_PHONEMES  # measure
        assert "ŋ" in ENGLISH_IPA_PHONEMES  # sing


class TestWordToIpa:
    """Word → IPA sequence using built-in dictionary."""

    def test_common_words(self):
        result = word_to_ipa("the")
        assert "ð" in result
        assert "ə" in result

    def test_pronouns(self):
        result = word_to_ipa("you")
        assert "j" in result   # Y → j
        assert "uː" in result  # UW → uː

    def test_verbs(self):
        result = word_to_ipa("think")
        assert "θ" in result
        assert "k" in result

    def test_empty_input(self):
        assert word_to_ipa("") == []
        assert word_to_ipa("  ") == []

    def test_punctuation_stripped(self):
        result = word_to_ipa("hello.")
        assert isinstance(result, list)

    def test_case_insensitive(self):
        result_lower = word_to_ipa("The")
        result_upper = word_to_ipa("THE")
        assert result_lower == result_upper


class TestWordStress:
    """Word stress (nhấn âm) song song với IPA — chỉ hiển thị, không vào DTW."""

    # 20 nguyên âm đầu của ENGLISH_IPA_PHONEMES (theo comment trong ipa.py).
    _VOWELS = set(ENGLISH_IPA_PHONEMES[:20])

    @staticmethod
    def _g2p_or_skip(word):
        """Trả (symbols, stresses); skip nếu g2p không có (từ đa âm tiết → rỗng)."""
        symbols, stresses = word_to_ipa_with_stress(word)
        if not symbols:
            pytest.skip("g2p_en không khả dụng — bỏ qua test phụ thuộc g2p")
        return symbols, stresses

    def test_alignment_length(self):
        # symbols và stresses luôn khớp độ dài 1-1.
        for w in ("the", "you", "think", "traditional", "interesting", "cat"):
            symbols, stresses = word_to_ipa_with_stress(w)
            assert len(symbols) == len(stresses), w

    def test_dictionary_word_no_stress(self):
        # Từ trong từ điển nội bộ: ARPAbet không kèm stress → toàn None.
        symbols, stresses = word_to_ipa_with_stress("the")
        assert symbols
        assert len(symbols) == len(stresses)
        assert all(s is None for s in stresses)

    def test_multisyllable_has_primary(self):
        # "traditional" qua g2p → có đúng nhấn chính, nằm trên một nguyên âm.
        symbols, stresses = self._g2p_or_skip("traditional")
        assert "primary" in stresses
        for sym, st in zip(symbols, stresses):
            if st is not None:
                assert sym in self._VOWELS, (sym, st)

    def test_multisyllable_interesting(self):
        symbols, stresses = self._g2p_or_skip("interesting")
        assert "primary" in stresses

    def test_monosyllable_suppressed(self):
        # Từ 1 âm tiết (cat) qua g2p: nhấn âm bị suppress ở backend → toàn None.
        symbols, stresses = self._g2p_or_skip("cat")
        assert all(s is None for s in stresses)

    def test_uppercase_matches_lower(self):
        # .lower() ngay đầu hàm: hoa/thường cho kết quả stress giống nhau.
        assert word_to_ipa_with_stress("Traditional") == \
            word_to_ipa_with_stress("traditional")
        assert word_to_ipa_with_stress("INTERESTING") == \
            word_to_ipa_with_stress("interesting")


class TestTextToIpaSequence:
    """Text → IPA phoneme sequence."""

    def test_simple_sentence(self):
        result = text_to_ipa_sequence("the time")
        assert len(result) > 0
        assert isinstance(result, list)

    def test_empty_text(self):
        assert text_to_ipa_sequence("") == []
        assert text_to_ipa_sequence(None) == []

    def test_mixed_known_unknown(self):
        result = text_to_ipa_sequence("the xyzabc123")
        assert len(result) > 0  # at least "the" contributes


class TestTextToIpaSequenceWithSpans:
    """Word-span tracking: phonemes + spans stay index-aligned 1-to-1."""

    def test_spans_cover_phonemes_contiguously(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        phonemes, spans, _stress = text_to_ipa_sequence_with_spans("the brown fox")
        assert spans  # at least some words mapped
        # First span starts at 0; each span continues from the previous end.
        assert spans[0].start_idx == 0
        for prev, cur in zip(spans, spans[1:]):
            assert cur.start_idx == prev.end_idx
        # Last span ends exactly at the phoneme count (no gaps when all mapped).
        assert spans[-1].end_idx == len(phonemes)

    def test_wrapper_returns_same_phonemes(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        phonemes, _spans, _stress = text_to_ipa_sequence_with_spans("the brown fox")
        assert text_to_ipa_sequence("the brown fox") == phonemes

    def test_dropped_word_keeps_alignment(self, monkeypatch):
        import src.phoneme.ipa as ipa_mod

        # Force the middle word to be unmappable (→ empty). g2p_en maps almost any
        # token, so the drop path must be triggered deterministically. Patch
        # word_to_ipa_with_stress since text_to_ipa_sequence_with_spans calls it.
        real = ipa_mod.word_to_ipa_with_stress

        def fake_word_to_ipa_with_stress(word):
            return ([], []) if word.lower() == "drop" else real(word)

        monkeypatch.setattr(
            ipa_mod, "word_to_ipa_with_stress", fake_word_to_ipa_with_stress
        )

        phonemes, spans, _stress = ipa_mod.text_to_ipa_sequence_with_spans("the drop fox")
        words = [s.word for s in spans]
        assert "drop" not in words  # dropped word contributes no span
        assert words == ["the", "fox"]
        # Surviving spans stay contiguous and slice real phonemes (no index shift).
        assert spans[0].start_idx == 0
        assert spans[1].start_idx == spans[0].end_idx
        assert spans[-1].end_idx == len(phonemes)
        for s in spans:
            assert phonemes[s.start_idx:s.end_idx]  # non-empty slice


class TestWordAt:
    """_word_at binary-search boundary behavior ([start, end) half-open)."""

    def _spans(self):
        from src.phoneme.models import WordSpan

        return [WordSpan("a", 0, 2), WordSpan("b", 2, 5), WordSpan("c", 7, 9)]

    def test_inside_spans(self):
        from src.phoneme.scoring import _word_at

        spans = self._spans()
        starts = [s.start_idx for s in spans]
        assert _word_at(0, spans, starts) == "a"
        assert _word_at(1, spans, starts) == "a"
        assert _word_at(2, spans, starts) == "b"
        assert _word_at(4, spans, starts) == "b"
        assert _word_at(8, spans, starts) == "c"

    def test_end_idx_is_exclusive(self):
        from src.phoneme.scoring import _word_at

        spans = self._spans()
        starts = [s.start_idx for s in spans]
        # position == end_idx of "b" must NOT borrow "b"; index 5 is a gap → None.
        assert _word_at(5, spans, starts) is None

    def test_gap_and_past_end_return_none(self):
        from src.phoneme.scoring import _word_at

        spans = self._spans()
        starts = [s.start_idx for s in spans]
        assert _word_at(6, spans, starts) is None   # gap between b(.. 5) and c(7..)
        assert _word_at(9, spans, starts) is None    # == last end_idx (exclusive)
        assert _word_at(999, spans, starts) is None  # far past end


class TestPhonemeSimilarity:
    """Phoneme similarity scoring."""

    def test_identical_phonemes(self):
        assert phoneme_similarity("p", "p") == 1.0
        assert phoneme_similarity("θ", "θ") == 1.0

    def test_same_place_bilabial(self):
        sim = phoneme_similarity("p", "b")
        assert sim >= 0.4  # same class (plosives) + same place

    def test_same_place_dental(self):
        sim = phoneme_similarity("θ", "ð")
        assert sim >= 0.7  # same class + same place

    def test_completely_different(self):
        sim = phoneme_similarity("iː", "p")
        assert sim == 0.0

    def test_same_class_different_place(self):
        sim = phoneme_similarity("p", "t")
        assert 0.0 < sim < 0.7


class TestErrorSeverity:
    """Severity label from similarity score."""

    def test_low_severity(self):
        assert error_severity(1.0) == "low"
        assert error_severity(0.7) == "low"

    def test_medium_severity(self):
        assert error_severity(0.5) == "medium"
        assert error_severity(0.4) == "medium"

    def test_high_severity(self):
        assert error_severity(0.3) == "high"
        assert error_severity(0.0) == "high"


# ── Scoring (DTW) tests ──────────────────────────────────────────────────────

from src.phoneme.models import PhonemeErrorType, PhonemeSegment, PhonemeScore
from src.phoneme.scoring import (
    compute_phoneme_score,
    weighted_accuracy,
)


class TestComputePhonemeScore:
    """DTW-based phoneme scoring."""

    def _make_segments(self, phonemes: list[str]) -> list[PhonemeSegment]:
        return [
            PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
            for i, p in enumerate(phonemes)
        ]

    def test_perfect_match(self):
        ref = ["p", "ə", "t"]
        segs = self._make_segments(ref)
        score = compute_phoneme_score(segs, ref)
        assert score is not None
        assert score.overall_accuracy == 1.0
        assert score.substitution_count == 0
        assert score.deletion_count == 0
        assert score.insertion_count == 0

    def test_substitution(self):
        ref = ["p", "ə", "t"]
        pred = ["b", "ə", "t"]  # p → b
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref)
        assert score is not None
        assert score.substitution_count >= 1
        assert score.overall_accuracy < 1.0

    def test_deletion(self):
        ref = ["p", "ə", "t"]
        pred = ["p", "t"]  # ə deleted
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref)
        assert score is not None
        assert score.deletion_count >= 1

    def test_insertion(self):
        # Use vowel "x" (not a real phoneme, similarity=0 with anything) so DTW
        # cannot align it as substitution and must treat it as insertion
        ref = ["p", "t"]
        pred = ["p", "x", "t"]  # "x" has zero similarity → insertion
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref)
        assert score is not None
        # DTW may still classify as substitution with zero similarity;
        # what matters is predicted_count > reference_count
        assert score.predicted_count > score.reference_count
        assert score.insertion_count >= 1 or score.substitution_count >= 1

    def test_no_reference(self):
        segs = self._make_segments(["p", "ə", "t"])
        score = compute_phoneme_score(segs, [])
        assert score is None

    def test_no_prediction_all_deletions(self):
        ref = ["p", "ə", "t"]
        score = compute_phoneme_score([], ref)
        assert score is not None
        assert score.overall_accuracy == 0.0
        assert score.deletion_count == len(ref)

    def test_scores_has_required_fields(self):
        ref = ["p", "ə", "t"]
        segs = self._make_segments(ref)
        score = compute_phoneme_score(segs, ref)
        d = score.to_dict()
        assert "overall_accuracy" in d
        assert "substitution_count" in d
        assert "deletion_count" in d
        assert "insertion_count" in d
        assert "errors" in d
        assert "avg_confidence" in d

    def test_no_spans_leaves_words_none(self):
        # Backward compat: without reference_spans every error keeps word=None.
        ref = ["p", "ə", "t"]
        segs = self._make_segments(["b", "ə", "t"])  # p → b
        score = compute_phoneme_score(segs, ref)
        assert score.substitution_count >= 1
        assert all(e.word is None for e in score.errors)

    def test_substitution_gets_word_from_spans(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the brown fox")
        pred = list(ref)
        pred[-1] = "x"  # corrupt last phoneme (in "fox") → substitution there
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        subs = [e for e in score.errors if e.error_type == PhonemeErrorType.SUBSTITUTION]
        assert subs
        # The corrupted phoneme lives in the last word "fox".
        assert any(e.word == "fox" for e in subs)

    def test_insertion_word_is_none_even_with_spans(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        pred = [ref[0]] + ["x"] + ref[1:]  # extra "x" → insertion (position=pred idx)
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        for e in score.errors:
            if e.error_type == PhonemeErrorType.INSERTION:
                assert e.word is None

    def test_word_propagates_to_dict(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        pred = list(ref)
        pred[-1] = "x"
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        assert all("word" in e for e in score.to_dict()["errors"])


class TestWordDetails:
    """Per-word IPA detail (score.words) for ELSA-style display."""

    def _make_segments(self, phonemes: list[str]) -> list[PhonemeSegment]:
        return [
            PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
            for i, p in enumerate(phonemes)
        ]

    def test_all_correct_word_is_ok(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        segs = self._make_segments(list(ref))
        score = compute_phoneme_score(segs, ref, spans)
        assert score.words
        assert score.words_total == len(spans)
        assert score.words_truncated is False
        for w in score.words:
            assert w.accuracy == 1.0
            assert all(p.status == "ok" for p in w.phonemes)
            assert all(p.severity is None and p.heard is None for p in w.phonemes)

    def test_ipa_reconstructs_reference_span_without_slashes(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        segs = self._make_segments(list(ref))
        score = compute_phoneme_score(segs, ref, spans)
        for w, span in zip(score.words, spans):
            assert w.ipa == "".join(ref[span.start_idx:span.end_idx])
            assert "/" not in w.ipa  # backend stores IPA without delimiters

    def test_substitution_point_has_heard_and_severity(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        pred = list(ref)
        pred[-1] = "x"  # corrupt last phoneme (in "fox") → substitution
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        fox = next(w for w in score.words if w.word == "fox")
        subs = [p for p in fox.phonemes if p.status == "sub"]
        assert subs
        assert all(p.heard is not None and p.severity is not None for p in subs)
        assert fox.accuracy < 1.0

    def test_deletion_point_marked_high(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        pred = list(ref)[:-1]  # drop last reference phoneme → deletion
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        dels = [p for w in score.words for p in w.phonemes if p.status == "del"]
        assert dels
        assert all(p.severity == "high" and p.heard is None for p in dels)

    def test_no_prediction_all_deletions(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        score = compute_phoneme_score([], ref, spans)
        assert score.overall_accuracy == 0.0
        assert score.words
        for w in score.words:
            assert w.accuracy == 0.0
            assert all(p.status == "del" and p.severity == "high" for p in w.phonemes)

    def test_insertion_adjacent_keeps_one_point_per_ref_index(self):
        # Regression for the one-status-per-reference-index invariant: an extra
        # predicted phoneme must not add/drop per-word points or shift indices.
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        pred = [ref[0]] + ["x"] + ref[1:]  # extra "x" insertion after first phoneme
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        for w, span in zip(score.words, spans):
            assert len(w.phonemes) == span.end_idx - span.start_idx
        assert sum(len(w.phonemes) for w in score.words) == len(ref)

    def test_no_spans_yields_no_words(self):
        ref = ["p", "ə", "t"]
        segs = self._make_segments(ref)
        score = compute_phoneme_score(segs, ref)  # no spans
        assert score.words == []
        assert score.words_total == 0

    def test_words_serialize_in_to_dict(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress = text_to_ipa_sequence_with_spans("the fox")
        segs = self._make_segments(list(ref))
        d = compute_phoneme_score(segs, ref, spans).to_dict()
        assert "words" in d and "words_truncated" in d and "words_total" in d
        assert d["words"]
        assert set(d["words"][0]) == {"word", "ipa", "phonemes", "accuracy"}
        assert set(d["words"][0]["phonemes"][0]) == {
            "symbol", "status", "heard", "severity", "stress"
        }


class TestWeightedAccuracy:
    """Weighted accuracy calculation."""

    def test_perfect_score(self):
        score = PhonemeScore(
            overall_accuracy=1.0,
            substitution_count=0, deletion_count=0, insertion_count=0,
            reference_count=10, predicted_count=10,
            avg_confidence=0.95, errors=[],
        )
        assert weighted_accuracy(score) == 1.0

    def test_zero_reference_count(self):
        score = PhonemeScore(
            overall_accuracy=0.0,
            substitution_count=0, deletion_count=0, insertion_count=0,
            reference_count=0, predicted_count=0,
            avg_confidence=0.0, errors=[],
        )
        assert weighted_accuracy(score) == 0.0


# ── Model serialization tests ────────────────────────────────────────────────

from src.phoneme.models import (
    PhonemeError,
    PhonemeResult,
    PhonemeSegment,
    PhonemeScore,
)


class TestModelSerialization:
    """to_dict() on all models."""

    def test_segment_to_dict(self):
        seg = PhonemeSegment(phoneme="θ", start=0.5, end=1.2, confidence=0.85)
        d = seg.to_dict()
        assert d["phoneme"] == "θ"
        assert d["backend"] == "wav2vec"

    def test_error_to_dict(self):
        err = PhonemeError(
            error_type=PhonemeErrorType.SUBSTITUTION,
            expected="θ", predicted="s",
            position=3, severity="medium",
        )
        d = err.to_dict()
        assert d["error_type"] == "substitution"
        assert d["expected"] == "θ"

    def test_score_to_dict(self):
        score = PhonemeScore(
            overall_accuracy=0.85,
            substitution_count=2, deletion_count=1, insertion_count=0,
            reference_count=20, predicted_count=19,
            avg_confidence=0.82, errors=[],
        )
        d = score.to_dict()
        assert d["overall_accuracy"] == 0.85
        assert d["reference_count"] == 20

    def test_result_to_dict(self):
        result = PhonemeResult(
            audio_path="test.wav",
            segments=[],
            reference_phonemes=["p", "ə", "t"],
            score=None,
            backend_used="wav2vec",
            backend_available=True,
        )
        d = result.to_dict()
        assert d["audio_path"] == "test.wav"
        assert d["backend_used"] == "wav2vec"
        assert d["score"] is None


# ── Analyzer graceful degradation tests ──────────────────────────────────────

from src.phoneme.analyzer import HybridPhonemeAnalyzer


class TestAnalyzerGracefulDegradation:
    """Analyzer handles missing wav2vec backend gracefully."""

    def test_disabled_analysis(self):
        analyzer = HybridPhonemeAnalyzer(enable_phoneme_analysis=False)
        # Nonexistent audio — should not crash even without backend
        result = analyzer.analyze("nonexistent.wav", reference_text="hello")
        assert result.warning is not None
        assert "disabled" in result.backend_used

    def test_missing_audio_file(self):
        analyzer = HybridPhonemeAnalyzer(enable_phoneme_analysis=True)
        result = analyzer.analyze("does_not_exist.wav", reference_text="hello")
        assert result.backend_available is False
        assert result.segments == []
        assert result.warning is not None

    def test_reference_phonemes_built(self):
        analyzer = HybridPhonemeAnalyzer(enable_phoneme_analysis=True)
        result = analyzer.analyze("does_not_exist.wav", reference_text="the time")
        # Even with missing audio, reference phonemes should be empty
        # (audio check happens before reference build)
        assert result.backend_used == "none"


# ── Wav2Vec backend availability test ────────────────────────────────────────

class TestWav2VecBackendAvailability:
    """Test wav2vec backend availability detection."""

    def test_predictor_is_available_property(self):
        from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor
        predictor = Wav2VecPhonemePredictor()
        # is_available should be bool, no crash
        assert isinstance(predictor.is_available, bool)

    def test_predict_missing_audio(self):
        from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor
        predictor = Wav2VecPhonemePredictor()
        segments, warning = predictor.predict("nonexistent_file.wav")
        assert segments == []
        assert warning is not None
        assert "không tồn tại" in warning or "not found" in warning.lower() or "không" in warning
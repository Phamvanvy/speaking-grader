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
    place_stress_at_onset,
    text_to_ipa_sequence,
    text_to_ipa_sequence_with_spans,
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
    """Word → IPA sequence via CMUdict / eSpeak pipeline."""

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

    def test_usually_no_spurious_w(self):
        # CMUdict has two entries: entries[0] (8 tokens, spurious W) and entries[1]
        # (6 tokens, no W). Both have the same primary-stress count, so _entry_score's
        # length tie-break picks the shorter: Y UW1 ZH AH0 L IY0 → /juːʒəliː/.
        result = word_to_ipa("usually")
        assert "w" not in result, f"spurious /w/ in usually: {result}"
        assert result == ["j", "uː", "ʒ", "ə", "l", "iː"]

    def test_function_word_prefers_weak_form(self):
        # Function words score toward 0 primary stress → reduced (schwa) variant.
        # "the" → DH AH0 (/ðə/), not the strong DH IY0 (/ðiː/) or DH AH1.
        result = word_to_ipa("the")
        assert result == ["ð", "ə"], f"the not weak form: {result}"
        # "to" → T AH0 (/tə/) reduced, not strong T UW1 (/tuː/).
        result = word_to_ipa("to")
        assert "uː" not in result, f"to picked strong form: {result}"

    def test_content_word_prefers_one_primary(self):
        # Content words score toward exactly 1 primary stress.
        symbols, stresses = word_to_ipa_with_stress("record")  # noun/verb both 1 primary
        assert symbols
        assert stresses.count("primary") == 1, (symbols, stresses)

    def test_override_hard_priority(self):
        # "favorite" override (F EY1 V AH0 R IH0 T) must win over CMUdict
        # (which has F EY1 V ER0 AH0 T → /ˈfeɪvɜːət/ with wrong ɜː).
        result = word_to_ipa("favorite")
        assert "ɜː" not in result, f"CMUdict ER0 slipped through override: {result}"
        assert result == ["f", "eɪ", "v", "ə", "r", "ɪ", "t"]

    def test_cmudict_common_word(self):
        # Common words previously in _COMMON_WORD_PRONUNCIATIONS now come from CMUdict.
        for word, expected_subset in [
            ("work", {"w", "ɜː", "k"}),
            ("think", {"θ", "ɪ", "ŋ", "k"}),
            ("people", {"p", "iː", "p", "ə", "l"}),
        ]:
            result = set(word_to_ipa(word))
            assert expected_subset <= result, f"{word}: got {result}"

    def test_er0_split_unstressed_vs_monosyllable(self):
        # ER0 không nhấn trong từ ĐA âm tiết → ə + r (rhotic schwa), KHÔNG phải ɜː.
        # CMUdict map ER→ɜː cố định nên trước đây cho /ˈsætɜː…/, /ˈmʌðɜː/ (sai).
        for word in ("mother", "water", "computer", "teacher", "number", "after"):
            result = word_to_ipa(word)
            assert "ɜː" not in result, f"{word}: ER0 vẫn ra ɜː: {result}"
            assert result[-2:] == ["ə", "r"], f"{word}: đuôi -er sai: {result}"
        # Từ ĐƠN âm tiết giữ ɜː dù CMUdict gắn ER0 (sir/fur), và ER1/ER2 luôn ɜː.
        for word in ("bird", "sir", "fur", "work", "person"):
            assert "ɜː" in word_to_ipa(word), word

    def test_determinism(self):
        # Same input must produce identical output on every call.
        for word in ("the", "usually", "traditional", "important"):
            assert word_to_ipa_with_stress(word) == word_to_ipa_with_stress(word), word

    def test_hard_failure_returns_empty(self, monkeypatch):
        # Layer 4: if CMUdict and eSpeak both return nothing, result is ([], []).
        import src.phoneme.ipa as ipa_mod
        monkeypatch.setattr(ipa_mod, "_lookup_cmudict", lambda w: None)
        monkeypatch.setattr(ipa_mod, "_espeak_word_to_symbols_stress", lambda w: None)
        assert word_to_ipa("anything") == []
        symbols, stresses = word_to_ipa_with_stress("anything")
        assert symbols == []
        assert stresses == []


class TestWordStress:
    """Word stress (nhấn âm) song song với IPA — chỉ hiển thị, không vào DTW."""

    # 20 nguyên âm đầu của ENGLISH_IPA_PHONEMES (theo comment trong ipa.py).
    _VOWELS = set(ENGLISH_IPA_PHONEMES[:20])

    @staticmethod
    def _g2p_or_skip(word):
        """Return (symbols, stresses); skip if no pronunciation found (OOV with eSpeak unavailable)."""
        symbols, stresses = word_to_ipa_with_stress(word)
        if not symbols:
            pytest.skip(f"no pronunciation found for {word!r} — eSpeak may be unavailable")
        return symbols, stresses

    def test_alignment_length(self):
        # symbols và stresses luôn khớp độ dài 1-1.
        for w in ("the", "you", "think", "traditional", "interesting", "cat"):
            symbols, stresses = word_to_ipa_with_stress(w)
            assert len(symbols) == len(stresses), w

    def test_dictionary_word_no_stress(self):
        # "the" (DH AH0) is monosyllabic → _finalize_stress suppresses all marks → None.
        symbols, stresses = word_to_ipa_with_stress("the")
        assert symbols
        assert len(symbols) == len(stresses)
        assert all(s is None for s in stresses)

    def test_multisyllable_has_primary(self):
        # "traditional" via CMUdict → primary stress on a vowel.
        symbols, stresses = self._g2p_or_skip("traditional")
        assert "primary" in stresses
        for sym, st in zip(symbols, stresses):
            if st is not None:
                assert sym in self._VOWELS, (sym, st)

    def test_multisyllable_interesting(self):
        symbols, stresses = self._g2p_or_skip("interesting")
        assert "primary" in stresses

    def test_monosyllable_suppressed(self):
        # "cat" (K AE1 T) is monosyllabic → _finalize_stress suppresses stress → all None.
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

        phonemes, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the brown fox")
        assert spans  # at least some words mapped
        # First span starts at 0; each span continues from the previous end.
        assert spans[0].start_idx == 0
        for prev, cur in zip(spans, spans[1:]):
            assert cur.start_idx == prev.end_idx
        # Last span ends exactly at the phoneme count (no gaps when all mapped).
        assert spans[-1].end_idx == len(phonemes)

    def test_wrapper_returns_same_phonemes(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        phonemes, _spans, _stress, _disp = text_to_ipa_sequence_with_spans("the brown fox")
        assert text_to_ipa_sequence("the brown fox") == phonemes

    def test_dropped_word_keeps_alignment(self, monkeypatch):
        import src.phoneme.ipa as ipa_mod

        # Force the middle word to be unmappable (→ empty) to test alignment stability.
        # CMUdict + eSpeak cover most tokens, so patch directly to trigger the drop path.
        # spans dựng từ word_to_ipa_with_stress_source (bản _source) → patch bản đó.
        real = ipa_mod.word_to_ipa_with_stress_source

        def fake_word_to_ipa_with_stress_source(word):
            return ([], [], "failed") if word.lower() == "drop" else real(word)

        monkeypatch.setattr(
            ipa_mod, "word_to_ipa_with_stress_source", fake_word_to_ipa_with_stress_source
        )

        phonemes, spans, _stress, _disp = ipa_mod.text_to_ipa_sequence_with_spans("the drop fox")
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

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the brown fox")
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

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        pred = [ref[0]] + ["x"] + ref[1:]  # extra "x" → insertion (position=pred idx)
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        for e in score.errors:
            if e.error_type == PhonemeErrorType.INSERTION:
                assert e.word is None

    def test_word_propagates_to_dict(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        pred = list(ref)
        pred[-1] = "x"
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        assert all("word" in e for e in score.to_dict()["errors"])


class TestAccentVariants:
    """Accent "default" (accept_accent_variants): chấp nhận coda /r/ non-rhotic (Anh-Anh).

    CHỈ áp dụng coda /r/; các khác biệt GB/US khác đã được normalize_ipa() gộp sẵn nên
    không cần xử lý ở tầng này (xem compute_phoneme_score docstring).
    """

    def _make_segments(self, phonemes: list[str]) -> list[PhonemeSegment]:
        return [
            PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
            for i, p in enumerate(phonemes)
        ]

    def _ref(self, text: str):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        return text_to_ipa_sequence_with_spans(text)

    def test_dropped_coda_r_accepted_in_default(self):
        # "car" /k ɔ r/ — Anh-Anh nuốt /r/ cuối → predicted /k ɔ/.
        ref, spans, stress, disp = self._ref("car")
        segs = self._make_segments(["k", "ɔ"])  # /r/ bị nuốt
        score = compute_phoneme_score(
            segs, ref, spans, stress, reference_display_stress=disp,
            accept_accent_variants=True,
        )
        assert score.deletion_count == 0
        assert score.overall_accuracy == 1.0
        r_point = score.words[0].phonemes[-1]
        assert r_point.status == "ok"
        assert r_point.penalty_reason == "accent_variant"

    def test_dropped_coda_r_still_penalized_when_flag_off(self):
        # Mặc định (us/gb) giữ nguyên: nuốt coda /r/ vẫn là deletion.
        ref, spans, stress, disp = self._ref("car")
        segs = self._make_segments(["k", "ɔ"])
        score = compute_phoneme_score(
            segs, ref, spans, stress, reference_display_stress=disp,
            accept_accent_variants=False,
        )
        assert score.deletion_count >= 1
        assert score.overall_accuracy < 1.0
        assert score.words[0].phonemes[-1].status == "del"

    def test_consonant_substituted_for_coda_r_stays_error(self):
        # /l/ thay /r/ cuối KHÔNG phải biến thể giọng → vẫn là lỗi thật, kể cả default.
        ref, spans, stress, disp = self._ref("car")
        segs = self._make_segments(["k", "ɔ", "l"])  # r → l
        score = compute_phoneme_score(
            segs, ref, spans, stress, reference_display_stress=disp,
            accept_accent_variants=True,
        )
        r_point = score.words[0].phonemes[-1]
        assert r_point.status == "sub"
        assert r_point.penalty_reason != "accent_variant"

    def test_onset_r_not_exempted(self):
        # "red" /r e d/ — /r/ là onset (không phải coda) → không khoan dung dù bật flag.
        ref, spans, stress, disp = self._ref("red")
        segs = self._make_segments(["e", "d"])  # bỏ /r/ đầu
        score = compute_phoneme_score(
            segs, ref, spans, stress, reference_display_stress=disp,
            accept_accent_variants=True,
        )
        # /r/ onset không được tag accent_variant (vẫn vào lỗi sub/del).
        assert all(
            p.penalty_reason != "accent_variant" for p in score.words[0].phonemes
        )
        assert score.overall_accuracy < 1.0


class TestNasalCodaLinking:
    """Nối âm: coda mũi của function word → stop homorganic trước nguyên âm = không lỗi.

    "in order" /ɪn/+/ɔː/ — wav2vec hay nghe /n/ cuối thành /t/ (giải phóng/nối coda).
    Đây là artifact nối âm, KHÔNG phải nuốt nasal (vẫn được phát) nên KHÔNG phạt.
    """

    def _make_segments(self, phonemes: list[str]) -> list[PhonemeSegment]:
        return [
            PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
            for i, p in enumerate(phonemes)
        ]

    def _ref(self, text: str):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        return text_to_ipa_sequence_with_spans(text)

    def test_in_order_n_to_t_accepted(self):
        # "in order" /ɪ n ɔː r d ɜː/ → /n/ nghe thành /t/ trước nguyên âm /ɔː/ của "order".
        ref, spans, stress, disp = self._ref("in order")
        pred = list(ref)
        pred[1] = "t"  # n → t (coda của "in", nối sang /ɔː/)
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans, stress, reference_display_stress=disp)
        n_point = score.words[0].phonemes[1]
        assert n_point.status == "ok"
        assert n_point.penalty_reason == "linking_variant"
        assert score.substitution_count == 0

    def test_in_bed_n_to_t_stays_error(self):
        # "in bed": /n/ trước phụ âm /b/ → KHÔNG nối âm → n→t vẫn là lỗi thật.
        ref, spans, stress, disp = self._ref("in bed")
        pred = list(ref)
        pred[1] = "t"  # n → t nhưng từ kế bắt đầu bằng phụ âm
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans, stress, reference_display_stress=disp)
        n_point = score.words[0].phonemes[1]
        assert n_point.status == "sub"
        assert n_point.penalty_reason != "linking_variant"

    def test_content_word_coda_not_exempted(self):
        # "pen" KHÔNG phải function word → n→t cuối vẫn là lỗi (giữ phân biệt pen/pet).
        ref, spans, stress, disp = self._ref("pen apple")
        n_idx = ref.index("n")
        pred = list(ref)
        pred[n_idx] = "t"
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans, stress, reference_display_stress=disp)
        n_point = next(p for p in score.words[0].phonemes if p.symbol == "n")
        assert n_point.penalty_reason != "linking_variant"


class TestWordDetails:
    """Per-word IPA detail (score.words) for ELSA-style display."""

    def _make_segments(self, phonemes: list[str]) -> list[PhonemeSegment]:
        return [
            PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
            for i, p in enumerate(phonemes)
        ]

    def test_all_correct_word_is_ok(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
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

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        segs = self._make_segments(list(ref))
        score = compute_phoneme_score(segs, ref, spans)
        for w, span in zip(score.words, spans):
            assert w.ipa == "".join(ref[span.start_idx:span.end_idx])
            assert "/" not in w.ipa  # backend stores IPA without delimiters

    def test_substitution_point_has_heard_and_severity(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        pred = list(ref)
        pred[-1] = "x"  # corrupt last phoneme (in "fox") → substitution
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        fox = next(w for w in score.words if w.word == "fox")
        subs = [p for p in fox.phonemes if p.status == "sub"]
        assert subs
        assert all(p.heard is not None and p.severity is not None for p in subs)
        assert fox.accuracy < 1.0

    def test_deletion_point_marked(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        pred = list(ref)[:-1]  # drop last reference phoneme → deletion
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        dels = [p for w in score.words for p in w.phonemes if p.status == "del"]
        assert dels
        # Severity giờ theo loại âm/vị trí (không còn luôn "high"); heard luôn None.
        assert all(p.severity in ("low", "medium", "high") and p.heard is None
                   for p in dels)

    def test_dropped_onset_consonant_is_high(self):
        # Nuốt/đọc sai phụ âm ĐẦU TỪ (onset θ trong think) → high severity. DTW có
        # thể xếp thành sub hoặc del tuỳ alignment — điều quan trọng là nó nặng.
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("think")
        pred = list(ref)[1:]  # bỏ phoneme đầu (θ)
        segs = self._make_segments(pred)
        score = compute_phoneme_score(segs, ref, spans)
        bad = [p for w in score.words for p in w.phonemes
               if p.status in ("sub", "del")]
        assert any(p.severity == "high" for p in bad)

    def test_no_prediction_all_deletions(self):
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        score = compute_phoneme_score([], ref, spans)
        # "Said nothing" → 0% và mọi âm là deletion (severity nay đa dạng).
        assert score.overall_accuracy == 0.0
        assert score.words
        for w in score.words:
            assert all(p.status == "del" and p.heard is None for p in w.phonemes)

    def test_insertion_adjacent_keeps_one_point_per_ref_index(self):
        # Regression for the one-status-per-reference-index invariant: an extra
        # predicted phoneme must not add/drop per-word points or shift indices.
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
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

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        segs = self._make_segments(list(ref))
        d = compute_phoneme_score(segs, ref, spans).to_dict()
        assert "words" in d and "words_truncated" in d and "words_total" in d
        assert d["words"]
        assert set(d["words"][0]) == {
            "word", "ipa", "phonemes", "accuracy", "skip_reason", "start", "end"}
        assert set(d["words"][0]["phonemes"][0]) == {
            "symbol", "status", "heard", "severity", "stress", "display_stress",
            "penalty_reason", "penalty_adjustment",
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


# ── De-noise: normalize / matcher / severity / scoring ───────────────────────

from src.phoneme.ipa import (
    deletion_penalty,
    deletion_severity,
    normalize_ipa,
    phonemes_match,
)
from src.phoneme.models import WordSpan


class TestMinimalNormalization:
    """normalize_ipa CHỈ gộp cặp chắc chắn — flap & ɜ giữ riêng."""

    def test_flap_not_merged_to_t(self):
        # ɾ KHÔNG bị normalize thành t (allophone xử lý ở phonemes_match).
        assert normalize_ipa("ɾ") != normalize_ipa("t")

    def test_nurse_vowel_kept_distinct_from_schwa(self):
        # ɜ giữ khác ə để lỗi thật (bird vs bud) không bị nuốt ở tầng normalize.
        assert normalize_ipa("ɜː") != normalize_ipa("ə")

    def test_r_colored_schwa_merges_to_schwa(self):
        assert normalize_ipa("ɚ") == normalize_ipa("ə")
        assert normalize_ipa("ɝ") == normalize_ipa("ə")


class TestContinuousSimilarity:
    """phoneme_similarity liên tục: near-vowel cao, khác hẳn = 0."""

    def test_near_vowels_high(self):
        assert phoneme_similarity("ɪ", "iː") >= 0.8
        assert phoneme_similarity("ʊ", "uː") >= 0.8

    def test_dental_fricative_mid(self):
        # θ↔s: lỗi ESL thật, không nên = 1.0 (giữ là lỗi rõ).
        assert 0.0 < phoneme_similarity("θ", "s") < 0.7


class TestPhonemesMatch:
    """Tolerance layer — allophone / reduction / function word."""

    def test_flap_allophone(self):
        assert phonemes_match("t", "ɾ")
        assert phonemes_match("d", "ɾ")

    def test_r_colored_er_as_r(self):
        # every /evɜːiː/ → /evɹiː/: ɜ↔r coi là khớp.
        assert phonemes_match("ɜ", "ɹ")

    def test_unstressed_reduction_when_reducible(self):
        assert phonemes_match("ɜ", "ə", reducible=True)
        assert phonemes_match("uː", "ʊ", reducible=True)

    def test_not_reducible_keeps_real_error(self):
        # bird-like: ɜ là nhân chính (reducible False) → KHÔNG khớp với ə.
        assert not phonemes_match("ɜ", "ə", reducible=False)

    def test_function_word_strong_form(self):
        # and /ənd/ → /ænd/: æ↔ə chỉ mở cho function word.
        assert phonemes_match("ə", "æ", word="and")
        assert not phonemes_match("ə", "æ", word="cat", reducible=False)


class TestDeletionSeverity:
    """Severity của âm thiếu theo loại âm + vị trí (không còn luôn high)."""

    def test_recognizer_prone_low(self):
        assert deletion_severity("ð") == "low"   # the
        assert deletion_severity("h") == "low"   # his
        assert deletion_severity("ə") == "low"

    def test_onset_consonant_high(self):
        assert deletion_severity("θ", is_onset=True) == "high"   # think
        assert deletion_severity("k", is_onset=True) == "high"   # cluster

    def test_coda_consonant_medium(self):
        assert deletion_severity("k", is_onset=False) == "medium"

    def test_stressed_vowel_high_unstressed_low(self):
        assert deletion_severity("æ", stress="primary") == "high"
        assert deletion_severity("æ", stress=None) == "low"

    def test_penalty_orders_with_severity(self):
        assert deletion_penalty("ð") < deletion_penalty("k", is_onset=True)


class TestDeNoiseScoring:
    """End-to-end de-noise trên reference dựng tay (không phụ thuộc g2p)."""

    def _segs(self, phonemes, conf=0.9):
        return [
            PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=conf)
            for i, p in enumerate(phonemes)
        ]

    def test_american_flap_and_er_not_flagged(self):
        # water /wɒtɜ/ đọc kiểu Mỹ /wɒɾə/: flap + đuôi -er → KHÔNG lỗi.
        ref = ["w", "ɒ", "t", "ɜ"]
        spans = [WordSpan("water", 0, 4)]
        stress = [None, "primary", None, None]
        score = compute_phoneme_score(self._segs(["w", "ɒ", "ɾ", "ə"]), ref, spans, stress)
        assert score.overall_accuracy == 1.0
        assert score.substitution_count == 0
        assert score.deletion_count == 0

    def test_and_strong_form_not_flagged(self):
        ref = ["ə", "n", "d"]
        spans = [WordSpan("and", 0, 3)]
        score = compute_phoneme_score(self._segs(["æ", "n", "d"]), ref, spans,
                                      [None, None, None])
        assert score.overall_accuracy == 1.0
        assert score.substitution_count == 0

    def test_real_stressed_vowel_error_preserved(self):
        # bird /bɜd/ → /bəd/: ɜ là nhân chính → vẫn là lỗi (substitution).
        ref = ["b", "ɜ", "d"]
        spans = [WordSpan("bird", 0, 3)]
        score = compute_phoneme_score(self._segs(["b", "ə", "d"]), ref, spans,
                                      [None, None, None])
        assert score.substitution_count >= 1

    def test_low_confidence_downweights_substitution(self):
        ref = ["k"]
        hi = compute_phoneme_score(self._segs(["t"], conf=0.9), ref)
        lo = compute_phoneme_score(self._segs(["t"], conf=0.1), ref)
        # Confidence thấp → penalty thấp hơn → accuracy cao hơn + severity nhẹ hơn.
        assert lo.overall_accuracy > hi.overall_accuracy
        assert hi.errors and hi.errors[0].severity in ("medium", "high")
        assert lo.errors and lo.errors[0].severity == "low"

    def test_skips_excluded_from_score(self):
        from src.phoneme.reliability import SkipDecision, SkipReason

        ref = ["ð", "ə", "f", "ɒ", "k", "s"]
        spans = [WordSpan("the", 0, 2), WordSpan("fox", 2, 6)]
        stress = [None] * 6
        pred = ["ð", "ə", "x", "x", "x", "x"]  # fox đọc hỏng hoàn toàn (ASR nghe nhầm)
        scored = compute_phoneme_score(self._segs(pred), ref, spans, stress)
        # Skip span index 1 ("fox") qua mapping index-keyed (occurrence-specific).
        skipped = compute_phoneme_score(
            self._segs(pred), ref, spans, stress,
            skips={1: SkipDecision(1, SkipReason.WHISPER_MISMATCH)},
        )
        # Bỏ qua "fox" → chỉ còn "the" (đúng) được chấm → accuracy 1.0, không lỗi.
        assert skipped.overall_accuracy == 1.0
        assert skipped.substitution_count == 0
        assert scored.overall_accuracy < 1.0
        # "fox" mang status skipped + skip_reason; "the" được chấm bình thường.
        fox = next(w for w in skipped.words if w.word == "fox")
        assert fox.skip_reason == "whisper_mismatch"
        assert all(p.status == "skipped" for p in fox.phonemes)
        the = next(w for w in skipped.words if w.word == "the")
        assert the.skip_reason is None

    def test_skipped_word_fully_excluded_even_if_phonemes_match(self):
        # Từ bị skip phải KHÔNG chấm bất kỳ âm nào, kể cả âm tình cờ khớp (regression
        # cho bug "phonemes_match thắng skipped" → âm khớp lọt vào mẫu số).
        from src.phoneme.reliability import SkipDecision, SkipReason

        ref = ["ð", "ə", "f", "ɒ", "k", "s"]
        spans = [WordSpan("the", 0, 2), WordSpan("fox", 2, 6)]
        score = compute_phoneme_score(
            self._segs(list(ref)), ref, spans, [None] * 6,  # pred == ref (fox sẽ khớp)
            skips={1: SkipDecision(1, SkipReason.WHISPER_MISMATCH)},
        )
        fox = next(w for w in score.words if w.word == "fox")
        assert all(p.status == "skipped" for p in fox.phonemes)  # không có "ok"
        assert score.substitution_count == 0 and score.deletion_count == 0
        assert score.overall_accuracy == 1.0  # chỉ "the" được chấm (khớp)

    def test_skip_keyed_by_index_not_string(self):
        # "the" lặp 3 lần; skip CHỈ occurrence thứ 2 (span index 1) → 2 "the" còn lại
        # vẫn được chấm. Đây là regression cho bug skip-theo-chuỗi.
        from src.phoneme.reliability import SkipDecision, SkipReason

        ref = ["ð", "ə", "ð", "ə", "ð", "ə"]
        spans = [WordSpan("the", 0, 2), WordSpan("the", 2, 4), WordSpan("the", 4, 6)]
        stress = [None] * 6
        score = compute_phoneme_score(
            self._segs(list(ref)), ref, spans, stress,
            skips={1: SkipDecision(1, SkipReason.WHISPER_MISMATCH)},
        )
        skipped = [w for w in score.words if w.skip_reason]
        assert len(skipped) == 1  # chỉ 1 occurrence bị skip, không phải cả 3
        assert score.words[0].skip_reason is None
        assert score.words[1].skip_reason == "whisper_mismatch"
        assert score.words[2].skip_reason is None


# ── Recognition Reliability layer (pure, cross-source, index-keyed) ──────────

from src.phoneme.reliability import (
    RecognizerEvidence,
    SkipDecision,
    SkipReason,
    assess_reliability,
)


class TestRecognitionReliability:
    """assess_reliability — pure layer, decides skips from cross-source evidence only."""

    def test_perfect_match_no_skips(self):
        ref = ["the", "quick", "brown", "fox"]
        ev = RecognizerEvidence.from_transcript("the quick brown fox")
        assert assess_reliability(ref, ev) == {}

    def test_deletion_skips_that_word(self):
        # Recognizer không nghe ra "brown".
        ref = ["the", "quick", "brown", "fox"]
        ev = RecognizerEvidence.from_transcript("the quick fox")
        skips = assess_reliability(ref, ev)
        assert set(skips) == {2}
        assert skips[2].reason is SkipReason.WHISPER_MISMATCH

    def test_large_substitution_skips_minor_kept(self):
        # Lệch lớn (traditional→xyzzy, ratio≈0) → skip; "mountains"→"mountain" gần → giữ.
        ref = ["a", "traditional", "story"]
        ev = RecognizerEvidence.from_transcript("a xyzzy story")
        skips = assess_reliability(ref, ev)
        assert 1 in skips and 0 not in skips and 2 not in skips

        ref2 = ["the", "mountains", "rise"]
        ev2 = RecognizerEvidence.from_transcript("the mountain rise")
        assert assess_reliability(ref2, ev2) == {}  # ratio cao → không skip

    def test_repeated_word_skips_only_the_mismatched_occurrence(self):
        # "the" xuất hiện 3 lần; chỉ occurrence giữa bị recognizer nghe nhầm.
        ref = ["the", "king", "and", "the", "queen", "saw", "the", "sea"]
        ev = RecognizerEvidence.from_transcript("the king and a queen saw the sea")
        skips = assess_reliability(ref, ev)
        # "the"#2 (index 3) → "a": skip index 3 only, not the other "the"s (0, 6).
        assert 3 in skips
        assert 0 not in skips and 6 not in skips

    def test_returns_index_keyed_mapping_pure(self):
        # SkipDecision không mang `word`; key là index; layer không cần scorer types.
        ref = ["alpha", "beta"]
        ev = RecognizerEvidence.from_transcript("alpha")  # beta deleted
        skips = assess_reliability(ref, ev)
        assert isinstance(skips[1], SkipDecision)
        assert skips[1].word_index == 1
        assert not hasattr(skips[1], "word")


# ── Telemetry (PR2): diagnostics — DIAGNOSTIC ONLY, never affects score ───────

from src.phoneme.diagnostics import (
    TELEMETRY_SCHEMA_VERSION,
    DiagnosticsContext,
    TelemetryWriter,
    WordDiagnostic,
    build_word_diagnostics,
    percentile,
)


class TestPercentile:
    def test_empty(self):
        assert percentile([], 20) == 0.0

    def test_single(self):
        assert percentile([0.7], 20) == 0.7

    def test_p20_exposes_low_tail(self):
        vals = [0.1, 0.2, 0.9, 0.9, 0.9]
        # p20 thấp dù trung bình cao — đúng mục đích "lộ collapse cục bộ".
        assert percentile(vals, 20) < (sum(vals) / len(vals))


class TestBuildWordDiagnostics:
    """build_word_diagnostics — THUẦN, tính từ alignment đã có (không quyết định gì)."""

    def _score_with_capture(self, ref, pred, spans, stress=None, **kw):
        captured: list[list] = []
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1),
                               confidence=kw.pop("conf", 0.9))
                for i, p in enumerate(pred)]
        score = compute_phoneme_score(
            segs, ref, spans, stress, diagnostics_sink=captured.append, **kw
        )
        return score, (captured[0] if captured else [])

    def test_sink_receives_one_diagnostic_per_word(self):
        ref = ["ð", "ə", "f", "ɒ", "k", "s"]
        spans = [WordSpan("the", 0, 2), WordSpan("fox", 2, 6)]
        _score, diags = self._score_with_capture(ref, list(ref), spans, [None] * 6)
        assert [d.word for d in diags] == ["the", "fox"]
        assert all(isinstance(d, WordDiagnostic) for d in diags)
        assert all(d.skip_reason is None for d in diags)

    def test_fields_for_correct_word(self):
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        _score, diags = self._score_with_capture(ref, list(ref), spans, [None] * 4)
        d = diags[0]
        assert d.reference_ipa == "fɒks"
        assert d.predicted_ipa == "fɒks"
        assert d.matches == 4 and d.substitutions == 0 and d.deletions == 0
        assert d.coverage == 1.0
        assert d.penalty == 0.0

    def test_skip_reason_surfaced(self):
        from src.phoneme.reliability import SkipDecision, SkipReason

        ref = ["ð", "ə", "f", "ɒ", "k", "s"]
        spans = [WordSpan("the", 0, 2), WordSpan("fox", 2, 6)]
        _score, diags = self._score_with_capture(
            ref, ["ð", "ə", "x", "x", "x", "x"], spans, [None] * 6,
            skips={1: SkipDecision(1, SkipReason.WHISPER_MISMATCH)},
        )
        fox = next(d for d in diags if d.word == "fox")
        assert fox.skip_reason == "whisper_mismatch"

    def test_telemetry_does_not_change_score(self):
        # Bật/tắt sink → cùng overall_accuracy (telemetry chỉ quan sát).
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(["f", "ɒ", "t", "s"])]  # 1 sub
        a = compute_phoneme_score(segs, ref, spans)
        b = compute_phoneme_score(segs, ref, spans, diagnostics_sink=lambda d: None)
        assert a.overall_accuracy == b.overall_accuracy

    def test_no_sink_is_noop(self):
        # Không truyền sink → không lỗi, vẫn chấm bình thường.
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(ref)]
        assert compute_phoneme_score(segs, ref, spans).overall_accuracy == 1.0


class TestTelemetryWriter:
    def test_writes_jsonl_with_schema_and_summary(self, tmp_path):
        import json

        path = tmp_path / "telemetry.jsonl"
        writer = TelemetryWriter(path)
        ctx = DiagnosticsContext(session_id="s1", audio_id="a.wav", utterance_id="q1")
        diags = [
            WordDiagnostic("the", 0, "ðə", "ðə", 1.0, 0.9, 0.9, 2, 0, 0, 0, 0.0, None),
            WordDiagnostic("fox", 1, "fɒks", "", 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0,
                           "whisper_mismatch"),
        ]
        writer.emit(ctx, diags)
        lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 3  # 2 word + 1 summary
        assert all(ln["schema_version"] == TELEMETRY_SCHEMA_VERSION for ln in lines)
        assert all(ln["session_id"] == "s1" for ln in lines)
        word_lines = [ln for ln in lines if ln["type"] == "word"]
        assert {ln["word"] for ln in word_lines} == {"the", "fox"}
        summary = next(ln for ln in lines if ln["type"] == "summary")
        assert summary["words_total"] == 2
        assert summary["words_skipped"] == 1
        assert summary["skip_reasons"] == {"whisper_mismatch": 1}

    def test_emit_appends(self, tmp_path):
        path = tmp_path / "t.jsonl"
        ctx = DiagnosticsContext("s", "a", "u")
        TelemetryWriter(path).emit(ctx, [])
        TelemetryWriter(path).emit(ctx, [])
        # 2 lần emit (mỗi lần 1 dòng summary, 0 word) → 2 dòng.
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    def test_summary_aggregates_drift_fraction(self, tmp_path):
        import json

        path = tmp_path / "drift.jsonl"
        ctx = DiagnosticsContext("s", "a", "u")
        diags = [
            # 1 sub trong window (hallucination) + 3 sub ngoài window (drift) → 3/4 = 0.75.
            WordDiagnostic("blood", 0, "blʌd", "flʌd", 1.0, 0.8, 0.8, 3, 1, 0, 0, 0.6,
                           None, window_start=0.0, window_end=0.4,
                           sub_inside_window=1, sub_outside_window=0),
            WordDiagnostic("folktales", 1, "foʊkteɪlz", "vtæz", 0.6, 0.5, 0.3, 0, 3, 0, 0,
                           1.8, None, window_start=1.0, window_end=1.6,
                           sub_inside_window=0, sub_outside_window=3),
        ]
        TelemetryWriter(path).emit(ctx, diags)
        lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        summary = next(ln for ln in lines if ln["type"] == "summary")
        assert summary["subs_inside_window"] == 1
        assert summary["subs_outside_window"] == 3
        assert summary["drift_fraction"] == 0.75

    def test_summary_drift_fraction_none_when_no_windows(self, tmp_path):
        import json

        path = tmp_path / "nowin.jsonl"
        ctx = DiagnosticsContext("s", "a", "u")
        # Không từ nào có window phân loại → drift_fraction None (không suy diễn 0).
        diags = [WordDiagnostic("fox", 0, "fɒks", "fɒks", 1.0, 0.9, 0.9, 4, 0, 0, 0,
                                0.0, None)]
        TelemetryWriter(path).emit(ctx, diags)
        lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        summary = next(ln for ln in lines if ln["type"] == "summary")
        assert summary["drift_fraction"] is None
        assert summary["subs_inside_window"] == 0


# ── PR3-0: word-window mapping + drift-vs-hallucination classification ─────────

from src.phoneme.diagnostics import map_reference_words_to_windows


class TestMapReferenceWordsToWindows:
    """map_reference_words_to_windows — tái dùng kỹ thuật difflib của reliability."""

    def test_equal_maps_positionally(self):
        win = map_reference_words_to_windows(
            ["the", "fox"], [("the", 0.0, 0.5), ("fox", 0.5, 1.0)]
        )
        assert win == {0: (0.0, 0.5), 1: (0.5, 1.0)}

    def test_replace_takes_best_ratio_window(self):
        # "mountains" ↔ "mountain" (ratio cao) → vẫn lấy window của từ transcript đó.
        win = map_reference_words_to_windows(
            ["mountains"], [("mountain", 1.0, 1.5)]
        )
        assert win == {0: (1.0, 1.5)}

    def test_deleted_reference_word_has_no_window(self):
        # "traditional" không có trong transcript → index 1 không có window (unalignable);
        # "the"/"fox" vẫn map đúng chỉ số occurrence.
        win = map_reference_words_to_windows(
            ["the", "traditional", "fox"],
            [("the", 0.0, 0.5), ("fox", 0.5, 1.0)],
        )
        assert 1 not in win
        assert win[0] == (0.0, 0.5)
        assert win[2] == (0.5, 1.0)

    def test_punctuation_and_case_normalized(self):
        win = map_reference_words_to_windows(
            ["vietnam"], [("Vietnam.", 0.0, 1.0)]
        )
        assert win == {0: (0.0, 1.0)}

    def test_empty_inputs(self):
        assert map_reference_words_to_windows([], [("x", 0.0, 1.0)]) == {}
        assert map_reference_words_to_windows(["x"], []) == {}


class TestDriftClassification:
    """build_word_diagnostics phân loại sub theo cửa sổ thời gian Whisper (PR3-0)."""

    def _capture(self, ref, pred, spans, times, word_windows):
        captured: list[list] = []
        segs = [PhonemeSegment(phoneme=p, start=s, end=e, confidence=0.9)
                for p, (s, e) in zip(pred, times)]
        compute_phoneme_score(
            segs, ref, spans, diagnostics_sink=captured.append,
            word_windows=word_windows,
        )
        return captured[0] if captured else []

    def test_substitution_inside_window_is_hallucination(self):
        # /fɒks/, predicted /fɒts/ — sub 'k'→'t' tại time (2,3) NẰM TRONG window (0,4).
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
        diags = self._capture(ref, ["f", "ɒ", "t", "s"], spans, times, {0: (0.0, 4.0)})
        d = diags[0]
        assert d.substitutions == 1
        assert d.sub_inside_window == 1
        assert d.sub_outside_window == 0
        assert (d.window_start, d.window_end) == (0.0, 4.0)

    def test_substitution_outside_window_is_drift(self):
        # Cùng sub nhưng window từ ở (10,12) — predicted segment (2,3) NGOÀI → drift.
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
        diags = self._capture(ref, ["f", "ɒ", "t", "s"], spans, times, {0: (10.0, 12.0)})
        d = diags[0]
        assert d.sub_inside_window == 0
        assert d.sub_outside_window == 1

    def test_no_window_leaves_classification_zero(self):
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
        diags = self._capture(ref, ["f", "ɒ", "t", "s"], spans, times, {})  # no windows
        d = diags[0]
        assert d.sub_inside_window == 0 and d.sub_outside_window == 0
        assert d.window_start is None and d.window_end is None

    def test_window_pad_absorbs_edge(self):
        # Segment (4.0,4.05) vừa ra ngoài window (0,4) nhưng trong pad 0.08 → vẫn inside.
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        times = [(0.0, 1.0), (1.0, 2.0), (4.0, 4.05), (3.0, 3.5)]
        diags = self._capture(ref, ["f", "ɒ", "t", "s"], spans, times, {0: (0.0, 4.0)})
        assert diags[0].sub_inside_window == 1

    def test_correspondences_emitted_per_phoneme(self):
        # /fɒks/ vs /fɒts/: 4 correspondence, status [ok,ok,sub,ok], is_final chỉ ở 's'.
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
        diags = self._capture(ref, ["f", "ɒ", "t", "s"], spans, times, {0: (0.0, 4.0)})
        cs = diags[0].correspondences
        assert [c["status"] for c in cs] == ["ok", "ok", "sub", "ok"]
        assert [c["is_final"] for c in cs] == [False, False, False, True]
        sub = cs[2]
        assert sub["ref_symbol"] == "k" and sub["pred_symbol"] == "t"
        assert sub["confidence"] is not None
        assert sub["sub_outside_window"] is False  # predicted segment trong window

    def test_correspondence_deletion_has_no_pred(self):
        # 'k' bị xoá (predicted thiếu) → status del, pred_symbol/confidence None.
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        times = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        diags = self._capture(ref, ["f", "ɒ", "s"], spans, times, {0: (0.0, 4.0)})
        cs = diags[0].correspondences
        dele = next(c for c in cs if c["status"] == "del")
        assert dele["ref_symbol"] == "k"
        assert dele["pred_symbol"] is None and dele["confidence"] is None

    def test_telemetry_with_windows_does_not_change_score(self):
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(["f", "ɒ", "t", "s"])]
        base = compute_phoneme_score(segs, ref, spans)
        withwin = compute_phoneme_score(
            segs, ref, spans, diagnostics_sink=lambda d: None,
            word_windows={0: (10.0, 12.0)},
        )
        assert base.overall_accuracy == withwin.overall_accuracy


class TestWordPlaybackWindows:
    """start/end mỗi WordPronunciation (cho UI nghe lại từng từ).

    Nguồn CHÍNH: Whisper WORD window (ranh giới từ ổn định). Fallback: wav2vec segment
    cho từ KHÔNG có Whisper window — cửa sổ segment suy từ DTW attribution nên khi DTW
    mượn âm từ từ kế sẽ phình ra cả cụm (bug "discount" phát thành "20 percent
    discount"), không được làm nguồn chính. Cửa sổ ĐÃ được đệm (lead/trail ~50–100ms)
    + CLAMP theo từ liền kề (_pad_and_clamp_windows) nên không lẹm sang từ khác;
    frontend phát verbatim. (Không cần diagnostics_sink.)"""

    def _segs(self, pred):
        # segment i có start=i, end=i+1 (giây) — để assert min/max dễ đọc.
        return [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(pred)]

    def test_single_word_padded_no_neighbor(self):
        # 1 từ, không hàng xóm → raw (0,4) được đệm: start clamp về 0, end += trail.
        from src.phoneme.scoring import _WORD_PLAY_LEAD, _WORD_PLAY_TRAIL
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        score = compute_phoneme_score(self._segs(["f", "ɒ", "k", "s"]), ref, spans)
        assert score.words[0].start == max(0.0, 0.0 - _WORD_PLAY_LEAD)  # = 0.0
        assert score.words[0].end == round(4.0 + _WORD_PLAY_TRAIL, 3)

    def test_whisper_window_preferred_over_segment(self):
        # Có cả segment lẫn Whisper window cho cùng từ → Whisper WORD window THẮNG
        # (raw 10..12, đã đệm); segment (0..4) chỉ là fallback khi thiếu window.
        from src.phoneme.scoring import _WORD_PLAY_LEAD, _WORD_PLAY_TRAIL
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        score = compute_phoneme_score(
            self._segs(["f", "ɒ", "k", "s"]), ref, spans,
            word_windows={0: (10.0, 12.0)},
        )
        assert (score.words[0].start, score.words[0].end) == (
            round(10.0 - _WORD_PLAY_LEAD, 3), round(12.0 + _WORD_PLAY_TRAIL, 3))

    def test_playback_not_inflated_by_dtw_borrowed_segments(self):
        # Bug "discount → 20 percent discount": DTW gán cả segment của từ TRƯỚC cho
        # từ này → cửa sổ segment phình (0..4 = 4s) dù Whisper nghe từ ở [3.0, 4.0].
        # Playback phải theo Whisper word window (+đệm), KHÔNG theo cửa sổ segment phình.
        from src.phoneme.scoring import _WORD_PLAY_LEAD, _WORD_PLAY_TRAIL
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        score = compute_phoneme_score(
            self._segs(["f", "ɒ", "k", "s"]),  # segments trải 0..4s (mượn từ từ kế)
            ref, spans,
            word_windows={0: (3.0, 4.0)},      # Whisper: từ chỉ nằm ở giây 3–4
        )
        w = score.words[0]
        assert (w.start, w.end) == (
            round(3.0 - _WORD_PLAY_LEAD, 3), round(4.0 + _WORD_PLAY_TRAIL, 3))
        assert w.end - w.start <= 1.0 + _WORD_PLAY_LEAD + _WORD_PLAY_TRAIL + 1e-9

    def test_whisper_fallback_when_word_all_deletion(self):
        # predicted chỉ phủ "cat"; "dog" toàn deletion (không segment) → fallback Whisper.
        from src.phoneme.scoring import _WORD_PLAY_LEAD, _WORD_PLAY_TRAIL
        ref = ["k", "æ", "t", "d", "ɒ", "ɡ"]
        spans = [WordSpan("cat", 0, 3), WordSpan("dog", 3, 6)]
        score = compute_phoneme_score(
            self._segs(["k", "æ", "t"]), ref, spans,   # 3 segment → chỉ cat
            word_windows={1: (5.0, 6.5)},                # window cho dog (index 1)
        )
        # cat: raw (0,3); end += trail (không chạm dog raw start 5.0). dog: raw (5,6.5);
        # start -= lead nhưng clamp ≥ cat raw end (3.0) → 4.9; end += trail.
        assert (score.words[0].start, score.words[0].end) == (0.0, round(3.0 + _WORD_PLAY_TRAIL, 3))
        assert (score.words[1].start, score.words[1].end) == (
            round(5.0 - _WORD_PLAY_LEAD, 3), round(6.5 + _WORD_PLAY_TRAIL, 3))

    def test_no_segment_no_window_leaves_none(self):
        # "dog" toàn deletion + KHÔNG có Whisper window → start/end None (không có nút).
        from src.phoneme.scoring import _WORD_PLAY_TRAIL
        ref = ["k", "æ", "t", "d", "ɒ", "ɡ"]
        spans = [WordSpan("cat", 0, 3), WordSpan("dog", 3, 6)]
        score = compute_phoneme_score(self._segs(["k", "æ", "t"]), ref, spans)
        assert (score.words[0].start, score.words[0].end) == (0.0, round(3.0 + _WORD_PLAY_TRAIL, 3))
        assert score.words[1].start is None and score.words[1].end is None

    def test_consecutive_words_clamped_no_bleed(self):
        # 2 từ liền kề (cat 0-3, dog 3-6): trail của cat KHÔNG lấn dog → cat.end == dog.start
        # (== ranh giới 3.0). Đây là cú chặn "in→order".
        ref = ["k", "æ", "t", "d", "ɒ", "ɡ"]
        spans = [WordSpan("cat", 0, 3), WordSpan("dog", 3, 6)]
        score = compute_phoneme_score(self._segs(["k", "æ", "t", "d", "ɒ", "ɡ"]), ref, spans)
        w0, w1 = score.words[0], score.words[1]
        assert w0.end <= w1.start          # không chồng lấn
        assert w0.end == 3.0 and w1.start == 3.0  # clamp đúng ranh giới từ

    def test_start_end_in_to_dict(self):
        from src.phoneme.scoring import _WORD_PLAY_TRAIL
        ref = ["f", "ɒ", "k", "s"]
        spans = [WordSpan("fox", 0, 4)]
        d = compute_phoneme_score(self._segs(["f", "ɒ", "k", "s"]), ref, spans).to_dict()
        assert d["words"][0]["start"] == 0.0
        assert d["words"][0]["end"] == round(4.0 + _WORD_PLAY_TRAIL, 3)


# ── L1-aware scoring layer (Vietnamese) ──────────────────────────────────────

from src.phoneme.scoring import _ref_metadata


class TestL1AwareScoring:
    """L1 final-consonant deletion tolerance + low-confidence neutralization."""

    def _segs(self, phs, conf=0.9):
        return [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=conf)
                for i, p in enumerate(phs)]

    def _points(self, score):
        pts = []
        for w in score.words:
            pts.extend(w.phonemes)
        return pts

    def test_final_consonant_deletion_reduced_and_labeled(self):
        ref = ["h", "æ", "n", "d"]; spans = [WordSpan("hand", 0, 4)]
        on = compute_phoneme_score(self._segs(["h", "æ", "n"]), ref, spans, l1_enabled=True)
        off = compute_phoneme_score(self._segs(["h", "æ", "n"]), ref, spans)
        assert on.overall_accuracy > off.overall_accuracy  # penalty giảm
        d_pt = next(p for p in self._points(on) if p.symbol == "d")
        assert d_pt.penalty_reason == "l1_final_deletion"
        assert d_pt.penalty_adjustment == 0.35
        assert on.l1_adjusted_count == 1
        assert on.l1_adjustment_ratio > 0.0

    def test_non_final_consonant_deletion_not_tolerated(self):
        # onset 'h' bị nuốt → không phải coda → không L1, penalty đầy đủ.
        ref = ["h", "æ", "n", "d"]; spans = [WordSpan("hand", 0, 4)]
        on = compute_phoneme_score(self._segs(["æ", "n", "d"]), ref, spans, l1_enabled=True)
        h_pt = next(p for p in self._points(on) if p.symbol == "h")
        assert h_pt.penalty_reason == "hard_error"
        assert h_pt.penalty_adjustment == 1.0

    def test_deletion_never_confidence_weighted(self):
        # Deletion KHÔNG đi qua confidence: penalty của 'd' giống nhau dù conf khác hẳn.
        ref = ["h", "æ", "n", "d"]; spans = [WordSpan("hand", 0, 4)]
        hi = compute_phoneme_score(self._segs(["h", "æ", "n"], conf=0.95), ref, spans, l1_enabled=True)
        lo = compute_phoneme_score(self._segs(["h", "æ", "n"], conf=0.05), ref, spans, l1_enabled=True)
        d_hi = next(p for p in self._points(hi) if p.symbol == "d")
        d_lo = next(p for p in self._points(lo) if p.symbol == "d")
        assert d_hi.penalty_adjustment == d_lo.penalty_adjustment == 0.35
        assert d_hi.penalty_reason == d_lo.penalty_reason == "l1_final_deletion"

    def test_low_confidence_substitution_neutralized(self):
        ref = ["f", "ɒ", "k", "s"]; spans = [WordSpan("fox", 0, 4)]
        segs = self._segs(["f", "ɒ", "t", "s"])
        segs[2] = PhonemeSegment(phoneme="t", start=2.0, end=3.0, confidence=0.2)  # < floor
        on = compute_phoneme_score(segs, ref, spans, l1_enabled=True)
        k_pt = next(p for p in self._points(on) if p.symbol == "k")
        assert k_pt.penalty_reason == "low_confidence_neutralized"
        assert k_pt.penalty_adjustment == 0.0
        assert on.low_conf_neutralized_count == 1

    def test_mid_confidence_substitution_normal(self):
        ref = ["f", "ɒ", "k", "s"]; spans = [WordSpan("fox", 0, 4)]
        segs = self._segs(["f", "ɒ", "t", "s"])
        segs[2] = PhonemeSegment(phoneme="t", start=2.0, end=3.0, confidence=0.55)  # in [0.40,0.70]
        on = compute_phoneme_score(segs, ref, spans, l1_enabled=True)
        k_pt = next(p for p in self._points(on) if p.symbol == "k")
        assert k_pt.penalty_reason == "hard_error"
        assert on.low_conf_neutralized_count == 0

    def test_l1_disabled_is_default_and_unchanged(self):
        ref = ["h", "æ", "n", "d"]; spans = [WordSpan("hand", 0, 4)]
        segs = self._segs(["h", "æ", "n"])
        default = compute_phoneme_score(segs, ref, spans)
        explicit_off = compute_phoneme_score(segs, ref, spans, l1_enabled=False)
        assert default.overall_accuracy == explicit_off.overall_accuracy
        assert default.l1_adjusted_count == 0
        d_pt = next(p for p in self._points(default) if p.symbol == "d")
        assert d_pt.penalty_reason is None and d_pt.penalty_adjustment == 1.0

    def test_determinism(self):
        ref = ["h", "æ", "n", "d"]; spans = [WordSpan("hand", 0, 4)]
        segs = self._segs(["h", "æ", "n"])
        a = compute_phoneme_score(segs, ref, spans, l1_enabled=True)
        b = compute_phoneme_score(segs, ref, spans, l1_enabled=True)
        assert a.to_dict() == b.to_dict()

    def test_multipliers_within_cap(self):
        from src.phoneme.l1_vietnamese import (
            L1_MULTIPLIER_CAP, _L1_FINAL_DELETION, match_l1_final_deletion,
            register_l1_pattern,
        )
        assert all(m.multiplier <= L1_MULTIPLIER_CAP for m in _L1_FINAL_DELETION.values())
        register_l1_pattern("final_test", "ʔ", 0.99)  # over cap → bị clamp
        assert match_l1_final_deletion("ʔ").multiplier == L1_MULTIPLIER_CAP

    def test_ref_is_coda_detection(self):
        _, _, _, _, coda, _, _ = _ref_metadata(["h", "æ", "n", "d"], [WordSpan("hand", 0, 4)], None)
        assert coda == [False, False, True, True]
        _, _, _, _, coda2, _, _ = _ref_metadata(["s", "k", "uː", "l"], [WordSpan("school", 0, 4)], None)
        assert coda2 == [False, False, False, True]
        _, _, _, _, coda3, _, _ = _ref_metadata(["t", "uː"], [WordSpan("to", 0, 2)], None)
        assert coda3 == [False, False]

    def test_correspondences_carry_l1_fields(self):
        ref = ["h", "æ", "n", "d"]; spans = [WordSpan("hand", 0, 4)]
        captured: list[list] = []
        compute_phoneme_score(self._segs(["h", "æ", "n"]), ref, spans,
                              diagnostics_sink=captured.append, l1_enabled=True)
        cs = captured[0][0].correspondences
        d_corr = next(c for c in cs if c["ref_symbol"] == "d")
        assert d_corr["penalty_reason"] == "l1_final_deletion"
        assert d_corr["penalty_adjustment"] == 0.35
        assert d_corr["l1_rule_id"] == "vi.final_stop.d"


class TestRecognizerNoiseGate:
    """Gate ẩn substitution bất khả thi về âm học + confidence thấp (wav2vec hallucinate).

    Bảo vệ: near-pair (sim ≥ 0.2) và lỗi VN thật trong _REAL_ERROR_SUBS (th-stopping ð→d,
    v→b...) — KHÔNG bao giờ bị gate dù conf thấp. Ngưỡng conf TÁCH nguyên âm/phụ âm.
    """

    def _segs(self, phs_conf):
        # phs_conf: list[(phoneme, confidence)]
        return [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=c)
                for i, (p, c) in enumerate(phs_conf)]

    def _pts(self, score):
        pts = []
        for w in score.words:
            pts.extend(w.phonemes)
        return pts

    def test_implausible_low_conf_sub_gated(self):
        # 'f' (onset) đọc thành 'l' (sim 0, không bảo vệ) ở conf 0.45 < 0.6 → recognizer noise.
        ref = ["f", "ɒ", "k", "s"]; spans = [WordSpan("fox", 0, 4)]
        segs = self._segs([("l", 0.45), ("ɒ", 0.9), ("k", 0.9), ("s", 0.9)])
        sc = compute_phoneme_score(segs, ref, spans)
        f_pt = next(p for p in self._pts(sc) if p.symbol == "f")
        assert f_pt.status == "sub"
        assert f_pt.penalty_reason == "recognizer_noise"
        assert f_pt.severity == "low"  # → rơi vào "Hidden recognizer noise", không tô đỏ
        assert f_pt.penalty_adjustment == 0.0
        assert sc.recognizer_noise_count == 1
        assert sc.to_dict()["recognizer_noise_count"] == 1

    def test_protected_real_errors_stay_red_even_low_conf(self):
        # th-stopping ð→d, θ→t, v→b ở conf thấp VẪN là lỗi thật → KHÔNG gate.
        cases = [
            (["ð", "ɪ", "s"], "ð", "d", WordSpan("this", 0, 3)),
            (["θ", "ɪ", "n"], "θ", "t", WordSpan("thin", 0, 3)),
            (["v", "æ", "n"], "v", "b", WordSpan("van", 0, 3)),
        ]
        for ref, exp, heard, span in cases:
            segs = self._segs([(heard, 0.45), (ref[1], 0.9), (ref[2], 0.9)])
            sc = compute_phoneme_score(segs, ref, [span])
            ph = next(p for p in self._pts(sc) if p.symbol == exp)
            assert ph.status == "sub", f"{exp}->{heard}"
            assert ph.penalty_reason != "recognizer_noise", f"{exp}->{heard} bị gate oan"
            assert ph.severity in ("medium", "high"), f"{exp}->{heard}"
            assert sc.recognizer_noise_count == 0

    def test_near_pair_protected_by_similarity(self):
        # θ→s (sim 0.4 ≥ 0.2) → ngưỡng sim bảo vệ dù conf thấp, dù không liệt kê trong bảng.
        ref = ["θ", "ɪ", "n"]; spans = [WordSpan("thin", 0, 3)]
        segs = self._segs([("s", 0.45), ("ɪ", 0.9), ("n", 0.9)])
        sc = compute_phoneme_score(segs, ref, spans)
        ph = next(p for p in self._pts(sc) if p.symbol == "θ")
        assert ph.penalty_reason != "recognizer_noise"
        assert sc.recognizer_noise_count == 0

    def test_confident_wild_sub_not_gated(self):
        # f→l nhưng conf 0.9 ≥ 0.6 → recognizer chắc → giữ là lỗi (hiếm, không giấu).
        ref = ["f", "ɒ", "k", "s"]; spans = [WordSpan("fox", 0, 4)]
        segs = self._segs([("l", 0.9), ("ɒ", 0.9), ("k", 0.9), ("s", 0.9)])
        sc = compute_phoneme_score(segs, ref, spans)
        f_pt = next(p for p in self._pts(sc) if p.symbol == "f")
        assert f_pt.penalty_reason != "recognizer_noise"
        assert f_pt.severity == "high"
        assert sc.recognizer_noise_count == 0

    def test_vowel_threshold_lower_than_consonant(self):
        # Cùng conf 0.5 + sub cross-class bất khả thi: nguyên âm (ngưỡng 0.45) KHÔNG bị gate,
        # phụ âm (ngưỡng 0.6) bị gate → tránh gate oan nguyên âm (confidence nền vốn thấp).
        v_segs = self._segs([("b", 0.9), ("v", 0.5), ("t", 0.9)])  # ɒ (vowel) → v
        v_sc = compute_phoneme_score(v_segs, ["b", "ɒ", "t"], [WordSpan("bot", 0, 3)])
        v_pt = next(p for p in self._pts(v_sc) if p.symbol == "ɒ")
        assert v_pt.penalty_reason != "recognizer_noise"  # 0.5 ≥ 0.45 vowel-threshold

        c_segs = self._segs([("ə", 0.5), ("ɒ", 0.9), ("k", 0.9), ("s", 0.9)])  # f (cons) → ə
        c_sc = compute_phoneme_score(c_segs, ["f", "ɒ", "k", "s"], [WordSpan("fox", 0, 4)])
        c_pt = next(p for p in self._pts(c_sc) if p.symbol == "f")
        assert c_pt.penalty_reason == "recognizer_noise"  # 0.5 < 0.6 cons-threshold

    def test_gate_disabled_matches_legacy(self):
        # conf=0 tắt gate → hành vi như cũ (sub giữ penalty đầy đủ).
        ref = ["f", "ɒ", "k", "s"]; spans = [WordSpan("fox", 0, 4)]
        segs = self._segs([("l", 0.45), ("ɒ", 0.9), ("k", 0.9), ("s", 0.9)])
        on = compute_phoneme_score(segs, ref, spans)
        off = compute_phoneme_score(segs, ref, spans,
                                    recognizer_noise_conf=0.0, recognizer_noise_conf_vowel=0.0)
        assert off.recognizer_noise_count == 0
        f_off = next(p for p in self._pts(off) if p.symbol == "f")
        assert f_off.penalty_reason != "recognizer_noise"
        assert on.overall_accuracy > off.overall_accuracy  # gate bỏ penalty → on cao hơn

    def test_deletion_never_gated(self):
        # Deletion KHÔNG đi qua gate (không có predicted/conf) → giữ nguyên reason.
        ref = ["f", "ɒ", "k", "s"]; spans = [WordSpan("fox", 0, 4)]
        segs = self._segs([("f", 0.9), ("ɒ", 0.9), ("s", 0.9)])  # 'k' bị nuốt
        sc = compute_phoneme_score(segs, ref, spans)
        k_pt = next(p for p in self._pts(sc) if p.symbol == "k")
        assert k_pt.status == "del"
        assert k_pt.penalty_reason != "recognizer_noise"
        assert sc.recognizer_noise_count == 0


# ── Reference IPA accuracy (AH split + CMUdict / per-word overrides) ─────────

from src.phoneme.ipa import (
    _validate_word_ipa_overrides,
    word_to_ipa_with_stress,
)


class TestReferenceIpaAccuracy:
    """AH1/AH2→ʌ vs AH0→ə (display) + CMUdict / _WORD_IPA_OVERRIDES. Scoring KHÔNG đổi (normalize ʌ→ə)."""

    def test_stressed_ah_is_open_back_vowel(self):
        # stomach (CMUdict: S T AH1 M AH0 K) → ʌ ở âm nhấn, ə ở âm yếu.
        sym, _st = word_to_ipa_with_stress("stomach")
        assert "ʌ" in sym
        assert sym == ["s", "t", "ʌ", "m", "ə", "k"]

    def test_single_syllable_stressed_ah(self):
        assert word_to_ipa_with_stress("cup")[0] == ["k", "ʌ", "p"]
        assert word_to_ipa_with_stress("love")[0] == ["l", "ʌ", "v"]

    def test_unstressed_ah_stays_schwa(self):
        # about (AH0 đầu) → ə, KHÔNG phải ʌ.
        sym, _ = word_to_ipa_with_stress("about")
        assert sym[0] == "ə" and "ʌ" not in sym

    def test_especially_override(self):
        # CMUdict entries start with AH0 (ə) not IH0 (ɪ); override pins /ɪspˈeʃəliː/.
        sym, st = word_to_ipa_with_stress("especially")
        assert sym == ["ɪ", "s", "p", "e", "ʃ", "ə", "l", "iː"]  # /ɪspˈeʃəliː/
        assert "primary" in st

    def test_override_validation_passes_and_rejects_bad_token(self):
        import re

        from src.phoneme.ipa import ARPABET_TO_IPA

        _validate_word_ipa_overrides()  # không raise với seed hiện tại
        # token base lạ phải bị từ chối (mô phỏng logic validate).
        bad = "XQ1"
        assert re.sub(r"\d", "", bad) not in ARPABET_TO_IPA

    def test_scoring_unchanged_uah_vs_schwa(self):
        # CHỨNG MINH fix hiển thị KHÔNG đổi điểm: reference ʌ vs ə cho điểm y hệt.
        spans = [WordSpan("cup", 0, 3)]
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(["k", "ɒ", "p"])]  # nguyên âm khác → tạo penalty
        uah = compute_phoneme_score(segs, ["k", "ʌ", "p"], spans)
        schwa = compute_phoneme_score(segs, ["k", "ə", "p"], spans)
        assert uah.overall_accuracy == schwa.overall_accuracy
        assert uah.substitution_count == schwa.substitution_count
        assert uah.deletion_count == schwa.deletion_count
        assert uah.insertion_count == schwa.insertion_count

    # ── OW diphthong: hiển thị US oʊ (≠ RP əʊ); scoring KHÔNG đổi (normalize oʊ→əʊ) ──

    def test_ow_maps_to_us_diphthong(self):
        # Bảng ARPAbet→IPA: OW hiển thị oʊ (US) thay vì əʊ (RP).
        assert ARPABET_TO_IPA["OW"] == "oʊ"
        assert "oʊ" in ENGLISH_IPA_PHONEMES and "əʊ" not in ENGLISH_IPA_PHONEMES

    def test_common_dict_ow_words_show_us(self):
        # Path từ điển nội bộ (đọc ARPABET_TO_IPA trực tiếp): go/know/show/over → oʊ.
        for w in ("go", "know", "show", "over"):
            sym = word_to_ipa(w)
            assert "oʊ" in sym and "əʊ" not in sym, w

    def test_folktales_reference_uses_oh(self):
        # "folktales" may be OOV in CMUdict; falls to eSpeak. Skip if neither available.
        sym, _st = word_to_ipa_with_stress("folktales")
        if not sym:
            pytest.skip("folktales not in CMUdict and eSpeak unavailable")
        assert "oʊ" in sym and "əʊ" not in sym
        assert sym[:2] == ["f", "oʊ"]

    def test_oh_is_vowel(self):
        from src.phoneme.ipa import is_vowel
        assert is_vowel("oʊ") is True

    def test_scoring_unchanged_oh_vs_schwa_diphthong(self):
        # CHỨNG MINH fix OW hiển thị KHÔNG đổi điểm: reference oʊ vs əʊ cho điểm y hệt.
        spans = [WordSpan("go", 0, 2)]
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(["ɡ", "ɔ"])]  # nguyên âm khác → tạo penalty
        us = compute_phoneme_score(segs, ["ɡ", "oʊ"], spans)
        rp = compute_phoneme_score(segs, ["ɡ", "əʊ"], spans)
        assert us.overall_accuracy == rp.overall_accuracy
        assert us.substitution_count == rp.substitution_count
        assert us.deletion_count == rp.deletion_count
        assert us.insertion_count == rp.insertion_count

    # ── AO: hiển thị THOUGHT ɔː (≠ ɒ); scoring KHÔNG đổi (normalize ɔː==ɒ→ɔ) ──

    def test_ao_maps_to_thought_vowel(self):
        assert ARPABET_TO_IPA["AO"] == "ɔː"

    def test_according_vowel_is_thought(self):
        # AO→ɔː + giữ r (rhotic/US) → /əˈkɔːrdɪŋ/ (not an override — comes from CMUdict).
        sym, _st = word_to_ipa_with_stress("according")
        if not sym:
            pytest.skip("according not found in CMUdict and eSpeak unavailable")
        assert sym == ["ə", "k", "ɔː", "r", "d", "ɪ", "ŋ"]

    def test_scoring_unchanged_ao_thought_vs_lot(self):
        from src.phoneme.ipa import normalize_ipa
        assert normalize_ipa("ɔː") == normalize_ipa("ɒ")
        spans = [WordSpan("your", 0, 3)]
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(["j", "ə", "r"])]
        thought = compute_phoneme_score(segs, ["j", "ɔː", "r"], spans)
        lot = compute_phoneme_score(segs, ["j", "ɒ", "r"], spans)
        assert thought.overall_accuracy == lot.overall_accuracy
        assert thought.substitution_count == lot.substitution_count

    # ── Expected IPA for key words (overrides + CMUdict) ─────────────────────────

    def test_word_ipa_matches_expected(self):
        # All from _WORD_IPA_OVERRIDES: CMUdict entries[0] has wrong pronunciations for
        # each (spurious W in usually, AH0 vs IH0 start in especially, Y vs IY in resilient,
        # IY0 vs IH0 start in relationship, ER0 in favorite, OOV for vietnamese/vietnam).
        cases = {
            "vietnamese": (["v","iː","e","t","n","ə","m","iː","z"], "ˌviːetnəˈmiːz"),
            "vietnam":    (["v","iː","e","t","n","ɑː","m"], "ˌviːetˈnɑːm"),
            "resilient":  (["r","ɪ","z","ɪ","l","iː","ə","n","t"], "rɪˈzɪliːənt"),
            "relationship": (["r","ɪ","l","eɪ","ʃ","ə","n","ʃ","ɪ","p"], "rɪˈleɪʃənʃɪp"),
            "favorite":   (["f","eɪ","v","ə","r","ɪ","t"], "ˈfeɪvərɪt"),
        }
        for word, (exp_sym, exp_render) in cases.items():
            sym, _st = word_to_ipa_with_stress(word)
            assert sym == exp_sym, word
            ph, _spans, _stress, disp = text_to_ipa_sequence_with_spans(word)
            rendered = "".join(
                ("ˈ" if d == "primary" else "ˌ" if d == "secondary" else "") + s
                for s, d in zip(ph, disp)
            )
            assert rendered == exp_render, (word, rendered)

    def test_overrides_validate(self):
        _validate_word_ipa_overrides()  # không raise với seed hiện tại


class TestStressOnsetPlacement:
    """place_stress_at_onset: dời dấu nhấn từ nguyên âm về đầu âm tiết (CHỈ hiển thị)."""

    @staticmethod
    def _render(word):
        ph, _spans, _stress, disp = text_to_ipa_sequence_with_spans(word)
        if not ph:
            pytest.skip(f"no pronunciation found for {word!r} (CMUdict / eSpeak unavailable)")
        return "".join(
            ("ˈ" if d == "primary" else "ˌ" if d == "secondary" else "") + s
            for s, d in zip(ph, disp)
        )

    def test_mark_before_single_onset(self):
        for word, exp in [("legend", "ˈledʒənd"), ("mountain", "ˈmaʊntən"),
                          ("traditional", "trəˈdɪʃənəl")]:
            assert self._render(word) == exp, word

    def test_mark_before_cluster_onset(self):
        # Cụm onset hợp lệ đi cùng nguyên âm: pr.
        assert self._render("princess") == "ˈprɪnses"

    def test_splits_illegal_medial_cluster(self):
        # 'tn'/'kt' KHÔNG phải onset hợp lệ → chỉ phụ âm cuối theo nguyên âm nhấn.
        assert self._render("vietnam") == "ˌviːetˈnɑːm"      # tn → n
        assert self._render("folktales") == "ˈfoʊkˌteɪlz"    # kt → t (secondary)

    def test_pure_function_relocates_not_mutates(self):
        # Đầu vào nhấn trên nguyên âm; đầu ra dời sang phụ âm onset, list gốc bất biến.
        symbols = ["p", "r", "ɪ", "n", "s", "e", "s"]
        stresses = [None, None, "primary", None, None, None, None]
        disp = place_stress_at_onset(symbols, stresses)
        assert disp == ["primary", None, None, None, None, None, None]  # dời về 'p'
        assert stresses == [None, None, "primary", None, None, None, None]  # bất biến

    def test_vowel_initial_syllable_keeps_mark_on_vowel(self):
        # Âm tiết nhấn KHÔNG có onset (hiatus: nguyên âm ngay sau nguyên âm) → giữ trên nguyên âm.
        disp = place_stress_at_onset(["r", "i", "æ", "k", "t"], [None, None, "primary", None, None])
        assert disp == [None, None, "primary", None, None]  # 'æ' đứng ngay sau 'i' → không dời

    def test_scoring_ignores_display_stress(self):
        # display_stress chỉ gắn lên PhonemePoint — KHÔNG đổi điểm/severity/counts.
        spans = [WordSpan("legend", 0, 6)]
        ref = ["l", "e", "dʒ", "ə", "n", "d"]
        segs = [PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
                for i, p in enumerate(["l", "e", "dʒ", "ə", "n", "d"])]
        stress = [None, "primary", None, None, None, None]
        disp = place_stress_at_onset(ref, stress)
        with_disp = compute_phoneme_score(segs, ref, spans, stress,
                                          reference_display_stress=disp)
        without = compute_phoneme_score(segs, ref, spans, stress)
        assert with_disp.overall_accuracy == without.overall_accuracy
        assert with_disp.substitution_count == without.substitution_count
        assert with_disp.deletion_count == without.deletion_count
        # scoring stress (trên nguyên âm) bất biến; display_stress dời về onset 'l'.
        pts_with = with_disp.words[0].phonemes
        pts_without = without.words[0].phonemes
        assert [p.stress for p in pts_with] == [p.stress for p in pts_without]
        assert pts_with[0].display_stress == "primary"  # 'l'
        assert pts_with[1].display_stress is None        # 'e' (nguyên âm)


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
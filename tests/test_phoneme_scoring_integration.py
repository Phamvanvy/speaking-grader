"""Test phoneme data integration into AI scoring payload.

Verifies that phoneme_result from wav2vec analysis is correctly included in the
scoring user prompt sent to the AI model.
"""

import json
from unittest.mock import MagicMock, patch

from src.config import Config
from src.features import Features
from src.gating import GatingResult
from src.phoneme.models import (
    PhonemeError,
    PhonemeErrorType,
    PhonemeResult,
    PhonemeScore,
    PhonemeSegment,
)
from src.scoring import (
    _build_system_prompt,
    _build_user_prompt,
    _drop_invalid_corrections,
    score,
)
from src.schema import (
    CriterionScore,
    LexicalCorrection,
    SpeakingResult,
)


class TestPhonemeInUserPrompt:
    """Phoneme data appears correctly in scoring payload."""

    def _build_features(self) -> Features:
        return Features(
            speech_rate_wpm=120.0,
            word_count=10,
            speaking_duration_sec=5.0,
            audio_duration_sec=6.0,
            silence_sec=1.0,
            pause_count=1,
            total_pause_sec=0.5,
            longest_pause_sec=0.5,
            filler_count=0,
            avg_word_probability=0.85,
            min_word_probability=0.3,
        )

    def _build_transcription(self):
        from src.asr import Transcription, Word

        words = [
            Word(text="Hello", start=0.0, end=0.5, probability=0.9),
            Word(text="world", start=0.6, end=1.0, probability=0.85),
        ]
        return Transcription(
            text="Hello world",
            words=words,
            duration=1.5,
        )

    def _build_gating(self) -> GatingResult:
        return GatingResult(
            is_empty=False,
            task_completion_floor=None,
            reasons=[],
            reference_coverage=None,
            fail_reference_match=False,
        )

    def _build_phoneme_result(self) -> PhonemeResult:
        return PhonemeResult(
            audio_path="test.wav",
            segments=[
                PhonemeSegment(phoneme="h", start=0.0, end=0.1, confidence=0.95, backend="wav2vec"),
                PhonemeSegment(phoneme="e", start=0.1, end=0.2, confidence=0.9, backend="wav2vec"),
                PhonemeSegment(phoneme="l", start=0.2, end=0.3, confidence=0.85, backend="wav2vec"),
                PhonemeSegment(phoneme="o", start=0.3, end=0.5, confidence=0.92, backend="wav2vec"),
            ],
            reference_phonemes=["h", "e", "l", "l", "o"],
            score=PhonemeScore(
                overall_accuracy=0.85,
                substitution_count=1,
                deletion_count=1,
                insertion_count=0,
                reference_count=5,
                predicted_count=4,
                avg_confidence=0.9,
                errors=[
                    PhonemeError(
                        error_type=PhonemeErrorType.DELETION,
                        expected="l",
                        predicted=None,
                        position=3,
                        severity="low",
                    ),
                ],
            ),
            backend_used="wav2vec",
            backend_available=True,
        )

    def test_phoneme_data_included_when_available(self):
        """phoneme_data key present in payload when phoneme_result provided."""
        from src.rubrics.toeic import get_question_type

        qt = get_question_type("read_aloud")
        gating = self._build_gating()
        feats = self._build_features()
        transcription = self._build_transcription()
        phoneme = self._build_phoneme_result()

        prompt = _build_user_prompt(
            qt=qt,
            prompt_text="Read this aloud",
            reference_script="Hello world",
            transcription=transcription,
            features=feats,
            gating=gating,
            phoneme_result=phoneme,
        )

        # Parse object JSON đầu tiên; raw_decode bỏ qua language reminder sau JSON.
        payload, _ = json.JSONDecoder().raw_decode(prompt, prompt.index("{"))

        assert "phoneme_data" in payload, "phoneme_data must be in payload"
        assert payload["phoneme_data"]["backend_used"] == "wav2vec"
        assert payload["phoneme_data"]["backend_available"] is True
        assert payload["phoneme_data"]["score"]["overall_accuracy"] == 0.85
        assert payload["phoneme_data"]["score"]["substitution_count"] == 1
        assert payload["phoneme_data"]["score"]["deletion_count"] == 1
        # Bản gọn cho prompt: KHÔNG kèm segments thô / reference_phonemes /
        # audio_path — đây là phần chiếm ~95% kích thước nhưng vô dụng với model
        # text. Chốt lại để không vô tình nhồi segments trở lại vào prompt.
        assert "segments" not in payload["phoneme_data"]
        assert "reference_phonemes" not in payload["phoneme_data"]
        assert "audio_path" not in payload["phoneme_data"]

    def test_prompt_excludes_per_word_breakdown(self):
        """Prompt LLM KHÔNG kèm `words` (per-word ELSA) — đó là khối phình prompt
        (~95% kích thước) mà model không dùng; bằng chứng per-word đã có ở errors[].word.

        Allowlist: chỉ summary + errors lọt vào prompt; words / penalty-metadata bị loại
        kể cả khi score có populate chúng (chống regression khi PhonemeScore thêm field UI).
        """
        from src.phoneme.models import PhonemePoint, WordPronunciation
        from src.rubrics.toeic import get_question_type

        phoneme = self._build_phoneme_result()
        # Populate words + penalty metadata để chắc chắn allowlist loại chúng khỏi prompt.
        object.__setattr__(phoneme.score, "words", [
            WordPronunciation(
                word="hello", ipa="hɛloʊ",
                phonemes=[PhonemePoint(symbol="h", status="ok")],
                accuracy=1.0, start=0.0, end=0.5,
            ),
        ])
        object.__setattr__(phoneme.score, "words_total", 1)
        object.__setattr__(phoneme.score, "raw_penalty", 1.23)

        prompt = _build_user_prompt(
            qt=get_question_type("read_aloud"),
            prompt_text="Read this aloud",
            reference_script="Hello world",
            transcription=self._build_transcription(),
            features=self._build_features(),
            gating=self._build_gating(),
            phoneme_result=phoneme,
        )
        payload, _ = json.JSONDecoder().raw_decode(prompt, prompt.index("{"))
        score_block = payload["phoneme_data"]["score"]
        # Allowlist: summary + errors có; per-word + penalty metadata KHÔNG.
        assert "errors" in score_block
        assert "overall_accuracy" in score_block
        for leaked in ("words", "words_total", "words_truncated", "raw_penalty",
                       "l1_adjustment_ratio", "recognizer_noise_count"):
            assert leaked not in score_block, f"{leaked} không được lọt vào prompt"
        # `word` của từng lỗi VẪN còn (ngữ cảnh per-word mà model thực sự dùng).
        assert all("word" in e for e in score_block["errors"])

    def test_phoneme_data_absent_when_none(self):
        """phoneme_data key NOT in payload when phoneme_result is None."""
        from src.rubrics.toeic import get_question_type

        qt = get_question_type("read_aloud")
        gating = self._build_gating()
        feats = self._build_features()
        transcription = self._build_transcription()

        prompt = _build_user_prompt(
            qt=qt,
            prompt_text="Read this aloud",
            reference_script="Hello world",
            transcription=transcription,
            features=feats,
            gating=gating,
            phoneme_result=None,
        )

        payload, _ = json.JSONDecoder().raw_decode(prompt, prompt.index("{"))

        assert "phoneme_data" not in payload, "phoneme_data must be absent when None"

    def test_system_prompt_includes_phoneme_rules(self):
        """System prompt contains phoneme evidence rules."""
        from src.rubrics.toeic import get_question_type

        qt = get_question_type("read_aloud")
        system = _build_system_prompt(qt, "vi")

        assert "PHONEME METRICS" in system, "System prompt must mention PHONEME METRICS"
        assert "phoneme_data" in system, "System prompt must reference phoneme_data"
        assert "overall_accuracy" in system, "Must mention overall_accuracy"
        assert "substitution" in system.lower(), "Must mention substitution errors"
        assert "severity" in system.lower(), "Must mention severity levels"


class TestScoreFunctionAcceptsPhonemeResult:
    """score() function correctly passes phoneme_result through."""

    def test_score_passes_phoneme_to_prompt(self):
        """phoneme_result is forwarded to _build_user_prompt."""
        config = Config(
            anthropic_api_key="fake-key",
            model="claude-sonnet-4-6",
            whisper_model="base",
            whisper_device="cpu",
            backend="anthropic",
            log_prompts=False,
        )
        from src.asr import Transcription, Word
        from src.rubrics.toeic import get_question_type

        qt = get_question_type("read_aloud")

        words = [Word(text="Hi", start=0.0, end=0.3, probability=0.9)]
        transcription = Transcription(
            text="Hi",
            words=words,
            duration=0.5,
        )
        feats = Features(
            speech_rate_wpm=120.0,
            word_count=1,
            speaking_duration_sec=0.3,
            audio_duration_sec=0.5,
            silence_sec=0.2,
            pause_count=0,
            total_pause_sec=0.0,
            longest_pause_sec=0.0,
            filler_count=0,
            avg_word_probability=0.9,
            min_word_probability=0.9,
        )
        gating = GatingResult(
            is_empty=False,
            task_completion_floor=None,
            reasons=[],
            reference_coverage=None,
            fail_reference_match=False,
        )
        phoneme = PhonemeResult(
            audio_path="test.wav",
            segments=[
                PhonemeSegment(phoneme="h", start=0.0, end=0.1, confidence=0.9, backend="wav2vec"),
            ],
            reference_phonemes=["h", "i"],
            score=PhonemeScore(
                overall_accuracy=0.9,
                substitution_count=0,
                deletion_count=0,
                insertion_count=0,
                reference_count=2,
                predicted_count=1,
                avg_confidence=0.9,
            ),
            backend_used="wav2vec",
            backend_available=True,
        )

        # Patch Anthropic client to avoid actual API call
        mock_response = MagicMock()
        # estimated_toeic_score do model trả KHÔNG còn được dùng — score() tính
        # lại tất định từ điểm tiêu chí. Đặt số khác (30) để chứng minh nó bị ghi
        # đè. read_aloud yêu cầu 2 tiêu chí: pronunciation + intonation_stress.
        mock_response.parsed_output = SpeakingResult(
            estimated_toeic_score=30,
            task_completion="medium",
            content_relevance="medium",
            question_type="read_aloud",
            criteria=[
                CriterionScore(criterion="pronunciation", score=2, justification="ok"),
                CriterionScore(
                    criterion="intonation_stress", score=2, justification="ok"
                ),
            ],
            score_rationale="Test",
            summary_feedback="Test",
        )
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
        mock_response.stop_reason = "stop"

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            mock_client.messages.parse.return_value = mock_response
            MockAnthropic.return_value = mock_client

            result, meta = score(
                config=config,
                qt=qt,
                prompt_text="Read this",
                reference_script="Hi",
                transcription=transcription,
                features=feats,
                gating=gating,
                phoneme_result=phoneme,
            )
            assert meta["backend_used"] == "anthropic"

            # Điểm tổng được TÍNH lại tất định, KHÔNG lấy 999 từ model:
            # base = (110 + 110) / 2 = 110; penalty = min(medium, medium) = 0.85
            # → 93.5 → làm tròn về bội số 10 = 90.
            assert result.estimated_toeic_score == 90
            # Verify _build_user_prompt was called (indirectly via the message content)
            call_args = mock_client.messages.parse.call_args
            user_content = call_args.kwargs["messages"][0]["content"]
            payload, _ = json.JSONDecoder().raw_decode(
                user_content, user_content.index("{")
            )
            assert "phoneme_data" in payload

class TestDropInvalidCorrections:
    """Post-LLM guard: drop corrections whose `said` isn't in the transcript."""

    def _result_with_corrections(self, corrections):
        return SpeakingResult(
            question_type="q",
            task_completion="medium",
            content_relevance="medium",
            criteria=[
                CriterionScore(
                    criterion="vocabulary",
                    score=6.5,
                    justification="x",
                    corrections=corrections,
                )
            ],
            score_rationale="r",
            summary_feedback="s",
        )

    def test_keeps_genuine_substring(self):
        res = self._result_with_corrections([
            LexicalCorrection(said="mountain goat", suggested="Mountain God", example="The Mountain God..."),
        ])
        _drop_invalid_corrections(res, "he said mountain goat to the village")
        assert [c.said for c in res.criteria[0].corrections] == ["mountain goat"]

    def test_drops_hallucinated_phrase(self):
        res = self._result_with_corrections([
            LexicalCorrection(said="create snowstorm", suggested="cause floods", example="Floods..."),
        ])
        _drop_invalid_corrections(res, "he wanted to create blood and chaos")
        assert res.criteria[0].corrections == []

    def test_match_is_case_and_whitespace_lenient(self):
        res = self._result_with_corrections([
            LexicalCorrection(said="Water  Goat", suggested="Water God", example="..."),
        ])
        _drop_invalid_corrections(res, "the water goat fought back")
        assert len(res.criteria[0].corrections) == 1


class TestCompactPhonemeOutput:
    """_compact_phoneme_output surfaces per-word detail for the frontend."""

    def _make_segments(self, phonemes):
        return [
            PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=0.9)
            for i, p in enumerate(phonemes)
        ]

    def test_words_survive_end_to_end(self):
        from src.core import _compact_phoneme_output
        from src.phoneme.ipa import text_to_ipa_sequence_with_spans
        from src.phoneme.scoring import compute_phoneme_score

        ref, spans, _stress, _disp = text_to_ipa_sequence_with_spans("the fox")
        pred = list(ref)
        pred[-1] = "x"  # corrupt the last phoneme in "fox"
        segs = self._make_segments(pred)
        result = PhonemeResult(
            audio_path="t.wav",
            segments=segs,
            reference_phonemes=ref,
            score=compute_phoneme_score(segs, ref, spans),
            backend_used="wav2vec",
            backend_available=True,
        )
        compact = _compact_phoneme_output(result)
        assert compact is not None
        s = compact["score"]
        assert "words" in s and "words_truncated" in s and "words_total" in s
        assert s["words"]
        # Per-word contract the frontend relies on.
        w0 = s["words"][0]
        assert set(w0) == {
            "word", "ipa", "phonemes", "accuracy", "skip_reason", "start", "end"}
        assert "/" not in w0["ipa"]
        bad = [
            p for w in s["words"] for p in w["phonemes"]
            if p["status"] in ("sub", "del")
        ]
        assert bad

    def test_none_when_no_score(self):
        from src.core import _compact_phoneme_output

        result = PhonemeResult(
            audio_path="t.wav",
            segments=[],
            reference_phonemes=[],
            score=None,
            backend_used="none",
            backend_available=False,
        )
        assert _compact_phoneme_output(result) is None

    def test_empty_words_when_no_spans(self):
        # Frontend must tolerate words: [] (documented contract).
        from src.core import _compact_phoneme_output
        from src.phoneme.scoring import compute_phoneme_score

        ref = ["p", "ə", "t"]
        segs = self._make_segments(ref)
        result = PhonemeResult(
            audio_path="t.wav",
            segments=segs,
            reference_phonemes=ref,
            score=compute_phoneme_score(segs, ref),  # no spans → words == []
            backend_used="wav2vec",
            backend_available=True,
        )
        compact = _compact_phoneme_output(result)
        assert compact["score"]["words"] == []
        assert compact["score"]["words_total"] == 0

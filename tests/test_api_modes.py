from __future__ import annotations

import types

import torch

from src import asr
from src import api
from src.config import Config
from src.rubrics.toeic import get_question_type


def _make_config() -> Config:
    return Config(
        anthropic_api_key=None,
        model="claude-sonnet-4-6",
        whisper_model="base",
        whisper_device="cpu",
        backend="anthropic",
        local_base_url="http://localhost:8080/v1",
        local_model="qwen3",
        local_api_key="no-key",
        feedback_lang="vi",
        max_tokens=30000,
        asr_engine_practice="faster_whisper",
        asr_engine_mock_test="whisperx",
        asr_model_practice="large-v3-turbo",
        asr_model_mock_test="large-v3",
        insanely_fast_model_id="openai/whisper-small",
        auto_confidence_threshold=0.75,
        auto_silence_ratio_threshold=0.35,
        auto_coverage_threshold=0.8,
    )


def _output(*, confidence: float, silence_ratio: float, coverage: float, score: int) -> dict:
    return {
        "audio_path": "tmp.wav",
        "question_id": "read_aloud",
        "question_type": "read_aloud",
        "features": {
            "avg_word_probability": confidence,
            "audio_duration_sec": 10.0,
            "silence_sec": 10.0 * silence_ratio,
            "speech_rate_wpm": 120.0,
            "accuracy_metrics": {"coverage": coverage},
        },
        "scores": {"estimated_toeic_score": score},
        "telemetry": {"transcription_time_ms": 111},
    }


def _grade(cfg, qt, mode, *, user_requested_review=False):
    return api._grade_bytes(
        b"audio-bytes",
        ".wav",
        cfg,
        qt,
        reference_script="hello world",
        image_b64=None,
        image_media_type=None,
        expected_duration_sec=12,
        prompt="",
        no_ai=False,
        mode=mode,
        user_requested_review=user_requested_review,
    )


def test_normalize_mode_maps_legacy_and_dirty_input():
    # Legacy aliases (client/bản ghi cũ) map sang mode mới.
    assert api._normalize_mode("auto") == "practice"
    assert api._normalize_mode("default") == "practice"
    assert api._normalize_mode("fast") == "practice"
    assert api._normalize_mode("review") == "mock_test"
    # Chuỗi bẩn (hoa/thường, khoảng trắng) vẫn map đúng.
    assert api._normalize_mode("Fast ") == "practice"
    assert api._normalize_mode("  Review") == "mock_test"
    # Mode mới đi thẳng.
    assert api._normalize_mode("practice") == "practice"
    assert api._normalize_mode("mock_test") == "mock_test"
    # Rỗng/None/lạ → practice (mặc định an toàn, không 400).
    assert api._normalize_mode("") == "practice"
    assert api._normalize_mode(None) == "practice"
    assert api._normalize_mode("nonsense") == "practice"


def test_practice_happy_path_uses_faster_whisper_no_escalation(monkeypatch):
    cfg = _make_config()
    qt = get_question_type("read_aloud")
    calls: list[dict] = []

    def _fake_grade_response(*args, **kwargs):
        calls.append(kwargs)
        # Tín hiệu tốt → không leo thang.
        return _output(confidence=0.9, silence_ratio=0.1, coverage=0.95, score=140)

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    out = _grade(cfg, qt, "practice")

    assert len(calls) == 1
    assert calls[0]["asr_backend"] == "faster_whisper"
    assert calls[0]["asr_model"] == "large-v3-turbo"
    assert out["telemetry"]["modeUsed"] == "practice"
    assert out["telemetry"]["reviewTriggered"] is False
    assert out["telemetry"]["fallbackReason"] is None


def test_mock_test_uses_whisperx_with_phoneme_no_escalation(monkeypatch):
    cfg = _make_config()
    qt = get_question_type("read_aloud")
    calls: list[dict] = []

    def _fake_grade_response(*args, **kwargs):
        calls.append(kwargs)
        return _output(confidence=0.85, silence_ratio=0.2, coverage=0.9, score=150)

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    out = _grade(cfg, qt, "mock_test")

    assert len(calls) == 1
    assert calls[0]["asr_backend"] == "whisperx"
    assert calls[0]["asr_model"] == "large-v3"
    assert calls[0]["phoneme_analysis"] is True
    assert out["telemetry"]["modeUsed"] == "mock_test"
    # Chọn mock_test trực tiếp KHÔNG phải auto-escalation.
    assert out["telemetry"]["reviewTriggered"] is False


def test_practice_escalates_to_mock_test_pipeline(monkeypatch):
    cfg = _make_config()
    qt = get_question_type("read_aloud")
    calls: list[dict] = []

    def _fake_grade_response(*args, **kwargs):
        calls.append(kwargs)
        if kwargs["asr_backend"] == "faster_whisper":
            # Trigger escalation: confidence thấp + silence ratio cao + coverage thấp.
            return _output(confidence=0.60, silence_ratio=0.40, coverage=0.60, score=80)
        if kwargs["asr_backend"] == "whisperx":
            return _output(confidence=0.85, silence_ratio=0.20, coverage=0.90, score=95)
        raise AssertionError(f"Unexpected backend: {kwargs['asr_backend']}")

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    out = _grade(cfg, qt, "practice")

    # Hai lượt gọi theo đúng thứ tự (call_args_list-style): practice → mock_test.
    assert len(calls) == 2
    assert calls[0]["asr_backend"] == "faster_whisper"
    assert calls[0]["asr_model"] == "large-v3-turbo"
    # Lượt escalation phải đổi CẢ engine LẪN model + bật phoneme.
    assert calls[1]["asr_backend"] == "whisperx"
    assert calls[1]["asr_model"] == "large-v3"
    assert calls[1]["phoneme_analysis"] is True

    assert out["telemetry"]["modeUsed"] == "mock_test"
    assert out["telemetry"]["reviewTriggered"] is True
    assert out["telemetry"]["scoreBeforeReview"] == 80
    assert out["telemetry"]["scoreAfterReview"] == 95
    assert out["telemetry"]["fallbackReason"] is None


def test_legacy_review_mode_routes_to_mock_test(monkeypatch):
    cfg = _make_config()
    qt = get_question_type("read_aloud")
    calls: list[dict] = []

    def _fake_grade_response(*args, **kwargs):
        calls.append(kwargs)
        return _output(confidence=0.85, silence_ratio=0.2, coverage=0.9, score=150)

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    out = _grade(cfg, qt, "review")  # legacy alias

    assert calls[0]["asr_backend"] == "whisperx"
    assert out["telemetry"]["modeRequested"] == "mock_test"
    assert out["telemetry"]["modeUsed"] == "mock_test"


def test_whisperx_audio_fallback_without_ffmpeg(monkeypatch):
    fake_waveform = torch.tensor([[0.0, 0.5, -0.5, 0.25]], dtype=torch.float32)

    import torchaudio

    monkeypatch.setattr(asr.shutil, "which", lambda _name, path=None: None)
    monkeypatch.setattr(torchaudio, "load", lambda _: (fake_waveform, 22050))
    monkeypatch.setattr(
        torchaudio.functional,
        "resample",
        lambda waveform, src_sr, dst_sr: waveform,
    )

    whisperx_module = types.SimpleNamespace(
        load_audio=lambda _: (_ for _ in ()).throw(AssertionError("ffmpeg path should not be used")),
    )

    audio = asr._load_audio_for_whisperx("dummy.wav", whisperx_module)

    assert audio.dtype.name == "float32"
    assert audio.shape == (4,)


def test_whisperx_audio_fallback_to_pyav_when_torchaudio_fails(monkeypatch):
    import torchaudio
    import numpy as np

    monkeypatch.setattr(asr.shutil, "which", lambda _name, path=None: None)
    monkeypatch.setattr(
        torchaudio,
        "load",
        lambda _: (_ for _ in ()).throw(RuntimeError("Format not recognised")),
    )
    monkeypatch.setattr(
        asr,
        "_load_audio_with_pyav",
        lambda _: np.array([0.0, 0.25, -0.25], dtype=np.float32),
    )

    whisperx_module = types.SimpleNamespace(
        load_audio=lambda _: (_ for _ in ()).throw(AssertionError("ffmpeg path should not be used")),
    )

    audio = asr._load_audio_for_whisperx("dummy.m4a", whisperx_module)

    assert audio.dtype.name == "float32"
    assert audio.tolist() == [0.0, 0.25, -0.25]


def test_resolve_torch_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = types.SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert asr._resolve_torch_device("cuda", "whisperx") == "cpu"
    assert asr._resolve_torch_device("auto", "whisperx") == "cpu"
    assert asr._resolve_torch_device("cpu", "whisperx") == "cpu"


def test_resolve_torch_device_auto_prefers_cuda(monkeypatch):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

    fake_torch = types.SimpleNamespace(
        __version__="2.8.0+cu128",
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=_FakeCuda(),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert asr._resolve_torch_device("auto", "whisperx") == "cuda"
    assert asr._resolve_torch_device("cuda", "whisperx") == "cuda"


def test_grade_bytes_passes_phoneme_strict(monkeypatch):
    """strict=true từ /grade (popup luyện 1 từ) phải xuống tới grade_response."""
    cfg = _make_config()
    qt = get_question_type("read_aloud")
    calls: list[dict] = []

    def _fake_grade_response(*args, **kwargs):
        calls.append(kwargs)
        return _output(confidence=0.9, silence_ratio=0.1, coverage=0.95, score=140)

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    api._grade_bytes(
        b"audio-bytes",
        ".wav",
        cfg,
        qt,
        reference_script="information",
        image_b64=None,
        image_media_type=None,
        expected_duration_sec=None,
        prompt="",
        no_ai=True,
        mode="mock_test",
        user_requested_review=False,
        phoneme_strict=True,
    )
    assert calls[0]["phoneme_strict"] is True

    # Không truyền cờ (mọi client hiện hành trừ popup) = False — không đổi đường chấm cũ.
    calls.clear()
    _grade(cfg, qt, "mock_test")
    assert calls[0]["phoneme_strict"] is False


def test_grade_response_strict_disables_sentence_leniency(monkeypatch):
    """phoneme_strict CHỈ tắt leniency câu dài (L1 + coverage/drift/collapse gate);
    guard nhiễu recognizer (noise gate, knee) giữ nguyên."""
    import dataclasses

    from src import core
    from src.asr import ASRRun, Transcription

    captured: list[dict] = []

    class _FakeAnalyzer:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def analyze(self, *args, **kwargs):
            return None  # phoneme là phụ trợ — None được pipeline chấp nhận

    monkeypatch.setattr(core, "HybridPhonemeAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(
        core.asr,
        "transcribe_with_backend",
        lambda *a, **k: ASRRun(
            transcription=Transcription(
                text="information", words=[], language="en", duration=1.0
            ),
            backend_used="whisperx",
            elapsed_ms=1,
        ),
    )

    cfg = dataclasses.replace(
        _make_config(),
        phoneme_analysis_enabled=True,
        phoneme_l1_enabled=True,
        phoneme_coverage_gate_enabled=True,
        phoneme_drift_cap_enabled=True,
        phoneme_collapse_gate_enabled=True,
    )
    qt = get_question_type("read_aloud")

    for strict in (False, True):
        core.grade_response(
            "audio.wav",
            cfg,
            qt,
            reference_script="information",
            no_ai=True,
            phoneme_analysis=True,
            save=False,
            phoneme_strict=strict,
        )

    assert len(captured) == 2
    default_kwargs, strict_kwargs = captured
    assert default_kwargs["l1_enabled"] is True
    assert default_kwargs["coverage_gate_enabled"] is True
    assert default_kwargs["drift_cap_enabled"] is True
    assert default_kwargs["collapse_gate_enabled"] is True
    assert strict_kwargs["l1_enabled"] is False
    assert strict_kwargs["coverage_gate_enabled"] is False
    assert strict_kwargs["drift_cap_enabled"] is False
    assert strict_kwargs["collapse_gate_enabled"] is False
    # Guard nhiễu recognizer không đổi trong strict.
    assert (
        strict_kwargs["recognizer_noise_conf"]
        == default_kwargs["recognizer_noise_conf"]
    )


def test_whisperx_no_speech_returns_empty_transcription(monkeypatch):
    """Silero VAD không thấy tiếng nói → whisperx nổ IndexError trên list segment
    rỗng (clip 1 từ quá ngắn/nhỏ tiếng). Phải trả transcript RỖNG thay vì 500."""
    import numpy as np

    fake_whisperx = types.SimpleNamespace()
    monkeypatch.setitem(__import__("sys").modules, "whisperx", fake_whisperx)
    monkeypatch.setattr(asr, "_resolve_torch_device", lambda d, b: "cpu")
    monkeypatch.setattr(
        asr,
        "_load_audio_for_whisperx",
        lambda path, mod: np.zeros(16000, dtype=np.float32),
    )

    class _FakeModel:
        def transcribe(self, audio, batch_size=16, language="en"):
            raise IndexError("list index out of range")

    monkeypatch.setitem(
        asr._whisperx_model_cache, ("base", "cpu", "int8"), _FakeModel()
    )

    out = asr._transcribe_whisperx("silence.webm", model_size="base", device="cpu")

    assert out.text == ""
    assert out.words == []
    assert out.duration == 1.0  # 16000 mẫu / 16kHz

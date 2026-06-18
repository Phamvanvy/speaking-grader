from __future__ import annotations

import types

import torch

from src import asr
from src import api
from src.config import Config
from src.rubrics.toeic import get_question_type


def _make_config(*, fast_enabled: bool = True) -> Config:
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
        asr_backend_default="faster_whisper",
        asr_backend_fast="insanely_fast_whisper",
        asr_backend_review="whisperx",
        fast_backend_enabled=fast_enabled,
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


def test_fast_available_uses_fast_backend(monkeypatch):
    cfg = _make_config(fast_enabled=True)
    qt = get_question_type("read_aloud")
    calls: list[str] = []

    def _fake_grade_response(*args, **kwargs):
        calls.append(kwargs["asr_backend"])
        return _output(confidence=0.9, silence_ratio=0.1, coverage=0.95, score=140)

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    out = api._grade_bytes(
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
        mode="fast",
        user_requested_review=False,
    )

    assert calls == ["insanely_fast_whisper"]
    assert out["telemetry"]["modeUsed"] == "fast"
    assert out["telemetry"]["fallbackReason"] is None


def test_fast_failure_fallbacks_to_default(monkeypatch):
    cfg = _make_config(fast_enabled=True)
    qt = get_question_type("read_aloud")
    calls: list[str] = []

    def _fake_grade_response(*args, **kwargs):
        backend = kwargs["asr_backend"]
        calls.append(backend)
        if backend == "insanely_fast_whisper":
            raise RuntimeError("fast backend crashed")
        return _output(confidence=0.88, silence_ratio=0.1, coverage=0.93, score=135)

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    out = api._grade_bytes(
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
        mode="fast",
        user_requested_review=False,
    )

    assert calls == ["insanely_fast_whisper", "faster_whisper"]
    assert out["telemetry"]["modeUsed"] == "default"
    assert out["telemetry"]["fallbackReason"] == "fast_backend_failed"


def test_auto_trigger_review_uses_whisperx(monkeypatch):
    cfg = _make_config(fast_enabled=True)
    qt = get_question_type("read_aloud")
    calls: list[str] = []

    def _fake_grade_response(*args, **kwargs):
        backend = kwargs["asr_backend"]
        calls.append(backend)
        if backend == "faster_whisper":
            # Trigger auto-review: confidence thấp + silence ratio cao + coverage thấp.
            return _output(confidence=0.60, silence_ratio=0.40, coverage=0.60, score=80)
        if backend == "whisperx":
            return _output(confidence=0.85, silence_ratio=0.20, coverage=0.90, score=95)
        raise AssertionError(f"Unexpected backend: {backend}")

    monkeypatch.setattr(api, "grade_response", _fake_grade_response)

    out = api._grade_bytes(
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
        mode="auto",
        user_requested_review=False,
    )

    assert calls == ["faster_whisper", "whisperx"]
    assert out["telemetry"]["modeUsed"] == "review"
    assert out["telemetry"]["reviewTriggered"] is True
    assert out["telemetry"]["scoreBeforeReview"] == 80
    assert out["telemetry"]["scoreAfterReview"] == 95
    assert out["telemetry"]["fallbackReason"] is None


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

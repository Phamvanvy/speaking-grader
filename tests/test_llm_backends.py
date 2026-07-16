"""Test tầng backend LLM (src/scoring/backends.py) sau refactor OpenAI-compat.

Mock client OpenAI để kiểm:
- payload local GIỮ NGUYÊN hành vi cũ (chat_template_kwargs + cache_prompt);
- payload openrouter KHÔNG lọt field llama.cpp, có provider.require_parameters;
- OpenRouter lỗi → fallback local (backend_used="local_fallback");
- fallback tắt → raise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.config import Config
from src.scoring import backends


class _Out(BaseModel):
    x: int


def _fake_response(content: str = '{"x": 1}'):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=content)
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class _FakeClient:
    """Ghi lại kwargs của chat.completions.create; fail=True → raise."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("boom")
        return _fake_response()


def _patch_clients(monkeypatch, clients: dict[str, _FakeClient]) -> None:
    """Map base_url → fake client (bỏ qua timeout/retries khi tra cứu)."""

    def _fake_get(base_url, api_key, timeout_sec=600.0, max_retries=0):
        return clients[base_url]

    monkeypatch.setattr(backends, "_get_openai_client", _fake_get)


_LOCAL_URL = "http://localhost:8080/v1"
_OR_URL = "https://openrouter.ai/api/v1"


def _local_config(**overrides) -> Config:
    base = dict(
        anthropic_api_key=None,
        model="claude-sonnet-4-6",
        whisper_model="base",
        whisper_device="cpu",
        backend="local",
        local_base_url=_LOCAL_URL,
        local_model="qwen3",
        local_api_key="no-key",
        local_enable_thinking=False,
        local_prefix_cache=True,
        max_tokens=180000,
        log_prompts=False,
    )
    base.update(overrides)
    return Config(**base)


def _openrouter_config(**overrides) -> Config:
    base = dict(
        anthropic_api_key=None,
        model="claude-sonnet-4-6",
        whisper_model="base",
        whisper_device="cpu",
        backend="openrouter",
        openrouter_api_key="sk-or-test",
        openrouter_model="anthropic/claude-haiku-4.5",
        openrouter_max_tokens=8000,
        local_base_url=_LOCAL_URL,
        local_model="qwen3",
        local_api_key="no-key",
        local_enable_thinking=False,
        local_prefix_cache=True,
        max_tokens=180000,
        log_prompts=False,
    )
    base.update(overrides)
    return Config(**base)


def test_local_payload_unchanged(monkeypatch):
    """Đích local gửi đúng extra_body cũ (bit-for-bit) + model/max_tokens local."""
    local = _FakeClient()
    _patch_clients(monkeypatch, {_LOCAL_URL: local})
    cfg = _local_config()

    result = backends._generate_local(
        cfg, "SYS", "USER", _Out, _Out.model_json_schema(), "Out"
    )
    assert isinstance(result, _Out) and result.x == 1

    kwargs = local.calls[0]
    assert kwargs["model"] == "qwen3"
    assert kwargs["max_tokens"] == 180000
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": True,
    }
    assert kwargs["response_format"]["json_schema"]["name"] == "Out"
    assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}


def test_openrouter_payload_no_llamacpp_fields(monkeypatch):
    """Đích openrouter: có provider.require_parameters, KHÔNG có field llama.cpp."""
    orc = _FakeClient()
    _patch_clients(monkeypatch, {_OR_URL: orc, _LOCAL_URL: _FakeClient()})
    cfg = _openrouter_config()

    result, meta = backends.generate(cfg, "SYS", "USER", _Out)
    assert isinstance(result, _Out)
    assert meta["backend_used"] == "openrouter"
    assert meta["model"] == "anthropic/claude-haiku-4.5"
    assert meta["fallback_reason"] is None

    kwargs = orc.calls[0]
    assert kwargs["model"] == "anthropic/claude-haiku-4.5"
    assert kwargs["max_tokens"] == 8000  # trần riêng, KHÔNG phải TOEIC_MAX_TOKENS
    extra = kwargs["extra_body"]
    assert extra["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
    }
    assert extra["reasoning"] == {"enabled": False}  # default "none"
    assert "cache_prompt" not in extra
    assert "chat_template_kwargs" not in extra


def test_openrouter_relaxed_provider_for_free_models(monkeypatch):
    """Knob test model free: require_parameters off + data_collection allow."""
    orc = _FakeClient()
    _patch_clients(monkeypatch, {_OR_URL: orc})
    cfg = _openrouter_config(
        openrouter_require_parameters=False,
        openrouter_data_collection="allow",
    )
    backends.generate(cfg, "SYS", "USER", _Out)
    assert orc.calls[0]["extra_body"]["provider"] == {
        "require_parameters": False,
        "data_collection": "allow",
    }


def test_openrouter_reasoning_effort(monkeypatch):
    orc = _FakeClient()
    _patch_clients(monkeypatch, {_OR_URL: orc})
    cfg = _openrouter_config(openrouter_reasoning="low")
    backends.generate(cfg, "SYS", "USER", _Out)
    assert orc.calls[0]["extra_body"]["reasoning"] == {"effort": "low"}


def test_openrouter_falls_back_to_local(monkeypatch):
    """OpenRouter lỗi → chấm lại bằng local, meta ghi rõ local_fallback + lý do."""
    orc = _FakeClient(fail=True)
    local = _FakeClient()
    _patch_clients(monkeypatch, {_OR_URL: orc, _LOCAL_URL: local})
    cfg = _openrouter_config()

    result, meta = backends.generate(cfg, "SYS", "USER", _Out)
    assert isinstance(result, _Out) and result.x == 1
    assert meta["backend_used"] == "local_fallback"
    assert meta["model"] == "qwen3"
    assert "boom" in meta["fallback_reason"]
    assert len(orc.calls) == 1 and len(local.calls) == 1
    # Lần chạy local dùng đúng policy local (extra_body llama.cpp).
    assert local.calls[0]["extra_body"]["cache_prompt"] is True


def test_openrouter_fallback_disabled_raises(monkeypatch):
    orc = _FakeClient(fail=True)
    local = _FakeClient()
    _patch_clients(monkeypatch, {_OR_URL: orc, _LOCAL_URL: local})
    cfg = _openrouter_config(openrouter_fallback_local=False)

    with pytest.raises(RuntimeError, match="boom"):
        backends.generate(cfg, "SYS", "USER", _Out)
    assert len(local.calls) == 0  # không được lén gọi local


def test_generate_local_backend_meta(monkeypatch):
    local = _FakeClient()
    _patch_clients(monkeypatch, {_LOCAL_URL: local})
    cfg = _local_config()
    result, meta = backends.generate(cfg, "SYS", "USER", _Out)
    assert meta["backend_used"] == "local"
    assert meta["model"] == "qwen3"

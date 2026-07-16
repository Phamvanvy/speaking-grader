"""Gọi LLM backend: Anthropic (Claude), OpenRouter, hoặc model local.

- anthropic : messages.parse + adaptive thinking (structured output Pydantic).
- local     : OpenAI-compatible (llama.cpp), ép schema bằng response_format
              json_schema (→ GBNF grammar).
- openrouter: cùng đường OpenAI-compatible như local nhưng trỏ OpenRouter
              (model trả phí, require_parameters để chỉ route provider hỗ trợ
              json_schema); lỗi/timeout → tự FALLBACK về local (xem generate()).

Điểm vào khuyến nghị là `generate()` — dispatcher 3 backend trả (result, meta)
với meta ghi backend nào THẬT SỰ chấm (telemetry). Các wrapper cũ
(_generate_local/_score_local/_score_anthropic) giữ nguyên signature.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pydantic import BaseModel

from ..config import Config
from ..rubrics.base import QuestionType
from ..schema import SpeakingResult
from .api_logging import _log_api_request, _log_messages, _log_response
from .prompts import _local_response_schema

logger = logging.getLogger("toeic.scoring")


@dataclass(frozen=True)
class OpenAICompatTarget:
    """Một đích OpenAI-compatible (local llama.cpp / OpenRouter) đủ thông tin gọi.

    Tách target khỏi Config để _generate_openai_compat dùng chung cho mọi đích —
    khác nhau chỉ ở base_url/model/trần token/timeout và extra_body (llama.cpp
    nhận chat_template_kwargs+cache_prompt; OpenRouter nhận provider+reasoning).
    """

    name: str  # "local" | "openrouter" — chọn nhánh extra_body + tag log
    base_url: str
    api_key: str
    model: str
    max_tokens: int
    timeout_sec: float
    max_retries: int


def _local_target(config: Config) -> OpenAICompatTarget:
    # timeout 600s = default cũ của SDK (local gen chậm là bình thường);
    # max_retries=0: retry 1 call local 10 phút không cứu được gì.
    return OpenAICompatTarget(
        name="local",
        base_url=config.local_base_url,
        api_key=config.local_api_key,
        model=config.local_model,
        max_tokens=config.max_tokens,
        timeout_sec=600.0,
        max_retries=0,
    )


def _openrouter_target(config: Config) -> OpenAICompatTarget:
    # max_retries=1: SDK tự retry 1 lần lỗi mạng/429/5xx — tối đa 1 lần tính
    # tiền trùng (model flash rẻ); hỏng tiếp thì generate() fallback local (free).
    return OpenAICompatTarget(
        name="openrouter",
        base_url=config.openrouter_base_url,
        api_key=config.openrouter_api_key or "",
        model=config.openrouter_model,
        max_tokens=config.openrouter_max_tokens,
        timeout_sec=config.openrouter_timeout_sec,
        max_retries=1,
    )


# Tái dùng client OpenAI-compatible (local + OpenRouter chung cache): mỗi bài
# tạo client mới = connection pool mới mỗi lần. Key gồm cả timeout/max_retries
# vì 2 target khác policy. Client openai-python thread-safe → dùng chung luồng.
_local_client_cache: dict[tuple, object] = {}


def _get_openai_client(
    base_url: str, api_key: str, timeout_sec: float = 600.0, max_retries: int = 0
):
    key = (base_url, api_key, timeout_sec, max_retries)
    client = _local_client_cache.get(key)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_sec,
            max_retries=max_retries,
        )
        _local_client_cache[key] = client
    return client


def _get_local_client(base_url: str, api_key: str):
    """Alias giữ signature cũ (exam_import import trực tiếp) — policy local."""
    return _get_openai_client(base_url, api_key)


# Client Anthropic cũng cache (trước đây tạo mới mỗi call → pool mới mỗi lần).
# timeout 300s: chấm + adaptive thinking thường <60s, 300s chặn socket treo vô
# hạn mà vẫn dư đuôi chậm; max_retries=2 = default SDK cũ.
_anthropic_client_cache: dict[str, object] = {}


def _get_anthropic_client(api_key: str):
    client = _anthropic_client_cache.get(api_key)
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=300.0, max_retries=2)
        _anthropic_client_cache[api_key] = client
    return client


def _openai_extra_body(config: Config, target: OpenAICompatTarget) -> dict:
    """extra_body theo đích — field llama.cpp KHÔNG được lọt sang OpenRouter."""
    if target.name == "openrouter":
        extra: dict = {
            # require_parameters: chỉ route tới provider tôn trọng response_format
            # json_schema (không thì JSON có thể sai schema âm thầm).
            # data_collection deny: transcript học viên không cho provider giữ lại.
            "provider": {"require_parameters": True, "data_collection": "deny"},
        }
        reasoning = (config.openrouter_reasoning or "").strip().lower()
        if reasoning == "none":
            extra["reasoning"] = {"enabled": False}
        elif reasoning in {"low", "medium", "high"}:
            extra["reasoning"] = {"effort": reasoning}
        # "" → không gửi field reasoning, model tự quyết.
        return extra
    # local (llama.cpp): giữ NGUYÊN hành vi cũ — bit-for-bit payload.
    extra = {
        "chat_template_kwargs": {"enable_thinking": config.local_enable_thinking}
    }
    if config.local_prefix_cache:
        extra["cache_prompt"] = True
    return extra


def _generate_anthropic(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    output_model: type[BaseModel],
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> BaseModel:
    """Gọi Claude với structured output GENERIC theo `output_model` (Pydantic).

    Dùng chung cho mọi tác vụ cần JSON đúng schema (chấm điểm: SpeakingResult;
    sinh bài mẫu: SampleAnswer...). Mọi logic invoke (content text/vision, log,
    messages.parse, xử lý max_tokens) tập trung ở đây — đổi backend chỉ sửa 1 chỗ.
    """
    if not config.has_api_key:
        raise RuntimeError(
            "Thiếu ANTHROPIC_API_KEY. Đặt trong .env, dùng TOEIC_BACKEND=local "
            "để chấm bằng model local, hoặc chạy với --no-ai để chỉ lấy "
            "transcript + features."
        )

    client = _get_anthropic_client(config.anthropic_api_key)

    # Không có ảnh → giữ nguyên content dạng chuỗi (hành vi cũ). Có ảnh → khối
    # image (base64) đứng trước, rồi khối text để Claude nhìn tranh trước khi đọc.
    if image_b64:
        content: object = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type or "image/jpeg",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": user_prompt},
        ]
    else:
        content = user_prompt

    messages = [{"role": "user", "content": content}]

    # Always log the messages being sent (sanitized). Anthropic giữ system prompt
    # tách riêng nên log kèm để thấy đủ system + user.
    _log_messages(
        logger, "anthropic", config.model, messages, system_prompt=system_prompt
    )

    # Log the full request payload (includes system prompt for Anthropic)
    if config.log_prompts:
        _log_api_request(
            config, "anthropic",
            model=config.model,
            base_url=None,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=0,
            extra_body=None,
            system_prompt=system_prompt,
        )

    t0 = time.monotonic()
    response = client.messages.parse(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=messages,
        output_format=output_model,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Claude trả structured output | model=%s | latency=%.2fs | "
        "input_tokens=%s output_tokens=%s",
        config.model,
        latency,
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )

    result = response.parsed_output

    # Log AI response
    if config.log_prompts:
        response_json = result.model_dump_json() if result else "null"
        _log_response(config, "anthropic", response_json)
    if result is None:
        # stop_reason refusal / max_tokens → parsed_output có thể None
        hint = (
            f" JSON bị cắt vì chạm trần max_tokens={config.max_tokens} — "
            f"tăng TOEIC_MAX_TOKENS."
            if response.stop_reason == "max_tokens"
            else ""
        )
        raise RuntimeError(
            f"Claude không trả về kết quả đúng schema "
            f"(stop_reason={response.stop_reason}).{hint}"
        )
    return result


def _score_anthropic(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    """Chấm điểm qua Claude (wrapper mỏng quanh _generate_anthropic)."""
    result = _generate_anthropic(
        config, system_prompt, user_prompt, SpeakingResult, image_b64, image_media_type
    )
    assert isinstance(result, SpeakingResult)
    return result


def _generate_openai_compat(
    config: Config,
    target: OpenAICompatTarget,
    system_prompt: str,
    user_prompt: str,
    output_model: type[BaseModel],
    json_schema: dict,
    schema_name: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> BaseModel:
    """Gọi một đích OpenAI-compatible (local / OpenRouter) với structured output.

    Ép đúng schema bằng response_format json_schema — llama.cpp chuyển schema
    thành GBNF grammar; OpenRouter route tới provider hỗ trợ (require_parameters
    trong extra_body). `json_schema`/`schema_name`/`output_model` đến từ caller
    nên hàm này dùng được cho cả chấm điểm (SpeakingResult, schema siết theo qt)
    lẫn sinh bài mẫu (SampleAnswer) — logic invoke tập trung một chỗ.
    """
    try:
        client = _get_openai_client(
            target.base_url, target.api_key, target.timeout_sec, target.max_retries
        )
    except ImportError as e:  # pragma: no cover - phụ thuộc tuỳ chọn
        raise RuntimeError(
            f"Backend {target.name} cần gói 'openai'. Cài: pip install openai"
        ) from e

    # Định dạng vision OpenAI-compatible: data URI base64. Cần model có thị
    # giác (vd Qwen2.5-VL); model thuần text sẽ bỏ qua/lỗi khối ảnh.
    if image_b64:
        data_uri = f"data:{image_media_type or 'image/jpeg'};base64,{image_b64}"
        user_content: object = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt

    # extra_body THEO ĐÍCH: local nhận chat_template_kwargs (tắt thinking Qwen3,
    # nhanh ~6.7×) + cache_prompt (KV-cache system prompt giữa các bài cùng đề);
    # OpenRouter nhận provider/reasoning — field llama.cpp không được gửi nhầm.
    extra_body = _openai_extra_body(config, target)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Quan sát được prefix caching (chỉ local): log tỉ trọng phần system prompt
    # ổn định so với tổng prompt — càng cao càng tiết kiệm prefill.
    if target.name == "local" and config.local_prefix_cache:
        user_chars = (
            len(user_content)
            if isinstance(user_content, str)
            else sum(len(p.get("text", "")) for p in user_content if isinstance(p, dict))
        )
        logger.info(
            "Prefix cache ON (cache_prompt=true) | system_chars=%d | user_chars=%d",
            len(system_prompt),
            user_chars,
        )

    # Always log the messages being sent (sanitized: image base64 stripped,
    # long text truncated) so ta thấy đúng prompt model nhận được.
    _log_messages(logger, target.name, target.model, messages)

    # Log the full request payload being sent to the API
    # (system prompt is already embedded in messages[0] for OpenAI-compatible)
    if config.log_prompts:
        _log_api_request(
            config, target.name,
            model=target.model,
            base_url=target.base_url,
            messages=messages,
            max_tokens=target.max_tokens,
            temperature=0,
            extra_body=extra_body,
        )

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=target.model,
        max_tokens=target.max_tokens,
        temperature=0,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                # Schema do caller cung cấp (vd SpeakingResult siết theo qt — ép
                # đúng N tiêu chí + enum key; hoặc SampleAnswer cho bài mẫu).
                "schema": json_schema,
                "strict": True,
            },
        },
        extra_body=extra_body,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Model %s trả structured output | model=%s | base_url=%s | latency=%.2fs | "
        "prompt_tokens=%s completion_tokens=%s",
        target.name,
        target.model,
        target.base_url,
        latency,
        getattr(usage, "prompt_tokens", "?"),
        getattr(usage, "completion_tokens", "?"),
    )

    finish = response.choices[0].finish_reason
    content = response.choices[0].message.content
    if finish == "length":
        hint = (
            "Tăng TOEIC_MAX_TOKENS, hoặc giảm độ dài nhận xét."
            if target.name == "local"
            else "Tăng TOEIC_OPENROUTER_MAX_TOKENS."
        )
        raise RuntimeError(
            f"Model {target.name} bị cắt vì chạm trần max_tokens="
            f"{target.max_tokens} (finish_reason=length) → JSON dở dang. {hint}"
        )
    if not content:
        raise RuntimeError(
            f"Model {target.name} không trả về nội dung (finish_reason={finish})."
        )

    # Log AI response
    if config.log_prompts:
        _log_response(config, target.name, content)

    try:
        return output_model.model_validate_json(content)
    except Exception as e:  # noqa: BLE001 - bọc lỗi parse cho rõ
        raise RuntimeError(
            f"Model {target.name} trả JSON không đúng schema {schema_name}: {e}\n"
            f"Nội dung: {content[:500]}"
        ) from e


def _generate_local(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    output_model: type[BaseModel],
    json_schema: dict,
    schema_name: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> BaseModel:
    """Wrapper mỏng giữ signature cũ: gọi đích local qua _generate_openai_compat."""
    return _generate_openai_compat(
        config,
        _local_target(config),
        system_prompt,
        user_prompt,
        output_model,
        json_schema,
        schema_name,
        image_b64,
        image_media_type,
    )


def _score_local(
    config: Config,
    qt: QuestionType,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    """Chấm điểm qua model local (wrapper mỏng quanh _generate_local).

    Truyền schema siết theo qt (_local_response_schema) — chống model nhỏ bỏ sót
    tiêu chí — và validate về SpeakingResult.
    """
    result = _generate_local(
        config,
        system_prompt,
        user_prompt,
        SpeakingResult,
        _local_response_schema(qt),
        "SpeakingResult",
        image_b64,
        image_media_type,
    )
    assert isinstance(result, SpeakingResult)
    return result


def generate(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    output_model: type[BaseModel],
    *,
    json_schema: dict | None = None,
    schema_name: str | None = None,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> tuple[BaseModel, dict]:
    """Dispatcher 3 backend (anthropic | local | openrouter) → (result, meta).

    meta = {backend_used, model, latency_ms, fallback_reason} — backend_used là
    backend THẬT SỰ tạo ra kết quả ("local_fallback" khi OpenRouter hỏng và
    local cứu), đưa vào telemetry để theo dõi fallback rate sau khi ship.

    openrouter: MỌI exception (APIError/timeout/JSON sai schema) → thử lại bằng
    local nếu openrouter_fallback_local bật — không viết retry loop riêng chống
    OpenRouter (SDK đã retry 1 lần); local miễn phí nên fail-over thẳng. Content
    -validation retry của score() nằm TRÊN hàm này, giữ nguyên như cũ.

    json_schema/schema_name mặc định suy từ output_model (đúng bằng giá trị các
    caller cũ tự truyền); score() truyền schema siết theo qt.
    """
    schema = json_schema if json_schema is not None else output_model.model_json_schema()
    name = schema_name or output_model.__name__

    def _meta(backend_used: str, model: str, started: float) -> dict:
        return {
            "backend_used": backend_used,
            "model": model,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "fallback_reason": None,
        }

    t0 = time.monotonic()
    if config.is_openrouter:
        try:
            result = _generate_openai_compat(
                config, _openrouter_target(config), system_prompt, user_prompt,
                output_model, schema, name, image_b64, image_media_type,
            )
            return result, _meta("openrouter", config.openrouter_model, t0)
        except Exception as e:  # noqa: BLE001 - mọi lỗi đều đáng fallback
            if not (config.openrouter_fallback_local and config.local_base_url):
                raise
            logger.error(
                "OpenRouter thất bại (%s: %s) → fallback model local %s",
                type(e).__name__, e, config.local_model,
            )
            t1 = time.monotonic()
            result = _generate_openai_compat(
                config, _local_target(config), system_prompt, user_prompt,
                output_model, schema, name, image_b64, image_media_type,
            )
            meta = _meta("local_fallback", config.local_model, t1)
            meta["fallback_reason"] = f"{type(e).__name__}: {e}"[:300]
            return result, meta
    if config.is_local:
        result = _generate_openai_compat(
            config, _local_target(config), system_prompt, user_prompt,
            output_model, schema, name, image_b64, image_media_type,
        )
        return result, _meta("local", config.local_model, t0)
    result = _generate_anthropic(
        config, system_prompt, user_prompt, output_model, image_b64, image_media_type
    )
    return result, _meta("anthropic", config.model, t0)

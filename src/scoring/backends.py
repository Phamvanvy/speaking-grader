"""Gọi LLM backend: Anthropic (Claude) hoặc model local (OpenAI-compatible).

_score_anthropic dùng messages.parse + adaptive thinking; _score_local ép schema bằng
response_format json_schema (llama.cpp → GBNF). Cả hai trả SpeakingResult đã parse.
"""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel

from ..config import Config
from ..rubrics.base import QuestionType
from ..schema import SpeakingResult
from .api_logging import _log_api_request, _log_messages, _log_response
from .prompts import _local_response_schema

logger = logging.getLogger("toeic.scoring")

# Tái dùng OpenAI client cho backend local: trước đây mỗi bài tạo client mới →
# connection pool mới mỗi lần. Cache theo (base_url, api_key) để batch dùng lại
# một pool. Client của openai-python thread-safe nên dùng chung giữa các luồng.
_local_client_cache: dict[tuple[str, str], object] = {}


def _get_local_client(base_url: str, api_key: str):
    key = (base_url, api_key)
    client = _local_client_cache.get(key)
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        _local_client_cache[key] = client
    return client


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

    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

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
    """Gọi model local (OpenAI-compatible) với structured output GENERIC.

    Ép đúng schema bằng response_format json_schema — llama.cpp chuyển schema
    thành GBNF grammar nên JSON trả về luôn hợp lệ. Không có 'thinking' của
    Claude; nếu model hỗ trợ reasoning (Qwen3) có thể bật qua chat template.

    `json_schema`/`schema_name`/`output_model` đến từ caller nên hàm này dùng được
    cho cả chấm điểm (SpeakingResult, schema siết theo qt) lẫn sinh bài mẫu
    (SampleAnswer) — logic invoke local tập trung một chỗ.
    """
    try:
        client = _get_local_client(config.local_base_url, config.local_api_key)
    except ImportError as e:  # pragma: no cover - phụ thuộc tuỳ chọn
        raise RuntimeError(
            "Backend local cần gói 'openai'. Cài: pip install openai"
        ) from e

    # Định dạng vision OpenAI-compatible: data URI base64. Cần model local có
    # thị giác (vd Qwen2.5-VL); model thuần text sẽ bỏ qua/lỗi khối ảnh.
    if image_b64:
        data_uri = f"data:{image_media_type or 'image/jpeg'};base64,{image_b64}"
        user_content: object = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt

    # Tắt reasoning cho model kiểu Qwen3 trừ khi bật rõ ràng. Truyền qua
    # chat_template_kwargs (llama.cpp với --jinja sẽ áp dụng vào chat template;
    # các server khác bỏ qua key lạ). Tắt thinking nhanh ~6.7× (xem Config).
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": config.local_enable_thinking}
    }
    # Prefix caching phía server (llama.cpp): tái dùng KV-cache của system prompt
    # (rubric) — giống nhau giữa mọi bài cùng đề trong batch nên prefill chỉ tính
    # 1 lần. Server không hỗ trợ key này sẽ bỏ qua. (vLLM bật bằng cờ server
    # --enable-prefix-caching, không qua field này.)
    if config.local_prefix_cache:
        extra_body["cache_prompt"] = True

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Quan sát được prefix caching: system prompt (rubric) là phần ổn định, đứng
    # đầu messages → server (llama.cpp) tái dùng KV-cache của nó giữa các bài cùng
    # đề. Log để xác nhận cache đang bật + tỉ trọng phần ổn định so với tổng prompt
    # (càng cao càng tiết kiệm prefill — sau khi cắt segments, phần này tăng mạnh).
    if config.local_prefix_cache:
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
    # long text truncated) so ta thấy đúng prompt model local nhận được.
    _log_messages(logger, "local", config.local_model, messages)

    # Log the full request payload being sent to the local API
    # (system prompt is already embedded in messages[0] for OpenAI-compatible)
    if config.log_prompts:
        _log_api_request(
            config, "local",
            model=config.local_model,
            base_url=config.local_base_url,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=0,
            extra_body=extra_body,
        )

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=config.local_model,
        max_tokens=config.max_tokens,
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
        "Model local trả structured output | model=%s | base_url=%s | latency=%.2fs | "
        "prompt_tokens=%s completion_tokens=%s",
        config.local_model,
        config.local_base_url,
        latency,
        getattr(usage, "prompt_tokens", "?"),
        getattr(usage, "completion_tokens", "?"),
    )

    finish = response.choices[0].finish_reason
    content = response.choices[0].message.content
    if finish == "length":
        raise RuntimeError(
            f"Model local bị cắt vì chạm trần max_tokens={config.max_tokens} "
            f"(finish_reason=length) → JSON dở dang. Tăng TOEIC_MAX_TOKENS, "
            f"hoặc giảm độ dài nhận xét."
        )
    if not content:
        raise RuntimeError(
            f"Model local không trả về nội dung (finish_reason={finish})."
        )

    # Log AI response
    if config.log_prompts:
        _log_response(config, "local", content)

    try:
        return output_model.model_validate_json(content)
    except Exception as e:  # noqa: BLE001 - bọc lỗi parse cho rõ
        raise RuntimeError(
            f"Model local trả JSON không đúng schema {schema_name}: {e}\n"
            f"Nội dung: {content[:500]}"
        ) from e


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

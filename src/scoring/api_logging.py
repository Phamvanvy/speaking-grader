"""Logging prompt/response LLM ra console + outputs/prompt_logs/ (debug, so model).

_log_messages luôn chạy (sanitize ảnh + cắt text); _log_api_request / _log_response chỉ
khi config.log_prompts. _ensure_prompt_log_dir xoay log theo từng lần chạy process.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..config import Config

logger = logging.getLogger("toeic.scoring")

# Directory to store prompt logs for debugging / model comparison.
# Enable by setting env var TOEIC_LOG_PROMPTS=1 or Config.log_prompts=True.
_PROMPT_LOG_DIR = Path("outputs/prompt_logs")

# Chính sách xoay log: mỗi LẦN CHẠY (process mới) sẽ dọn sạch log của lần chạy
# trước ở lần ghi đầu tiên — nên chạy lại app = log mới "ghi đè" log cũ. Nhưng
# trong CÙNG một phiên, các lần chấm tiếp theo chỉ GHI THÊM (append), không xoá
# nhau. Cờ dưới đảm bảo bước dọn chạy đúng một lần cho mỗi process.
_prompt_log_reset_done = False


def _ensure_prompt_log_dir() -> None:
    """Tạo thư mục log; lần ĐẦU trong process thì xoá log của lần chạy trước.

    Hiệu ứng: chạy lại app → log mới ghi đè (dọn) log cũ; nhiều lần chấm trong
    cùng một phiên → tích luỹ thêm (append), không đè lên nhau.
    """
    global _prompt_log_reset_done
    _PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _prompt_log_reset_done:
        for old in _PROMPT_LOG_DIR.glob("*.json"):
            try:
                old.unlink()
            except OSError as e:  # pragma: no cover - file bị khoá/đang mở hiếm gặp
                logger.warning("Không xoá được log cũ %s: %s", old, e)
        _prompt_log_reset_done = True


# Độ dài tối đa của mỗi đoạn text khi log ra console (tránh ngập log).
_LOG_TEXT_PREVIEW = 4000


def _preview_content(content: object) -> object:
    """Rút gọn content của 1 message để log: bỏ base64 ảnh, cắt text dài."""
    if isinstance(content, str):
        if len(content) > _LOG_TEXT_PREVIEW:
            return content[:_LOG_TEXT_PREVIEW] + f"... [+{len(content) - _LOG_TEXT_PREVIEW} chars]"
        return content
    if isinstance(content, list):
        parts: list[object] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype in ("image_url", "image"):
                    parts.append({"type": ptype, "data": "[IMAGE REDACTED]"})
                elif ptype == "text":
                    parts.append({"type": "text", "text": _preview_content(part.get("text", ""))})
                else:
                    parts.append(part)
            else:
                parts.append(part)
        return parts
    return content


def _log_messages(
    log: logging.Logger,
    backend: str,
    model: str,
    messages: list[dict],
    *,
    system_prompt: str | None = None,
) -> None:
    """Log nội dung messages gửi lên LLM (sanitize ảnh + cắt text dài).

    Luôn chạy (không phụ thuộc config.log_prompts) để debug nhanh prompt thực tế
    model nhận. Ảnh base64 bị thay bằng [IMAGE REDACTED]; text > _LOG_TEXT_PREVIEW
    ký tự bị cắt.
    """
    preview = [
        {"role": m.get("role"), "content": _preview_content(m.get("content"))}
        for m in messages
    ]
    if system_prompt is not None:
        # Anthropic truyền system tách khỏi messages → log riêng cho đủ ngữ cảnh.
        preview.insert(0, {"role": "system", "content": _preview_content(system_prompt)})
    log.info(
        "LLM request | backend=%s | model=%s | messages=%s",
        backend,
        model,
        json.dumps(preview, ensure_ascii=False, indent=2),
    )


def _log_api_request(
    config: Config,
    backend: str,
    *,
    model: str,
    base_url: str | None,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    extra_body: dict | None,
    system_prompt: str | None = None,
) -> None:
    """Log the full API request payload (messages, params) to outputs/prompt_logs/."""
    _ensure_prompt_log_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Sanitize messages: strip image base64 data, keep structure
    sanitized_messages = []
    for msg in messages:
        sanitized_msg = dict(msg)
        content = sanitized_msg.get("content")
        if isinstance(content, list):
            # Vision message: strip image data
            sanitized_content = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype in ("image_url", "image"):
                        sanitized_content.append({"type": ptype, "[...]": "[IMAGE REDACTED]"})
                    elif ptype == "text":
                        text = part.get("text", "")
                        if len(text) > 5000:
                            text = text[:5000] + "... [truncated]"
                        sanitized_content.append({"type": "text", "text": text})
                    else:
                        sanitized_content.append(part)
                else:
                    sanitized_content.append(part)
            sanitized_msg["content"] = sanitized_content
        elif isinstance(content, str) and len(content) > 5000:
            sanitized_msg["content"] = content[:5000] + "... [truncated]"
        sanitized_messages.append(sanitized_msg)

    # Build a single payload for the request
    # Use a combined hash for the stem to avoid collisions
    content_hash = hash(json.dumps(messages, ensure_ascii=False)[:1000]) % 100000
    stem = f"{ts}_{backend}_req_{content_hash:05d}"
    log_file = _PROMPT_LOG_DIR / f"{stem}.json"

    payload = {
        "backend": backend,
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": extra_body,
    }
    # Include system prompt (Anthropic passes it as a separate parameter)
    if system_prompt:
        payload["system_prompt"] = system_prompt[:5000]
    payload["messages"] = sanitized_messages

    log_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("API request logged to %s", log_file)


def _log_response(config: Config, backend: str, response_json: str) -> None:
    """Log AI response JSON to outputs/prompt_logs/."""
    _ensure_prompt_log_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{backend}_resp_{hash(response_json) % 100000:05d}"
    log_file = _PROMPT_LOG_DIR / f"{stem}.json"

    try:
        pretty = json.loads(response_json)
        content = json.dumps(pretty, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        content = response_json

    log_file.write_text(content, encoding="utf-8")
    logger.info("Response logged to %s", log_file)

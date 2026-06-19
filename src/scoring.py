"""Chấm điểm bằng LLM với structured output.

Gửi đề bài + (script) + transcript + số liệu khách quan + cờ gating cho model,
nhận về SpeakingResult đúng schema (không phải tự parse JSON).

Hai backend (xem Config.backend):
- "anthropic": Claude qua Anthropic SDK (messages.parse + adaptive thinking).
- "local": model local (vd Qwen3 qua llama.cpp server) qua API
  OpenAI-compatible, ép schema bằng response_format json_schema.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .asr import Transcription
from .config import Config, resolve_language_name
from .features import Features
from .gating import GatingResult
from .phoneme.models import PhonemeResult
from .rubrics.toeic import QuestionType
from .schema import SpeakingResult

logger = logging.getLogger("toeic.scoring")

# Directory to store prompt logs for debugging / model comparison.
# Enable by setting env var TOEIC_LOG_PROMPTS=1 or Config.log_prompts=True.
_PROMPT_LOG_DIR = Path("outputs/prompt_logs")


def _build_system_prompt(qt: QuestionType, feedback_lang: str) -> str:
    criteria_lines = "\n".join(
        f"- {c.key} ({c.label}): {c.description}" for c in qt.criteria
    )
    language_name = resolve_language_name(feedback_lang)
    return f"""You are an experienced TOEIC Speaking examiner. Score one spoken \
response for the task type: {qt.label}.

TASK GUIDANCE:
{qt.guidance}

CRITERIA TO SCORE (only these):
{criteria_lines}

SCORING SCALE:
{qt.scale_description}

EVIDENCE RULES (important):
- Use the OBJECTIVE METRICS provided (speech_rate_wpm, pause_count, \
longest_pause_sec, filler_count, and for read-aloud the accuracy_metrics: wer, \
substitutions, insertions, deletions) as PRIMARY evidence.
- Use analysis of the transcript text as SECONDARY evidence.
- Do NOT rely solely on ASR confidence (avg_word_probability / \
min_word_probability). It is affected by microphone quality, accent, and \
background noise — treat it ONLY as weak supporting evidence, never as the \
pronunciation score itself.
- For read-aloud, accuracy_metrics.word_issues lists places where the ASR \
transcript diverged from the script (substitution / insertion / deletion), e.g. \
expected "morning" but recognized "warning". These are NOT confirmed \
mispronunciations — the ASR may have misheard due to noise, accent, or its own \
limits. Use them ONLY as "words worth reviewing": you may say the ASR may have \
misheard a word and suggest the test-taker double-check it. NEVER state with \
certainty that the test-taker pronounced a specific word wrong based on a \
word_issue alone.
- PHONEME METRICS (if available): phoneme_data provides deep pronunciation \
evidence at the phoneme level (IPA). Use overall_accuracy as a STRONG signal for \
the pronunciation score. High-severity errors indicate clear mispronunciations. \
Pay special attention to substitution errors where similar-sounding phonemes are \
confused (e.g. /θ/ → /s/, /æ/ → /ɛ/) — these are common ESL mistakes. Low \
severity errors may be acceptable regional variants. If phoneme_data is null or \
disabled, rely on word-level evidence only.

TASK COMPLETION:
- task_completion reflects whether the response actually fulfils the prompt \
(answered fully, long enough, on-topic). A grammatically perfect but far too \
short or off-topic answer must get a LOW task_completion.
- If a completion floor is provided by upstream rule-based checks, do not score \
task_completion higher than that floor.

Map the per-criterion scores to estimated_toeic_score on the 0-200 TOEIC \
Speaking scale (TOEIC does NOT use IELTS bands). Be consistent and calibrated.
Give concrete, actionable suggestions for each criterion.

EXPLAIN YOUR REASONING (important):
- Each criterion's `justification` must be a clear, logical chain: cite the \
specific objective metric or transcript evidence, say what it implies, then why \
that lands the criterion at this 0-3 score and not one higher or lower.
- `score_rationale` must explain step by step how the per-criterion scores \
combine into the final estimated_toeic_score: which criteria pulled the score \
up or down, how task_completion / content_relevance and any gating floor were \
applied, and why the result falls in this 0-200 band rather than higher/lower. \
Do not just restate the number — justify it.

OUTPUT LANGUAGE (important):
- Write ALL human-readable text — every `justification`, every entry in \
`suggestions`, `score_rationale`, and `summary_feedback` — in {language_name}.
- Keep machine fields unchanged and in English: the `criterion` field must stay \
the lowercase English key (e.g. "pronunciation", "intonation_stress"), and the \
enum values for task_completion / content_relevance (very_low/low/medium/high) \
stay as-is. Only the explanatory prose is translated."""


def _build_user_prompt(
    qt: QuestionType,
    prompt_text: str,
    reference_script: str | None,
    transcription: Transcription,
    features: Features,
    gating: GatingResult,
    phoneme_result: PhonemeResult | None = None,
    has_image: bool = False,
) -> str:
    payload: dict = {
        "task_prompt": prompt_text,
        "reference_script": reference_script if qt.uses_reference_script else None,
        "transcript": transcription.text,
        "objective_metrics": features.to_dict(),
        "rule_based_gating": {
            "task_completion_floor": gating.task_completion_floor,
            "reasons": gating.reasons,
            "reference_coverage": gating.reference_coverage,
            "fail_reference_match": gating.fail_reference_match,
        },
    }

    # Include phoneme data if available
    if phoneme_result is not None:
        payload["phoneme_data"] = phoneme_result.to_dict()
        logger.info(
            "Phoneme data included in scoring payload: "
            "backend=%s | available=%s | segments=%d",
            phoneme_result.backend_used,
            phoneme_result.backend_available,
            len(phoneme_result.segments),
        )

    image_note = (
        "An IMAGE of the picture the test-taker was asked to describe is attached "
        "to this message. Judge whether the spoken transcript accurately and "
        "completely describes what is actually in the picture (objects, people, "
        "actions, setting). A description that does not match the picture must "
        "lower content_relevance / relevance.\n\n"
        if has_image
        else ""
    )
    return (
        "Score the following TOEIC Speaking response. All numeric metrics are "
        "pre-computed and objective.\n\n"
        + image_note
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# Ký tự mở ngoặc ở cuối chuỗi → dấu hiệu text bị cắt giữa chừng (JSON degenerate).
_DANGLING_OPEN = ("(", "[", "{", "（", "［", "｛")


def _is_truncated(text: str) -> bool:
    """True nếu chuỗi rỗng hoặc kết thúc bằng dấu mở ngoặc (bị cắt giữa chừng)."""
    s = (text or "").strip()
    return not s or s.endswith(_DANGLING_OPEN)


def _validate_result(result: SpeakingResult, qt: QuestionType) -> list[str]:
    """Bắt output 'hợp lệ schema nhưng rác' mà Pydantic không chặn được.

    Trả về danh sách mô tả lỗi (rỗng nếu OK). Chỉ gắn cờ 3 dạng hỏng đã quan
    sát thực tế: thiếu tiêu chí bắt buộc, suggestions điền nhầm tên key tiêu chí,
    và text bị cắt/rỗng. KHÔNG bắt suggestions rỗng — model trả thiếu suggestions
    vẫn là output hợp lệ.
    """
    problems: list[str] = []
    required = {c.key for c in qt.criteria}

    present = {c.criterion for c in result.criteria}
    missing = required - present
    if missing:
        problems.append(f"thiếu tiêu chí bắt buộc: {sorted(missing)}")

    for c in result.criteria:
        polluted = [s for s in c.suggestions if s in required]
        if polluted:
            problems.append(
                f"suggestions của '{c.criterion}' chứa tên tiêu chí: {polluted}"
            )
        if _is_truncated(c.justification):
            problems.append(f"justification của '{c.criterion}' bị cắt/rỗng")

    if _is_truncated(result.score_rationale):
        problems.append("score_rationale bị cắt/rỗng")
    if _is_truncated(result.summary_feedback):
        problems.append("summary_feedback bị cắt/rỗng")

    return problems


def score(
    config: Config,
    qt: QuestionType,
    prompt_text: str,
    reference_script: str | None,
    transcription: Transcription,
    features: Features,
    gating: GatingResult,
    phoneme_result: PhonemeResult | None = None,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    """Gọi LLM (Claude hoặc model local) và trả về SpeakingResult.

    phoneme_result: kết quả phoneme analysis từ wav2vec/MFA (optional).
        Nếu có thì thêm vào payload để AI dùng làm evidence cho pronunciation.
    image_b64/image_media_type: ảnh đề bài (vd Describe Picture) gửi kèm dạng
    vision. Cả hai backend đều hỗ trợ; bỏ trống nếu không có ảnh.
    """
    system_prompt = _build_system_prompt(qt, config.feedback_lang)
    user_prompt = _build_user_prompt(
        qt,
        prompt_text,
        reference_script,
        transcription,
        features,
        gating,
        phoneme_result=phoneme_result,
        has_image=bool(image_b64),
    )

    # Gọi backend rồi validate; nếu output rác thì retry 1 lần và raise rõ ràng
    # thay vì âm thầm lưu điểm hỏng. Bắt glitch JSON hiếm (thiếu tiêu chí /
    # suggestions lẫn tên key / text cụt) mà schema Pydantic không chặn được.
    max_attempts = 2
    last_problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        if config.is_local:
            result = _score_local(
                config, system_prompt, user_prompt, image_b64, image_media_type
            )
        else:
            result = _score_anthropic(
                config, system_prompt, user_prompt, image_b64, image_media_type
            )
        last_problems = _validate_result(result, qt)
        if not last_problems:
            return result
        logger.warning(
            "Kết quả chấm không hợp lệ (lần %d/%d): %s",
            attempt,
            max_attempts,
            "; ".join(last_problems),
        )
    raise RuntimeError(
        f"LLM trả kết quả hỏng sau {max_attempts} lần (schema hợp lệ nhưng "
        f"nội dung rác): {'; '.join(last_problems)}"
    )


def _score_anthropic(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
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
        output_format=SpeakingResult,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Claude chấm xong | model=%s | latency=%.2fs | "
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


def _score_local(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    """Chấm bằng model local qua API OpenAI-compatible (vd llama.cpp server).

    Ép đúng schema bằng response_format json_schema — llama.cpp chuyển schema
    thành GBNF grammar nên JSON trả về luôn hợp lệ. Không có 'thinking' của
    Claude; nếu model hỗ trợ reasoning (Qwen3) có thể bật qua chat template.
    """
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - phụ thuộc tuỳ chọn
        raise RuntimeError(
            "Backend local cần gói 'openai'. Cài: pip install openai"
        ) from e

    client = OpenAI(base_url=config.local_base_url, api_key=config.local_api_key)

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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

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
                "name": "SpeakingResult",
                "schema": SpeakingResult.model_json_schema(),
                "strict": True,
            },
        },
        extra_body=extra_body,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Model local chấm xong | model=%s | base_url=%s | latency=%.2fs | "
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
        return SpeakingResult.model_validate_json(content)
    except Exception as e:  # noqa: BLE001 - bọc lỗi parse cho rõ
        raise RuntimeError(
            f"Model local trả JSON không đúng schema SpeakingResult: {e}\n"
            f"Nội dung: {content[:500]}"
        ) from e


# ---- Prompt logging helpers -------------------------------------------------

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
    _PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
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
    _PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
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

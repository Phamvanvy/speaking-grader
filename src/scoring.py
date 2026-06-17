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

from .asr import Transcription
from .config import Config, resolve_language_name
from .features import Features
from .gating import GatingResult
from .rubrics.toeic import QuestionType
from .schema import SpeakingResult

logger = logging.getLogger("toeic.scoring")


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
) -> str:
    payload = {
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
    return (
        "Score the following TOEIC Speaking response. All numeric metrics are "
        "pre-computed and objective.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def score(
    config: Config,
    qt: QuestionType,
    prompt_text: str,
    reference_script: str | None,
    transcription: Transcription,
    features: Features,
    gating: GatingResult,
) -> SpeakingResult:
    """Gọi LLM (Claude hoặc model local) và trả về SpeakingResult."""
    system_prompt = _build_system_prompt(qt, config.feedback_lang)
    user_prompt = _build_user_prompt(
        qt, prompt_text, reference_script, transcription, features, gating
    )

    if config.is_local:
        return _score_local(config, system_prompt, user_prompt)
    return _score_anthropic(config, system_prompt, user_prompt)


def _score_anthropic(
    config: Config, system_prompt: str, user_prompt: str
) -> SpeakingResult:
    if not config.has_api_key:
        raise RuntimeError(
            "Thiếu ANTHROPIC_API_KEY. Đặt trong .env, dùng TOEIC_BACKEND=local "
            "để chấm bằng model local, hoặc chạy với --no-ai để chỉ lấy "
            "transcript + features."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    t0 = time.monotonic()
    response = client.messages.parse(
        model=config.model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
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
    if result is None:
        # stop_reason refusal / max_tokens → parsed_output có thể None
        raise RuntimeError(
            f"Claude không trả về kết quả đúng schema "
            f"(stop_reason={response.stop_reason})."
        )
    return result


def _score_local(
    config: Config, system_prompt: str, user_prompt: str
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

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=config.local_model,
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SpeakingResult",
                "schema": SpeakingResult.model_json_schema(),
                "strict": True,
            },
        },
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

    content = response.choices[0].message.content
    if not content:
        finish = response.choices[0].finish_reason
        raise RuntimeError(
            f"Model local không trả về nội dung (finish_reason={finish})."
        )
    try:
        return SpeakingResult.model_validate_json(content)
    except Exception as e:  # noqa: BLE001 - bọc lỗi parse cho rõ
        raise RuntimeError(
            f"Model local trả JSON không đúng schema SpeakingResult: {e}\n"
            f"Nội dung: {content[:500]}"
        ) from e

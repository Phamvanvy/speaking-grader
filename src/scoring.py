"""Chấm điểm bằng Claude API với structured output.

Gửi đề bài + (script) + transcript + số liệu khách quan + cờ gating cho Claude,
nhận về SpeakingResult đúng schema (không phải tự parse JSON).
"""

from __future__ import annotations

import json
import logging
import time

from .asr import Transcription
from .config import Config
from .features import Features
from .gating import GatingResult
from .rubrics.toeic import QuestionType
from .schema import SpeakingResult

logger = logging.getLogger("toeic.scoring")


def _build_system_prompt(qt: QuestionType) -> str:
    criteria_lines = "\n".join(
        f"- {c.key} ({c.label}): {c.description}" for c in qt.criteria
    )
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

TASK COMPLETION:
- task_completion reflects whether the response actually fulfils the prompt \
(answered fully, long enough, on-topic). A grammatically perfect but far too \
short or off-topic answer must get a LOW task_completion.
- If a completion floor is provided by upstream rule-based checks, do not score \
task_completion higher than that floor.

Map the per-criterion scores to estimated_toeic_score on the 0-200 TOEIC \
Speaking scale (TOEIC does NOT use IELTS bands). Be consistent and calibrated.
Give concrete, actionable suggestions for each criterion."""


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
    """Gọi Claude và trả về SpeakingResult."""
    if not config.has_api_key:
        raise RuntimeError(
            "Thiếu ANTHROPIC_API_KEY. Đặt trong .env, hoặc chạy với --no-ai để "
            "chỉ lấy transcript + features."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    system_prompt = _build_system_prompt(qt)
    user_prompt = _build_user_prompt(
        qt, prompt_text, reference_script, transcription, features, gating
    )

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

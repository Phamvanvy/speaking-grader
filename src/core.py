"""Lõi pipeline chấm điểm — dùng chung cho CLI ([main.py]) và API ([api.py]).

Tách khỏi main() để cùng một luồng ASR → features → gating → scoring → report
phục vụ được cả dòng lệnh lẫn HTTP, không phụ thuộc ngân hàng câu hỏi: đầu vào
(script tham chiếu / ảnh / thời lượng kỳ vọng) được truyền thẳng vào.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import asr, features as features_mod, gating, report, scoring
from .config import Config
from .rubrics.toeic import QuestionType

logger = logging.getLogger("toeic.core")


def grade_response(
    audio_path: str,
    config: Config,
    qt: QuestionType,
    *,
    prompt_text: str = "",
    reference_script: str | None = None,
    expected_duration_sec: float | None = None,
    image_b64: str | None = None,
    image_media_type: str | None = None,
    no_ai: bool = False,
    question_id: str = "adhoc",
    save: bool = True,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline cho 1 audio và trả về dict kết quả (build_output).

    - qt: dạng câu (quyết định tiêu chí + có dùng script tham chiếu không).
    - reference_script: text dùng cho Read Aloud (để so transcript ra WER/coverage).
    - image_b64 / image_media_type: ảnh đề bài cho Describe Picture (gửi LLM dạng vision).
    - expected_duration_sec: optional, vào features (reading_pace) + gating.
    - no_ai: chỉ chạy ASR + features, bỏ qua LLM.
    - save: ghi JSON ra outputs/ (CLI cần; API có thể tắt).
    """
    active_model = config.local_model if config.is_local else config.model
    logger.info(
        "Chấm | audio=%s | question=%s | type=%s | backend=%s | model=%s | no_ai=%s",
        audio_path,
        question_id,
        qt.key,
        config.backend,
        active_model,
        no_ai,
    )

    # [1] ASR
    transcription = asr.transcribe(
        audio_path,
        model_size=config.whisper_model,
        device=config.whisper_device,
    )

    # [2] Features
    feats = features_mod.extract_features(
        transcription,
        reference_script=reference_script,
        expected_duration_sec=expected_duration_sec,
    )

    # [3] Gating
    gate = gating.evaluate(
        transcription,
        feats,
        expected_duration_sec=expected_duration_sec,
        question_type=qt,
    )
    for reason in gate.reasons:
        logger.info("Gating: %s", reason)

    # [4] Scoring (trừ khi no_ai hoặc audio rỗng)
    scores_dict = None
    if no_ai:
        logger.info("Bỏ qua chấm điểm (no_ai).")
    elif gate.should_skip_ai:
        logger.warning("Audio rỗng/không nhận ra lời — không gọi LLM.")
    else:
        result = scoring.score(
            config=config,
            qt=qt,
            prompt_text=prompt_text,
            reference_script=reference_script,
            transcription=transcription,
            features=feats,
            gating=gate,
            image_b64=image_b64,
            image_media_type=image_media_type,
        )
        scores_dict = result.model_dump(mode="json")

    # [5] Report
    output = report.build_output(
        audio_path=audio_path,
        question_id=question_id,
        question_type=qt.key,
        transcript=transcription.text,
        features=feats.to_dict(),
        scores=scores_dict,
    )
    if save:
        stem = f"{Path(audio_path).stem}__{question_id}"
        out_path = report.save_json(output, stem=stem)
        logger.info("Đã lưu kết quả: %s", out_path)
    return output

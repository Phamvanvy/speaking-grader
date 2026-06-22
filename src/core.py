"""Lõi pipeline chấm điểm — dùng chung cho CLI ([main.py]) và API ([api.py]).

Tách khỏi main() để cùng một luồng ASR → features → gating → scoring → report
phục vụ được cả dòng lệnh lẫn HTTP, không phụ thuộc ngân hàng câu hỏi: đầu vào
(script tham chiếu / ảnh / thời lượng kỳ vọng) được truyền thẳng vào.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from . import asr, features as features_mod, gating, report, scoring
from .config import Config
from .phoneme.analyzer import HybridPhonemeAnalyzer
from .phoneme.models import PhonemeResult
from .rubrics.base import QuestionType

logger = logging.getLogger("toeic.core")


def _compact_phoneme_output(phoneme_result: PhonemeResult | None) -> dict | None:
    """Bản gọn của phoneme analysis cho JSON output + UI.

    Chỉ giữ headline + lỗi đã gắn từ (KHÔNG kèm segments thô / reference_phonemes
    đầy đủ). Shape nested theo `score` để khớp model backend và frontend reader
    (`data.phoneme.score.errors`). None nếu không có score.
    """
    if phoneme_result is None or phoneme_result.score is None:
        return None
    score = phoneme_result.score.to_dict()
    return {
        "backend_used": phoneme_result.backend_used,
        "warning": phoneme_result.warning,
        "score": {
            "overall_accuracy": score["overall_accuracy"],
            "errors": score["errors"],
            "words": score["words"],
            "words_truncated": score["words_truncated"],
            "words_total": score["words_total"],
        },
    }


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
    provided_info: str | None = None,
    asr_backend: str = "faster_whisper",
    asr_model: str | None = None,
    no_ai: bool = False,
    phoneme_analysis: bool | None = None,
    question_id: str = "adhoc",
    save: bool = True,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline cho 1 audio và trả về dict kết quả (build_output).

    - qt: dạng câu (quyết định tiêu chí + có dùng script tham chiếu không).
    - reference_script: text dùng cho Read Aloud (để so transcript ra WER/coverage).
    - image_b64 / image_media_type: ảnh đề bài cho Describe Picture (gửi LLM dạng vision).
    - provided_info: tài liệu cho sẵn (Q8-10) dạng text; vào payload chấm khi dạng câu
      có uses_provided_info.
    - expected_duration_sec: optional, vào features (reading_pace) + gating.
    - no_ai: chỉ chạy ASR + features, bỏ qua LLM.
    - asr_model: model ASR cho lần chấm này (vd "large-v3-turbo" cho practice,
      "large-v3" cho mock_test). None = dùng config.whisper_model chung.
    - phoneme_analysis: ép bật/tắt phoneme analysis (wav2vec) cho lần chấm này,
      bất kể config. None = theo config.phoneme_analysis_enabled. API dùng cờ này
      để gắn wav2vec theo mode: mock_test → True, practice → None (theo config),
      và True khi practice tự leo lên mock_test.
    - save: ghi JSON ra outputs/ (CLI cần; API có thể tắt).
    """
    phoneme_enabled = (
        config.phoneme_analysis_enabled
        if phoneme_analysis is None
        else phoneme_analysis
    )
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
    pipeline_started = time.perf_counter()
    step_timings_ms: dict[str, int] = {}

    # [1] ASR
    step_started = time.perf_counter()
    asr_run = asr.transcribe_with_backend(
        audio_path,
        backend=asr_backend,
        model_size=asr_model or config.whisper_model,
        device=config.whisper_device,
    )
    transcription = asr_run.transcription
    step_timings_ms["asr"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=asr | backend=%s | duration_ms=%d | words=%d | audio_sec=%.2f",
        question_id,
        asr_run.backend_used,
        step_timings_ms["asr"],
        transcription.word_count,
        transcription.duration,
    )

    # [2] Features
    step_started = time.perf_counter()
    feats = features_mod.extract_features(
        transcription,
        reference_script=reference_script,
        expected_duration_sec=expected_duration_sec,
    )
    step_timings_ms["features"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=features | duration_ms=%d | wpm=%.1f | pauses=%d",
        question_id,
        step_timings_ms["features"],
        feats.speech_rate_wpm,
        feats.pause_count,
    )

    # [2b] Phoneme analysis (optional — Phase 1: wav2vec)
    phoneme_result = None
    if phoneme_enabled and transcription.text.strip():
        step_started = time.perf_counter()
        phoneme_analyzer = HybridPhonemeAnalyzer(
            wav2vec_model=config.phoneme_wav2vec_model,
            device=config.phoneme_device,
        )
        # Read Aloud có script mẫu → so phát âm với script. Câu nói tự do (IELTS
        # Speaking, Describe Picture, Respond...) không có script → fallback về
        # transcript ASR: đo phát âm của chính những từ thí sinh đã nói (kiểu ELSA).
        phoneme_result = phoneme_analyzer.analyze(
            audio_path,
            reference_text=reference_script or transcription.text,
        )
        step_timings_ms["phoneme"] = int((time.perf_counter() - step_started) * 1000)
        if phoneme_result.score:
            logger.info(
                "Phoneme | question=%s | accuracy=%.2f | substitutions=%d | deletions=%d | insertions=%d",
                question_id,
                phoneme_result.score.overall_accuracy,
                phoneme_result.score.substitution_count,
                phoneme_result.score.deletion_count,
                phoneme_result.score.insertion_count,
            )
        else:
            logger.info(
                "Phoneme | question=%s | skipped (%s)",
                question_id,
                phoneme_result.warning or "no reference",
            )
    else:
        step_timings_ms["phoneme"] = 0
        if not phoneme_enabled:
            logger.info("Phoneme | question=%s | disabled (mode/config)", question_id)

    # [3] Gating
    step_started = time.perf_counter()
    gate = gating.evaluate(
        transcription,
        feats,
        expected_duration_sec=expected_duration_sec,
        question_type=qt,
    )
    step_timings_ms["gating"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=gating | duration_ms=%d | skip_ai=%s | floor=%s",
        question_id,
        step_timings_ms["gating"],
        gate.should_skip_ai,
        gate.task_completion_floor,
    )
    for reason in gate.reasons:
        logger.info("Gating: %s", reason)

    # [4] Scoring (trừ khi no_ai hoặc audio rỗng)
    scores_dict = None
    scoring_status = "skipped"
    if no_ai:
        logger.info("Bỏ qua chấm điểm (no_ai).")
    elif gate.should_skip_ai:
        logger.warning("Audio rỗng/không nhận ra lời — không gọi LLM.")
    else:
        step_started = time.perf_counter()
        result = scoring.score(
            config=config,
            qt=qt,
            prompt_text=prompt_text,
            reference_script=reference_script,
            transcription=transcription,
            features=feats,
            gating=gate,
            phoneme_result=phoneme_result,
            image_b64=image_b64,
            image_media_type=image_media_type,
            provided_info=provided_info,
        )
        step_timings_ms["scoring"] = int((time.perf_counter() - step_started) * 1000)
        scores_dict = result.model_dump(mode="json")
        scoring_status = "completed"
        _score_field = (
            "estimated_ielts_band"
            if qt.exam == "ielts"
            else "estimated_toeic_score"
        )
        logger.info(
            "Timing | question=%s | step=scoring | duration_ms=%d | exam=%s | score=%s",
            question_id,
            step_timings_ms["scoring"],
            qt.exam,
            scores_dict.get(_score_field),
        )

    if scoring_status != "completed":
        step_timings_ms["scoring"] = 0
        logger.info(
            "Timing | question=%s | step=scoring | duration_ms=0 | status=%s",
            question_id,
            scoring_status if no_ai else "skipped_by_gating",
        )

    # [5] Report
    step_started = time.perf_counter()
    output = report.build_output(
        audio_path=audio_path,
        question_id=question_id,
        question_type=qt.key,
        exam=qt.exam,
        transcript=transcription.text,
        features=feats.to_dict(),
        scores=scores_dict,
        phoneme=_compact_phoneme_output(phoneme_result),
        telemetry={
            "asr_backend_used": asr_run.backend_used,
            "transcription_time_ms": asr_run.elapsed_ms,
            "step_timings_ms": step_timings_ms,
        },
    )
    step_timings_ms["report_build"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=report_build | duration_ms=%d",
        question_id,
        step_timings_ms["report_build"],
    )
    if save:
        save_started = time.perf_counter()
        stem = f"{Path(audio_path).stem}__{question_id}"
        out_path = report.save_json(output, stem=stem)
        step_timings_ms["report_save"] = int((time.perf_counter() - save_started) * 1000)
        logger.info("Đã lưu kết quả: %s", out_path)
        logger.info(
            "Timing | question=%s | step=report_save | duration_ms=%d",
            question_id,
            step_timings_ms["report_save"],
        )
    else:
        step_timings_ms["report_save"] = 0

    total_ms = int((time.perf_counter() - pipeline_started) * 1000)
    output["telemetry"]["step_timings_ms"] = step_timings_ms
    output["telemetry"]["pipeline_total_ms"] = total_ms
    logger.info(
        "Timing | question=%s | total_ms=%d | asr=%d | features=%d | phoneme=%d | gating=%d | scoring=%d | report_build=%d | report_save=%d",
        question_id,
        total_ms,
        step_timings_ms["asr"],
        step_timings_ms["features"],
        step_timings_ms.get("phoneme", 0),
        step_timings_ms["gating"],
        step_timings_ms["scoring"],
        step_timings_ms["report_build"],
        step_timings_ms["report_save"],
    )
    return output

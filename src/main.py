"""CLI: chấm 1 file audio TOEIC Speaking.

Ví dụ:
    python -m src.main --audio data/audio/sample.wav --question q1_read_aloud
    python -m src.main --audio data/audio/sample.wav --question q1_read_aloud --no-ai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ép stdout/stderr sang UTF-8 để in được tiếng Việt trên console Windows
# (mặc định có thể là cp1258 và không mã hoá được một số ký tự).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from . import asr, features as features_mod, gating, report, scoring
from .config import load_config
from .logging_setup import setup_logging
from .questions import get_question
from .rubrics.toeic import get_question_type


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chấm bài TOEIC Speaking từ 1 file audio."
    )
    parser.add_argument("--audio", required=True, help="Đường dẫn file audio (.wav/.mp3)")
    parser.add_argument("--question", required=True, help="ID câu hỏi trong ngân hàng")
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Chỉ chạy ASR + features (bỏ qua chấm bằng Claude).",
    )
    args = parser.parse_args(argv)

    logger = setup_logging()
    config = load_config()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("Không tìm thấy file audio: %s", audio_path)
        return 2

    try:
        question = get_question(args.question)
        qt = get_question_type(question.type)
    except (KeyError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 2

    active_model = config.local_model if config.is_local else config.model
    logger.info(
        "Bắt đầu | audio=%s | question=%s | type=%s | backend=%s | model=%s | no_ai=%s",
        audio_path,
        question.id,
        question.type,
        config.backend,
        active_model,
        args.no_ai,
    )

    # [1] ASR
    transcription = asr.transcribe(
        str(audio_path),
        model_size=config.whisper_model,
        device=config.whisper_device,
    )

    # [2] Features
    feats = features_mod.extract_features(
        transcription,
        reference_script=question.reference_script,
        expected_duration_sec=question.expected_duration_sec,
    )

    # [3] Gating
    gate = gating.evaluate(
        transcription,
        feats,
        expected_duration_sec=question.expected_duration_sec,
        question_type=qt,
    )
    for reason in gate.reasons:
        logger.info("Gating: %s", reason)

    # [4] Scoring (trừ khi --no-ai hoặc audio rỗng)
    scores_dict = None
    if args.no_ai:
        logger.info("Bỏ qua chấm điểm (--no-ai).")
    elif gate.should_skip_ai:
        logger.warning("Audio rỗng/không nhận ra lời — không gọi Claude.")
    else:
        try:
            result = scoring.score(
                config=config,
                qt=qt,
                prompt_text=question.prompt,
                reference_script=question.reference_script,
                transcription=transcription,
                features=feats,
                gating=gate,
            )
            scores_dict = result.model_dump(mode="json")
        except Exception as e:  # noqa: BLE001 - báo lỗi rõ ràng cho người dùng
            logger.error("Lỗi khi chấm bằng Claude: %s", e)
            return 1

    # [5] Report
    output = report.build_output(
        audio_path=str(audio_path),
        question_id=question.id,
        question_type=question.type,
        transcript=transcription.text,
        features=feats.to_dict(),
        scores=scores_dict,
    )
    out_path = report.save_json(output, stem=f"{audio_path.stem}__{question.id}")
    logger.info("Đã lưu kết quả: %s", out_path)
    report.print_report(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

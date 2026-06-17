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

from . import report
from .config import load_config
from .core import grade_response
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

    try:
        output = grade_response(
            str(audio_path),
            config,
            qt,
            prompt_text=question.prompt,
            reference_script=question.reference_script,
            expected_duration_sec=question.expected_duration_sec,
            no_ai=args.no_ai,
            question_id=question.id,
        )
    except Exception as e:  # noqa: BLE001 - báo lỗi rõ ràng cho người dùng
        logger.error("Lỗi khi chấm: %s", e)
        return 1

    report.print_report(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

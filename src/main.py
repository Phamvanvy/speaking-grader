"""CLI: chấm 1 file audio Speaking (TOEIC mặc định, hoặc IELTS qua --exam).

Ví dụ:
    python -m src.main --audio data/audio/sample.wav --question q1_read_aloud
    python -m src.main --audio data/audio/sample.wav --question q1_read_aloud --no-ai
    python -m src.main --audio data/audio/answer.wav --question q3_describe_picture
    python -m src.main --audio data/audio/answer.wav --question q3_describe_picture --image data/images/q3_sample.jpg
    python -m src.main --exam ielts --audio data/audio/answer.wav --question ielts_p2_memorable_trip
"""

from __future__ import annotations

import argparse
import base64
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
from .rubrics import resolve_question_type

_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _load_image(image_path: str) -> tuple[str, str]:
    """Đọc file ảnh, trả về (base64_data, media_type)."""
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {p}")
    media_type = _IMAGE_MEDIA_TYPES.get(p.suffix.lower(), "image/jpeg")
    return base64.standard_b64encode(p.read_bytes()).decode(), media_type


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chấm bài TOEIC Speaking từ 1 file audio."
    )
    parser.add_argument("--audio", required=True, help="Đường dẫn file audio (.wav/.mp3)")
    parser.add_argument("--question", required=True, help="ID câu hỏi trong ngân hàng")
    parser.add_argument(
        "--exam",
        default=None,
        help="Kỳ thi: toeic | ielts (mặc định theo config.default_exam).",
    )
    parser.add_argument(
        "--image",
        help="Đường dẫn file ảnh đề bài (ghi đè image_path trong ngân hàng câu hỏi).",
    )
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

    exam = (args.exam or config.default_exam).strip().lower()
    try:
        question = get_question(args.question, exam=exam)
        qt = resolve_question_type(question.type, exam=exam)
    except (KeyError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 2

    # Xác định ảnh: --image flag > image_path trong ngân hàng câu hỏi
    image_b64: str | None = None
    image_media_type: str | None = None
    raw_image_path = args.image or question.image_path
    if raw_image_path:
        try:
            image_b64, image_media_type = _load_image(raw_image_path)
        except FileNotFoundError as e:
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
            image_b64=image_b64,
            image_media_type=image_media_type,
            provided_info=question.provided_info,
            asr_backend=config.asr_engine_practice,
            asr_model=config.asr_model_practice,
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

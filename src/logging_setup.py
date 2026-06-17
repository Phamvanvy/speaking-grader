"""Cấu hình logging — ghi ra console và logs/app.log.

Mỗi lần chạy ta log audio_path, question_id, duration, model, token usage,
latency... để sau này tối ưu prompt/chi phí dễ dàng.
"""

from __future__ import annotations

import logging
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "app.log"

_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Cấu hình root logger một lần; trả về logger của app."""
    global _configured
    logger = logging.getLogger("toeic")
    if _configured:
        return logger

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    _configured = True
    return logger

"""Cấu hình tập trung — đọc từ biến môi trường / .env.

Không hardcode API key ở bất kỳ đâu. Model chấm điểm cấu hình được để dễ
chuyển giữa Sonnet (rẻ, dùng khi tinh chỉnh prompt) và Opus (benchmark).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # nạp .env nếu có

# Version của cách tính features — đổi khi thay đổi logic trong features.py
# để dễ so sánh kết quả cũ/mới khi debug.
FEATURES_VERSION = "v1"


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None
    model: str
    whisper_model: str
    whisper_device: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)


def load_config() -> Config:
    return Config(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("TOEIC_MODEL", "claude-sonnet-4-6"),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
    )

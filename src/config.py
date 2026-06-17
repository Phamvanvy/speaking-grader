"""Cấu hình tập trung — đọc từ biến môi trường / .env.

Không hardcode API key ở bất kỳ đâu. Model chấm điểm cấu hình được để dễ
chuyển giữa Sonnet (rẻ, dùng khi tinh chỉnh prompt) và Opus (benchmark).

Hỗ trợ 2 backend chấm điểm (TOEIC_BACKEND):
- "anthropic" (mặc định): gọi Claude qua Anthropic SDK.
- "local": gọi model local (vd Qwen3 qua llama.cpp server) bằng API
  OpenAI-compatible. Dùng khi phát triển: miễn phí, offline. Lưu ý điểm số
  sẽ lệch calibration so với Claude — chốt benchmark thì quay lại "anthropic".
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # nạp .env nếu có

# Version của cách tính features — đổi khi thay đổi logic trong features.py
# để dễ so sánh kết quả cũ/mới khi debug.
FEATURES_VERSION = "v1"

# Ánh xạ mã ngôn ngữ -> tên ngôn ngữ để đưa vào prompt. Chỉ cần cho phần
# nhận xét người đọc (justification/suggestions/summary_feedback); tiêu chí
# và thang điểm vẫn dùng key tiếng Anh ổn định.
_LANGUAGE_NAMES = {
    "vi": "Vietnamese (tiếng Việt)",
    "en": "English",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "zh": "Chinese (中文)",
    "fr": "French (français)",
    "es": "Spanish (español)",
    "de": "German (Deutsch)",
}


def resolve_language_name(lang: str) -> str:
    """Trả về tên ngôn ngữ để nhúng vào prompt.

    Nhận mã ngắn ('vi', 'en', 'ja'...) hoặc tên tự do ('Italian', '日本語').
    Mã đã biết -> tên đầy đủ; còn lại trả nguyên (đã strip) để model tự hiểu.
    """
    key = (lang or "").strip()
    return _LANGUAGE_NAMES.get(key.lower(), key) or _LANGUAGE_NAMES["en"]


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str | None
    model: str
    whisper_model: str
    whisper_device: str
    # "anthropic" (Claude) hoặc "local" (model local, OpenAI-compatible)
    backend: str = "anthropic"
    # Cấu hình backend local (vd llama.cpp server)
    local_base_url: str = "http://localhost:8080/v1"
    local_model: str = "qwen3"
    local_api_key: str = "no-key"  # llama.cpp không kiểm tra key
    # Ngôn ngữ cho phần nhận xét (justification/suggestions/summary_feedback).
    # Mã ngắn ('vi', 'en', 'ja'...) hoặc tên tự do. Mặc định tiếng Việt vì
    # người dùng chính là người học VN.
    feedback_lang: str = "vi"

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def is_local(self) -> bool:
        return self.backend == "local"


def load_config() -> Config:
    return Config(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("TOEIC_MODEL", "claude-sonnet-4-6"),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        backend=(os.getenv("TOEIC_BACKEND", "anthropic") or "anthropic").lower(),
        local_base_url=os.getenv("TOEIC_LOCAL_BASE_URL", "http://localhost:8080/v1"),
        local_model=os.getenv("TOEIC_LOCAL_MODEL", "qwen3"),
        local_api_key=os.getenv("TOEIC_LOCAL_API_KEY", "no-key"),
        feedback_lang=os.getenv("TOEIC_FEEDBACK_LANG", "vi") or "vi",
    )

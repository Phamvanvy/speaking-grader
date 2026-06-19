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
    # Bật reasoning ("thinking") cho model local kiểu Qwen3. Đo thực tế: bật
    # thinking sinh ~6300 token reasoning → ~63s/bài; tắt còn ~960 token → ~9s
    # (nhanh ~6.7×) mà vẫn trả đủ JSON justification/score_rationale. Mặc định
    # TẮT cho local (vốn là backend dev, đã lệch calibration). Bật lại bằng
    # TOEIC_LOCAL_ENABLE_THINKING=true khi cần reasoning kỹ hơn.
    local_enable_thinking: bool = False
    # Ngôn ngữ cho phần nhận xét (justification/suggestions/summary_feedback).
    # Mã ngắn ('vi', 'en', 'ja'...) hoặc tên tự do. Mặc định tiếng Việt vì
    # người dùng chính là người học VN.
    feedback_lang: str = "vi"
    # Trần token sinh ra của LLM. Nhận xét tiếng Việt nhiều tiêu chí + rationale
    # dễ vượt 4096 → JSON bị cắt. Để rộng; cả 2 backend đều dừng sớm khi xong
    # nên đặt cao không tốn thêm (chỉ tính token thực sinh ra).
    max_tokens: int = 30000
    # ASR backends cho routing nhiều tầng:
    # - default: production mặc định
    # - fast: lane tối ưu throughput (có thể fallback default nếu không sẵn sàng)
    # - review: lane chấm chi tiết (chạy khi mode=review hoặc auto trigger)
    asr_backend_default: str = "faster_whisper"
    asr_backend_fast: str = "insanely_fast_whisper"
    asr_backend_review: str = "whisperx"
    # Bật/tắt fast lane ở runtime (dùng khi rollout production theo từng môi trường).
    fast_backend_enabled: bool = True
    # Model HF cho adapter Insanely Fast Whisper (transformers pipeline).
    insanely_fast_model_id: str = "openai/whisper-small"
    # Ngưỡng auto-review.
    auto_confidence_threshold: float = 0.75
    auto_silence_ratio_threshold: float = 0.35
    auto_coverage_threshold: float = 0.80
    # Phoneme analysis (wav2vec 2.0 backend — Phase 1, hybrid-ready for MFA Phase 2)
    phoneme_analysis_enabled: bool = False
    phoneme_wav2vec_model: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    phoneme_device: str = "cpu"
    phoneme_confidence_threshold: float = 0.1
    phoneme_min_duration_sec: float = 0.1
    # Log prompts and AI responses to outputs/prompt_logs/ for debugging.
    # Enable with TOEIC_LOG_PROMPTS=1.
    log_prompts: bool = False

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def is_local(self) -> bool:
        return self.backend == "local"


def load_config() -> Config:
    fast_enabled_raw = (os.getenv("TOEIC_FAST_BACKEND_ENABLED", "true") or "true").strip().lower()
    return Config(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("TOEIC_MODEL", "claude-sonnet-4-6"),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        # auto: ưu tiên CUDA khi môi trường torch hỗ trợ, fallback CPU.
        whisper_device=os.getenv("WHISPER_DEVICE", "auto"),
        backend=(os.getenv("TOEIC_BACKEND", "anthropic") or "anthropic").lower(),
        local_base_url=os.getenv("TOEIC_LOCAL_BASE_URL", "http://localhost:8080/v1"),
        local_model=os.getenv("TOEIC_LOCAL_MODEL", "qwen3"),
        local_api_key=os.getenv("TOEIC_LOCAL_API_KEY", "no-key"),
        local_enable_thinking=(
            os.getenv("TOEIC_LOCAL_ENABLE_THINKING", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        feedback_lang=os.getenv("TOEIC_FEEDBACK_LANG", "vi") or "vi",
        max_tokens=int(os.getenv("TOEIC_MAX_TOKENS", "30000")),
        asr_backend_default=(
            os.getenv("TOEIC_ASR_BACKEND_DEFAULT", "faster_whisper")
            or "faster_whisper"
        ).lower(),
        asr_backend_fast=(
            os.getenv("TOEIC_ASR_BACKEND_FAST", "insanely_fast_whisper")
            or "insanely_fast_whisper"
        ).lower(),
        asr_backend_review=(
            os.getenv("TOEIC_ASR_BACKEND_REVIEW", "whisperx")
            or "whisperx"
        ).lower(),
        fast_backend_enabled=fast_enabled_raw not in {"0", "false", "no", "off"},
        insanely_fast_model_id=(
            os.getenv("TOEIC_INSANELY_FAST_MODEL_ID", "openai/whisper-small")
            or "openai/whisper-small"
        ),
        auto_confidence_threshold=float(
            os.getenv("TOEIC_AUTO_CONFIDENCE_THRESHOLD", "0.75")
        ),
        auto_silence_ratio_threshold=float(
            os.getenv("TOEIC_AUTO_SILENCE_RATIO_THRESHOLD", "0.35")
        ),
        auto_coverage_threshold=float(
            os.getenv("TOEIC_AUTO_COVERAGE_THRESHOLD", "0.80")
        ),
        # Phoneme analysis config
        phoneme_analysis_enabled=(
            os.getenv("TOEIC_PHONEME_ANALYSIS_ENABLED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_wav2vec_model=(
            os.getenv(
                "TOEIC_PHONEME_WAV2VEC_MODEL",
                "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
            )
            or "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ),
        phoneme_device=os.getenv("TOEIC_PHONEME_DEVICE", "cpu") or "cpu",
        phoneme_confidence_threshold=float(
            os.getenv("TOEIC_PHONEME_CONFIDENCE_THRESHOLD", "0.1")
        ),
        phoneme_min_duration_sec=float(
            os.getenv("TOEIC_PHONEME_MIN_DURATION_SEC", "0.1")
        ),
        log_prompts=(
            os.getenv("TOEIC_LOG_PROMPTS", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
    )

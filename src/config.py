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

from .rubrics import EXAM_REGISTRIES
from .rubrics.base import Exam

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
    # Kỳ thi mặc định khi request/CLI không nêu rõ ("toeic" | "ielts"). Validate
    # trong load_config() theo EXAM_REGISTRIES.
    default_exam: str = Exam.TOEIC.value
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
    # ASR engine + model theo user mode (tách rời business mode khỏi engine):
    # - practice: lane nhanh, có thể tự leo lên mock_test khi tín hiệu kém.
    # - mock_test: lane chấm chi tiết (engine tốt nhất + phoneme).
    # Model rỗng → fallback về whisper_model (WHISPER_MODEL) để container cũ
    # chỉ đặt WHISPER_MODEL vẫn chạy, không bị âm thầm nâng lên model nặng.
    asr_engine_practice: str = "faster_whisper"
    asr_engine_mock_test: str = "whisperx"
    asr_model_practice: str = "base"
    asr_model_mock_test: str = "base"
    # Model HF cho adapter Insanely Fast Whisper (transformers pipeline) — engine
    # tuỳ chọn, giữ lại để có thể trỏ asr_engine_* sang insanely_fast_whisper.
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
    # Số từ tối đa hiển thị trong phoneme word-detail. Cắt theo ranh giới từ để
    # tránh payload quá lớn với bài dài. Đặt qua TOEIC_PHONEME_MAX_WORDS.
    phoneme_max_words: int = 200
    # Confidence knee: lỗi sub của phoneme có confidence < knee bị hạ penalty (recognizer
    # không chắc → ít khả năng lỗi người đọc). Đặt qua TOEIC_PHONEME_CONFIDENCE_KNEE.
    phoneme_confidence_knee: float = 0.5
    # Ngưỡng SequenceMatcher.ratio để coi 1 từ ASR-nghe-nhầm là "lệch lớn" → bỏ qua
    # không chấm phoneme (vd Son Tinh→Andy). Ratio cao (mountains→mountain) vẫn chấm.
    # Đặt qua TOEIC_PHONEME_SKIP_RATIO.
    phoneme_skip_ratio: float = 0.6
    # Telemetry phoneme (PR2): ghi per-word diagnostic JSONL để hiệu chỉnh trên golden
    # corpus. Chỉ quan sát, KHÔNG ảnh hưởng điểm. Bật qua TOEIC_PHONEME_TELEMETRY=1.
    phoneme_telemetry_enabled: bool = False
    phoneme_telemetry_path: str = "outputs/phoneme_telemetry.jsonl"
    # L1-aware scoring layer (Vietnamese). Mặc định TẮT → điểm giữ nguyên cho tới khi
    # validate trên golden corpus. Bật qua TOEIC_PHONEME_L1_ENABLED. Giảm penalty nuốt
    # phụ âm cuối kiểu L1 + trung hoà sub confidence rất thấp; vẫn hiển thị (accent note),
    # KHÔNG skip (Recognition Reliability mới được skip).
    phoneme_l1_enabled: bool = False
    phoneme_l1_language: str = "vi"
    phoneme_l1_min_confidence: float = 0.70
    phoneme_l1_low_conf_floor: float = 0.40
    # Log prompts and AI responses to outputs/prompt_logs/ for debugging.
    # Enable with TOEIC_LOG_PROMPTS=1.
    log_prompts: bool = False
    # Số bài chấm song song trong /grade-batch. 0 = tự chọn (1 cho local, 4 cho
    # cloud). Với local có thể đặt 2-3: ASR (Whisper) đã được serialize bằng lock
    # riêng nên Whisper vẫn chạy 1 lúc/lần, phần chồng lấn là tầng LLM của bài đã
    # xong ASR với ASR của bài kế. Đặt qua TOEIC_BATCH_CONCURRENCY.
    batch_concurrency: int = 0
    # Bật prefix caching phía server local (llama.cpp): gửi cache_prompt=true để
    # tái dùng KV-cache của phần system prompt (rubric) — giống nhau giữa mọi bài
    # cùng đề trong batch. Server không hỗ trợ sẽ bỏ qua key này. Tắt bằng
    # TOEIC_LOCAL_PREFIX_CACHE=false. (Không ảnh hưởng backend Anthropic.)
    local_prefix_cache: bool = True

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def is_local(self) -> bool:
        return self.backend == "local"


def load_config() -> Config:
    # Model ASR theo mode: lấy env riêng, rỗng → fallback WHISPER_MODEL chung.
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    # Kỳ thi mặc định: tên env mới (đúng nghĩa multi-exam) → fallback tên cũ → "toeic".
    default_exam = (
        os.getenv("SPEAKING_GRADER_DEFAULT_EXAM")
        or os.getenv("TOEIC_DEFAULT_EXAM")
        or Exam.TOEIC.value
    ).strip().lower()
    if default_exam not in EXAM_REGISTRIES:
        raise ValueError(
            f"default_exam không hợp lệ: '{default_exam}'. "
            f"Hợp lệ: {sorted(EXAM_REGISTRIES)} "
            f"(đặt qua SPEAKING_GRADER_DEFAULT_EXAM)."
        )
    return Config(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("TOEIC_MODEL", "claude-sonnet-4-6"),
        whisper_model=whisper_model,
        # auto: ưu tiên CUDA khi môi trường torch hỗ trợ, fallback CPU. Chấp nhận
        # 'cuda:N' để ghim Whisper vào 1 GPU cụ thể (vd WHISPER_DEVICE=cuda:0 còn
        # TOEIC_PHONEME_DEVICE=cuda:1 → ASR và wav2vec ở 2 card khác nhau).
        whisper_device=os.getenv("WHISPER_DEVICE", "auto"),
        backend=(os.getenv("TOEIC_BACKEND", "anthropic") or "anthropic").lower(),
        default_exam=default_exam,
        local_base_url=os.getenv("TOEIC_LOCAL_BASE_URL", "http://localhost:8080/v1"),
        local_model=os.getenv("TOEIC_LOCAL_MODEL", "qwen3"),
        local_api_key=os.getenv("TOEIC_LOCAL_API_KEY", "no-key"),
        local_enable_thinking=(
            os.getenv("TOEIC_LOCAL_ENABLE_THINKING", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        feedback_lang=os.getenv("TOEIC_FEEDBACK_LANG", "vi") or "vi",
        max_tokens=int(os.getenv("TOEIC_MAX_TOKENS", "30000")),
        asr_engine_practice=(
            os.getenv("TOEIC_ASR_ENGINE_PRACTICE", "faster_whisper")
            or "faster_whisper"
        ).lower(),
        asr_engine_mock_test=(
            os.getenv("TOEIC_ASR_ENGINE_MOCK_TEST", "whisperx")
            or "whisperx"
        ).lower(),
        asr_model_practice=(
            os.getenv("TOEIC_ASR_MODEL_PRACTICE") or whisper_model
        ),
        asr_model_mock_test=(
            os.getenv("TOEIC_ASR_MODEL_MOCK_TEST") or whisper_model
        ),
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
        # 'cpu' | 'cuda' | 'cuda:N'. Đặt 'cuda:1' để wav2vec chạy GPU thứ 2, tách
        # khỏi Whisper (WHISPER_DEVICE=cuda:0) → tận dụng 2 card song song khi chấm batch.
        phoneme_device=os.getenv("TOEIC_PHONEME_DEVICE", "cpu") or "cpu",
        phoneme_confidence_threshold=float(
            os.getenv("TOEIC_PHONEME_CONFIDENCE_THRESHOLD", "0.1")
        ),
        phoneme_min_duration_sec=float(
            os.getenv("TOEIC_PHONEME_MIN_DURATION_SEC", "0.1")
        ),
        phoneme_max_words=int(os.getenv("TOEIC_PHONEME_MAX_WORDS", "200")),
        phoneme_confidence_knee=float(
            os.getenv("TOEIC_PHONEME_CONFIDENCE_KNEE", "0.5")
        ),
        phoneme_skip_ratio=float(os.getenv("TOEIC_PHONEME_SKIP_RATIO", "0.6")),
        phoneme_telemetry_enabled=(
            os.getenv("TOEIC_PHONEME_TELEMETRY", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_telemetry_path=(
            os.getenv("TOEIC_PHONEME_TELEMETRY_PATH", "outputs/phoneme_telemetry.jsonl")
        ),
        phoneme_l1_enabled=(
            os.getenv("TOEIC_PHONEME_L1_ENABLED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_l1_language=os.getenv("TOEIC_PHONEME_L1_LANGUAGE", "vi") or "vi",
        phoneme_l1_min_confidence=float(
            os.getenv("TOEIC_PHONEME_L1_MIN_CONFIDENCE", "0.70")
        ),
        phoneme_l1_low_conf_floor=float(
            os.getenv("TOEIC_PHONEME_L1_LOW_CONF_FLOOR", "0.40")
        ),
        log_prompts=(
            os.getenv("TOEIC_LOG_PROMPTS", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        batch_concurrency=int(os.getenv("TOEIC_BATCH_CONCURRENCY", "0") or "0"),
        local_prefix_cache=(
            os.getenv("TOEIC_LOCAL_PREFIX_CACHE", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
    )

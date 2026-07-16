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
    # Backend local CÓ vision (llama.cpp chạy kèm --mmproj, vd Qwen-VL): cho phép
    # bước bóc tách đề (/exam/import) gửi ẢNH trang cho model kể cả khi tài liệu đã
    # có text-layer → đọc được tranh/bảng/đề scan. Mặc định TẮT vì model local
    # text-thuần (không mmproj) sẽ LỖI nếu nhận ảnh. Bật qua TOEIC_LOCAL_VISION_EXTRACT.
    local_vision_extract: bool = False
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
    # Chấm nói TIẾNG HÀN (TOPIK/practice lang=ko). Default OFF — lang=ko khi flag
    # tắt → HTTP 400. Bật qua TOEIC_LANG_KO_ENABLED sau khi bench M2 chốt model.
    lang_ko_enabled: bool = False
    # Acoustic model cho tiếng Hàn — mặc định dùng CHUNG xlsr-espeak (đa ngôn ngữ,
    # emit IPA); bench M2 quyết định có đổi sang model Korean-phone riêng không.
    # Model cache key theo model_id:device nên EN/KO coexist không tốn gì thêm.
    # Bench 2026-07-16 (data/bench/ko/BENCH_REPORT.md): phone-mfa thắng rõ espeak
    # trên cả TTS lẫn người thật (native acc 0.99/0.96 vs 0.89/0.87, false-error
    # ~10×/3× ít hơn, chấm được âm căng).
    phoneme_wav2vec_model_ko: str = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"
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
    # L1 vi→ko (M5): tolerance chuyển di cho người VIỆT nói tiếng HÀN (tense→plain,
    # ʌ↔o, coda l→n/nuốt — xem src/phoneme/l1/vi_ko.py). Flag RIÊNG với
    # phoneme_l1_enabled (bảng vi→en): default OFF, chỉ bật sau khi tune multiplier
    # bằng telemetry học viên thật. Bật qua TOEIC_PHONEME_L1_KO_ENABLED.
    phoneme_l1_ko_enabled: bool = False
    # Recognizer-noise gate (ĐỘC LẬP với L1): sub bất khả thi về âm học (sim < _sim &
    # không nằm trong _REAL_ERROR_SUBS) + recognizer không chắc (conf < ngưỡng theo loại
    # âm) → coi là wav2vec hallucinate → ẩn khỏi đỏ + bỏ penalty. Ngưỡng conf TÁCH nguyên
    # âm/phụ âm vì nguyên âm vốn confidence thấp hơn (hiệu chỉnh từ telemetry). Đặt conf=0
    # để TẮT. Đặt qua TOEIC_PHONEME_RECOGNIZER_NOISE_SIM/CONF/CONF_VOWEL.
    phoneme_recognizer_noise_sim: float = 0.2
    phoneme_recognizer_noise_conf: float = 0.6
    phoneme_recognizer_noise_conf_vowel: float = 0.45
    # Free-speech ASR-confidence gate: khi KHÔNG có script (reference == transcript),
    # từ mà chính Whisper không chắc (word probability < ngưỡng) → reference không
    # đáng tin → bỏ qua không chấm phoneme (skip khỏi cả tử lẫn mẫu accuracy).
    # faster-whisper: từ đúng thường ≥0.85, từ đoán mò (tên riêng nghe nhầm) ~0.2-0.6
    # → 0.55 cắt đuôi dưới. whisperx dùng thang khác (wav2vec char-align, thấp hơn)
    # → ngưỡng riêng. Đặt <=0 để TẮT. Qua TOEIC_PHONEME_ASR_CONF_MIN(_WHISPERX).
    phoneme_asr_conf_min: float = 0.55
    phoneme_asr_conf_min_whisperx: float = 0.40
    # Chấp nhận nuốt stop cuối từ khi nối từ (connected speech): "test preparation"
    # → /tes-prep/ là phát âm bản xứ đúng, không tính lỗi. Tắt (false) → hành vi cũ.
    # Đặt qua TOEIC_PHONEME_CONNECTED_SPEECH.
    phoneme_connected_speech_enabled: bool = True
    # Coverage gate (Track A): từ bị "del" 100% + wav2vec im lặng trong Whisper window
    # + Whisper word prob ≥ min_asr_prob → cap penalty (severity "low", coverage_collapse).
    # Default OFF = bit-for-bit như cũ. Qua TOEIC_PHONEME_COVERAGE_GATE_ENABLED/CAP/
    # MAX_LEN/MIN_ASR_PROB. Xem scoring/constants.py cho rationale + số telemetry.
    phoneme_coverage_gate_enabled: bool = False
    phoneme_coverage_gate_cap: float = 0.2
    phoneme_coverage_gate_max_len: int = 4
    phoneme_coverage_gate_min_asr_prob: float = 0.60
    # whisperx dùng alignment score (wav2vec char-align) — thang THẤP hơn hẳn logprob
    # của faster-whisper → ngưỡng riêng, cùng lý do phoneme_asr_conf_min_whisperx.
    phoneme_coverage_gate_min_asr_prob_whisperx: float = 0.40
    # Drift cap (Track B): sub có predicted segment NGOÀI window Whisper của từ (±pad)
    # → nghi DTW mượn âm từ kế → cap penalty (severity "low", drift_suspected).
    # Default OFF. Qua TOEIC_PHONEME_DRIFT_CAP_ENABLED/SUB_CAP/WINDOW_PAD.
    phoneme_drift_cap_enabled: bool = False
    phoneme_drift_sub_cap: float = 0.2
    phoneme_drift_window_pad: float = 0.08
    # Boundary refinement: segment bị DTW gán nhầm sang từ kề (bleed biên — case
    # "our eyes" → "eyes" /z z/) được re-pair về đúng từ TRÊN path trước khi chấm
    # → display + score + playback nhất quán (xem alignment._refine_boundary_bleed).
    # Default OFF = bit-for-bit như cũ. Qua TOEIC_PHONEME_BOUNDARY_REFINE.
    phoneme_boundary_refine_enabled: bool = False
    # Multi-reference homograph: từ đa-entry CMUdict có cửa sổ Whisper → chọn LẠI
    # entry khớp acoustic nhất trước DTW (fix "project" luôn bị so với dạng động từ
    # /prədʒekt/; blast radius: 6,068 từ material — xem scoring/homograph.py +
    # scripts/analyze_homographs.py). Default OFF = bit-for-bit như cũ.
    # Qua TOEIC_PHONEME_MULTIREF.
    phoneme_homograph_multiref: bool = False
    # S-cluster leniency: /p t k/ sau /s/ đầu từ (speak, stay, school) là âm KHÔNG bật
    # hơi → wav2vec hay gán nhầm chỗ cấu âm (sp→st) hoặc voicing (p→b). 2 bậc: voiced
    # cùng chỗ (p→b/t→d/k→ɡ) = "ok" (s_cluster_variant); plosive khác chỗ = cap penalty
    # 0.1 → severity "low" (s_cluster_unaspirated). Default OFF = bit-for-bit như cũ.
    # Qua TOEIC_PHONEME_S_CLUSTER.
    phoneme_s_cluster_enabled: bool = False
    # Recognizer-collapse gate: mở rộng coverage gate cho collapse TỪNG PHẦN — cap del/
    # sub bị wav2vec CTC blank-collapse (âm THAM CHIẾU có mass posterior ≥ floor nhưng
    # argmax=<pad> trong Whisper window) về COVERAGE_COLLAPSE. Cần deletion-evidence
    # posteriors (ON) + word_windows/probs. Default OFF = bit-for-bit như cũ.
    # Qua TOEIC_PHONEME_COLLAPSE_GATE_ENABLED. Xem scoring/word_details.py.
    phoneme_collapse_gate_enabled: bool = False
    # Deletion-evidence probe (SHADOW): giữ frame posteriors wav2vec để đo bằng chứng
    # âm học cho mỗi âm bị thiếu (phân biệt "thiếu âm thật" vs "recognizer hallucinate
    # deletion"). CHỈ telemetry/log — KHÔNG BAO GIỜ đổi điểm → default ON an toàn.
    # Tắt (false) để tiết kiệm RAM. Qua TOEIC_PHONEME_DELETION_EVIDENCE.
    phoneme_deletion_evidence_enabled: bool = True
    # Chunk audio TRƯỚC wav2vec theo Whisper word timestamps — fix IPA "lem" trên
    # audio dài (model suy giảm khi nhận cả bài trong 1 forward pass; xem
    # phoneme/chunking.py). "off" (default, bit-for-bit như cũ) | "pause" (cắt tại
    # khoảng lặng ≥ min_pause) | "hybrid" (pause → ranh giới câu → hard-cut khi chunk
    # vượt max). KHÔNG đổi scoring. Qua TOEIC_PHONEME_CHUNKING/CHUNK_MAX_SEC/
    # CHUNK_MIN_PAUSE_SEC/CHUNK_PAD_SEC.
    phoneme_chunking_strategy: str = "off"
    phoneme_chunk_max_sec: float = 30.0
    phoneme_chunk_min_pause_sec: float = 0.5
    phoneme_chunk_pad_sec: float = 0.25
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
    # ── TTS "nghe phát âm đúng" (Piper) ──────────────────────────────────
    # Đường dẫn file voice .onnx (file config .onnx.json cùng tên đặt cạnh nó).
    # Rỗng → endpoint /tts trả 503 (tính năng tắt). Voice tải tách khỏi pip để
    # không phình cài đặt mặc định — xem README. Đặt qua TTS_VOICE_US / TTS_VOICE_GB.
    tts_voice_us: str = ""
    tts_voice_gb: str = ""
    # accent 'default'/'auto' (client) map sang giọng này. Mặc định US vì IPA tham
    # chiếu (g2p_en/CMUdict) là giọng Mỹ → nhất quán với IPA hiển thị. KHÔNG phải
    # fallback tuỳ tiện. Đặt qua TTS_DEFAULT_ACCENT (us | gb).
    tts_default_accent: str = "us"
    # Thư mục cache WAV đã tổng hợp. Key cache có version (xem src/tts.py:CACHE_VERSION)
    # → đổi voice/normalization tự "miss" không cần xoá tay. Đặt qua TTS_CACHE_DIR.
    tts_cache_dir: str = "outputs/tts_cache"
    # ── CORS (gọi API từ origin khác) ────────────────────────────────────
    # Danh sách origin được phép gọi API qua trình duyệt, ngăn cách bằng dấu
    # phẩy (vd "https://app.example.com,https://foo.bar"). Mặc định "*" = mọi
    # origin (tiện expose API ra ngoài). Khi để "*", trình duyệt CẤM kèm
    # credentials (cookie) nên allow_credentials tự tắt — API này không dùng
    # cookie nên không ảnh hưởng. Đặt qua CORS_ALLOW_ORIGINS.
    cors_allow_origins: str = "*"
    # ── Lịch sử chấm bài (per-user, SQLite + audio trên đĩa) ─────────────
    # Bật/tắt toàn bộ tính năng lưu lịch sử. Tắt → các endpoint /history trả 404
    # và mọi request chấm bỏ qua việc lưu (kể cả khi client gửi user_id).
    history_enabled: bool = True
    # File SQLite (WAL) chứa metadata + result JSON. Docker mount ./data/history
    # và override 2 path này để dữ liệu sống qua rebuild (xem docker-compose.yml).
    history_db_path: str = "data/history.db"
    # Thư mục chứa audio gốc của từng bản ghi: {record_id}/audio{suffix}.
    history_audio_dir: str = "data/history_audio"
    # Quota: giữ tối đa N bản ghi mới nhất mỗi user, bản cũ hơn bị xoá (kèm audio)
    # sau mỗi lần lưu. 0 = không giới hạn.
    history_max_records_per_user: int = 1000
    # Từ đã lưu để luyện tập (per-user) + cache định nghĩa LLM — DB file riêng
    # (không đụng schema versioning của history.db). Xem src/words.py.
    words_db_path: str = "data/words.db"
    # Tài khoản đăng nhập (users + sessions) — DB file RIÊNG. QUAN TRỌNG: trong
    # Docker phải trỏ vào subtree đã mount (data/history/) để tài khoản sống qua
    # rebuild; xem docker-compose.yml (TOEIC_AUTH_DB_PATH). Xem src/auth.py.
    auth_db_path: str = "data/auth.db"
    # OAuth Client ID cho "Đăng nhập với Google" (dùng chung với app khác cùng
    # project GCP → cùng tài khoản Google = cùng người ở các app). Rỗng = tắt nút
    # Google trên UI. CLIENT_ID là thông tin CÔNG KHAI (không phải secret); flow
    # id_token phía web KHÔNG cần client secret. Xem /auth/google trong api.py.
    google_client_id: str = ""
    # ── Admission control cho các endpoint chấm bài ──────────────────────
    # Giới hạn số bài chấm ĐỒNG THỜI mỗi uvicorn worker (Dockerfile chạy 2 worker
    # → tổng = 2×N). ASR đã serialize bằng lock riêng nên slot >1 chỉ chồng lấn
    # tầng LLM/feature; giữ ≤~10 để không cạn threadpool anyio (40 thread/process).
    # 0 = TẮT admission control (hành vi cũ: nhận không giới hạn → quá tải là
    # request treo). Đặt qua TOEIC_GRADE_CONCURRENCY.
    grade_concurrency: int = 4
    # Số request được XẾP HÀNG chờ slot mỗi worker; vượt → 429 ngay (chặn
    # retry-storm). 2×(4+40)=88 > 50 học viên đồng thời → bình thường không ai
    # bị 429, hàng đợi chỉ sắp thứ tự. Qua TOEIC_GRADE_QUEUE_MAX.
    grade_queue_max: int = 40
    # Chờ trong hàng quá N giây → 429 (chỉ shed load không thể phục vụ sớm;
    # worst-case chờ thật ≈ 40 bài × 15s / 4 slot = 150s). Qua TOEIC_GRADE_QUEUE_TIMEOUT.
    grade_queue_timeout_sec: float = 180.0
    # Giá trị header Retry-After khi trả 429 (client tự thêm jitter).
    # Qua TOEIC_GRADE_RETRY_AFTER.
    grade_retry_after_sec: int = 15
    # ── Backend "openrouter" (TOEIC_BACKEND=openrouter) ──────────────────
    # Chấm điểm qua OpenRouter (OpenAI-compatible, model trả phí) để giải phóng
    # CPU/GPU máy host cho ASR; llama.cpp local giữ làm FALLBACK khi OpenRouter
    # lỗi/timeout — không học viên nào bị kẹt vì dịch vụ ngoài. ĐỔI MODEL CHẤM
    # = ĐIỂM THAY ĐỔI → bắt buộc bench (scripts/bench_llm_scoring.py) trước khi
    # flip .env sang backend này.
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Model do bench quyết định (vd "anthropic/claude-haiku-4.5") — không hardcode.
    openrouter_model: str = ""
    # Trần completion RIÊNG: nhiều provider reject max_tokens quá lớn — KHÔNG
    # dùng chung TOEIC_MAX_TOKENS (đặt 180k cho thinking budget của local).
    openrouter_max_tokens: int = 8000
    # Model flash-class trả JSON chấm điểm trong 5-20s; 90s đủ cover đuôi chậm.
    openrouter_timeout_sec: float = 90.0
    # OpenRouter lỗi → tự chạy lại bằng backend local (config local_* sẵn có).
    openrouter_fallback_local: bool = True
    # Reasoning tokens: "none" (tắt — rẻ + nhanh, JSON không bị chờ think dài),
    # "low"/"medium"/"high" (effort), "" = để model tự quyết.
    openrouter_reasoning: str = "none"
    # require_parameters: chỉ route tới provider hỗ trợ ĐỦ mọi tham số request
    # (đặc biệt structured_outputs cho json_schema strict). Model free thường
    # thiếu capability → 404 "No endpoints found"; tắt (false) để test model
    # free, nhưng provider có thể lờ schema → JSON sai → fallback local.
    openrouter_require_parameters: bool = True
    # "deny" (production: provider không được giữ transcript học viên để train)
    # | "allow" (endpoint free thường THU THẬP data — bắt buộc allow mới route
    # được; chỉ dùng khi test với audio không nhạy cảm).
    openrouter_data_collection: str = "deny"
    # Bóc tách đề (vision) qua OpenRouter chỉ khi bật — model chấm điểm đã bench
    # có thể KHÔNG có vision; default off = exam import vẫn đi local như prod.
    openrouter_vision_extract: bool = False

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse cors_allow_origins (CSV) → list origin đã strip, bỏ rỗng."""
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def is_local(self) -> bool:
        return self.backend == "local"

    @property
    def is_openrouter(self) -> bool:
        return self.backend == "openrouter"


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
    backend = (os.getenv("TOEIC_BACKEND", "anthropic") or "anthropic").lower()
    if backend not in {"anthropic", "local", "openrouter"}:
        raise ValueError(
            f"TOEIC_BACKEND không hợp lệ: '{backend}'. "
            "Hợp lệ: anthropic | local | openrouter."
        )
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or None
    openrouter_model = (os.getenv("TOEIC_OPENROUTER_MODEL", "") or "").strip()
    if backend == "openrouter":
        # Fail fast lúc khởi động thay vì 500 ở bài chấm đầu tiên.
        if not openrouter_api_key:
            raise ValueError(
                "TOEIC_BACKEND=openrouter cần OPENROUTER_API_KEY trong .env."
            )
        if not openrouter_model:
            raise ValueError(
                "TOEIC_BACKEND=openrouter cần TOEIC_OPENROUTER_MODEL "
                "(vd 'anthropic/claude-haiku-4.5' — chọn qua bench)."
            )
    return Config(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        model=os.getenv("TOEIC_MODEL", "claude-sonnet-4-6"),
        whisper_model=whisper_model,
        # auto: ưu tiên CUDA khi môi trường torch hỗ trợ, fallback CPU. Chấp nhận
        # 'cuda:N' để ghim Whisper vào 1 GPU cụ thể (vd WHISPER_DEVICE=cuda:0 còn
        # TOEIC_PHONEME_DEVICE=cuda:1 → ASR và wav2vec ở 2 card khác nhau).
        whisper_device=os.getenv("WHISPER_DEVICE", "auto"),
        backend=backend,
        default_exam=default_exam,
        local_base_url=os.getenv("TOEIC_LOCAL_BASE_URL", "http://localhost:8080/v1"),
        local_model=os.getenv("TOEIC_LOCAL_MODEL", "qwen3"),
        local_api_key=os.getenv("TOEIC_LOCAL_API_KEY", "no-key"),
        local_enable_thinking=(
            os.getenv("TOEIC_LOCAL_ENABLE_THINKING", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        local_vision_extract=(
            os.getenv("TOEIC_LOCAL_VISION_EXTRACT", "false") or "false"
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
        lang_ko_enabled=(
            os.getenv("TOEIC_LANG_KO_ENABLED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_wav2vec_model_ko=(
            os.getenv(
                "TOEIC_PHONEME_WAV2VEC_MODEL_KO",
                "slplab/wav2vec2-xls-r-300m_phone-mfa_korean",
            )
            or "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"
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
        phoneme_l1_ko_enabled=(
            os.getenv("TOEIC_PHONEME_L1_KO_ENABLED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_recognizer_noise_sim=float(
            os.getenv("TOEIC_PHONEME_RECOGNIZER_NOISE_SIM", "0.2")
        ),
        phoneme_recognizer_noise_conf=float(
            os.getenv("TOEIC_PHONEME_RECOGNIZER_NOISE_CONF", "0.6")
        ),
        phoneme_recognizer_noise_conf_vowel=float(
            os.getenv("TOEIC_PHONEME_RECOGNIZER_NOISE_CONF_VOWEL", "0.45")
        ),
        phoneme_asr_conf_min=float(
            os.getenv("TOEIC_PHONEME_ASR_CONF_MIN", "0.55")
        ),
        phoneme_asr_conf_min_whisperx=float(
            os.getenv("TOEIC_PHONEME_ASR_CONF_MIN_WHISPERX", "0.40")
        ),
        phoneme_connected_speech_enabled=(
            os.getenv("TOEIC_PHONEME_CONNECTED_SPEECH", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_coverage_gate_enabled=(
            os.getenv("TOEIC_PHONEME_COVERAGE_GATE_ENABLED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_coverage_gate_cap=float(
            os.getenv("TOEIC_PHONEME_COVERAGE_GATE_CAP", "0.2")
        ),
        phoneme_coverage_gate_max_len=int(
            os.getenv("TOEIC_PHONEME_COVERAGE_GATE_MAX_LEN", "4")
        ),
        phoneme_coverage_gate_min_asr_prob=float(
            os.getenv("TOEIC_PHONEME_COVERAGE_GATE_MIN_ASR_PROB", "0.60")
        ),
        phoneme_coverage_gate_min_asr_prob_whisperx=float(
            os.getenv("TOEIC_PHONEME_COVERAGE_GATE_MIN_ASR_PROB_WHISPERX", "0.40")
        ),
        phoneme_drift_cap_enabled=(
            os.getenv("TOEIC_PHONEME_DRIFT_CAP_ENABLED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_drift_sub_cap=float(
            os.getenv("TOEIC_PHONEME_DRIFT_SUB_CAP", "0.2")
        ),
        phoneme_drift_window_pad=float(
            os.getenv("TOEIC_PHONEME_DRIFT_WINDOW_PAD", "0.08")
        ),
        phoneme_boundary_refine_enabled=(
            os.getenv("TOEIC_PHONEME_BOUNDARY_REFINE", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_homograph_multiref=(
            os.getenv("TOEIC_PHONEME_MULTIREF", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_s_cluster_enabled=(
            os.getenv("TOEIC_PHONEME_S_CLUSTER", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_collapse_gate_enabled=(
            os.getenv("TOEIC_PHONEME_COLLAPSE_GATE_ENABLED", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_deletion_evidence_enabled=(
            os.getenv("TOEIC_PHONEME_DELETION_EVIDENCE", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        phoneme_chunking_strategy=(
            os.getenv("TOEIC_PHONEME_CHUNKING", "off") or "off"
        ).strip().lower(),
        phoneme_chunk_max_sec=float(
            os.getenv("TOEIC_PHONEME_CHUNK_MAX_SEC", "30.0")
        ),
        phoneme_chunk_min_pause_sec=float(
            os.getenv("TOEIC_PHONEME_CHUNK_MIN_PAUSE_SEC", "0.5")
        ),
        phoneme_chunk_pad_sec=float(
            os.getenv("TOEIC_PHONEME_CHUNK_PAD_SEC", "0.25")
        ),
        log_prompts=(
            os.getenv("TOEIC_LOG_PROMPTS", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        batch_concurrency=int(os.getenv("TOEIC_BATCH_CONCURRENCY", "0") or "0"),
        local_prefix_cache=(
            os.getenv("TOEIC_LOCAL_PREFIX_CACHE", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        tts_voice_us=os.getenv("TTS_VOICE_US", "") or "",
        tts_voice_gb=os.getenv("TTS_VOICE_GB", "") or "",
        tts_default_accent=(
            os.getenv("TTS_DEFAULT_ACCENT", "us") or "us"
        ).strip().lower(),
        tts_cache_dir=(
            os.getenv("TTS_CACHE_DIR", "outputs/tts_cache") or "outputs/tts_cache"
        ),
        cors_allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*") or "*",
        history_enabled=(
            os.getenv("TOEIC_HISTORY_ENABLED", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        history_db_path=(
            os.getenv("TOEIC_HISTORY_DB_PATH", "data/history.db")
            or "data/history.db"
        ),
        history_audio_dir=(
            os.getenv("TOEIC_HISTORY_AUDIO_DIR", "data/history_audio")
            or "data/history_audio"
        ),
        history_max_records_per_user=int(
            os.getenv("TOEIC_HISTORY_MAX_RECORDS", "1000") or "1000"
        ),
        words_db_path=(
            os.getenv("TOEIC_WORDS_DB_PATH", "data/words.db") or "data/words.db"
        ),
        auth_db_path=(
            os.getenv("TOEIC_AUTH_DB_PATH", "data/auth.db") or "data/auth.db"
        ),
        google_client_id=(os.getenv("GOOGLE_CLIENT_ID", "") or "").strip(),
        grade_concurrency=int(os.getenv("TOEIC_GRADE_CONCURRENCY", "4") or "4"),
        grade_queue_max=int(os.getenv("TOEIC_GRADE_QUEUE_MAX", "40") or "40"),
        grade_queue_timeout_sec=float(
            os.getenv("TOEIC_GRADE_QUEUE_TIMEOUT", "180") or "180"
        ),
        grade_retry_after_sec=int(
            os.getenv("TOEIC_GRADE_RETRY_AFTER", "15") or "15"
        ),
        openrouter_api_key=openrouter_api_key,
        openrouter_base_url=(
            os.getenv("TOEIC_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            or "https://openrouter.ai/api/v1"
        ),
        openrouter_model=openrouter_model,
        openrouter_max_tokens=int(
            os.getenv("TOEIC_OPENROUTER_MAX_TOKENS", "8000") or "8000"
        ),
        openrouter_timeout_sec=float(
            os.getenv("TOEIC_OPENROUTER_TIMEOUT", "90") or "90"
        ),
        openrouter_fallback_local=(
            os.getenv("TOEIC_OPENROUTER_FALLBACK_LOCAL", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        openrouter_reasoning=(
            os.getenv("TOEIC_OPENROUTER_REASONING", "none") or ""
        ).strip().lower(),
        openrouter_require_parameters=(
            os.getenv("TOEIC_OPENROUTER_REQUIRE_PARAMETERS", "true") or "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        openrouter_data_collection=(
            os.getenv("TOEIC_OPENROUTER_DATA_COLLECTION", "deny") or "deny"
        ).strip().lower(),
        openrouter_vision_extract=(
            os.getenv("TOEIC_OPENROUTER_VISION_EXTRACT", "false") or "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
    )

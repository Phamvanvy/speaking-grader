# TOEIC Speaking Grader — image GPU (NVIDIA).
# GPU chạy được trên base slim vì:
#   - NVIDIA driver/libcuda do nvidia-container-runtime inject (khối `deploy` trong
#     docker-compose.yml) → GPU nhìn thấy trong container.
#   - cuBLAS/cuDNN đến từ wheel nvidia-*-cu12 trong requirements.txt.
# Vấn đề duy nhất: trên Linux, thư viện .so của các wheel đó nằm trong
# site-packages/nvidia/*/lib và KHÔNG nằm trên loader path → ctranslate2 báo
# "Library libcublas.so.12 is not found". Ta thêm chúng vào LD_LIBRARY_PATH bên dưới.
# (Hàm _register_cuda_dll_dirs trong src/asr.py chỉ xử lý Windows nên không giúp gì ở đây.)
FROM python:3.11-slim

# ffmpeg: faster-whisper đọc audio qua ffmpeg (bắt buộc khi chấm file thật).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài deps trước (tận dụng layer cache khi chỉ đổi source).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cho loader Linux tìm thấy libcublas.so.12 / libcudnn*.so.9 từ các wheel nvidia-*-cu12.
# (ctranslate2 nạp qua dlopen lúc encode; site-packages/nvidia/*/lib không nằm sẵn trên path.)
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# Prefetch model Whisper vào image → không phải tải lúc cold start.
# device=cuda (giá trị hợp lệ cho ctranslate2; "gpu" KHÔNG hợp lệ).
ENV WHISPER_MODEL=base \
    WHISPER_DEVICE=cuda
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base')"

# ── Phoneme analysis (wav2vec 2.0 — optional, Phase 1) ──────────────────────
# Model wav2vec cache tự động qua volume `whisper-cache` → /root/.cache/huggingface
# (kể cả docker compose up --build, volume giữ nguyên → không tải lại ~900MB model).
# Prefetch model wav2vec vào image (optional — comment nếu muốn image nhẹ hơn).
ENV PHONEME_WAV2VEC_MODEL=facebook/wav2vec2-lg-960h \
    TOEIC_PHONEME_ANALYSIS_ENABLED=false \
    TOEIC_WAV2VEC_DEVICE=cuda
# Prefetch: chỉ chạy nếu torch/transformers đã cài (sẽ fail nếu không — nên để
# sau COPY src && RUN pip install phoneme deps; hiện tại bỏ comment khi cần).
# RUN python -c "from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor; \
#     p = Wav2VecPhonemePredictor(); p._load_model()" 2>/dev/null || true

COPY src ./src

# Whisper KHÔNG an toàn đa luồng → 1 worker/process; scale bằng nhiều replica.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]

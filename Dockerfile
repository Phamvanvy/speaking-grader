# TOEIC Speaking Grader — image GPU (NVIDIA).
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
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# ❌ ĐÃ XÓA dòng RUN python -c tải trước model ở đây.
# Hãy để cơ chế Volume tự động lưu cache ở lần chạy đầu tiên trên máy local.

COPY src ./src
COPY web ./web

# ── ĐỒNG BỘ CẤU HÌNH VỚI FILE .ENV ─────────────────────────────────────────
ENV WHISPER_MODEL=large-v3-turbo \
    WHISPER_DEVICE=cuda \
    PHONEME_WAV2VEC_MODEL=facebook/wav2vec2-xlsr-53-espeak-cv-ft \
    TOEIC_PHONEME_ANALYSIS_ENABLED=true \
    TOEIC_PHONEME_DEVICE=cuda

# Tăng worker để tận dụng i5-14400F và 64GB RAM xử lý đa luồng gối đầu (Concurrency)
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers 2"]

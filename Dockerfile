# syntax=docker/dockerfile:1
# TOEIC Speaking Grader — image GPU (NVIDIA).

# ── Stage build frontend React (Vite) ──────────────────────────────────────
# Chuẩn bị cutover (M5): build ra web/dist với base '/' (production). Ở giai đoạn
# này production '/' VẪN serve legacy (COPY web ./web bên dưới) — dist chỉ để /beta
# dogfood và sẵn sàng cho lúc flip _WEB_DIR → web/dist. npm ci cache theo package-lock.
FROM node:20-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY frontend ./
# outDir = ../web/dist (vite.config.ts) → ghi vào /build/web/dist. base mặc định '/'.
RUN npm run build

# ── Stage runtime (Python GPU) ─────────────────────────────────────────────
FROM python:3.11-slim

# ffmpeg: faster-whisper đọc audio qua ffmpeg (bắt buộc khi chấm file thật).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg espeak-ng espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── ÉP CỨNG ĐƯỜNG DẪN CACHE VỀ THƯ MỤC SẼ MOUNT VOLUME ────────────────────
# Đảm bảo cả HuggingFace, Torch Hub và các thư viện khác ghi chung vào một chỗ
ENV HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    XDG_CACHE_HOME=/root/.cache

# Cài deps trước (tận dụng layer cache khi chỉ đổi source).
# --mount=type=cache: pip cache sống độc lập với layer cache của image, nên dù
# layer RUN này có bị Docker Desktop tự dọn (disk pressure) và phải chạy lại,
# pip vẫn lấy wheel từ cache mount thay vì tải lại từ PyPI.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Cho loader Linux tìm thấy libcublas.so.12 / libcudnn*.so.9 từ các wheel nvidia-*-cu12.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

COPY src ./src
COPY web ./web
# React build từ stage frontend-build. Hiện chỉ phục vụ /beta (dogfood) + sẵn cho
# cutover; khi flip _WEB_DIR → web/dist thì đây thành production. Xem src/api.py.
COPY --from=frontend-build /build/web/dist ./web/dist
# Ngân hàng câu hỏi + ảnh đề mẫu cho /exam/builtin ("dùng đề có sẵn"). Chỉ JSON câu
# hỏi và ảnh mẫu, các phần khác của data/ vẫn bị .dockerignore loại ra.
COPY data/questions ./data/questions
COPY data/image ./data/image

# ── ĐỒNG BỘ CẤU HÌNH VỚI FILE .ENV ─────────────────────────────────────────
ENV WHISPER_MODEL=large-v3-turbo \
    WHISPER_DEVICE=cuda \
    PHONEME_WAV2VEC_MODEL=facebook/wav2vec2-xlsr-53-espeak-cv-ft \
    TOEIC_PHONEME_ANALYSIS_ENABLED=true \
    TOEIC_PHONEME_DEVICE=cuda

# ── TTS "nghe phát âm đúng" (Piper) ─────────────────────────────────────────
# piper-tts đã cài từ requirements.txt; espeak-ng (Piper cần) đã có ở apt trên.
# Voice .onnx KHÔNG bake vào image (giữ image nhẹ) → mount qua volume vào /app/voices
# (xem docker-compose.yml). File vắng → /tts trả 503, phần còn lại app vẫn chạy.
# TTS_CACHE_DIR đặt dưới /root/.cache để WAV đã tổng hợp persist cùng volume cache model.
ENV TTS_VOICE_US=/app/voices/en_US-lessac-medium.onnx \
    TTS_VOICE_GB=/app/voices/en_GB-alan-medium.onnx \
    TTS_CACHE_DIR=/root/.cache/tts

# Tăng worker để tận dụng i5-14400F và 64GB RAM xử lý đa luồng gối đầu (Concurrency)
ENV PORT=8000
EXPOSE 8000
# --timeout-keep-alive 75 > idle keepalive mặc định của Caddy → tránh uvicorn
# đóng connection giữa lúc proxy tái dùng (502 lẻ tẻ khi tải cao).
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers 2 --timeout-keep-alive 75"]
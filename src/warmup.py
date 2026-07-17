"""Warmup model lúc startup: nạp sẵn ASR + wav2vec vào GPU thay vì đợi request đầu.

Không có warmup, request chấm đầu tiên sau `docker compose up` phải trả giá nạp
model (Whisper vài GB + wav2vec trên từng GPU) — chậm hàng chục giây tới vài
phút. Warmup nạp đúng những model mà config đang trỏ tới, vào ĐÚNG cache mà
đường request thật dùng, nên request đến trong lúc warmup chỉ chờ lock rồi hit
cache (không nạp trùng, không đổi kết quả chấm).

Bật bằng TOEIC_WARMUP_MODELS=true (docker-compose.yml đặt sẵn; local dev mặc
định tắt để `uvicorn --reload` không nạp model mỗi lần code đổi). Chạy trong
thread nền daemon: uvicorn bind port + /health sống ngay, model nạp dần phía
sau; mọi lỗi warmup chỉ log — model lỗi sẽ nạp lazy như cũ khi có request.
"""

from __future__ import annotations

import logging
import os
import threading
from time import perf_counter
from typing import Callable

from .config import Config

logger = logging.getLogger("toeic.warmup")


def warmup_enabled() -> bool:
    return (os.getenv("TOEIC_WARMUP_MODELS", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _warm_one(label: str, load: Callable[[], object]) -> None:
    """Nạp 1 model best-effort: lỗi chỉ log, KHÔNG hỏng startup/các model sau."""
    started = perf_counter()
    try:
        load()
        logger.info("Warmup %s xong (%.1fs).", label, perf_counter() - started)
    except Exception:  # noqa: BLE001
        logger.exception("Warmup %s lỗi (bỏ qua — sẽ nạp lazy khi có request).", label)


def run_warmup(config: Config) -> None:
    """Nạp tuần tự mọi model mà config hiện tại sẽ dùng khi chấm bài.

    Tuần tự (1 thread) là chủ đích: nạp song song nhiều model vài GB lên cùng
    GPU dễ tranh VRAM/đĩa — đúng lý do asr.py serialize bằng lock nạp.
    """
    from . import asr

    # ASR theo mode (practice / mock_test) — gộp trùng khi 2 mode cùng engine+model.
    seen: set[tuple[str, str]] = set()
    for backend, model in (
        (config.asr_engine_practice, config.asr_model_practice),
        (config.asr_engine_mock_test, config.asr_model_mock_test),
    ):
        if (backend, model) in seen:
            continue
        seen.add((backend, model))
        _warm_one(
            f"ASR {backend}:{model}",
            lambda b=backend, m=model: asr.warm_asr_backend(
                b, m, config.whisper_device
            ),
        )

    if not config.phoneme_analysis_enabled:
        return

    from .phoneme.wav2vec_backend import _get_wav2vec_model

    # Cùng danh sách device mà chunk-parallel dùng (TOEIC_PHONEME_DEVICES),
    # device chính luôn có mặt — khớp cách Wav2VecPhonemePredictor ghép list.
    devices = [d.strip() for d in config.phoneme_devices.split(",") if d.strip()]
    if config.phoneme_device not in devices:
        devices = [config.phoneme_device, *devices]

    model_ids = [config.phoneme_wav2vec_model]
    if config.lang_ko_enabled:
        model_ids.append(config.phoneme_wav2vec_model_ko)

    for model_id in model_ids:
        for device in devices:
            _warm_one(
                f"wav2vec {model_id} @ {device}",
                lambda m=model_id, d=device: _get_wav2vec_model(m, d),
            )

    logger.info("Warmup hoàn tất — model đã sẵn trên GPU, request đầu không chờ nạp.")


def start_background_warmup(config: Config) -> None:
    """Khởi động warmup trong thread nền nếu TOEIC_WARMUP_MODELS bật."""
    if not warmup_enabled():
        logger.info("Warmup model tắt (đặt TOEIC_WARMUP_MODELS=true để bật).")
        return
    threading.Thread(
        target=run_warmup, args=(config,), name="model-warmup", daemon=True
    ).start()

"""Admission control cho các endpoint chấm bài (chống nghẽn khi đông học viên).

Trước đây /grade nhận không giới hạn: 50 người bấm cùng lúc → 50 thread chấm
chen nhau GPU/threadpool → mọi request đều chậm dần rồi treo (proxy 502).
Giờ mỗi worker chỉ chấm `grade_concurrency` bài đồng thời; số còn lại XẾP HÀNG
(tối đa `grade_queue_max` request, chờ tối đa `grade_queue_timeout_sec` giây),
quá ngưỡng → 429 + Retry-After để frontend tự retry với backoff.

Trạng thái là per-process (uvicorn --workers 2 → tổng slot = 2×N). Semaphore
tạo LAZY khi request đầu tiên tới — lúc import module, event loop của uvicorn
chưa chạy.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

from fastapi import HTTPException

from .config import Config

logger = logging.getLogger("toeic.admission")

_sem: asyncio.Semaphore | None = None
_waiting = 0
_running = 0


def _get_sem(config: Config) -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(config.grade_concurrency)
    return _sem


def _reset_for_tests() -> None:
    """Xoá state module để mỗi test tạo semaphore mới theo config riêng."""
    global _sem, _waiting, _running
    _sem = None
    _waiting = 0
    _running = 0


def _too_busy(config: Config) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=(
            "Server đang chấm quá nhiều bài cùng lúc — vui lòng thử lại sau "
            f"~{config.grade_retry_after_sec}s."
        ),
        headers={"Retry-After": str(config.grade_retry_after_sec)},
    )


def admission_stats(config: Config) -> dict:
    """Snapshot cho /health: số bài đang chấm / đang chờ / sức chứa mỗi worker."""
    return {
        "running": _running,
        "waiting": _waiting,
        "capacity": config.grade_concurrency,
        "queue_max": config.grade_queue_max,
    }


@asynccontextmanager
async def admission_slot(
    config: Config, *, queue: bool = True
) -> AsyncIterator[int]:
    """Giữ 1 slot chấm bài; yield số ms đã chờ trong hàng.

    queue=True (request lẻ /grade, /suggest): hàng đầy hoặc chờ quá timeout
    → 429. queue=False (từng bài TRONG /grade-batch, /exam/grade): batch đã
    được nhận rồi — chờ vô hạn, không 429 giữa chừng để 1 câu không làm hỏng
    cả đề; semaphore local của batch đã chặn fan-out.

    grade_concurrency <= 0 → admission control TẮT (hành vi cũ).
    """
    global _waiting, _running
    if config.grade_concurrency <= 0:
        yield 0
        return

    sem = _get_sem(config)
    t0 = perf_counter()
    if queue and _waiting >= config.grade_queue_max:
        raise _too_busy(config)
    _waiting += 1
    try:
        if queue:
            try:
                await asyncio.wait_for(
                    sem.acquire(), timeout=config.grade_queue_timeout_sec
                )
            except asyncio.TimeoutError:
                raise _too_busy(config) from None
        else:
            await sem.acquire()
    finally:
        _waiting -= 1

    wait_ms = int((perf_counter() - t0) * 1000)
    _running += 1
    if wait_ms > 1000:
        logger.info(
            "Admission | queue_wait_ms=%d | running=%d | waiting=%d",
            wait_ms, _running, _waiting,
        )
    try:
        yield wait_ms
    finally:
        _running -= 1
        sem.release()

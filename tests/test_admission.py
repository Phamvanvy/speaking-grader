"""Test admission control (src/admission.py) — giới hạn tải endpoint chấm bài.

Test trực tiếp admission_slot bằng asyncio.run (không cần pytest-asyncio,
không cần dựng FastAPI app / load model).
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
from fastapi import HTTPException

from src import admission
from src.config import Config


def _config(**overrides) -> Config:
    base = Config(
        anthropic_api_key=None,
        model="claude-sonnet-4-6",
        whisper_model="base",
        whisper_device="cpu",
        backend="anthropic",
        local_base_url="http://localhost:8080/v1",
        local_model="qwen3",
        local_api_key="no-key",
        feedback_lang="vi",
        max_tokens=30000,
        asr_engine_practice="faster_whisper",
        asr_engine_mock_test="whisperx",
        asr_model_practice="large-v3-turbo",
        asr_model_mock_test="large-v3",
        insanely_fast_model_id="openai/whisper-small",
        auto_confidence_threshold=0.75,
        auto_silence_ratio_threshold=0.35,
        auto_coverage_threshold=0.8,
    )
    return dataclasses.replace(base, **overrides)


@pytest.fixture(autouse=True)
def _fresh_state():
    admission._reset_for_tests()
    yield
    admission._reset_for_tests()


def test_concurrency_bounded():
    """Không bao giờ có nhiều hơn grade_concurrency bài chạy đồng thời."""
    cfg = _config(grade_concurrency=2, grade_queue_max=10,
                  grade_queue_timeout_sec=5.0)
    running = 0
    max_running = 0

    async def _one():
        nonlocal running, max_running
        async with admission.admission_slot(cfg):
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.02)
            running -= 1

    async def _main():
        await asyncio.gather(*(_one() for _ in range(6)))

    asyncio.run(_main())
    assert max_running == 2


def test_queue_full_returns_429_with_retry_after():
    """Hàng đợi đầy → 429 ngay, kèm header Retry-After."""
    cfg = _config(grade_concurrency=1, grade_queue_max=1,
                  grade_queue_timeout_sec=5.0, grade_retry_after_sec=7)

    async def _main():
        holder_started = asyncio.Event()
        release = asyncio.Event()

        async def _holder():
            async with admission.admission_slot(cfg):
                holder_started.set()
                await release.wait()

        async def _waiter():
            async with admission.admission_slot(cfg):
                pass

        h = asyncio.create_task(_holder())
        await holder_started.wait()
        w = asyncio.create_task(_waiter())
        await asyncio.sleep(0.01)  # để _waiter chiếm chỗ chờ duy nhất

        # Slot bận + hàng chờ đầy (1/1) → request thứ 3 bị 429 tức thì.
        with pytest.raises(HTTPException) as exc:
            async with admission.admission_slot(cfg):
                pass
        assert exc.value.status_code == 429
        assert exc.value.headers["Retry-After"] == "7"

        release.set()
        await asyncio.gather(h, w)

    asyncio.run(_main())


def test_queue_timeout_returns_429():
    """Chờ trong hàng quá grade_queue_timeout_sec → 429."""
    cfg = _config(grade_concurrency=1, grade_queue_max=5,
                  grade_queue_timeout_sec=0.05, grade_retry_after_sec=3)

    async def _main():
        holder_started = asyncio.Event()
        release = asyncio.Event()

        async def _holder():
            async with admission.admission_slot(cfg):
                holder_started.set()
                await release.wait()

        h = asyncio.create_task(_holder())
        await holder_started.wait()

        with pytest.raises(HTTPException) as exc:
            async with admission.admission_slot(cfg):
                pass
        assert exc.value.status_code == 429

        release.set()
        await h

    asyncio.run(_main())


def test_batch_slot_waits_without_429():
    """queue=False (item trong batch): chờ qua cả queue-full lẫn timeout, không 429."""
    cfg = _config(grade_concurrency=1, grade_queue_max=0,
                  grade_queue_timeout_sec=0.01)

    async def _main():
        holder_started = asyncio.Event()
        done: list[str] = []

        async def _holder():
            async with admission.admission_slot(cfg, queue=False):
                holder_started.set()
                await asyncio.sleep(0.05)  # lâu hơn queue_timeout
                done.append("holder")

        async def _batch_item():
            await holder_started.wait()
            async with admission.admission_slot(cfg, queue=False):
                done.append("item")

        await asyncio.gather(_holder(), _batch_item())
        assert done == ["holder", "item"]

    asyncio.run(_main())


def test_disabled_passes_through():
    """grade_concurrency=0 → admission control tắt, không giới hạn gì."""
    cfg = _config(grade_concurrency=0, grade_queue_max=0,
                  grade_queue_timeout_sec=0.0)

    async def _main():
        async with admission.admission_slot(cfg) as wait_ms:
            assert wait_ms == 0

    asyncio.run(_main())
    assert admission._sem is None  # không tạo semaphore khi tắt


def test_queue_wait_ms_reported():
    """Thời gian chờ hàng đợi được trả về (đưa vào telemetry)."""
    cfg = _config(grade_concurrency=1, grade_queue_max=5,
                  grade_queue_timeout_sec=5.0)

    async def _main():
        holder_started = asyncio.Event()
        waited: list[int] = []

        async def _holder():
            async with admission.admission_slot(cfg):
                holder_started.set()
                await asyncio.sleep(0.05)

        async def _waiter():
            await holder_started.wait()
            async with admission.admission_slot(cfg) as wait_ms:
                waited.append(wait_ms)

        await asyncio.gather(_holder(), _waiter())
        assert waited and waited[0] >= 40  # chờ ~50ms (trừ hao scheduler)

    asyncio.run(_main())

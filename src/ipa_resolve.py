"""Cascade tra IPA theo yêu cầu (cache → CMUdict → Cambridge → eSpeak).

Ưu tiên (user-specified, exact):
  ── Cache ──
    Cambridge/CMUdict đã cache → trả (cache hit)
  ── MISS ──
    CMUdict (G2P sẵn có)
      ├── found:     lưu cache(source=cmudict) + ENQUEUE Cambridge warm (nền) → trả CMUdict NGAY
      └── not found: Cambridge fetch (ĐỒNG BỘ — lấp OOV) → success? lưu cache : eSpeak (fallback cuối)

Master flag TẮT (cfg.ipa_cache_enabled=False) → resolve_ipa() rơi về đúng
word_ipa_display() cũ (bit-for-bit, KHÔNG mạng/DB) — không regression.

An toàn đồng thời: khoá per-word (dogpile — N miss cùng từ chỉ fetch 1 lần) +
threading.Semaphore trần fetch Cambridge (lịch sự với dịch vụ ngoài). Warm chạy
nền qua asyncio.create_task ở luồng async; luồng sync (/words) bỏ warm-on-hit để
khỏi quản lý thread (Cambridge vẫn được thử ĐỒNG BỘ khi CMUdict miss).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

from fastapi.concurrency import run_in_threadpool

from . import ipa_cache
from .cambridge import fetch_cambridge
from .config import Config
from .ipa_cache import (
    CAMBRIDGE_ERROR,
    CAMBRIDGE_NOT_FOUND,
    CAMBRIDGE_SUCCESS,
    CAMBRIDGE_UNTRIED,
    IPACacheRow,
)
from .phoneme.ipa import (
    format_ipa_with_stress,
    place_stress_at_onset,
    word_ipa_display,
    word_to_ipa_with_stress_source,
)

logger = logging.getLogger("toeic.ipa_resolve")

# Nguồn G2P coi là "tra từ điển được" (nhánh CMUdict-found); espeak/failed = miss.
_DICT_SOURCES = frozenset({"cmudict", "override", "context"})


@dataclass
class IPAResult:
    word: str
    uk_ipa: str | None = None
    us_ipa: str | None = None
    source: str | None = None
    cached: bool = False
    # Chuỗi IPA hiển thị gọn (tương thích word_ipa_display) cho cột saved_words.ipa.
    display: str = ""

    def to_dict(self) -> dict:
        return {
            "uk_ipa": self.uk_ipa,
            "us_ipa": self.us_ipa,
            "ipa_source": self.source,
            "ipa_cached": self.cached,
        }


# ── helpers thuần ────────────────────────────────────────────────────────


def _normalize(word: str) -> str:
    return " ".join((word or "").split()).lower()


def _g2p(word: str) -> tuple[str, str]:
    """(display_ipa, source) từ pipeline G2P sẵn có. display rỗng nếu không tra được."""
    symbols, stress, source = word_to_ipa_with_stress_source(word)
    if not symbols:
        return "", source  # source == "failed"
    disp = format_ipa_with_stress(symbols, place_stress_at_onset(symbols, stress))
    return disp, source


def _row_to_result(row: IPACacheRow, *, cached: bool) -> IPAResult:
    display = row.us_ipa or row.uk_ipa or ""
    return IPAResult(
        word=row.word, uk_ipa=row.uk_ipa, us_ipa=row.us_ipa,
        source=row.source, cached=cached, display=display,
    )


def _display_only(word: str) -> IPAResult:
    """Đường degrade khi master flag TẮT: đúng hành vi word_ipa_display cũ."""
    disp = word_ipa_display(word)
    return IPAResult(word=word, us_ipa=disp or None, display=disp,
                     source="g2p" if disp else None, cached=False)


def _is_terminal_hit(row: IPACacheRow | None) -> bool:
    """Hàng cache có được coi là kết quả cuối (không cần tính lại/ fetch lại)?
    - ERROR (lỗi tạm thời Cambridge) → KHÔNG terminal (cho phép thử lại lần sau).
    - Có phiên âm → terminal.
    - Không phiên âm nhưng đã NOT_FOUND → terminal (negative cache).
    """
    if row is None:
        return False
    if row.cambridge_status == CAMBRIDGE_ERROR:
        return False
    if row.uk_ipa or row.us_ipa:
        return True
    return row.cambridge_status == CAMBRIDGE_NOT_FOUND


# ── trần fetch Cambridge đồng thời (threading — dùng chung sync-miss & warm) ──

_fetch_sem: threading.Semaphore | None = None
_fetch_sem_lock = threading.Lock()


def _get_fetch_sem(cfg: Config) -> threading.Semaphore:
    global _fetch_sem
    if _fetch_sem is None:
        with _fetch_sem_lock:
            if _fetch_sem is None:
                _fetch_sem = threading.Semaphore(max(1, cfg.ipa_max_concurrency))
    return _fetch_sem


def _fetch_guarded(word: str, cfg: Config):
    sem = _get_fetch_sem(cfg)
    with sem:
        return fetch_cambridge(word, cfg)


# ── cascade lõi (đồng bộ — chạy trong threadpool) ────────────────────────


def _resolve_core(word: str, cfg: Config) -> tuple[IPAResult, bool]:
    """Cascade đầy đủ cho 1 TỪ đã chuẩn hoá. Trả (result, needs_warm).
    needs_warm=True chỉ khi CMUdict-found tươi + Cambridge bật (caller lên lịch warm nền)."""
    row = ipa_cache.get(cfg, word)
    if _is_terminal_hit(row):
        return _row_to_result(row, cached=True), False

    disp, source = _g2p(word)

    if source in _DICT_SOURCES:
        # CMUdict-found → lưu ngay, đánh dấu Cambridge chưa thử, tín hiệu warm.
        new = IPACacheRow(word=word, us_ipa=disp or None, source=source,
                          cambridge_status=CAMBRIDGE_UNTRIED)
        ipa_cache.put(cfg, new)
        needs_warm = cfg.ipa_cambridge_enabled and source == "cmudict"
        return _row_to_result(new, cached=False), needs_warm

    # CMUdict MISS → Cambridge đồng bộ (trước eSpeak) nếu bật.
    if cfg.ipa_cambridge_enabled:
        res = _fetch_guarded(word, cfg)
        if res.status == "success" and res.entry:
            e = res.entry
            new = IPACacheRow(
                word=word, uk_ipa=e.uk_ipa, us_ipa=e.us_ipa,
                source="cambridge", cambridge_status=CAMBRIDGE_SUCCESS,
            )
            ipa_cache.put(cfg, new)
            return _row_to_result(new, cached=False), False
        # not_found / error → dùng eSpeak (disp) làm fallback cuối, nhớ trạng thái.
        status = CAMBRIDGE_NOT_FOUND if res.status == "not_found" else CAMBRIDGE_ERROR
        new = IPACacheRow(word=word, us_ipa=disp or None,
                          source=(source if disp else None),
                          cambridge_status=status)
        ipa_cache.put(cfg, new)
        return _row_to_result(new, cached=False), False

    # Cambridge tắt → eSpeak (disp có thể rỗng nếu 'failed'). Lưu để lần sau nhanh.
    new = IPACacheRow(word=word, us_ipa=disp or None,
                      source=(source if disp else None),
                      cambridge_status=CAMBRIDGE_UNTRIED)
    ipa_cache.put(cfg, new)
    return _row_to_result(new, cached=False), False


def _warm_sync(word: str, cfg: Config) -> None:
    """Nâng cache lên nguồn Cambridge (chạy nền sau khi đã trả CMUdict)."""
    res = _fetch_guarded(word, cfg)
    if res.status == "success" and res.entry:
        e = res.entry
        ipa_cache.put(cfg, IPACacheRow(
            word=word, uk_ipa=e.uk_ipa, us_ipa=e.us_ipa,
            source="cambridge", cambridge_status=CAMBRIDGE_SUCCESS,
        ))
    elif res.status == "not_found":
        ipa_cache.set_cambridge_status(cfg, word, CAMBRIDGE_NOT_FOUND)
    else:
        ipa_cache.set_cambridge_status(cfg, word, CAMBRIDGE_ERROR)


async def _warm(word: str, cfg: Config) -> None:
    try:
        await run_in_threadpool(_warm_sync, word, cfg)
    except Exception:  # noqa: BLE001 - warm nền không được làm sập request
        logger.exception("ipa warm lỗi word=%r", word)


# ── khoá per-word (dogpile) cho luồng async ──────────────────────────────

_locks: dict[str, asyncio.Lock] = {}
_locks_guard = asyncio.Lock()


async def _get_lock(word: str) -> asyncio.Lock:
    async with _locks_guard:
        lock = _locks.get(word)
        if lock is None:
            lock = asyncio.Lock()
            _locks[word] = lock
        return lock


# ── API công khai ────────────────────────────────────────────────────────


async def resolve_ipa(
    word: str, cfg: Config, *, wait_cambridge: bool = False
) -> IPAResult:
    """Tra IPA đầy đủ (uk/us + audio) cho 1 từ — dùng ở endpoint async (/word-info).

    Master flag tắt → degrade word_ipa_display. Cụm nhiều từ → chỉ ghép display
    theo token (không Cambridge/audio). Đồng thời-an toàn qua khoá per-word.

    `wait_cambridge`: khi CMUdict-found tươi (chỉ có us_ipa), CHỜ Cambridge ĐỒNG BỘ
    để có luôn uk_ipa ngay lần gọi đầu (thay vì warm nền rồi phải gọi lại). Dùng cho
    /word-info: popup hiện cả UK/US ngay lần mở đầu. Đánh đổi ~vài trăm ms cho TỪ MỚI
    (từ đã cache Cambridge = tức thì). KHÔNG áp cụm từ (Cambridge demand-driven 1 từ).
    """
    word = _normalize(word)
    if not word:
        return IPAResult(word="")
    if not cfg.ipa_cache_enabled:
        return _display_only(word)
    if " " in word:  # cụm → hiển thị theo token, không tra Cambridge/audio
        disp = await run_in_threadpool(_phrase_display, word)
        return IPAResult(word=word, us_ipa=disp or None, display=disp,
                         source="g2p" if disp else None)
    lock = await _get_lock(word)
    async with lock:
        result, needs_warm = await run_in_threadpool(_resolve_core, word, cfg)
        if needs_warm and wait_cambridge:
            # Nâng cache lên Cambridge NGAY (đồng bộ) rồi đọc lại → có uk_ipa. Trong
            # cùng lock để 2 request cùng từ không fetch đôi (dogpile).
            await run_in_threadpool(_warm_sync, word, cfg)
            result, needs_warm = await run_in_threadpool(_resolve_core, word, cfg)
    if needs_warm:
        asyncio.create_task(_warm(word, cfg))
    return result


def _phrase_display(text: str) -> str:
    return " ".join(filter(None, (word_ipa_display(t) for t in text.split()))) or ""


def resolve_ipa_display(text: str, cfg: Config) -> str:
    """Chuỗi IPA hiển thị cho 1 từ/cụm — dùng ở endpoint SYNC (/words upsert).

    Cascade đồng bộ cache→CMUdict→Cambridge(khi miss)→eSpeak cho từng token rồi
    ghép. KHÔNG warm-on-hit ở luồng sync (tránh quản lý thread); Cambridge vẫn
    được thử đồng bộ khi CMUdict miss. Master flag tắt → word_ipa_display cũ.
    """
    if not cfg.ipa_cache_enabled:
        return _phrase_display(text)
    parts: list[str] = []
    for tok in text.split():
        result, _needs_warm = _resolve_core(_normalize(tok), cfg)
        if result.display:
            parts.append(result.display)
    return " ".join(parts)

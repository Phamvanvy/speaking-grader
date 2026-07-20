"""Cascade tra IPA (src/ipa_resolve.py) + cache (src/ipa_cache.py).

Không mạng: fetch Cambridge được monkeypatch. CMUdict dùng thật (dependency lõi).
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from src import ipa_cache, ipa_resolve
from src.cambridge import CambridgeEntry, CambridgeResult
from src.config import load_config
from src.phoneme.ipa import word_ipa_display


def _cfg(tmp_path, **over):
    base = load_config()
    return dataclasses.replace(
        base,
        ipa_db_path=str(tmp_path / "ipa.db"),
        ipa_cache_enabled=over.pop("ipa_cache_enabled", True),
        ipa_cambridge_enabled=over.pop("ipa_cambridge_enabled", True),
        ipa_max_retries=1,
        ipa_backoff_base_sec=0.0,
        **over,
    )


# ── bit-for-bit khi master flag TẮT ──────────────────────────────────────


def test_display_bit_for_bit_when_disabled(tmp_path):
    cfg = _cfg(tmp_path, ipa_cache_enabled=False)
    for w in ["hello", "pronounce", "borrow a book"]:
        expected = " ".join(filter(None, (word_ipa_display(t) for t in w.split())))
        assert ipa_resolve.resolve_ipa_display(w, cfg) == expected


def test_disabled_does_not_touch_db(tmp_path):
    cfg = _cfg(tmp_path, ipa_cache_enabled=False)
    ipa_resolve.resolve_ipa_display("hello", cfg)
    # Không bật cache → không tạo file DB.
    assert not (tmp_path / "ipa.db").exists()


# ── cache hit / miss + CMUdict-found warm ────────────────────────────────


def test_cmudict_found_caches_and_signals_warm(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    called = []
    monkeypatch.setattr(
        ipa_resolve, "fetch_cambridge",
        lambda w, c: called.append(w) or CambridgeResult("not_found"),
    )
    result, needs_warm = ipa_resolve._resolve_core("hello", cfg)
    assert result.source == "cmudict"
    assert result.display
    assert needs_warm is True          # CMUdict-found + Cambridge bật → warm nền
    assert called == []                # warm CHƯA chạy trong _resolve_core (đồng bộ)
    # Đã lưu cache với cambridge chưa thử.
    row = ipa_cache.get(cfg, "hello")
    assert row is not None and row.source == "cmudict"
    assert row.cambridge_status == ipa_cache.CAMBRIDGE_UNTRIED


def test_second_lookup_is_cache_hit(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ipa_resolve, "fetch_cambridge",
                        lambda w, c: CambridgeResult("not_found"))
    r1, _ = ipa_resolve._resolve_core("hello", cfg)
    assert r1.cached is False
    r2, warm2 = ipa_resolve._resolve_core("hello", cfg)
    assert r2.cached is True
    assert warm2 is False


def test_normalization_uppercase_and_spaces(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ipa_resolve, "fetch_cambridge",
                        lambda w, c: CambridgeResult("not_found"))
    ipa_resolve._resolve_core("hello", cfg)
    # "  HELLO " chuẩn hoá về "hello" → cache hit qua resolve_ipa (async).
    res = asyncio.run(ipa_resolve.resolve_ipa("  HELLO ", cfg))
    assert res.cached is True
    assert res.word == "hello"


# ── CMUdict miss → Cambridge đồng bộ (trước eSpeak) ──────────────────────


def _force_cmudict_miss(monkeypatch):
    """Ép nhánh CMUdict-miss: G2P trả 'failed' (không phụ thuộc espeak trên máy dev)."""
    monkeypatch.setattr(
        ipa_resolve, "word_to_ipa_with_stress_source",
        lambda w: ([], [], "failed"),
    )


def test_cmudict_miss_fetches_cambridge_synchronously(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _force_cmudict_miss(monkeypatch)
    monkeypatch.setattr(
        ipa_resolve, "fetch_cambridge",
        lambda w, c: CambridgeResult(
            "success", CambridgeEntry(w, uk_ipa="prəˈnaʊns", us_ipa="prəˈnɑʊns")
        ),
    )
    result, needs_warm = ipa_resolve._resolve_core("pronounce", cfg)
    assert result.source == "cambridge"
    assert result.uk_ipa == "prəˈnaʊns" and result.us_ipa == "prəˈnɑʊns"
    assert needs_warm is False
    row = ipa_cache.get(cfg, "pronounce")
    assert row.source == "cambridge"
    assert row.cambridge_status == ipa_cache.CAMBRIDGE_SUCCESS


def test_negative_cache_not_found_is_terminal(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _force_cmudict_miss(monkeypatch)
    calls = []
    monkeypatch.setattr(
        ipa_resolve, "fetch_cambridge",
        lambda w, c: calls.append(w) or CambridgeResult("not_found"),
    )
    ipa_resolve._resolve_core("asdfqwer", cfg)
    ipa_resolve._resolve_core("asdfqwer", cfg)   # lần 2 KHÔNG được fetch lại
    assert calls == ["asdfqwer"]
    row = ipa_cache.get(cfg, "asdfqwer")
    assert row.cambridge_status == ipa_cache.CAMBRIDGE_NOT_FOUND


def test_transient_error_is_retried_next_time(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _force_cmudict_miss(monkeypatch)
    calls = []
    monkeypatch.setattr(
        ipa_resolve, "fetch_cambridge",
        lambda w, c: calls.append(w) or CambridgeResult("error"),
    )
    ipa_resolve._resolve_core("flaky", cfg)
    ipa_resolve._resolve_core("flaky", cfg)   # ERROR không terminal → thử lại
    assert calls == ["flaky", "flaky"]
    row = ipa_cache.get(cfg, "flaky")
    assert row.cambridge_status == ipa_cache.CAMBRIDGE_ERROR


def test_cambridge_disabled_skips_fetch(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, ipa_cambridge_enabled=False)
    _force_cmudict_miss(monkeypatch)
    monkeypatch.setattr(
        ipa_resolve, "fetch_cambridge",
        lambda w, c: pytest.fail("không được gọi Cambridge khi tắt"),
    )
    result, warm = ipa_resolve._resolve_core("whatever", cfg)
    assert warm is False


# ── dogpile: N request đồng thời cùng từ chỉ fetch 1 lần ──────────────────


def test_dogpile_guard_single_fetch(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _force_cmudict_miss(monkeypatch)
    calls = []

    def _slow(w, c):
        calls.append(w)
        return CambridgeResult("success", CambridgeEntry(w, us_ipa="ˈtɛst"))

    monkeypatch.setattr(ipa_resolve, "fetch_cambridge", _slow)

    async def _run():
        return await asyncio.gather(*[
            ipa_resolve.resolve_ipa("dogpile", cfg) for _ in range(8)
        ])

    results = asyncio.run(_run())
    assert all(r.us_ipa == "ˈtɛst" for r in results)
    assert calls == ["dogpile"]   # khoá per-word → đúng 1 lần fetch


# ── warm nền nâng CMUdict → Cambridge ────────────────────────────────────


def test_warm_sync_upgrades_cmudict_to_cambridge(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(ipa_resolve, "fetch_cambridge",
                        lambda w, c: CambridgeResult("not_found"))
    ipa_resolve._resolve_core("hello", cfg)             # lưu source=cmudict
    monkeypatch.setattr(
        ipa_resolve, "fetch_cambridge",
        lambda w, c: CambridgeResult("success", CambridgeEntry(w, us_ipa="həˈloʊ")),
    )
    ipa_resolve._warm_sync("hello", cfg)                # warm → nâng cấp
    row = ipa_cache.get(cfg, "hello")
    assert row.source == "cambridge"
    assert row.us_ipa == "həˈloʊ"
    assert row.cambridge_status == ipa_cache.CAMBRIDGE_SUCCESS

"""Test offline cho gợi ý từ luyện âm (tab Từ đã lưu, endpoint /words/suggestions).

Kiểm tra: tally hồ sơ âm từ result_json (severity/skip đúng luật), con trỏ quét
composite (created_at, id) không bỏ sót record cùng giây, migration words.db
v1→v2, inverted index chỉ nhận từ CMUdict (không chạm eSpeak), cache LLM
(TTL/version/fallback ngắn hạn), và orchestrator loại từ đã lưu. Không gọi LLM
thật, không cần server.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src import history, phoneme_profile, word_suggest, words
from src.config import load_config


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        anthropic_api_key=None,  # LLM thật không bao giờ được gọi trong test
        words_db_path=str(tmp_path / "words.db"),
        history_db_path=str(tmp_path / "history.db"),
        history_audio_dir=str(tmp_path / "history_audio"),
    )


@pytest.fixture()
def small_index(monkeypatch):
    """Index nhỏ deterministic: monkeypatch wordlist + reset singleton."""
    wl = ["the", "think", "this", "that", "cat", "see", "zoo", "ship",
          "chair", "job", "van", "sit", "month", "three", "bath", "thin"]
    monkeypatch.setattr(word_suggest, "_load_wordlist", lambda: wl)
    monkeypatch.setattr(word_suggest, "_index", None)
    yield wl
    word_suggest._index = None  # không rò index giả sang test khác


def _pt(symbol, status="ok", severity=None, heard=None):
    return {"symbol": symbol, "status": status, "severity": severity, "heard": heard}


def _result(words_points, skip_reason=None):
    return {"phoneme": {"score": {"words": [
        {"word": f"w{i}", "skip_reason": skip_reason, "phonemes": pts}
        for i, pts in enumerate(words_points)
    ]}}}


def _insert_record(cfg, user_id, created_at, rec_id, result):
    conn = history._connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT INTO history_records (id, user_id, kind, result_json, created_at)"
                " VALUES (?, ?, 'single', ?, ?)",
                (rec_id, user_id, json.dumps(result), created_at),
            )
    finally:
        conn.close()


# ── Tally ─────────────────────────────────────────────────────────────────


def test_tally_result_counts_by_severity():
    tallies = {}
    result = _result([[
        _pt("θ", "ok"),
        _pt("θ", "sub", "high", heard="t"),
        _pt("θ", "sub", "low", heard="s"),      # low = pass → err 0, không tally heard
        _pt("s", "del", "medium"),
        _pt("z", "skipped"),                     # bỏ qua hoàn toàn
    ]])
    phoneme_profile._tally_result(result, tallies)
    t = tallies["θ"]
    assert t["attempts"] == 3 and t["ok"] == 1 and t["sub"] == 2
    assert t["err_weighted"] == pytest.approx(1.0)  # high=1.0, low=0.0
    assert t["heard"] == {"t": 1.0}
    s = tallies["s"]
    assert s["del"] == 1 and s["err_weighted"] == pytest.approx(0.6)
    assert "z" not in tallies


def test_tally_skips_skip_reason_and_summary_blobs():
    tallies = {}
    phoneme_profile._tally_result(_result([[_pt("θ", "sub", "high")]], skip_reason="asr"), tallies)
    assert tallies == {}
    # Record cha của exam chỉ là summary — không có key phoneme → không crash.
    phoneme_profile._tally_result({"overall_score": 120}, tallies)
    phoneme_profile._tally_result({"phoneme": None}, tallies)
    assert tallies == {}


def test_tally_weight_scales_counts():
    tallies = {}
    phoneme_profile._tally_result(_result([[_pt("θ", "sub", "high")]]), tallies, weight=0.25)
    assert tallies["θ"]["attempts"] == pytest.approx(0.25)
    assert tallies["θ"]["err_weighted"] == pytest.approx(0.25)


def test_recency_weight_steps():
    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    assert phoneme_profile._recency_weight((now - timedelta(days=5)).strftime(fmt), now) == 1.0
    assert phoneme_profile._recency_weight((now - timedelta(days=60)).strftime(fmt), now) == 0.5
    assert phoneme_profile._recency_weight((now - timedelta(days=120)).strftime(fmt), now) == 0.25
    assert phoneme_profile._recency_weight("garbage", now) == 1.0


# ── Con trỏ quét history ──────────────────────────────────────────────────


def test_refresh_profile_cursor_flow(cfg):
    uid = "u1"
    _insert_record(cfg, uid, "2026-07-01T00:00:01Z", "aaa", _result([[_pt("θ", "sub", "high")]]))
    _insert_record(cfg, uid, "2026-07-02T00:00:01Z", "bbb", _result([[_pt("θ", "ok")]]))
    phoneme_profile.refresh_profile(cfg, uid)
    stats = words.get_phoneme_stats(cfg, uid)
    assert stats["θ"]["attempts"] == pytest.approx(2.0)
    assert words.get_profile_cursor(cfg, uid) == ("2026-07-02T00:00:01Z", "bbb")
    # Quét lại → no-op, không double-count.
    phoneme_profile.refresh_profile(cfg, uid)
    assert words.get_phoneme_stats(cfg, uid)["θ"]["attempts"] == pytest.approx(2.0)


def test_list_results_since_same_second_not_skipped(cfg):
    uid = "u1"
    ts = "2026-07-01T00:00:01Z"
    _insert_record(cfg, uid, ts, "aaa", _result([[_pt("s", "ok")]]))
    _insert_record(cfg, uid, ts, "bbb", _result([[_pt("z", "ok")]]))
    # Batch 1 dòng: dừng giữa 2 record cùng giây.
    results, cursor = history.list_results_since(cfg, uid, "", "", limit=1)
    assert len(results) == 1 and cursor == (ts, "aaa")
    # Resume từ con trỏ composite → record cùng giây id lớn hơn KHÔNG bị bỏ sót.
    results2, cursor2 = history.list_results_since(cfg, uid, *cursor, limit=10)
    assert len(results2) == 1 and cursor2 == (ts, "bbb")
    # Hết dữ liệu → con trỏ đứng yên.
    results3, cursor3 = history.list_results_since(cfg, uid, *cursor2, limit=10)
    assert results3 == [] and cursor3 == cursor2


def test_apply_tallies_concurrent_guard(cfg):
    uid = "u1"
    tallies = {"θ": {"attempts": 1.0, "ok": 0.0, "sub": 1.0, "del": 0.0,
                     "err_weighted": 1.0, "heard": {"t": 1.0}}}
    assert words.apply_phoneme_tallies(cfg, uid, tallies, ("2026-07-01T00:00:01Z", "aaa"))
    # Request thứ 2 cùng đoạn quét (cursor không tiến) → bị guard bỏ qua.
    assert not words.apply_phoneme_tallies(cfg, uid, tallies, ("2026-07-01T00:00:01Z", "aaa"))
    assert words.get_phoneme_stats(cfg, uid)["θ"]["attempts"] == pytest.approx(1.0)


# ── Migration words.db v1 → v2 ────────────────────────────────────────────

_V1_DDL = """
CREATE TABLE IF NOT EXISTS saved_words (
  user_id TEXT NOT NULL, word TEXT NOT NULL, ipa TEXT, phonemes_json TEXT,
  accuracy REAL, last_score REAL,
  saved_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  last_practiced_at TEXT, PRIMARY KEY (user_id, word));
CREATE TABLE IF NOT EXISTS word_info_cache (
  word TEXT NOT NULL, lang TEXT NOT NULL, definition_en TEXT, example_en TEXT,
  meaning TEXT, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (word, lang));
"""


def test_words_db_migrates_v1_to_v2(cfg):
    conn = sqlite3.connect(cfg.words_db_path)
    with conn:
        conn.executescript(_V1_DDL)
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT INTO saved_words (user_id, word, ipa) VALUES ('u1', 'think', 'θɪŋk')"
        )
    conn.close()
    # Mở qua words._connect → tự migrate lên v2, dữ liệu cũ nguyên vẹn.
    assert words.list_words(cfg, "u1")["words"][0]["word"] == "think"
    conn = sqlite3.connect(cfg.words_db_path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"saved_words", "word_info_cache", "phoneme_stats",
            "phoneme_profile_state", "suggestion_cache"} <= tables


# ── Inverted index ────────────────────────────────────────────────────────


def test_index_candidates_frequency_order_and_no_espeak(small_index, monkeypatch):
    # Từ OOV không được vào index → eSpeak không bao giờ bị gọi.
    monkeypatch.setattr(word_suggest, "_load_wordlist", lambda: [*small_index, "xqzvt"])
    import src.phoneme.ipa as ipa
    def _boom(*a, **k):  # pragma: no cover - fail nếu espeak bị chạm
        raise AssertionError("eSpeak không được phép bị gọi khi build index")
    monkeypatch.setattr(ipa, "_espeak_word_to_symbols_stress", _boom, raising=False)
    cands = word_suggest.candidates_for("θ")
    # Đúng thứ tự tần suất (thứ tự wordlist), chỉ từ chứa /θ/.
    assert cands == ["think", "month", "three", "bath", "thin"]
    _, by_word = word_suggest._get_index()
    assert "xqzvt" not in by_word
    assert "the" in by_word  # 3 ký tự, 2 phoneme — vẫn hợp lệ


# ── Cache LLM ─────────────────────────────────────────────────────────────


def _set_cache_created_at(cfg, phoneme, lang, dt):
    conn = sqlite3.connect(cfg.words_db_path)
    with conn:
        conn.execute(
            "UPDATE suggestion_cache SET created_at = ? WHERE phoneme = ? AND lang = ?",
            (dt.strftime("%Y-%m-%dT%H:%M:%SZ"), phoneme, lang),
        )
    conn.close()


def test_cached_suggestions_hit_skips_llm(cfg, small_index, monkeypatch):
    calls = []
    def fake_rank(config, symbol, candidates):
        calls.append(symbol)
        return [{"word": candidates[0], "reason": "vị trí đầu từ"}]
    monkeypatch.setattr(word_suggest, "rank_with_llm", fake_rank)
    got1, llm1 = word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    got2, llm2 = word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    assert got1 == got2 == [{"word": "think", "reason": "vị trí đầu từ"}]
    assert llm1 and not llm2 and calls == ["θ"]


def test_cached_suggestions_llm_ttl_expiry(cfg, small_index, monkeypatch):
    calls = []
    monkeypatch.setattr(
        word_suggest, "rank_with_llm",
        lambda c, s, cand: calls.append(s) or [{"word": cand[0], "reason": "x"}],
    )
    word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    _set_cache_created_at(cfg, "θ", "vi", datetime.now(timezone.utc) - timedelta(days=31))
    word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    assert len(calls) == 2  # entry LLM quá 30 ngày → re-fetch


def test_cached_suggestions_fallback_short_ttl(cfg, small_index, monkeypatch):
    calls = []
    def fail_rank(config, symbol, candidates):
        calls.append(symbol)
        raise RuntimeError("backend down")
    monkeypatch.setattr(word_suggest, "rank_with_llm", fail_rank)
    got, llm_called = word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    assert llm_called and got[0] == {"word": "think", "reason": None}
    # Fallback được cache ngắn hạn → gọi lại NGAY không hammer LLM.
    got2, llm2 = word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    assert not llm2 and got2 == got and calls == ["θ"]
    # Quá TTL 1h → retry LLM.
    _set_cache_created_at(cfg, "θ", "vi", datetime.now(timezone.utc) - timedelta(hours=2))
    word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    assert calls == ["θ", "θ"]


def test_cached_suggestions_version_mismatch_is_miss(cfg, small_index, monkeypatch):
    words.put_suggestion_cache(cfg, "θ", "vi", [{"word": "old", "reason": None}],
                               "m", word_suggest._CACHE_VERSION - 1)
    monkeypatch.setattr(
        word_suggest, "rank_with_llm",
        lambda c, s, cand: [{"word": cand[0], "reason": "mới"}],
    )
    got, llm_called = word_suggest.cached_suggestions(cfg, cfg, "θ", "vi")
    assert llm_called and got[0]["word"] == "think"


def test_cached_suggestions_no_budget_no_cache_write(cfg, small_index):
    got, llm_called = word_suggest.cached_suggestions(cfg, cfg, "θ", "vi", allow_llm=False)
    assert not llm_called and got[0] == {"word": "think", "reason": None}
    assert words.get_suggestion_cache(cfg, "θ", "vi") is None  # request sau thử LLM lại


# ── Orchestrator ──────────────────────────────────────────────────────────


def _seed_weak(cfg, uid, symbol, attempts=10.0, err=8.0, seq=0):
    # Cursor id tăng dần theo seq — guard của apply_phoneme_tallies yêu cầu
    # cursor mới phải LỚN hơn cursor hiện tại (chống double-count).
    words.apply_phoneme_tallies(
        cfg, uid,
        {symbol: {"attempts": attempts, "ok": attempts - err, "sub": err,
                  "del": 0.0, "err_weighted": err, "heard": {}}},
        ("2026-07-01T00:00:01Z", f"seed-{seq:03d}"),
    )


def test_get_suggestions_excludes_saved_words(cfg, small_index, monkeypatch):
    uid = "u1"
    # Cần ≥2 âm yếu organic để source="history" (dưới 2 → pad fallback);
    # θ lỗi nặng hơn ð nên đứng đầu.
    _seed_weak(cfg, uid, "θ", err=9.0, seq=0)
    _seed_weak(cfg, uid, "ð", err=6.0, seq=1)
    words.upsert_word(cfg, uid, "think")  # đã lưu → không được gợi ý lại
    monkeypatch.setattr(
        word_suggest, "rank_with_llm",
        lambda c, s, cand: [{"word": w, "reason": "r"} for w in cand[:3]],
    )
    out = word_suggest.get_suggestions(cfg, cfg, uid, limit=5, lang="vi")
    assert out["source"] == "history"
    assert out["weak_phonemes"][0]["symbol"] == "θ"
    sugg_words = [s["word"] for s in out["suggestions"]]
    assert "think" not in sugg_words and sugg_words  # loại từ đã lưu, vẫn có gợi ý
    for s in out["suggestions"]:
        assert "θ" in s["target_phonemes"] or s["phoneme"] != "θ"
        assert s["ipa"]


def test_get_suggestions_empty_history_falls_back(cfg, small_index, monkeypatch):
    monkeypatch.setattr(
        word_suggest, "rank_with_llm",
        lambda c, s, cand: [{"word": cand[0], "reason": "r"}],
    )
    out = word_suggest.get_suggestions(cfg, cfg, "fresh-user", limit=5, lang="vi")
    assert out["source"] == "fallback"
    assert all(w["fallback"] for w in out["weak_phonemes"])
    assert out["suggestions"], "fallback vẫn phải có gợi ý"


def test_get_suggestions_llm_budget_cap(cfg, small_index, monkeypatch):
    uid = "u1"
    # 4 âm yếu → chỉ 3 call LLM đầu được phép, âm thứ 4 dùng fallback tần suất.
    for i, sym in enumerate(("θ", "ð", "s", "z")):
        _seed_weak(cfg, uid, sym, seq=i)
    calls = []
    monkeypatch.setattr(
        word_suggest, "rank_with_llm",
        lambda c, s, cand: calls.append(s) or [{"word": cand[0], "reason": "r"}],
    )
    word_suggest.get_suggestions(cfg, cfg, uid, limit=12, lang="vi")
    assert len(calls) == word_suggest._MAX_LLM_CALLS_PER_REQUEST

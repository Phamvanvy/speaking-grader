"""Gợi ý từ mới để luyện âm yếu — tab "Từ đã lưu" (endpoint /words/suggestions).

Hybrid 2 tầng:
1. LOCAL: hồ sơ âm yếu (src/phoneme_profile.py) + inverted index phoneme→từ
   thông dụng (src/data/common_words.txt ∩ CMUdict) → ứng viên deterministic.
2. LLM: chỉ CHỌN/xếp hạng ~10 từ hay cho MỖI phoneme từ ứng viên — kết quả
   user-agnostic, cache SQLite theo (phoneme, lang) nên hiếm khi gọi. Cá nhân
   hoá (loại từ đã lưu, ghép theo âm yếu của user) làm SAU khi đọc cache.

Index CHỈ nhận từ có trong CMUdict (hoặc _WORD_IPA_OVERRIDES) nên word_to_ipa
không bao giờ rơi xuống eSpeak fallback (máy dev Windows không có espeak-ng).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import phoneme_profile, words
from .config import Config
from .phoneme.ipa import word_to_ipa
from .phoneme.ipa.g2p import _get_cmudict
from .phoneme.ipa.phoneme_set import _WORD_IPA_OVERRIDES, normalize_ipa
from .schema import PhonemePracticeList
from .scoring.backends import _generate_anthropic, _generate_local
from .words import _WORD_RE

logger = logging.getLogger("toeic.word_suggest")

# Bump khi đổi prompt/schema LLM — vô hiệu toàn bộ suggestion_cache cũ, không
# cần migration.
_CACHE_VERSION = 1

# TTL cache: entry LLM sống lâu (nội dung ổn định); entry fallback tần suất
# (model=NULL, ghi khi LLM lỗi) sống NGẮN để user vẫn có gợi ý ngay mà không
# hammer LLM mỗi request khi backend down — tự retry LLM sau 1h.
_TTL_LLM_SECONDS = 30 * 86400
_TTL_FALLBACK_SECONDS = 3600

# Chặn latency: tối đa N call LLM (cache-miss) mỗi request; các âm còn lại dùng
# fallback tần suất trong request này, cache đầy dần qua các lần refresh sau.
_MAX_LLM_CALLS_PER_REQUEST = 3

# Số âm yếu lấy ứng viên + số ứng viên gửi LLM + số từ LLM chọn.
_TOP_WEAK_PHONEMES = 4
_CANDIDATES_PER_PHONEME = 60
_LLM_PICKS = 10

_WORDLIST_PATH = Path(__file__).parent / "data" / "common_words.txt"

# Index xây lazy 1 lần (như _get_cmudict): by_symbol[sym] = [(rank, word), ...]
# theo tần suất tăng dần rank; by_word[word] = (rank, ipa_str, frozenset syms).
_index_lock = threading.Lock()
_index: tuple[dict[str, list[tuple[int, str]]], dict[str, tuple[int, str, frozenset]]] | None = None


def _load_wordlist() -> list[str]:
    lines = _WORDLIST_PATH.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        w = line.strip().lower()
        if not w or w.startswith("#") or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _build_index(
    wordlist: list[str],
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, tuple[int, str, frozenset]]]:
    cmu = _get_cmudict()
    by_symbol: dict[str, list[tuple[int, str]]] = {}
    by_word: dict[str, tuple[int, str, frozenset]] = {}
    for rank, word in enumerate(wordlist):
        if not _WORD_RE.match(word) or not 3 <= len(word) <= 9:
            continue
        # Chỉ từ CMUdict-backed → word_to_ipa không bao giờ chạm eSpeak.
        if word not in cmu and word not in _WORD_IPA_OVERRIDES:
            continue
        symbols = word_to_ipa(word)
        if not 2 <= len(symbols) <= 7:
            continue
        syms = frozenset(normalize_ipa(s) for s in symbols)
        by_word[word] = (rank, "".join(symbols), syms)
        for s in syms:
            by_symbol.setdefault(s, []).append((rank, word))
    return by_symbol, by_word


def _get_index() -> tuple[dict[str, list[tuple[int, str]]], dict[str, tuple[int, str, frozenset]]]:
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = _build_index(_load_wordlist())
                logger.info(
                    "word_suggest index: %d từ, %d phoneme",
                    len(_index[1]), len(_index[0]),
                )
    return _index


def candidates_for(symbol: str, n: int = _CANDIDATES_PER_PHONEME) -> list[str]:
    """n từ thông dụng nhất chứa phoneme (đã normalize) — user-agnostic."""
    by_symbol, _ = _get_index()
    return [w for _rank, w in by_symbol.get(normalize_ipa(symbol), [])[:n]]


# ── LLM rank + cache ──────────────────────────────────────────────────────


def rank_with_llm(config: Config, symbol: str, candidates: list[str]) -> list[dict]:
    """LLM chọn ~10 từ luyện tập tốt nhất cho 1 phoneme từ candidates.

    Kết quả post-validate: chỉ giữ từ CÓ trong candidates (LLM không được bịa),
    dedupe, cap _LLM_PICKS. Raise khi backend lỗi — caller tự degrade.
    """
    system_prompt = (
        "You are a pronunciation coach for Vietnamese TOEIC learners. You will "
        "be given a target IPA phoneme and a CANDIDATES list of common English "
        f"words containing it. Choose the {_LLM_PICKS} BEST words to practice "
        "that sound:\n"
        "- common everyday / TOEIC-office vocabulary a learner should know;\n"
        "- mix of positions (word-initial / medial / final) where possible;\n"
        "- short words that are easy to drill;\n"
        "- no near-duplicate inflections (of think/thinks/thinking pick ONE).\n"
        "For each chosen word write `reason` in VIETNAMESE, ≤12 từ (vì sao từ "
        "này tốt để luyện âm đó — vd vị trí âm, cặp tối thiểu).\n"
        "Echo the phoneme in `phoneme` (no slashes). Do NOT invent words "
        "outside CANDIDATES."
    )
    user_prompt = (
        f"PHONEME: /{symbol}/\n"
        f"CANDIDATES: {', '.join(candidates)}\n\n"
        "Now produce the structured JSON."
    )
    if config.is_local:
        result = _generate_local(
            config, system_prompt, user_prompt, PhonemePracticeList,
            PhonemePracticeList.model_json_schema(), "PhonemePracticeList",
            None, None,
        )
    else:
        result = _generate_anthropic(
            config, system_prompt, user_prompt, PhonemePracticeList, None, None
        )
    assert isinstance(result, PhonemePracticeList)
    allowed = set(candidates)
    picked: list[dict] = []
    seen: set[str] = set()
    for item in result.words:
        w = (item.word or "").strip().lower()
        if w in allowed and w not in seen:
            seen.add(w)
            picked.append({"word": w, "reason": (item.reason or "").strip() or None})
        if len(picked) >= _LLM_PICKS:
            break
    if not picked:
        raise RuntimeError(f"LLM không chọn được từ hợp lệ nào cho /{symbol}/.")
    return picked


def _cache_age_seconds(created_at: str | None) -> float:
    try:
        ts = datetime.strptime(created_at or "", "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return float("inf")
    return (datetime.now(timezone.utc) - ts).total_seconds()


def cached_suggestions(
    cfg: Config, config: Config, symbol: str, lang: str, *, allow_llm: bool = True
) -> tuple[list[dict], bool]:
    """Danh sách [{"word","reason"}] cho 1 phoneme + cờ "đã tốn 1 call LLM".

    Thứ tự: cache hợp lệ (đúng _CACHE_VERSION, chưa hết TTL theo loại entry) →
    LLM (nếu allow_llm) → fallback tần suất. Fallback do LLM LỖI được cache
    ngắn hạn (model=NULL) để không hammer backend; fallback do HẾT budget
    (allow_llm=False) thì KHÔNG cache — request sau thử LLM lại.
    """
    entry = words.get_suggestion_cache(cfg, symbol, lang)
    if entry and entry.get("cache_version") == _CACHE_VERSION:
        ttl = _TTL_LLM_SECONDS if entry.get("model") else _TTL_FALLBACK_SECONDS
        if _cache_age_seconds(entry.get("created_at")) < ttl:
            return entry["words"], False

    candidates = candidates_for(symbol)
    if not candidates:
        return [], False
    fallback = [{"word": w, "reason": None} for w in candidates[:_LLM_PICKS]]
    if not allow_llm:
        return fallback, False

    try:
        picked = rank_with_llm(config, symbol, candidates)
    except Exception:  # noqa: BLE001 - LLM lỗi không được chặn gợi ý
        logger.exception("LLM rank lỗi cho /%s/ — dùng fallback tần suất", symbol)
        try:
            words.put_suggestion_cache(cfg, symbol, lang, fallback, None, _CACHE_VERSION)
        except Exception:  # noqa: BLE001
            logger.exception("Lỗi ghi suggestion_cache fallback (bỏ qua)")
        return fallback, True
    model = config.local_model if config.is_local else config.model
    try:
        words.put_suggestion_cache(cfg, symbol, lang, picked, model, _CACHE_VERSION)
    except Exception:  # noqa: BLE001 - cache hỏng không được chặn response
        logger.exception("Lỗi ghi suggestion_cache (bỏ qua)")
    return picked, True


# ── Orchestrator ──────────────────────────────────────────────────────────


def get_suggestions(
    cfg: Config, config: Config, user_id: str, *, limit: int = 12, lang: str = "vi"
) -> dict:
    """Payload cho GET /words/suggestions: âm yếu + từ gợi ý đã cá nhân hoá.

    Cá nhân hoá cục bộ SAU cache: loại từ đã lưu (từ đã ok nằm sẵn trong danh
    sách trên, từ đã ok thì âm của nó cũng không còn yếu → tự giảm gợi ý),
    ưu tiên từ trúng nhiều âm yếu, round-robin giữa các âm để 1 âm không
    chiếm hết danh sách.
    """
    weak, source = phoneme_profile.get_weak_phonemes(cfg, user_id)
    weak_syms = [w["symbol"] for w in weak]
    saved = {e["word"] for e in words.list_words(cfg, user_id)["words"]}
    _, by_word = _get_index()

    llm_budget = _MAX_LLM_CALLS_PER_REQUEST
    pools: list[list[dict]] = []  # theo thứ tự âm yếu
    seen_words: set[str] = set()
    for item in weak[:_TOP_WEAK_PHONEMES]:
        sym = item["symbol"]
        picks, llm_called = cached_suggestions(
            cfg, config, sym, lang, allow_llm=llm_budget > 0
        )
        if llm_called:
            llm_budget -= 1
        pool: list[dict] = []
        for p in picks:
            w = p["word"]
            if w in saved or w in seen_words or w not in by_word:
                continue
            seen_words.add(w)
            rank, ipa, syms = by_word[w]
            pool.append({
                "word": w, "ipa": ipa,
                "target_phonemes": [s for s in weak_syms if s in syms],
                "phoneme": sym, "reason": p.get("reason"),
                "_rank": rank,
            })
        # Toàn bộ picks đã lưu/trùng → bù từ ứng viên tần suất kế tiếp.
        if not pool:
            for w in candidates_for(sym):
                if w in saved or w in seen_words or w not in by_word:
                    continue
                seen_words.add(w)
                rank, ipa, syms = by_word[w]
                pool.append({
                    "word": w, "ipa": ipa,
                    "target_phonemes": [s for s in weak_syms if s in syms],
                    "phoneme": sym, "reason": None,
                    "_rank": rank,
                })
                if len(pool) >= _LLM_PICKS:
                    break
        # Trong 1 âm: ưu tiên từ trúng nhiều âm yếu, rồi tần suất.
        pool.sort(key=lambda x: (-len(x["target_phonemes"]), x["_rank"]))
        pools.append(pool)

    # Round-robin giữa các âm (thứ tự âm yếu dần) để chia đều danh sách.
    suggestions: list[dict] = []
    i = 0
    while len(suggestions) < max(1, limit) and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            item = pool.pop(0)
            item.pop("_rank", None)
            suggestions.append(item)
        i += 1
        if i > 10_000:  # pragma: no cover - phòng hờ
            break
        if all(not p for p in pools):
            break

    return {
        "weak_phonemes": weak,
        "suggestions": suggestions,
        "source": source,
        "total": len(suggestions),
    }

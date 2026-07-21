"""Lắp nội dung bài học (Phase 1) — tái sử dụng tối đa hạ tầng sẵn có.

Dispatch theo dimension của Lesson:
- pronunciation: từ luyện cho từng âm trong nhóm (word_suggest.cached_suggestions
  → cache per-phoneme), kèm IPA + lý do; loại từ đã lưu.
- rubric: gộp suggestions/corrections CHÍNH của user cho tiêu chí đó từ history
  gần đây (history.list_recent_results) + tips tĩnh của lesson.
- question_type: bài nói MẪU (suggest.suggest_answer) + thang điểm + guidance của
  dạng câu; cache user-agnostic ở course.db (LLM chỉ gọi 1 lần/(lesson,lang)).

Điểm vào: get_lesson_content(cfg, config, user_id, lesson_id, lang). Route
GET /course/lesson/{id} gọi qua run_in_threadpool (có thể chạm LLM).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .. import history, word_suggest, words as words_mod
from ..config import Config
from ..phoneme.ipa.ko import word_to_ipa_ko
from ..phoneme.ipa.ko.phoneme_set_ko import normalize_ipa_ko
from ..phoneme.ipa.phoneme_set import normalize_ipa
from ..rubrics import resolve_question_type
from ..rubrics.base import exam_language
from ..suggest import suggest_answer
from . import practice as _practice
from . import store
from .generate import DONE_THRESHOLD
from .syllabus import KOREAN_PRACTICE_WORDS, PHONEME_GROUPS, get_lesson

logger = logging.getLogger("toeic.course.content")

_CONTENT_CACHE_VERSION = 1
_CONTENT_TTL_SECONDS = 30 * 86400
_RECENT_RESULTS_LIMIT = 40
_WORDS_PER_SYMBOL = 6
_MAX_PRON_WORDS = 15
_MAX_RUBRIC_ITEMS = 8


def get_lesson_content(
    cfg: Config, config: Config, user_id: str, lesson_id: str, lang: str
) -> dict:
    """Nội dung đầy đủ 1 lesson để render + luyện. Raise ValueError nếu id sai."""
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise ValueError(f"Không có lesson '{lesson_id}'.")

    prog = store.get_progress(cfg, user_id).get(lesson_id)
    base = {
        "id": lesson.id,
        "title": lesson.title,
        "dimension": lesson.dimension,
        "target": lesson.target,
        "exam": lesson.exam,
        "description": lesson.description,
        "est_minutes": lesson.est_minutes,
        "done_threshold": DONE_THRESHOLD.get(lesson.dimension, 0.7),
        "progress": prog,
    }

    if lesson.dimension == "pronunciation":
        base.update(_pron_content(cfg, config, user_id, lesson, lang))
    elif lesson.dimension == "rubric":
        base.update(_rubric_content(cfg, user_id, lesson))
        base["practice"] = _practice.build_practice(cfg, config, lesson, lang)
    else:  # question_type
        base.update(_qtype_content(cfg, config, lesson, lang))
        base["practice"] = _practice.build_practice(cfg, config, lesson, lang)
    return base


def _pron_content(cfg, config, user_id, lesson, lang) -> dict:
    # Tiếng Hàn: từ luyện curated (KOREAN_PRACTICE_WORDS) + IPA sinh qua G2P Hàn;
    # KHÔNG dùng word_suggest (index CMUdict tiếng Anh).
    if exam_language(lesson.exam) == "ko":
        return _pron_content_ko(lesson)
    symbols = PHONEME_GROUPS.get(lesson.target, [])
    saved = {e["word"] for e in words_mod.list_words(cfg, user_id)["words"]}
    _by_symbol, by_word = word_suggest._get_index()
    out_words: list[dict] = []
    seen: set[str] = set()
    for sym in symbols:
        picks, _llm = word_suggest.cached_suggestions(cfg, config, sym, lang)
        added = 0
        for p in picks:
            w = p["word"]
            if w in saved or w in seen or w not in by_word:
                continue
            seen.add(w)
            _rank, ipa, _syms = by_word[w]
            out_words.append({
                "word": w, "ipa": ipa,
                "phoneme": normalize_ipa(sym), "reason": p.get("reason"),
            })
            added += 1
            if added >= _WORDS_PER_SYMBOL or len(out_words) >= _MAX_PRON_WORDS:
                break
        if len(out_words) >= _MAX_PRON_WORDS:
            break
    return {
        "phonemes": [normalize_ipa(s) for s in symbols],
        "words": out_words,
    }


def _pron_content_ko(lesson) -> dict:
    """Từ luyện tiếng Hàn curated: hangul + IPA (word_to_ipa_ko, khớp reference
    recognizer) + nghĩa tiếng Việt (đưa vào `reason` để UI hiển thị)."""
    out_words: list[dict] = []
    for hangul, meaning in KOREAN_PRACTICE_WORDS.get(lesson.target, []):
        try:
            ipa = "".join(normalize_ipa_ko(s) for s in word_to_ipa_ko(hangul))
        except Exception:  # noqa: BLE001 - IPA lỗi không chặn từ luyện
            ipa = ""
        out_words.append({
            "word": hangul, "ipa": ipa,
            "phoneme": lesson.target, "reason": meaning,
        })
    return {
        "phonemes": [normalize_ipa_ko(s) for s in PHONEME_GROUPS.get(lesson.target, [])],
        "words": out_words,
    }


def _rubric_content(cfg, user_id, lesson) -> dict:
    results = history.list_recent_results(cfg, user_id, _RECENT_RESULTS_LIMIT)
    suggestions: list[str] = []
    corrections: list[dict] = []
    seen_sugg: set[str] = set()
    seen_corr: set[str] = set()
    for r in results:
        if (r.get("exam") or "toeic") != lesson.exam:
            continue
        for c in (r.get("scores") or {}).get("criteria") or []:
            if not isinstance(c, dict) or c.get("criterion") != lesson.target:
                continue
            for s in c.get("suggestions") or []:
                key = (s or "").strip()
                if key and key not in seen_sugg:
                    seen_sugg.add(key)
                    suggestions.append(key)
            for corr in c.get("corrections") or []:
                if not isinstance(corr, dict):
                    continue
                key = f"{corr.get('said')}→{corr.get('suggested')}"
                if key not in seen_corr:
                    seen_corr.add(key)
                    corrections.append({
                        "said": corr.get("said"),
                        "suggested": corr.get("suggested"),
                        "example": corr.get("example"),
                    })
    return {
        "tips": list(lesson.tips),
        "learner_suggestions": suggestions[:_MAX_RUBRIC_ITEMS],
        "corrections": corrections[:_MAX_RUBRIC_ITEMS],
    }


def _cache_age_seconds(created_at: str | None) -> float:
    try:
        ts = datetime.strptime(created_at or "", "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return float("inf")
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _qtype_content(cfg, config, lesson, lang) -> dict:
    qt = resolve_question_type(lesson.target, lesson.exam)
    meta = {"scale_description": qt.scale_description, "guidance": qt.guidance}

    entry = store.get_lesson_content_cache(cfg, lesson.id, lang)
    if (
        entry
        and entry.get("cache_version") == _CONTENT_CACHE_VERSION
        and _cache_age_seconds(entry.get("created_at")) < _CONTENT_TTL_SECONDS
    ):
        return {**meta, "sample_answer": entry["content"].get("sample_answer")}

    sample = None
    try:
        ans = suggest_answer(config, qt)
        sample = {
            "answer": ans.answer,
            "outline": list(getattr(ans, "outline", []) or []),
            "highlights": list(ans.highlights or []),
            "target_band": ans.target_band,
        }
    except Exception:  # noqa: BLE001 - LLM lỗi không chặn nội dung bài
        logger.exception("Lỗi sinh bài mẫu cho %s (bỏ qua)", lesson.id)

    if sample is not None:
        model = config.local_model if config.is_local else config.model
        try:
            store.put_lesson_content_cache(
                cfg, lesson.id, lang, {"sample_answer": sample}, model,
                _CONTENT_CACHE_VERSION,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Lỗi ghi lesson_content_cache (bỏ qua)")
    return {**meta, "sample_answer": sample}

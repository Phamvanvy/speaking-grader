"""Nội dung Boss cuối chặng (Phase 3A) — thử thách nói TỔNG HỢP cả Unit.

Boss CHỈ mở khóa khi mọi lesson trong Unit đã `done` (gate ở generate.build_course
+ endpoint) → nội dung đã học/đã cache ấm sẵn, nên build_boss KHÔNG sinh LLM mới:
- pronunciation: gộp từ luyện mọi nhóm âm trong unit (reuse content._pron_content).
- rubric / question_type: ghép đoạn đọc-to từ nội dung ĐÃ CACHE của các lesson
  (sample_answer / đề read_aloud) — đọc trực tiếp cache, KHÔNG generate.

Chấm = phát âm read-aloud với reference_text (đúng lựa chọn "chỉ phát âm"): học
viên đọc to reference_text → /grade như Boss lesson. Điểm vào: build_boss.
"""

from __future__ import annotations

import logging

from ..config import Config
from . import content as _content, store
from .syllabus import Unit

logger = logging.getLogger("toeic.course.boss")

_MAX_BOSS_WORDS = 18          # cap số từ đọc (pronunciation boss)
_MAX_SENTENCES_PER_LESSON = 1  # rubric/qtype: lấy 1 câu/đại diện mỗi lesson
_MAX_READ_ALOUD_WORDS = 60     # cap tổng độ dài đọc-to (rubric/qtype boss)


def _first_sentences(text: str, n: int) -> str:
    """n câu đầu của `text` (tách thô theo . ! ? — đủ cho cắt ngắn đọc-to)."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(p for p in parts[:n] if p).strip()


def _cap_words(text: str, max_words: int) -> str:
    ws = (text or "").split()
    return " ".join(ws[:max_words])


def _aggregate_pron_words(cfg, config, user_id, unit: Unit, lang: str) -> list[dict]:
    """Gộp từ luyện của mọi lesson pronunciation trong unit (dedupe, cap)."""
    out: list[dict] = []
    seen: set[str] = set()
    for lesson in unit.lessons:
        try:
            words = _content._pron_content(cfg, config, user_id, lesson, lang).get("words", [])
        except Exception:  # noqa: BLE001 — 1 lesson lỗi không chặn cả Boss
            logger.exception("Lỗi gộp từ Boss cho lesson %s (bỏ qua)", lesson.id)
            continue
        for w in words:
            key = (w.get("word") or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(w)
            if len(out) >= _MAX_BOSS_WORDS:
                return out
    return out


def _lesson_read_aloud_text(cfg, lesson, lang: str) -> str:
    """Đoạn đọc-to đại diện của 1 lesson rubric/qtype, LẤY TỪ CACHE (không generate).

    - question_type: cache (lesson_id, lang) → sample_answer.answer.
    - rubric: cache ('<lesson_id>#practice', lang) → reference (read_aloud) | prompt.
    Trả '' nếu chưa cache (lesson chưa mở) — caller bỏ qua.
    """
    if lesson.dimension == "question_type":
        entry = store.get_lesson_content_cache(cfg, lesson.id, lang)
        sample = (entry or {}).get("content", {}).get("sample_answer") if entry else None
        return _first_sentences((sample or {}).get("answer", ""), _MAX_SENTENCES_PER_LESSON)
    if lesson.dimension == "rubric":
        entry = store.get_lesson_content_cache(cfg, f"{lesson.id}#practice", lang)
        c = (entry or {}).get("content", {}) if entry else {}
        return _first_sentences(c.get("reference") or c.get("prompt") or "", _MAX_SENTENCES_PER_LESSON)
    return ""


def _aggregate_read_aloud(cfg, unit: Unit, lang: str) -> str:
    """Ghép đoạn đọc-to đã cache của các lesson trong unit; cap tổng độ dài."""
    chunks: list[str] = []
    for lesson in unit.lessons:
        t = _lesson_read_aloud_text(cfg, lesson, lang)
        if t:
            chunks.append(t)
    return _cap_words(" ".join(chunks), _MAX_READ_ALOUD_WORDS)


def build_boss(cfg: Config, config: Config, user_id: str, unit: Unit, lang: str) -> dict:
    """Nội dung Boss của `unit`: {reference_text, words}. reference_text = chuỗi để
    đọc to (chấm phát âm). words (chỉ pronunciation) = [{word, ipa, ...}] để hiển thị.
    KHÔNG sinh LLM (đọc nội dung đã học/đã cache)."""
    if unit.dimension == "pronunciation":
        words = _aggregate_pron_words(cfg, config, user_id, unit, lang)
        return {"words": words, "reference_text": " ".join(w["word"] for w in words)}
    return {"words": [], "reference_text": _aggregate_read_aloud(cfg, unit, lang)}

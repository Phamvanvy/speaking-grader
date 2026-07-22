"""Quest nội dung nâng cao cho khóa học (Phase 3B/3C) — LLM sinh, chấm phát âm.

Khác lesson (chẩn đoán, tính mastery), Quest là lớp **BONUS**: hội thoại nhập vai
(Role-play, 3B) / truyện đọc-to (Story, 3C) để giữ người học quay lại. LLM CHỈ
sinh NỘI DUNG (kịch bản/truyện) — chấm nói vẫn qua ĐÚNG gradePronunciation dùng
chung (một đường chấm), KHÔNG thêm judge mới.

- Topic curated theo kỳ thi (ROLEPLAY_TOPICS) — EN trước (TOEIC/IELTS); TOPIK để
  sau (thiếu nguồn nội dung Hàn).
- Sinh 1 lần qua LLM rồi cache USER-AGNOSTIC theo (exam, topic) trong course.db
  (lesson_content_cache, id tổng hợp '<exam>.<topic>#roleplay') — mọi user dùng
  chung, version+TTL+guard như practice.py.
- Fail-soft: LLM lỗi / nội dung không hợp lệ → trả None; caller/endpoint trả null
  để frontend ẩn quest, KHÔNG chặn khóa học.

Điểm vào: list_roleplay_topics(exam), build_roleplay(cfg, config, exam, topic, lang).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from ..config import Config
from ..schema import RolePlayScript, StoryQuest
from . import store

logger = logging.getLogger("toeic.course.quests")

_CACHE_TTL_SECONDS = 30 * 86400
_ROLEPLAY_CACHE_VERSION = 1
_STORY_CACHE_VERSION = 1

# Số lượt tối thiểu để kịch bản Role-play coi là hợp lệ (đủ dài để thành 1 phiên).
_MIN_ROLEPLAY_TURNS = 2
# Story: số đoạn tối thiểu + số từ tối thiểu mỗi đoạn (guard nội dung LLM).
_MIN_STORY_SEGMENTS = 3
_MIN_STORY_SEGMENT_WORDS = 4


@dataclass(frozen=True)
class QuestTopic:
    """Một chủ đề Quest curated — slug ổn định + tiêu đề hiển thị + gợi ý cho LLM."""

    slug: str          # ổn định (dùng trong quest_id + cache id): '<exam>.<slug>'
    title: str         # hiển thị trên node/thẻ quest
    setting: str       # mô tả bối cảnh cho LLM sinh kịch bản


# Chủ đề Role-play theo kỳ thi. TOEIC = tình huống công sở/đời sống thực dụng;
# IELTS = chủ đề hội thoại xã hội/học thuật nhẹ. Giữ ~4 chủ đề/kỳ để UI gọn.
ROLEPLAY_TOPICS: dict[str, tuple[QuestTopic, ...]] = {
    "toeic": (
        QuestTopic("hotel_checkin", "Nhận phòng khách sạn",
                   "checking in at a hotel front desk while traveling for work"),
        QuestTopic("job_interview", "Phỏng vấn xin việc",
                   "a job interview for an office position"),
        QuestTopic("restaurant_order", "Gọi món ở nhà hàng",
                   "ordering food at a restaurant with a waiter"),
        QuestTopic("phone_meeting", "Hẹn lịch họp qua điện thoại",
                   "a phone call to schedule a business meeting"),
    ),
    "ielts": (
        QuestTopic("hometown_chat", "Nói về quê hương",
                   "a friendly chat about the learner's hometown and daily life"),
        QuestTopic("travel_plan", "Bàn kế hoạch du lịch",
                   "discussing plans for an upcoming trip with a friend"),
        QuestTopic("shop_return", "Đổi trả hàng ở cửa hàng",
                   "returning a faulty product at a shop and asking for a refund"),
        QuestTopic("study_group", "Rủ bạn học nhóm",
                   "arranging a study group with a classmate before an exam"),
    ),
}

# Chủ đề Story (đọc-to tuyến tính) theo kỳ thi. TOEIC = mẩu chuyện công sở/đời
# sống; IELTS = câu chuyện đời thường/du lịch (từ vựng phong phú hơn).
STORY_TOPICS: dict[str, tuple[QuestTopic, ...]] = {
    "toeic": (
        QuestTopic("first_day", "Ngày đầu đi làm",
                   "an employee's first day at a new office job"),
        QuestTopic("business_trip", "Chuyến công tác",
                   "a short business trip that does not go as planned"),
        QuestTopic("product_launch", "Ra mắt sản phẩm",
                   "a team preparing for an important product launch"),
    ),
    "ielts": (
        QuestTopic("lost_wallet", "Chiếc ví thất lạc",
                   "someone who loses their wallet while travelling and what happens next"),
        QuestTopic("new_hobby", "Sở thích mới",
                   "a person discovering and learning a new hobby"),
        QuestTopic("kind_stranger", "Người lạ tốt bụng",
                   "a small act of kindness from a stranger that changes someone's day"),
    ),
}


def list_roleplay_topics(exam: str) -> tuple[QuestTopic, ...]:
    """Chủ đề Role-play của kỳ thi (rỗng nếu chưa hỗ trợ — vd topik)."""
    return ROLEPLAY_TOPICS.get(exam, ())


def get_roleplay_topic(exam: str, slug: str) -> QuestTopic | None:
    """QuestTopic role-play theo (exam, slug); None nếu không có."""
    for t in list_roleplay_topics(exam):
        if t.slug == slug:
            return t
    return None


def list_story_topics(exam: str) -> tuple[QuestTopic, ...]:
    """Chủ đề Story của kỳ thi (rỗng nếu chưa hỗ trợ)."""
    return STORY_TOPICS.get(exam, ())


def get_story_topic(exam: str, slug: str) -> QuestTopic | None:
    """QuestTopic story theo (exam, slug); None nếu không có."""
    for t in list_story_topics(exam):
        if t.slug == slug:
            return t
    return None


def roleplay_quest_id(exam: str, slug: str) -> str:
    """quest_id ổn định của 1 Role-play quest (khóa idempotent + badge)."""
    return f"{exam}.{slug}#roleplay"


def story_quest_id(exam: str, slug: str) -> str:
    """quest_id ổn định của 1 Story quest."""
    return f"{exam}.{slug}#story"


def _cache_age_seconds(created_at: str | None) -> float:
    try:
        ts = datetime.strptime(created_at or "", "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return float("inf")
    return (datetime.now(timezone.utc) - ts).total_seconds()


def _get_or_generate(
    cfg: Config, config: Config, cache_id: str, lang: str, *,
    version: int, generate_fn, valid_fn, kind_label: str,
) -> dict | None:
    """Nội dung quest (dict) — cache-first, USER-AGNOSTIC; None nếu không dựng được.

    Khuôn chung Role-play/Story (practice.py-style): đọc cache còn hạn + hợp lệ →
    trả luôn; else gọi `generate_fn()` (LLM), validate qua `valid_fn`, ghi cache.
    LLM lỗi / nội dung không hợp lệ → None (fail-soft, caller ẩn quest)."""
    entry = store.get_lesson_content_cache(cfg, cache_id, lang)
    if (
        entry
        and entry.get("cache_version") == version
        and _cache_age_seconds(entry.get("created_at")) < _CACHE_TTL_SECONDS
        and valid_fn(entry["content"])
    ):
        return entry["content"]

    try:
        content = generate_fn()
    except Exception:  # noqa: BLE001 — LLM lỗi không chặn khóa học
        logger.exception("Lỗi sinh %s cho %s (bỏ qua)", kind_label, cache_id)
        return None

    if not valid_fn(content):
        logger.warning("%s %s không hợp lệ — bỏ qua", kind_label, cache_id)
        return None

    try:
        model = config.local_model if config.is_local else config.model
        store.put_lesson_content_cache(cfg, cache_id, lang, content, model, version)
    except Exception:  # noqa: BLE001
        logger.exception("Lỗi ghi cache %s (bỏ qua)", kind_label)
    return content


# ── Role-play (Phase 3B) ─────────────────────────────────────────────────


def _roleplay_valid(content: dict) -> bool:
    """Kịch bản hợp lệ khi có ≥_MIN lượt và mỗi expected_user KHÔNG rỗng."""
    turns = content.get("turns") or []
    if len(turns) < _MIN_ROLEPLAY_TURNS:
        return False
    return all((t.get("expected_user") or "").strip() for t in turns)


def _roleplay_content(script: RolePlayScript) -> dict:
    return {
        "scenario": script.scenario or "",
        "role_user": script.role_user or "",
        "role_npc": script.role_npc or "",
        "turns": [
            {"npc": t.npc or "", "expected_user": t.expected_user or "", "hint": t.hint or ""}
            for t in script.turns
        ],
    }


def build_roleplay(
    cfg: Config, config: Config, exam: str, slug: str, lang: str
) -> dict | None:
    """Kịch bản Role-play cho (exam, topic) — cache-first; None nếu không dựng được.

    Trả dict {scenario, role_user, role_npc, turns:[{npc, expected_user, hint}]}.
    LLM lỗi / kịch bản không hợp lệ → None (caller ẩn quest, KHÔNG chặn)."""
    topic = get_roleplay_topic(exam, slug)
    if topic is None:
        return None
    from ..suggest import suggest_roleplay

    return _get_or_generate(
        cfg, config, roleplay_quest_id(exam, slug), lang,
        version=_ROLEPLAY_CACHE_VERSION,
        generate_fn=lambda: _roleplay_content(suggest_roleplay(config, exam, topic.setting)),
        valid_fn=_roleplay_valid,
        kind_label="Kịch bản Role-play",
    )


# ── Story (Phase 3C) ─────────────────────────────────────────────────────


def _story_valid(content: dict) -> bool:
    """Truyện hợp lệ khi có ≥_MIN đoạn và mỗi đoạn ≥_MIN từ."""
    segments = content.get("segments") or []
    if len(segments) < _MIN_STORY_SEGMENTS:
        return False
    return all(
        len((s.get("text") or "").split()) >= _MIN_STORY_SEGMENT_WORDS for s in segments
    )


def _story_content(story: "StoryQuest") -> dict:
    return {
        "title": story.title or "",
        "segments": [{"text": s.text or ""} for s in story.segments],
    }


def build_story(
    cfg: Config, config: Config, exam: str, slug: str, lang: str
) -> dict | None:
    """Truyện đọc-to cho (exam, topic) — cache-first; None nếu không dựng được.

    Trả dict {title, segments:[{text}]}. LLM lỗi / truyện không hợp lệ → None."""
    topic = get_story_topic(exam, slug)
    if topic is None:
        return None
    from ..suggest import suggest_story

    return _get_or_generate(
        cfg, config, story_quest_id(exam, slug), lang,
        version=_STORY_CACHE_VERSION,
        generate_fn=lambda: _story_content(suggest_story(config, exam, topic.setting)),
        valid_fn=_story_valid,
        kind_label="Truyện đọc-to",
    )

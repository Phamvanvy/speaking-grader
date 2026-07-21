"""Sinh khóa học cá nhân hóa — hàm THUẦN: syllabus + hồ sơ chẩn đoán + tiến độ
đã lưu → view model cho frontend.

Cá nhân hóa (chốt với user):
- priority mỗi lesson 0-1 (cao = yếu = ưu tiên):
  · pronunciation: MAX weakness của âm trong nhóm xuất hiện ở ranking
    get_weak_phonemes (âm hiếm đã bị lọc _MIN_ATTEMPTS ở phoneme_profile nên
    không đẩy sai); fallback (chưa đủ data) → 0.5.
  · rubric / question_type: 1 - mastery của target; chưa đủ data → 0.5.
- Thứ tự & mở khóa: Unit song song, TUẦN TỰ trong Unit. Unit xếp theo priority
  trung bình (yếu trước). Trong mỗi Unit, lesson xếp theo priority (yếu trước) và
  mở khóa tuần tự: lesson kế available chỉ khi lesson liền trước trong CÙNG Unit
  đã done. Tiến độ đã lưu (done/in_progress) ghi đè.
- Badge "Nên học": top-K lesson đang mở (chưa done) yếu nhất TOÀN course.

Ngưỡng "done" theo dimension (điểm đã chuẩn hóa 0-1) — mark_lesson_complete
(src/course/__init__.py) dùng bảng này.
"""

from __future__ import annotations

from ..phoneme.ipa.ko.phoneme_set_ko import normalize_ipa_ko
from ..phoneme.ipa.phoneme_set import normalize_ipa
from ..rubrics.base import exam_language
from .syllabus import PHONEME_GROUPS, SYLLABUS, Lesson, Unit


def _norm(symbol: str, lang: str) -> str:
    return normalize_ipa_ko(symbol) if lang == "ko" else normalize_ipa(symbol)

# Ngưỡng đạt để đánh dấu lesson 'done' — điểm đã chuẩn hóa 0-1. Phát âm khớt hơn
# (âm rõ đúng/sai); rubric & dạng câu 0.67 ≈ 2/3 TOEIC / 6/9 IELTS (mức khá).
DONE_THRESHOLD: dict[str, float] = {
    "pronunciation": 0.80,
    "rubric": 0.67,
    "question_type": 0.67,
}

# Số lesson gắn badge "Nên học" (ưu tiên cao nhất, đang mở).
_PRIORITY_BADGE_K = 3
# Số lần chấm thật tối thiểu để TỰ ĐỘNG hoàn thành 1 lesson rubric/qtype từ
# mastery (đủ bằng chứng, không phải may rủi 1 bài).
AUTO_COMPLETE_MIN_ATTEMPTS = 3.0
# Priority mặc định khi chưa đủ dữ liệu để biết yếu hay không.
_UNKNOWN_PRIORITY = 0.5
# Priority khi 1 nhóm âm hoàn toàn không xuất hiện trong hồ sơ (không tín hiệu).
_NO_SIGNAL_PRIORITY = 0.3


def _weak_map(weak_phonemes: list[dict]) -> dict[str, float]:
    """symbol → weakness 0-1. Symbol GIỮ NGUYÊN (get_weak_phonemes đã chuẩn hoá
    theo lang đúng). error_rate làm weakness; entry fallback (None) → _UNKNOWN."""
    out: dict[str, float] = {}
    for w in weak_phonemes or []:
        sym = w.get("symbol")
        if not sym:
            continue
        rate = w.get("error_rate")
        out[sym] = float(rate) if isinstance(rate, (int, float)) else _UNKNOWN_PRIORITY
    return out


def _lesson_priority(
    lesson: Lesson, mastery: dict, weak: dict[str, float], lang: str
) -> float:
    """priority 0-1 của 1 lesson theo dimension. `lang` chuẩn hoá symbol nhóm âm
    khớp với key của weak (en: normalize_ipa; ko: normalize_ipa_ko)."""
    if lesson.dimension == "pronunciation":
        present = [
            weak[_norm(s, lang)]
            for s in PHONEME_GROUPS.get(lesson.target, [])
            if _norm(s, lang) in weak
        ]
        return max(present) if present else _NO_SIGNAL_PRIORITY
    bucket = "criteria" if lesson.dimension == "rubric" else "question_types"
    entry = (mastery.get(bucket) or {}).get(lesson.target)
    if not entry or entry.get("weakness") is None:
        return _UNKNOWN_PRIORITY
    return float(entry["weakness"])


def _lesson_view(
    lesson: Lesson, status: str, priority: float, prog: dict | None
) -> dict:
    return {
        "id": lesson.id,
        "title": lesson.title,
        "dimension": lesson.dimension,
        "target": lesson.target,
        "description": lesson.description,
        "est_minutes": lesson.est_minutes,
        "status": status,
        "priority": round(priority, 3),
        "focus": False,  # badge "Nên học" — gắn sau khi xếp toàn course
        "best_score": (prog or {}).get("best_score"),
        "attempts": (prog or {}).get("attempts", 0),
    }


def auto_completions(exam: str, mastery: dict) -> list[tuple[str, float]]:
    """(lesson_id, mastery) cho các lesson rubric/qtype ĐÃ đạt ngưỡng done từ bài
    chấm THẬT (đủ AUTO_COMPLETE_MIN_ATTEMPTS lần). Phát âm không tự hoàn thành —
    người học tự luyện. Dùng để khép vòng "khóa học theo kết quả test"."""
    out: list[tuple[str, float]] = []
    for unit in SYLLABUS.get(exam, []):
        if unit.dimension == "pronunciation":
            continue
        bucket = "criteria" if unit.dimension == "rubric" else "question_types"
        threshold = DONE_THRESHOLD[unit.dimension]
        for ls in unit.lessons:
            entry = (mastery.get(bucket) or {}).get(ls.target)
            if not entry:
                continue
            if (
                entry.get("attempts", 0) >= AUTO_COMPLETE_MIN_ATTEMPTS
                and entry.get("mastery", 0) >= threshold
            ):
                out.append((ls.id, float(entry["mastery"])))
    return out


def build_course(
    exam: str,
    mastery: dict,
    weak_phonemes: list[dict],
    progress: dict[str, dict],
    activity: dict,
) -> dict:
    """View model khóa học cá nhân hóa cho 1 kỳ thi."""
    weak = _weak_map(weak_phonemes)
    lang = exam_language(exam)
    units_src: list[Unit] = SYLLABUS.get(exam, [])

    # 1) priority mỗi lesson.
    prio: dict[str, float] = {}
    for unit in units_src:
        for ls in unit.lessons:
            prio[ls.id] = _lesson_priority(ls, mastery, weak, lang)

    # 2) Xếp Unit theo priority: MAX lesson (điểm yếu mạnh nhất nổi lên — tránh 1
    #    tín hiệu mạnh bị loãng bởi các lesson no-signal), rồi mean, rồi thứ tự
    #    syllabus. Nhất quán với chủ trương "max weakness" của mảng phát âm.
    def _unit_rank(iu: tuple[int, Unit]) -> tuple[float, float, int]:
        idx, unit = iu
        ps = [prio[ls.id] for ls in unit.lessons] or [0.0]
        return (-max(ps), -(sum(ps) / len(ps)), idx)

    unit_order = sorted(enumerate(units_src), key=_unit_rank)

    done_count = 0
    total = 0
    by_dim: dict[str, dict[str, int]] = {}
    out_units: list[dict] = []
    all_lesson_views: list[dict] = []

    for orig_idx, unit in unit_order:
        # 3) Trong Unit: xếp lesson theo priority (yếu trước), tie-break syllabus.
        ordered = sorted(
            enumerate(unit.lessons), key=lambda il: (-prio[il[1].id], il[0])
        )
        prev_done = True  # lesson đầu trong Unit luôn mở
        lesson_views: list[dict] = []
        for _, ls in ordered:
            prog = progress.get(ls.id)
            if prog:
                status = prog["status"]  # 'done' | 'in_progress'
            elif prev_done:
                status = "available"
            else:
                status = "locked"
            view = _lesson_view(ls, status, prio[ls.id], prog)
            lesson_views.append(view)
            all_lesson_views.append(view)

            total += 1
            dim = ls.dimension
            by_dim.setdefault(dim, {"done": 0, "total": 0})
            by_dim[dim]["total"] += 1
            if status == "done":
                done_count += 1
                by_dim[dim]["done"] += 1
            # Gate mở khóa lesson kế: chỉ khi lesson này ĐÃ done.
            prev_done = status == "done"

        out_units.append({
            "id": unit.id,
            "title": unit.title,
            "dimension": unit.dimension,
            "lessons": lesson_views,
        })

    # 4) Badge "Nên học": top-K lesson đang mở (available/in_progress) yếu nhất.
    open_lessons = [
        v for v in all_lesson_views if v["status"] in ("available", "in_progress")
    ]
    for v in sorted(open_lessons, key=lambda v: -v["priority"])[:_PRIORITY_BADGE_K]:
        v["focus"] = True

    return {
        "exam": exam,
        "progress": {
            "done": done_count,
            "total": total,
            "pct": round(done_count / total, 3) if total else 0.0,
            "by_dimension": by_dim,
        },
        "streak": {
            "days": activity.get("streak_days", 0),
            "longest": activity.get("longest_streak", 0),
            "last_active_day": activity.get("last_active_day"),
            "total_completed": activity.get("total_completed", 0),
        },
        "units": out_units,
    }

"""Hồ sơ mastery per-user cho khóa học — tổng hợp CHỈ-ĐỌC từ output chấm điểm.

Song song với src/phoneme_profile.py (mảng phát âm), module này tổng hợp 2 mảng
CÒN LẠI của khóa học:
- Tiêu chí rubric (grammar/vocab/fluency…): điểm `scores.criteria[*].score`.
- Dạng câu: điểm tổng ước tính của bài (`estimated_*_score`).

Nguồn evidence: history result_json quét TĂNG DẦN bằng con trỏ composite
(created_at, id) lưu ở course.db (course_scan_state), tally cộng dồn vào
criterion_stats/qtype_stats (mirror phoneme_profile → phoneme_stats). Không hook
vào save path: tự backfill history cũ, xoá record chỉ ngừng thêm evidence.

QUYẾT ĐỊNH CHỊU LỰC — chuẩn hóa thang + tách theo đề: điểm tiêu chí 0-3 (TOEIC)
vs 0-9 (IELTS), tên tiêu chí trùng (pronunciation ở cả hai). Nên mọi tally key
theo (exam, …) và điểm chuẩn hóa về 0-1 TRƯỚC khi cộng. `mastery =
score_sum/attempts` là số 0-1; `weakness = 1 - mastery`.

Consumer: src/course/generate.py (build_course).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .. import history
from ..config import Config
from ..phoneme_profile import _recency_weight  # tái dùng: già hóa evidence như nhau
from ..rubrics.base import exam_score_field, exam_score_max
from . import store

logger = logging.getLogger("toeic.course.profile")

# Thang điểm TỐI ĐA của MỖI tiêu chí theo kỳ thi (khác max điểm TỔNG). Chỉ các
# kỳ có khóa học; exam khác → bỏ qua tally (chưa hỗ trợ). TOPIK: rubric 0-5.
CRITERION_MAX: dict[str, float] = {"toeic": 3.0, "ielts": 9.0, "topik": 5.0}

# Ngưỡng đủ mẫu để coi mastery là đáng tin (dưới ngưỡng → weakness=None =
# "chưa đủ dữ liệu"); thấp hơn phoneme (âm cần nhiều mẫu hơn tiêu chí/bài).
_MIN_ATTEMPTS = 2.0


def _norm_criteria(result: dict, exam: str) -> list[tuple[str, float]]:
    """(criterion, score_chuẩn_hóa_0-1) từ scores.criteria; [] nếu thiếu/không hỗ trợ."""
    crit_max = CRITERION_MAX.get(exam)
    if not crit_max:
        return []
    scores = result.get("scores") or {}
    out: list[tuple[str, float]] = []
    for c in scores.get("criteria") or []:
        if not isinstance(c, dict):
            continue
        name = c.get("criterion")
        raw = c.get("score")
        if not name or not isinstance(raw, (int, float)):
            continue
        out.append((name, max(0.0, min(1.0, raw / crit_max))))
    return out


def _norm_overall(result: dict, exam: str) -> float | None:
    """Điểm TỔNG của bài chuẩn hóa 0-1 (overall/exam_score_max); None nếu thiếu."""
    scores = result.get("scores") or {}
    raw = scores.get(exam_score_field(exam))
    if not isinstance(raw, (int, float)):
        return None
    return max(0.0, min(1.0, raw / exam_score_max(exam)))


def _tally_result(
    result: dict,
    crit_t: dict[tuple[str, str], dict],
    qt_t: dict[tuple[str, str], dict],
    weight: float,
) -> None:
    """Tally 1 grading result vào crit_t/qt_t (weighted).

    Guard .get mọi tầng: record cha của exam chỉ là summary (không scores) → bỏ
    qua tự nhiên. exam mặc định 'toeic' khớp report.build_output.
    """
    if not isinstance(result, dict):
        return
    exam = (result.get("exam") or "toeic").strip().lower()
    if exam not in CRITERION_MAX:
        return  # kỳ thi chưa có khóa học (vd topik) → không tally
    for name, norm in _norm_criteria(result, exam):
        t = crit_t.setdefault((exam, name), {"attempts": 0.0, "score_sum": 0.0})
        t["attempts"] += weight
        t["score_sum"] += weight * norm

    qtype = result.get("question_type")
    overall = _norm_overall(result, exam)
    if qtype and overall is not None:
        t = qt_t.setdefault((exam, qtype), {"attempts": 0.0, "score_sum": 0.0})
        t["attempts"] += weight
        t["score_sum"] += weight * overall


def refresh_mastery(cfg: Config, user_id: str) -> None:
    """Quét phần history mới hơn con trỏ, cộng dồn vào criterion_stats/qtype_stats.

    Mỗi lượt tối đa 1 batch (300 blob) — backfill account lớn hội tụ qua vài
    request; steady-state no-op O(records mới). (Mirror
    phoneme_profile.refresh_profile.)
    """
    since_at, since_id = store.get_scan_cursor(cfg, user_id)
    results, cursor = history.list_results_since(cfg, user_id, since_at, since_id)
    if cursor == (since_at, since_id):
        return  # không có gì mới
    now = datetime.now(timezone.utc)
    crit_t: dict[tuple[str, str], dict] = {}
    qt_t: dict[tuple[str, str], dict] = {}
    for r in results:
        _tally_result(
            r["result"], crit_t, qt_t, weight=_recency_weight(r["created_at"], now)
        )
    # Apply cả khi tallies rỗng (blob không có scores) để con trỏ vẫn tiến.
    store.apply_mastery_tallies(cfg, user_id, crit_t, qt_t, cursor)


def get_mastery(cfg: Config, user_id: str, exam: str) -> dict:
    """Mastery của user cho 1 kỳ thi (đã refresh trước).

    Trả {'criteria': {crit: {mastery, attempts, weakness}},
         'question_types': {qtype: {mastery, attempts, weakness}}}.
    mastery = score_sum/attempts (0-1); weakness = 1 - mastery. Dưới _MIN_ATTEMPTS
    → weakness=None (chưa đủ dữ liệu — build_course coi là "cần chẩn đoán").
    """
    stats = store.get_mastery_stats(cfg, user_id, exam)

    def _summ(raw: dict[str, dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for key, s in raw.items():
            attempts = s["attempts"]
            if attempts <= 0:
                continue
            mastery = s["score_sum"] / attempts
            enough = attempts >= _MIN_ATTEMPTS
            out[key] = {
                "mastery": round(mastery, 3),
                "attempts": round(attempts, 1),
                "weakness": round(1.0 - mastery, 3) if enough else None,
            }
        return out

    return {
        "criteria": _summ(stats["criteria"]),
        "question_types": _summ(stats["question_types"]),
    }

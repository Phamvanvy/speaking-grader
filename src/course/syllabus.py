"""Giáo trình khóa học (data TĨNH) — Unit → Lesson cho TOEIC + IELTS.

Module data Python (không JSON) để tham chiếu hằng `rubrics` và được type-check.
Giáo trình là KHUNG cố định; thứ tự/ưu tiên/mở khóa cá nhân hóa nằm ở
src/course/generate.py (build_course), KHÔNG ở đây.

3 loại Unit mỗi kỳ thi:
- pronunciation: mỗi Lesson = 1 nhóm âm (PHONEME_GROUPS); khớp âm yếu từ
  phoneme_profile.get_weak_phonemes.
- rubric: mỗi Lesson = 1 tiêu chí chấm (criterion) của kỳ thi, lấy từ registry
  rubrics (nguồn hằng duy nhất — không chép tay label).
- question_type: mỗi Lesson = 1 dạng câu (question type) của kỳ thi.

Thêm TOPIK sau = thêm PHONEME_GROUPS âm Hàn + entry SYLLABUS['topik'], KHÔNG đổi
logic (đúng hợp đồng mở rộng của EXAM_REGISTRIES).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..rubrics import EXAM_REGISTRIES, list_question_types, resolve_question_type

# ── Nhóm âm cho Unit phát âm ──────────────────────────────────────────────
# key nhóm → danh sách IPA. Seed từ phoneme_profile.FALLBACK_WEAK_PHONEMES (điểm
# nóng L1 Việt: fricative/affricate + nguyên âm ngắn hay nhầm). Khớp với symbol
# của get_weak_phonemes qua normalize_ipa (generate.py chuẩn hóa cả hai vế).
PHONEME_GROUPS: dict[str, list[str]] = {
    "th_family": ["θ", "ð"],
    "s_z": ["s", "z"],
    "sh_ch_j": ["ʃ", "tʃ", "dʒ"],
    "v_w": ["v", "w"],
    "short_vowels": ["ɪ", "æ", "ʊ"],
}

# Tiêu đề + mô tả nhóm âm (tiếng Việt) cho UI.
_PHONEME_GROUP_META: dict[str, tuple[str, str]] = {
    "th_family": ("Âm /θ/ và /ð/ (th)", "Cặp âm 'th' hữu thanh/vô thanh — hay bị thay bằng /t/, /d/, /s/, /z/."),
    "s_z": ("Âm /s/ và /z/", "Phân biệt /s/–/z/, nhất là âm cuối từ (miss vs is)."),
    "sh_ch_j": ("Âm /ʃ/, /tʃ/, /dʒ/", "Nhóm âm xát/tắc-xát: sh, ch, j — dễ lẫn với /s/, /z/."),
    "v_w": ("Âm /v/ và /w/", "Âm môi-răng /v/ vs môi-môi /w/ — người Việt hay lẫn."),
    "short_vowels": ("Nguyên âm ngắn /ɪ/, /æ/, /ʊ/", "Nguyên âm ngắn hay bị kéo dài hoặc nhầm (ship/sheep, bad/bed)."),
}


@dataclass(frozen=True)
class Lesson:
    id: str                       # duy nhất, ổn định: '<exam>.<dim>.<slug>'
    title: str
    dimension: str                # 'pronunciation' | 'rubric' | 'question_type'
    target: str                   # phoneme-group key | criterion key | question_type key
    exam: str                     # 'toeic' | 'ielts'
    description: str = ""
    tips: tuple[str, ...] = ()     # coaching tĩnh (rubric); phát âm dùng phonemeTips
    est_minutes: int = 5


@dataclass(frozen=True)
class Unit:
    id: str
    title: str
    exam: str
    dimension: str
    lessons: tuple[Lesson, ...] = field(default_factory=tuple)


# ── Coaching tĩnh cho Unit rubric ─────────────────────────────────────────
# Vài gạch đầu dòng "cần cải thiện gì" theo criterion key (dùng chung 2 kỳ).
_CRITERION_TIPS: dict[str, tuple[str, ...]] = {
    "pronunciation": (
        "Đọc rõ từng âm, chú ý âm cuối từ (endings) hay bị nuốt.",
        "Ghi âm và so với audio mẫu để nghe khác biệt.",
    ),
    "intonation_stress": (
        "Nhấn đúng trọng âm từ (record → nhấn syllable nào).",
        "Lên/xuống giọng cuối câu: câu hỏi Yes/No lên giọng.",
    ),
    "grammar": (
        "Chú ý thì động từ và chia số ít/nhiều.",
        "Nói câu đầy đủ chủ–vị, tránh câu cụt.",
    ),
    "vocabulary": (
        "Đa dạng từ vựng, tránh lặp một từ nhiều lần.",
        "Học collocation theo chủ đề (make a decision, heavy rain).",
    ),
    "cohesion": (
        "Dùng từ nối: first, then, because, however, for example.",
        "Sắp ý theo trình tự, tránh liệt kê rời rạc.",
    ),
    "relevance": (
        "Trả lời đúng trọng tâm câu hỏi trước khi mở rộng.",
        "Đủ ý: nêu quan điểm + lý do + ví dụ.",
    ),
    "organization": (
        "Bố cục mở – thân – kết rõ ràng.",
        "Nêu lập trường → lý do → ví dụ → kết luận ngắn.",
    ),
    "fluency_coherence": (
        "Nói liền mạch, giảm ngập ngừng và 'ừm/à'.",
        "Nối ý bằng linking words để mạch lạc.",
    ),
    "lexical_resource": (
        "Paraphrase thay vì lặp từ; dùng collocation tự nhiên.",
        "Mở rộng vốn từ theo chủ đề thường gặp.",
    ),
    "grammatical_range": (
        "Trộn câu đơn và câu phức (mệnh đề quan hệ, điều kiện).",
        "Giữ đúng thì khi kể/mô tả.",
    ),
}


def _pronunciation_unit(exam: str) -> Unit:
    lessons = []
    for gkey in PHONEME_GROUPS:
        title, desc = _PHONEME_GROUP_META.get(gkey, (gkey, ""))
        lessons.append(
            Lesson(
                id=f"{exam}.pron.{gkey}",
                title=title,
                dimension="pronunciation",
                target=gkey,
                exam=exam,
                description=desc,
                est_minutes=5,
            )
        )
    return Unit(
        id=f"{exam}.pron",
        title="Phát âm (âm khó)",
        exam=exam,
        dimension="pronunciation",
        lessons=tuple(lessons),
    )


def _unique_criteria(exam: str) -> list[tuple[str, str, str]]:
    """(key, label, description) các tiêu chí của kỳ thi, gộp từ mọi dạng câu,
    giữ thứ tự xuất hiện đầu tiên (nguồn: registry rubrics — không chép tay)."""
    seen: dict[str, tuple[str, str, str]] = {}
    for qt in EXAM_REGISTRIES[exam].values():
        for c in qt.criteria:
            if c.key not in seen:
                seen[c.key] = (c.key, c.label, c.description)
    return list(seen.values())


def _rubric_unit(exam: str) -> Unit:
    lessons = []
    for key, label, desc in _unique_criteria(exam):
        lessons.append(
            Lesson(
                id=f"{exam}.rubric.{key}",
                title=label,
                dimension="rubric",
                target=key,
                exam=exam,
                description=desc,
                tips=_CRITERION_TIPS.get(key, ()),
                est_minutes=6,
            )
        )
    return Unit(
        id=f"{exam}.rubric",
        title="Tiêu chí chấm điểm",
        exam=exam,
        dimension="rubric",
        lessons=tuple(lessons),
    )


def _question_type_unit(exam: str) -> Unit:
    lessons = []
    for key in list_question_types(exam):
        qt = resolve_question_type(key, exam)
        lessons.append(
            Lesson(
                id=f"{exam}.qtype.{key}",
                title=qt.label,
                dimension="question_type",
                target=key,
                exam=exam,
                description="Luyện theo dạng câu này với câu trả lời mẫu.",
                est_minutes=8,
            )
        )
    return Unit(
        id=f"{exam}.qtype",
        title="Theo dạng câu",
        exam=exam,
        dimension="question_type",
        lessons=tuple(lessons),
    )


def _build_syllabus() -> dict[str, list[Unit]]:
    out: dict[str, list[Unit]] = {}
    for exam in ("toeic", "ielts"):
        out[exam] = [
            _pronunciation_unit(exam),
            _rubric_unit(exam),
            _question_type_unit(exam),
        ]
    return out


SYLLABUS: dict[str, list[Unit]] = _build_syllabus()

# Kỳ thi có khóa học (dùng để validate exam param ở API).
SUPPORTED_EXAMS: tuple[str, ...] = tuple(SYLLABUS)


def all_lessons(exam: str) -> list[Lesson]:
    """Mọi Lesson của 1 kỳ thi (phẳng, theo thứ tự Unit)."""
    return [ls for unit in SYLLABUS.get(exam, []) for ls in unit.lessons]


def get_lesson(lesson_id: str) -> Lesson | None:
    """Lesson theo id (tra qua index dựng sẵn); None nếu không có."""
    return _LESSON_INDEX.get(lesson_id)


def _build_index() -> dict[str, Lesson]:
    index: dict[str, Lesson] = {}
    for exam, units in SYLLABUS.items():
        for unit in units:
            for ls in unit.lessons:
                if ls.id in index:
                    raise ValueError(f"Lesson id trùng trong syllabus: {ls.id}")
                # target phải resolve được (fail-fast lúc import).
                if ls.dimension == "pronunciation" and ls.target not in PHONEME_GROUPS:
                    raise ValueError(f"Lesson {ls.id}: nhóm âm '{ls.target}' không tồn tại.")
                index[ls.id] = ls
    return index


_LESSON_INDEX: dict[str, Lesson] = _build_index()

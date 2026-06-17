"""Rubric TOEIC Speaking dưới dạng config.

Mỗi dạng câu (question type) ánh xạ sang tập tiêu chí áp dụng + mô tả thang
điểm. Thêm IELTS sau chỉ cần tạo rubrics/ielts.py, không sửa logic chấm.

Bản đầu (Phase 1) implement đầy đủ READ_ALOUD; các dạng khác đã khai báo
khung để mở rộng dần.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class QuestionType:
    key: str
    label: str
    criteria: list[Criterion]
    scale_description: str
    # Hướng dẫn riêng cho dạng câu này (đưa vào system prompt)
    guidance: str = ""
    uses_reference_script: bool = False


# --- Các tiêu chí dùng chung -------------------------------------------------

PRONUNCIATION = Criterion(
    "pronunciation",
    "Pronunciation",
    "Độ rõ ràng và chính xác của âm; người nghe có hiểu dễ dàng không.",
)
INTONATION_STRESS = Criterion(
    "intonation_stress",
    "Intonation & stress",
    "Ngữ điệu, trọng âm từ/câu, nhịp điệu tự nhiên.",
)
GRAMMAR = Criterion(
    "grammar",
    "Grammar",
    "Độ chính xác và đa dạng của cấu trúc ngữ pháp.",
)
VOCABULARY = Criterion(
    "vocabulary",
    "Vocabulary",
    "Sự phong phú, chính xác và phù hợp của từ vựng.",
)
COHESION = Criterion(
    "cohesion",
    "Cohesion",
    "Tính mạch lạc, liên kết ý, dùng từ nối hợp lý.",
)
RELEVANCE = Criterion(
    "relevance",
    "Relevance & completeness",
    "Trả lời đúng và đủ các phần của yêu cầu.",
)
ORGANIZATION = Criterion(
    "organization",
    "Organization",
    "Bố cục câu trả lời (mở-thân-kết), lập luận có trật tự.",
)


SCALE_0_3 = (
    "Mỗi tiêu chí chấm theo thang 0-3: "
    "0 = không đạt/không trả lời; 1 = yếu, nhiều lỗi gây cản trở; "
    "2 = đạt, một vài lỗi nhưng vẫn hiểu; 3 = tốt, gần như không lỗi đáng kể."
)


# --- Định nghĩa các dạng câu -------------------------------------------------

READ_ALOUD = QuestionType(
    key="read_aloud",
    label="Read a text aloud (Q1-2)",
    criteria=[PRONUNCIATION, INTONATION_STRESS],
    scale_description=SCALE_0_3,
    uses_reference_script=True,
    guidance=(
        "Đây là bài đọc to một đoạn văn cho sẵn. Có script tham chiếu, vì vậy "
        "accuracy_metrics (WER, deletions = từ bị bỏ, substitutions = đọc sai, "
        "insertions = thêm từ, coverage = hits/reference_word_count) là bằng "
        "chứng KHÁCH QUAN quan trọng. Từ bị bỏ (deletions) nghiêm trọng hơn vì "
        "là phần đáng lẽ phải đọc. Chỉ chấm Pronunciation và Intonation & "
        "stress; KHÔNG chấm nội dung vì nội dung đã cho sẵn.\n\n"
        "HOÀN THÀNH BÀI (task_completion) cho Read Aloud — KHÁC các dạng khác:\n"
        "- KHÔNG tính theo thời lượng nói. Đọc nhanh nhưng đọc HẾT đoạn vẫn là "
        "hoàn thành đầy đủ; tuyệt đối KHÔNG hạ task_completion chỉ vì "
        "speaking_duration_sec ngắn hơn expected_duration_sec.\n"
        "- Thước đo hoàn thành là độ phủ script (coverage). coverage >= 0.95 và "
        "ít deletions → task_completion = high.\n"
        "- Nếu fail_reference_match = true (coverage < 0.50): thí sinh KHÔNG đọc "
        "đúng đoạn được giao (đọc nhầm đoạn khác hoặc thiếu phần lớn). Khi đó: "
        "task_completion = very_low; estimated_toeic_score phải bị phạt nặng, "
        "KHÔNG vượt khoảng thấp (≲ 80/200). Vẫn được mô tả pronunciation/"
        "intonation nhưng phát âm tốt KHÔNG được kéo điểm tổng lên.\n\n"
        "NHỊP ĐỌC (reading_pace.pace_ratio = thời lượng thực / kỳ vọng): CHỈ là "
        "bằng chứng phụ cho Intonation & stress (nhịp điệu/tốc độ), TUYỆT ĐỐI "
        "KHÔNG dùng cho task_completion. pace_ratio < 1 nghĩa là đọc nhanh hơn "
        "tham chiếu — có thể nhận xét về tốc độ/nhịp, nhưng đọc nhanh mà đủ nội "
        "dung (coverage cao) vẫn là hoàn thành đầy đủ."
    ),
)

DESCRIBE_PICTURE = QuestionType(
    key="describe_picture",
    label="Describe a picture (Q3-4)",
    criteria=[PRONUNCIATION, INTONATION_STRESS, GRAMMAR, VOCABULARY, COHESION],
    scale_description=SCALE_0_3,
    guidance="Mô tả tranh. Đánh giá thêm grammar, vocabulary, cohesion.",
)

RESPOND_QUESTIONS = QuestionType(
    key="respond_questions",
    label="Respond to questions (Q5-7)",
    criteria=[PRONUNCIATION, INTONATION_STRESS, GRAMMAR, VOCABULARY, RELEVANCE],
    scale_description=SCALE_0_3,
    guidance="Trả lời câu hỏi. Đặc biệt chú ý relevance & completeness.",
)

RESPOND_WITH_INFO = QuestionType(
    key="respond_with_info",
    label="Respond using information provided (Q8-10)",
    criteria=[PRONUNCIATION, INTONATION_STRESS, GRAMMAR, VOCABULARY, RELEVANCE],
    scale_description=SCALE_0_3,
    guidance="Trả lời dựa trên thông tin cho sẵn. Chú ý dùng đúng thông tin.",
)

EXPRESS_OPINION = QuestionType(
    key="express_opinion",
    label="Express an opinion (Q11)",
    criteria=[
        PRONUNCIATION,
        INTONATION_STRESS,
        GRAMMAR,
        VOCABULARY,
        RELEVANCE,
        ORGANIZATION,
    ],
    scale_description=SCALE_0_3,
    guidance=(
        "Trình bày ý kiến. Đánh giá đầy đủ các tiêu chí + organization. "
        "Trả lời quá ngắn (vd 'Yes, I think so.') phải bị hạ task_completion "
        "dù grammar/vocabulary có tốt."
    ),
)


_REGISTRY: dict[str, QuestionType] = {
    qt.key: qt
    for qt in [
        READ_ALOUD,
        DESCRIBE_PICTURE,
        RESPOND_QUESTIONS,
        RESPOND_WITH_INFO,
        EXPRESS_OPINION,
    ]
}


def get_question_type(key: str) -> QuestionType:
    if key not in _REGISTRY:
        raise KeyError(
            f"Không biết dạng câu '{key}'. Hợp lệ: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]

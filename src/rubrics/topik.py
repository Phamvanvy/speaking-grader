"""Rubric TOPIK 말하기 평가 (TOPIK Speaking) — registry tối thiểu M1.

M1 chỉ cần practice mode chạy được với ngữ cảnh exam hợp lệ: read_aloud (đọc to
câu tiếng Hàn, có script tham chiếu → chấm phát âm là chính) + q1_answer_question
(질문에 대답하기 — dạng 1 chính thức). Bộ 6 dạng đầy đủ (q2 그림 보고 역할 수행하기
… q6 의견 제시하기) + công thức estimated_topik_score đến ở M3.

Mirror cấu trúc toeic.py/ielts.py. 3 tiêu chí theo rubric chính thức của NIIED:
내용 및 과제 수행 / 언어 사용 / 발화 전달력.
"""

from __future__ import annotations

from .base import Criterion, Exam, QuestionType

# --- 3 tiêu chí chính thức TOPIK 말하기 ---------------------------------------

CONTENT_TASK = Criterion(
    "content_task",
    "내용 및 과제 수행 (Content & Task)",
    "Mức độ hoàn thành yêu cầu của đề: trả lời đúng trọng tâm, nội dung phong "
    "phú, triển khai ý mạch lạc và đủ độ dài yêu cầu.",
)
LANGUAGE_USE = Criterion(
    "language_use",
    "언어 사용 (Language Use)",
    "Độ chính xác và đa dạng của từ vựng/ngữ pháp tiếng Hàn; dùng cấu trúc và "
    "mức kính ngữ (높임말) phù hợp ngữ cảnh.",
)
DELIVERY = Criterion(
    "delivery",
    "발화 전달력 (Delivery)",
    "Phát âm rõ và tự nhiên (bao gồm biến âm chuẩn: 연음, 비음화, 경음화...), "
    "ngữ điệu phù hợp, tốc độ và độ trôi chảy ổn định.",
)

_ALL_CRITERIA = [CONTENT_TASK, LANGUAGE_USE, DELIVERY]

SCALE_TOPIK = (
    "Chấm MỖI tiêu chí trên thang 0-5 theo rubric TOPIK 말하기: "
    "5 = hoàn thành xuất sắc, ngôn ngữ chính xác/tự nhiên, phát âm rất rõ; "
    "4 = hoàn thành tốt, vài lỗi nhỏ không cản trở; "
    "3 = hoàn thành ở mức khá, lỗi thỉnh thoảng gây khó hiểu cục bộ; "
    "2 = hoàn thành một phần, lỗi thường xuyên, người nghe phải đoán; "
    "1 = nội dung rất hạn chế, khó hiểu; 0 = không trả lời/không liên quan. "
    "KHÔNG tự cho điểm tổng — hệ thống tính (M3)."
)


READ_ALOUD_KO = QuestionType(
    key="read_aloud",
    label="낭독 — Đọc to đoạn văn (practice)",
    criteria=[DELIVERY],
    scale_description=SCALE_TOPIK,
    exam=Exam.TOPIK.value,
    uses_reference_script=True,
    guidance=(
        "Thí sinh đọc to đoạn văn tiếng Hàn cho sẵn. Trọng tâm là ĐỘ CHÍNH XÁC "
        "PHÁT ÂM so với chuẩn phát âm (표준 발음법): biến âm bắt buộc (연음, "
        "비음화, 경음화, 구개음화...), phân biệt âm thường/căng/bật hơi "
        "(ㄱ/ㄲ/ㅋ), nguyên âm ㅓ/ㅗ, ㅡ/ㅜ. KHÔNG chấm nội dung — chỉ delivery."
    ),
    display_inputs=("reference",),
    required_inputs=("reference",),
)

Q1_ANSWER_QUESTION = QuestionType(
    key="q1_answer_question",
    label="문항 1 — 질문에 대답하기 (Trả lời câu hỏi)",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_TOPIK,
    exam=Exam.TOPIK.value,
    guidance=(
        "Dạng 1 TOPIK 말하기 (sơ cấp): nghe một câu hỏi đời thường ngắn (sở "
        "thích, gia đình, thời tiết...) và trả lời tự nhiên trong ~30 giây. "
        "KỲ VỌNG: trả lời ĐÚNG câu hỏi bằng 2-4 câu đơn giản, đúng đuôi câu "
        "(-아요/-어요/-습니다), từ vựng sơ cấp chính xác. Trả lời một từ / lạc "
        "đề → hạ content_task."
    ),
)

TOPIK_QUESTION_TYPES: dict[str, QuestionType] = {
    qt.key: qt
    for qt in (READ_ALOUD_KO, Q1_ANSWER_QUESTION)
}

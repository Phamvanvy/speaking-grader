"""Rubric TOPIK 말하기 평가 (TOPIK Speaking) — đủ 6 dạng câu chính thức (M3).

Mirror cấu trúc toeic.py/ielts.py. 3 tiêu chí theo rubric chính thức của NIIED:
내용 및 과제 수행 / 언어 사용 / 발화 전달력. LLM chấm mỗi tiêu chí 0-5; điểm tổng
0-200 (level 1-6) do code tính tất định trong scoring/compute.py
(_compute_topik_score), KHÔNG để LLM bốc số.

6 dạng chính thức chia 3 mức: 문항 1-2 sơ cấp (초급), 3-4 trung cấp (중급),
5-6 cao cấp (고급) — mức câu quyết định trần điểm per-question
(TOPIK_LEVEL_CAP) và trọng số khi gộp cả đề (TOPIK_OVERALL_WEIGHT), vì một
câu sơ cấp làm hoàn hảo không thể là bằng chứng cho năng lực 6급.
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
    "KHÔNG tự cho điểm tổng — hệ thống tính từ điểm tiêu chí."
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

Q2_ROLE_PLAY = QuestionType(
    key="q2_role_play",
    label="문항 2 — 그림 보고 역할 수행하기 (Nhìn tranh, thực hiện vai)",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_TOPIK,
    exam=Exam.TOPIK.value,
    guidance=(
        "Dạng 2 TOPIK 말하기 (sơ cấp): nhìn tranh mô tả một tình huống đời "
        "thường (hỏi đường, mua đồ, đặt món...) và nói ĐÚNG VAI được giao "
        "trong tình huống đó, ~40 giây. KỲ VỌNG: câu nói phù hợp tình huống "
        "trong tranh và vai được giao (vd khách hỏi nhân viên), mức kính ngữ "
        "đúng quan hệ vai, từ vựng/mẫu câu sơ cấp chính xác. Nói lạc khỏi "
        "tình huống trong tranh hoặc sai vai → hạ content_task."
    ),
    display_inputs=("prompt", "image"),
    required_inputs=("prompt", "image"),
)

Q3_PICTURE_STORY = QuestionType(
    key="q3_picture_story",
    label="문항 3 — 그림 보고 이야기하기 (Nhìn tranh, kể chuyện)",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_TOPIK,
    exam=Exam.TOPIK.value,
    guidance=(
        "Dạng 3 TOPIK 말하기 (trung cấp): nhìn chuỗi tranh (thường 4 tranh) "
        "và kể lại thành một câu chuyện liền mạch theo đúng trình tự, ~60 "
        "giây. KỲ VỌNG: kể ĐỦ các tranh theo thứ tự, dùng thì quá khứ và từ "
        "nối trình tự (그래서, 그런데, -고 나서...), nhân vật/hành động nhất "
        "quán. Bỏ sót tranh, kể sai trình tự, hoặc chỉ liệt kê rời rạc → hạ "
        "content_task."
    ),
    display_inputs=("prompt", "image"),
    required_inputs=("prompt", "image"),
)

Q4_COMPLETE_DIALOGUE = QuestionType(
    key="q4_complete_dialogue",
    label="문항 4 — 대화 완성하기 (Hoàn thành hội thoại)",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_TOPIK,
    exam=Exam.TOPIK.value,
    uses_provided_info=True,
    guidance=(
        "Dạng 4 TOPIK 말하기 (trung cấp): nghe một đoạn hội thoại dở dang về "
        "chủ đề xã hội quen thuộc (đoạn thoại nằm trong provided_info) và nói "
        "TIẾP lượt lời của một nhân vật sao cho mạch lạc với nội dung trước "
        "đó, ~60 giây. KỲ VỌNG: bám đúng lập trường/tình huống của nhân vật "
        "được giao, phản hồi trực tiếp ý người kia vừa nói, lập luận có lý "
        "do, mức kính ngữ nhất quán với quan hệ trong thoại. Trả lời chung "
        "chung không ăn nhập với đoạn thoại → hạ content_task."
    ),
    display_inputs=("prompt", "provided_info"),
    required_inputs=("prompt", "provided_info"),
)

Q5_INTERPRET_DATA = QuestionType(
    key="q5_interpret_data",
    label="문항 5 — 자료 해석하기 (Diễn giải tư liệu)",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_TOPIK,
    exam=Exam.TOPIK.value,
    uses_provided_info=True,
    guidance=(
        "Dạng 5 TOPIK 말하기 (cao cấp): nhìn tư liệu (biểu đồ, bảng số liệu, "
        "poster — ảnh và/hoặc provided_info) về một vấn đề xã hội và trình "
        "bày ~80 giây: mô tả xu hướng/số liệu chính, rồi phân tích nguyên "
        "nhân hoặc triển vọng. KỲ VỌNG: nêu ĐÚNG các con số/xu hướng trong tư "
        "liệu (sai số liệu → hạ content_task), dùng văn phong trình bày trang "
        "trọng (-ㅂ니다, 것으로 보입니다, -는 것으로 나타났습니다), từ vựng "
        "phân tích (증가하다, 감소하다, 원인, 전망...). Bịa số liệu không có "
        "trong tư liệu → hạ content_task."
    ),
    display_inputs=("prompt", "image", "provided_info"),
    required_inputs=("prompt", "image", "provided_info"),
)

Q6_PRESENT_OPINION = QuestionType(
    key="q6_present_opinion",
    label="문항 6 — 의견 제시하기 (Trình bày ý kiến)",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_TOPIK,
    exam=Exam.TOPIK.value,
    guidance=(
        "Dạng 6 TOPIK 말하기 (cao cấp): nêu và bảo vệ ý kiến về một vấn đề xã "
        "hội trừu tượng (giáo dục, môi trường, công nghệ...), ~80 giây. KỲ "
        "VỌNG: lập trường rõ ràng, 2-3 lý do/ví dụ được triển khai logic, có "
        "mở-thân-kết, văn phong trang trọng (-ㅂ니다 hoặc -아/어요 nhất quán), "
        "ngữ pháp cao cấp (-(으)ㄹ 뿐만 아니라, -기 마련이다, 만약 -는다면...). "
        "Chỉ nêu lập trường mà không có lý do, hoặc lý do lặp/rời rạc → hạ "
        "content_task."
    ),
)

TOPIK_QUESTION_TYPES: dict[str, QuestionType] = {
    qt.key: qt
    for qt in (
        READ_ALOUD_KO,
        Q1_ANSWER_QUESTION,
        Q2_ROLE_PLAY,
        Q3_PICTURE_STORY,
        Q4_COMPLETE_DIALOGUE,
        Q5_INTERPRET_DATA,
        Q6_PRESENT_OPINION,
    )
}

# --- Mức câu → trần điểm + trọng số (dùng bởi scoring/compute.py) --------------
# Điểm tổng TOPIK official được scale IRT (không công bố công thức); đây là ƯỚC
# TÍNH NỘI BỘ theo nguyên tắc: câu sơ cấp chỉ là bằng chứng cho năng lực tối đa
# ~4급 (trần 130 = vừa chạm sàn 5급), câu trung cấp ~5급 (trần 170), câu cao cấp
# không trần. Cut-lines chính thức (công văn Bộ GD Hàn 2026, exam.topik.go.kr):
# 1급 20-49, 2급 50-89, 3급 90-109, 4급 110-129, 5급 130-159, 6급 160-200.
#
# read_aloud là practice mode (không phải câu thi official) — chấm delivery
# thuần nên không áp trần level (điểm hiển thị là chất lượng phát âm, không
# phải claim năng lực tổng hợp).
TOPIK_LEVEL_CAP: dict[str, int] = {
    Q1_ANSWER_QUESTION.key: 130,
    Q2_ROLE_PLAY.key: 130,
    Q3_PICTURE_STORY.key: 170,
    Q4_COMPLETE_DIALOGUE.key: 170,
    Q5_INTERPRET_DATA.key: 200,
    Q6_PRESENT_OPINION.key: 200,
    READ_ALOUD_KO.key: 200,
}

# Trọng số gộp cả đề: 초급:중급:고급 = 1:2:3 (câu khó đóng góp nhiều hơn vào
# nhận định level — phỏng theo cơ cấu 배점 tăng dần của đề official). Câu lạ /
# read_aloud (nếu lọt vào một đề tự soạn) → weight 1.
TOPIK_OVERALL_WEIGHT: dict[str, float] = {
    Q1_ANSWER_QUESTION.key: 1.0,
    Q2_ROLE_PLAY.key: 1.0,
    Q3_PICTURE_STORY.key: 2.0,
    Q4_COMPLETE_DIALOGUE.key: 2.0,
    Q5_INTERPRET_DATA.key: 3.0,
    Q6_PRESENT_OPINION.key: 3.0,
}

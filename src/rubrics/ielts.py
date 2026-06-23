"""Rubric IELTS Speaking dưới dạng config.

Mỗi Part (dạng câu) ánh xạ sang tập tiêu chí + mô tả thang band. IELTS Speaking
dùng 4 tiêu chí chính thức như nhau cho cả 3 Part, chấm trên band 0–9; điểm
overall = trung bình 4 tiêu chí làm tròn về 0.5 gần nhất (tính trong scoring.py).

Mirror cấu trúc của `toeic.py`; chỉ khác bộ tiêu chí, thang điểm và guidance.
"""

from __future__ import annotations

from .base import Criterion, Exam, QuestionType

# --- 4 tiêu chí chính thức IELTS Speaking ------------------------------------
# Key tiếng Anh ổn định (machine field), mô tả tiếng Việt cho người chấm.

FLUENCY_COHERENCE = Criterion(
    "fluency_coherence",
    "Fluency and Coherence",
    "Độ trôi chảy (nói liền mạch, ít ngập ngừng/tự sửa) và tính mạch lạc "
    "(triển khai ý logic, dùng từ nối hợp lý).",
)
LEXICAL_RESOURCE = Criterion(
    "lexical_resource",
    "Lexical Resource",
    "Sự phong phú, chính xác và linh hoạt của vốn từ; khả năng diễn giải "
    "(paraphrase) và dùng collocation/thành ngữ phù hợp.",
)
GRAMMATICAL_RANGE = Criterion(
    "grammatical_range",
    "Grammatical Range and Accuracy",
    "Độ đa dạng của cấu trúc (đơn/phức) và độ chính xác ngữ pháp; lỗi ảnh "
    "hưởng nghĩa bị trừ.",
)
PRONUNCIATION = Criterion(
    "pronunciation",
    "Pronunciation",
    "Độ rõ ràng, trọng âm, ngữ điệu và nhịp điệu; mức độ dễ nghe đối với "
    "người bản ngữ.",
)


SCALE_0_9 = (
    "Chấm MỖI tiêu chí trên thang BAND 0-9 của IELTS (cho phép bước 0.5): "
    "9 = expert (gần như hoàn hảo); 8 = very good (chỉ vài lỗi không hệ thống); "
    "7 = good (kiểm soát tốt, đôi lỗi); 6 = competent (hiệu quả dù còn lỗi, vẫn "
    "hiểu được); 5 = modest (lỗi thường xuyên gây khó khăn một phần); "
    "4 = limited (chỉ giao tiếp được chủ đề quen, lỗi nhiều); "
    "3-2 = extremely/intermittently limited; 1-0 = không đủ ngôn ngữ để đánh giá. "
    "Điểm overall do hệ thống tự tính (trung bình 4 tiêu chí, làm tròn 0.5) — "
    "KHÔNG tự cho điểm tổng."
)


# --- Định nghĩa 3 Part -------------------------------------------------------
# Cả 3 Part dùng đủ 4 tiêu chí; khác nhau ở kỳ vọng nội dung / độ sâu lập luận.

_ALL_CRITERIA = [
    FLUENCY_COHERENCE,
    LEXICAL_RESOURCE,
    GRAMMATICAL_RANGE,
    PRONUNCIATION,
]


PART1_INTERVIEW = QuestionType(
    key="part1_interview",
    label="Part 1 — Introduction & interview",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_0_9,
    exam=Exam.IELTS.value,
    guidance=(
        "Part 1: giám khảo hỏi các câu về chủ đề quen thuộc (bản thân, nhà cửa, "
        "công việc, sở thích...). Thí sinh trả lời ngắn-vừa, mở rộng câu trả lời "
        "bằng lý do/ví dụ. KHÔNG có script tham chiếu.\n\n"
        "KỲ VỌNG NỘI DUNG:\n"
        "- Trả lời ĐÚNG câu hỏi và có MỞ RỘNG (không chỉ 'Yes/No'). Trả lời cụt "
        "lủn, một từ → hạ Fluency & Coherence và task_completion.\n"
        "- Lạc đề / không hiểu câu hỏi → hạ content_relevance.\n\n"
        "CHẤM 4 TIÊU CHÍ trên band 0-9:\n"
        "- Fluency & Coherence: nói liền mạch, ít ngập ngừng dài; căn cứ "
        "speech_rate_wpm, pause_count, filler_count (bằng chứng phụ).\n"
        "- Lexical Resource: từ vựng đời thường chính xác, có cố gắng paraphrase.\n"
        "- Grammatical Range & Accuracy: trộn câu đơn/phức, đúng thì.\n"
        "- Pronunciation: rõ ràng, dễ nghe; dùng phoneme_data nếu có làm bằng "
        "chứng mạnh."
    ),
)

PART2_LONG_TURN = QuestionType(
    key="part2_long_turn",
    label="Part 2 — Long turn (cue card)",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_0_9,
    uses_provided_info=True,
    # Cue card nhập qua ô prompt (UI chưa có ô provided_info riêng); chấp nhận
    # cả provided_info nếu client gửi qua API.
    display_inputs=("prompt",),
    required_inputs=("prompt", "provided_info"),
    exam=Exam.IELTS.value,
    guidance=(
        "Part 2: thí sinh nhận một cue card (đưa qua provided_info — chủ đề + các "
        "ý gợi ý) và nói ĐỘC THOẠI liên tục 1-2 phút sau ~1 phút chuẩn bị. KHÔNG "
        "có script tham chiếu.\n\n"
        "KỲ VỌNG NỘI DUNG (đối chiếu provided_info):\n"
        "- Bao quát các ý gợi ý trên cue card và nói ĐỦ THỜI LƯỢNG (~1.5-2 phút). "
        "Nói quá ngắn so với expected_duration_sec, bỏ phần lớn ý gợi ý, hoặc "
        "dừng sớm → hạ task_completion (very_low/low) DÙ ngôn ngữ tốt.\n"
        "- Lạc khỏi chủ đề cue card → hạ content_relevance.\n\n"
        "CHẤM 4 TIÊU CHÍ trên band 0-9:\n"
        "- Fluency & Coherence (TRỌNG TÂM Part 2): duy trì lời nói liên tục, có "
        "mở-thân-kết, dùng từ nối; ngập ngừng/đứt quãng nhiều làm giảm mạnh. Căn "
        "cứ speech_rate_wpm, pause_count, longest_pause_sec, filler_count.\n"
        "- Lexical Resource: từ vựng đa dạng theo chủ đề, paraphrase, collocation.\n"
        "- Grammatical Range & Accuracy: câu phức, đúng thì khi kể chuyện/mô tả.\n"
        "- Pronunciation: rõ, ngữ điệu tự nhiên; dùng phoneme_data nếu có."
    ),
)

PART3_DISCUSSION = QuestionType(
    key="part3_discussion",
    label="Part 3 — Two-way discussion",
    criteria=_ALL_CRITERIA,
    scale_description=SCALE_0_9,
    exam=Exam.IELTS.value,
    guidance=(
        "Part 3: thảo luận các câu hỏi TRỪU TƯỢNG, mở rộng từ chủ đề Part 2 "
        "(xu hướng xã hội, so sánh, nguyên nhân-hệ quả, quan điểm). Đòi hỏi LẬP "
        "LUẬN SÂU hơn Part 1. KHÔNG có script tham chiếu.\n\n"
        "KỲ VỌNG NỘI DUNG:\n"
        "- Phải PHÁT TRIỂN ý: nêu quan điểm + lý do + ví dụ/giải thích, biết so "
        "sánh và suy đoán (speculate). Trả lời hời hợt, ngắn, không lập luận → hạ "
        "Fluency & Coherence và task_completion.\n"
        "- Né câu hỏi / lạc đề → hạ content_relevance.\n\n"
        "CHẤM 4 TIÊU CHÍ trên band 0-9:\n"
        "- Fluency & Coherence: triển khai ý dài mạch lạc, lập luận có liên kết; "
        "căn cứ speech_rate_wpm, pause_count, filler_count.\n"
        "- Lexical Resource: từ vựng học thuật/trừu tượng chính xác, linh hoạt.\n"
        "- Grammatical Range & Accuracy: cấu trúc phức (điều kiện, giả định, "
        "mệnh đề quan hệ) để diễn đạt ý trừu tượng.\n"
        "- Pronunciation: rõ ràng kể cả khi nói ý phức; dùng phoneme_data nếu có."
    ),
)


_REGISTRY: dict[str, QuestionType] = {
    qt.key: qt
    for qt in [
        PART1_INTERVIEW,
        PART2_LONG_TURN,
        PART3_DISCUSSION,
    ]
}

# Public alias — truy cập tiện theo dict (vd IELTS_QUESTION_TYPES["part2_long_turn"])
IELTS_QUESTION_TYPES: dict[str, QuestionType] = _REGISTRY


def get_question_type(key: str) -> QuestionType:
    if key not in _REGISTRY:
        raise KeyError(
            f"Không biết dạng câu IELTS '{key}'. Hợp lệ: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]

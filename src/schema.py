"""Pydantic models cho kết quả chấm điểm (structured output của Claude).

Claude trả JSON đúng schema này qua client.messages.parse(...).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CompletionLevel(str, Enum):
    very_low = "very_low"
    low = "low"
    medium = "medium"
    high = "high"


class LexicalCorrection(BaseModel):
    """Một lỗi dùng từ cụ thể + cách sửa, cho tiêu chí lexical_resource/vocabulary.

    `said` PHẢI là chuỗi con xuất hiện đúng trong transcript (được validate lại
    sau khi model trả lời — xem _drop_invalid_corrections trong scoring.py).
    """
    said: str = Field(
        description="Cụm từ thí sinh đã nói (trích NGUYÊN VĂN từ transcript)"
    )
    suggested: str = Field(description="Từ/cụm từ đúng nên dùng thay thế")
    reason: str | None = Field(
        default=None, description="Lý do ngắn gọn vì sao nên sửa"
    )
    example: str = Field(
        description="Một câu ví dụ tự nhiên dùng từ/cụm từ được đề xuất"
    )


class CriterionScore(BaseModel):
    criterion: str = Field(
        description="Tên tiêu chí, vd 'pronunciation', 'intonation_stress'"
    )
    score: float = Field(
        description="Điểm tiêu chí: thang 0-3 cho TOEIC, band 0-9 cho IELTS"
    )
    justification: str = Field(description="Lý do chấm, dựa trên số liệu + transcript")
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "BẮT BUỘC cho MỌI tiêu chí: 2-4 gợi ý cải thiện cụ thể, hành động "
            "được (mỗi phần tử là một câu). KHÔNG để trống — luôn nêu được ít "
            "nhất vài điều thí sinh có thể luyện để lên điểm. Mỗi gợi ý phải bám "
            "vào điểm yếu/bằng chứng đã nêu trong justification, không nói chung chung."
        ),
    )
    corrections: list[LexicalCorrection] = Field(
        default_factory=list,
        description=(
            "Sửa lỗi dùng từ cụ thể (said → suggested + example). Chỉ điền cho "
            "tiêu chí lexical_resource (IELTS) / vocabulary (TOEIC); để rỗng cho "
            "các tiêu chí khác."
        ),
    )


class SampleAnswer(BaseModel):
    """Bài nói mẫu chất lượng cao do LLM sinh cho một dạng câu mở.

    Dùng cho nút "Gợi ý bài mẫu" (endpoint /suggest). KHÁC SpeakingResult: không
    chấm điểm, chỉ tạo bài mẫu để người học tham khảo/luyện theo.
    """

    answer: str = Field(
        description=(
            "Bài nói mẫu hoàn chỉnh bằng TIẾNG ANH, văn phong NÓI tự nhiên (như "
            "thí sinh nói thật, không phải văn viết), độ dài hợp với thời lượng "
            "kỳ vọng của dạng câu."
        )
    )
    target_band: str = Field(
        description="Mức nhắm tới của bài mẫu (echo lại), vd '9.0' hoặc 'TOEIC mức cao nhất'."
    )
    highlights: list[str] = Field(
        default_factory=list,
        description=(
            "3-5 điểm nhấn band cao đáng học từ bài mẫu (collocation, cấu trúc câu "
            "phức, từ nối, cách triển khai ý...). Viết bằng ngôn ngữ nhận xét được "
            "cấu hình (feedback_lang)."
        ),
    )
    outline: list[str] = Field(
        default_factory=list,
        description=(
            "Dàn ý ngắn (mở-thân-kết / các ý chính) — hữu ích cho độc thoại dài "
            "(IELTS Part 2). Có thể để rỗng với câu trả lời ngắn."
        ),
    )


class PracticeTask(BaseModel):
    """Đề luyện tập (task-context) cho 1 dạng câu — dùng để CHẤM THẬT lesson
    rubric/dạng câu (khóa học P2), thay nút "Đã học xong" thủ công.

    Do LLM sinh 1 lần rồi cache user-agnostic theo (lesson_id, lang) trong
    course.db (xem src/course/practice.py). Chỉ điền field liên quan dạng câu:
    read_aloud → `reference`; respond_with_info → `provided_info` + `prompt`;
    các dạng mở khác → chỉ `prompt`.
    """

    prompt: str = Field(
        default="",
        description=(
            "Đề bài / câu hỏi mà thí sinh phải TRẢ LỜI BẰNG LỜI NÓI. Với dạng "
            "đọc to (read_aloud) để trống (đề là `reference`). Viết bằng ngôn ngữ "
            "của kỳ thi (TOEIC/IELTS: tiếng Anh; TOPIK: tiếng Hàn)."
        ),
    )
    reference: str = Field(
        default="",
        description=(
            "CHỈ dạng đọc to (read_aloud): đoạn văn (2-4 câu) để thí sinh đọc to. "
            "Các dạng khác để rỗng."
        ),
    )
    provided_info: str = Field(
        default="",
        description=(
            "CHỈ dạng trả lời theo tài liệu (respond_with_info): tài liệu cho sẵn "
            "dạng text (lịch trình/agenda/bảng thông tin) để thí sinh dựa vào mà "
            "trả lời `prompt`. Các dạng khác để rỗng."
        ),
    )


class RolePlayTurn(BaseModel):
    """Một lượt hội thoại trong kịch bản nhập vai (Phase 3B).

    NPC nói `npc` → học viên đáp lại. `expected_user` là câu THAM CHIẾU học viên
    nên nói (để chấm phát âm); `hint` là gợi ý NGẮN hiển thị trong lúc học viên
    trả lời (KHÔNG lộ `expected_user` cho tới khi chấm xong).
    """

    npc: str = Field(
        description=(
            "Lời thoại của NHÂN VẬT (NPC) mở đầu lượt này — một câu tự nhiên, "
            "đúng vai và ngữ cảnh kịch bản. Viết bằng ngôn ngữ của kỳ thi (tiếng "
            "Anh cho TOEIC/IELTS)."
        )
    )
    expected_user: str = Field(
        description=(
            "Câu THAM CHIẾU học viên nên nói để đáp lại NPC (1 câu, tự nhiên, "
            "≤20 từ, đọc to được). Dùng làm text chấm phát âm — KHÔNG bao giờ rỗng."
        )
    )
    hint: str = Field(
        default="",
        description=(
            "Gợi ý NGẮN (mẹo/ý cần nói) hiển thị trong lúc học viên trả lời, "
            "KHÔNG phải câu mẫu. Ví dụ 'Chào lại và hỏi giá phòng.'"
        ),
    )


class RolePlayScript(BaseModel):
    """Kịch bản hội thoại nhập vai (Role-play Quest, Phase 3B).

    LLM sinh 1 lần rồi cache USER-AGNOSTIC theo (exam, topic) trong course.db
    (id tổng hợp '<exam>.<topic>#roleplay'). Chấm = phát âm từng `expected_user`
    qua gradePronunciation dùng chung — LLM CHỈ sinh nội dung, không chấm.
    """

    scenario: str = Field(
        description=(
            "Mô tả NGẮN bối cảnh hội thoại (1-2 câu) đặt học viên vào tình huống "
            "— ví dụ 'Bạn đang nhận phòng khách sạn ở nước ngoài.'"
        )
    )
    role_user: str = Field(
        description="Vai của HỌC VIÊN (ví dụ 'khách du lịch', 'ứng viên phỏng vấn')."
    )
    role_npc: str = Field(
        description="Vai của NHÂN VẬT đối thoại (ví dụ 'lễ tân khách sạn')."
    )
    turns: list[RolePlayTurn] = Field(
        default_factory=list,
        description=(
            "Chuỗi lượt hội thoại (≥2 lượt), NPC nói trước mỗi lượt rồi học viên "
            "đáp. Diễn tiến mạch lạc theo bối cảnh."
        ),
    )


class StorySegment(BaseModel):
    """Một đoạn của truyện đọc-to (Story Quest, Phase 3C)."""

    text: str = Field(
        description=(
            "MỘT câu/đoạn ngắn của truyện để học viên đọc to (≥4 từ, tự nhiên, "
            "≤25 từ). Viết bằng ngôn ngữ nói của kỳ thi (tiếng Anh cho TOEIC/IELTS)."
        )
    )


class StoryQuest(BaseModel):
    """Truyện đọc-to tuyến tính (Story Quest, Phase 3C) — KHÔNG nhánh/lựa chọn.

    LLM sinh 1 lần rồi cache USER-AGNOSTIC theo (exam, topic) trong course.db
    (id tổng hợp '<exam>.<topic>#story'). Học viên đọc to lần lượt từng đoạn;
    chấm phát âm mỗi đoạn qua gradePronunciation dùng chung — LLM CHỈ sinh truyện.
    """

    title: str = Field(description="Tiêu đề ngắn của truyện.")
    segments: list[StorySegment] = Field(
        default_factory=list,
        description=(
            "Các đoạn truyện theo THỨ TỰ (≥3 đoạn), mạch lạc thành một câu chuyện "
            "hoàn chỉnh. Mỗi đoạn học viên đọc to một lần."
        ),
    )


class WordInfo(BaseModel):
    """Định nghĩa + ví dụ cho 1 từ (popup luyện phát âm, kiểu ELSA).

    Do LLM sinh 1 lần rồi cache SQLite theo (word, lang) — xem src/words.py.
    """

    word: str = Field(description="Từ được tra (echo lại, lowercase).")
    definition_en: str = Field(
        description=(
            "Định nghĩa TIẾNG ANH ngắn gọn (1 câu, learner-friendly, kiểu từ điển "
            "Oxford Learner's), theo nghĩa THÔNG DỤNG nhất của từ."
        )
    )
    example_en: str = Field(
        description=(
            "MỘT câu ví dụ tiếng Anh tự nhiên, ngắn (≤20 từ), dùng đúng nghĩa đã "
            "định nghĩa ở trên."
        )
    )
    meaning: str = Field(
        description=(
            "Nghĩa của từ bằng ngôn ngữ đích (feedback_lang, vd tiếng Việt) — "
            "ngắn gọn kiểu từ điển (vd 'tính phí; sạc điện'), khớp nghĩa đã chọn."
        )
    )


class PhonemePracticeWord(BaseModel):
    """1 từ luyện tập cho 1 phoneme — do LLM chọn từ danh sách CANDIDATES."""

    word: str = Field(
        description="Từ được chọn (lowercase, PHẢI nằm trong danh sách CANDIDATES)."
    )
    reason: str = Field(
        description=(
            "Vì sao từ này tốt để luyện âm đó — TIẾNG VIỆT, ≤12 từ (vd vị trí "
            "âm đầu/giữa/cuối từ, cặp tối thiểu)."
        )
    )


class PhonemePracticeList(BaseModel):
    """Danh sách từ luyện tập tốt nhất cho 1 phoneme (gợi ý tab Từ đã lưu).

    Do LLM chọn 1 lần rồi cache SQLite theo (phoneme, lang) — xem src/words.py
    (suggestion_cache) + src/word_suggest.py.
    """

    phoneme: str = Field(
        description="IPA phoneme được luyện (echo lại, không kèm dấu / /)."
    )
    words: list[PhonemePracticeWord] = Field(
        description="10 từ luyện tập tốt nhất, chọn TỪ CANDIDATES, không lặp biến thể."
    )


class SpeakingResult(BaseModel):
    question_type: str
    # task_completion là tiêu chí hạng nhất: trả lời đúng/đủ yêu cầu hay không.
    task_completion: CompletionLevel
    content_relevance: CompletionLevel
    criteria: list[CriterionScore]
    # Điểm tổng theo từng thang đo — KHÔNG do LLM sinh, được TÍNH TỰ ĐỘNG trong
    # scoring.py từ điểm tiêu chí + task_completion + content_relevance, để cùng
    # một bộ điểm luôn ra cùng một số (loại bỏ dao động do model tự "bốc" số).
    # Chỉ MỘT field được set tuỳ kỳ thi (qt.exam); field còn lại để None. Cả hai
    # optional (default None) nên không bắt buộc trong schema gửi cho model; giá
    # trị model trả (nếu có) sẽ bị ghi đè.
    #
    # TOEIC Speaking: thang 0-200 (báo theo bước 10).
    estimated_toeic_score: int | None = Field(
        default=None,
        ge=0,
        le=200,
        description=(
            "Điểm TOEIC Speaking (0-200) — TÍNH TỰ ĐỘNG từ điểm tiêu chí, model "
            "KHÔNG cần điền. None nếu kỳ thi không phải TOEIC."
        ),
    )
    # IELTS Speaking: band 0-9 (bước 0.5) — trung bình 4 tiêu chí làm tròn 0.5.
    estimated_ielts_band: float | None = Field(
        default=None,
        ge=0,
        le=9,
        description=(
            "Band IELTS Speaking (0-9, bước 0.5) — TÍNH TỰ ĐỘNG từ band từng "
            "tiêu chí, model KHÔNG cần điền. None nếu kỳ thi không phải IELTS."
        ),
    )
    # TOPIK 말하기: thang 0-200 (level 1-6) — tính từ điểm tiêu chí 0-5, áp trần
    # theo mức câu (sơ/trung/cao cấp — xem rubrics/topik.py + scoring/compute.py).
    estimated_topik_score: int | None = Field(
        default=None,
        ge=0,
        le=200,
        description=(
            "Điểm TOPIK 말하기 (0-200) — TÍNH TỰ ĐỘNG từ điểm tiêu chí, model "
            "KHÔNG cần điền. None nếu kỳ thi không phải TOPIK."
        ),
    )
    # Giải thích logic chấm: tiêu chí nào mạnh/yếu và mức độ hoàn thành tổng thể.
    # KHÔNG nêu một con số 0-200 cụ thể (số tổng do code tính). Phải viết bằng
    # ngôn ngữ nhận xét được cấu hình.
    score_rationale: str = Field(
        description=(
            "Lập luận từng bước: tiêu chí nào kéo chất lượng lên/xuống, mức "
            "task_completion / content_relevance, và vì sao bài ở mức này. "
            "KHÔNG cần nêu con số tổng 0-200 — số đó được tính tự động."
        )
    )
    summary_feedback: str

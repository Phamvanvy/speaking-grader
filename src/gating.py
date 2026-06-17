"""Rule-based gating — kiểm tra rẻ, ổn định, KHÔNG gọi AI.

Bắt sớm các trường hợp hiển nhiên (audio quá ngắn/rỗng, nói quá ít so với
thời lượng kỳ vọng) để gán task_completion mà không cần Claude suy luận,
vừa rẻ vừa ổn định.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .asr import Transcription
from .features import Features

if TYPE_CHECKING:
    from .rubrics.toeic import QuestionType

# Ngưỡng coi là "không hoàn thành": thời lượng nói tối thiểu & số từ tối thiểu.
MIN_DURATION_SEC = 5.0
MIN_WORD_COUNT = 3
# Nếu nói ngắn hơn tỉ lệ này so với expected_duration_sec → coi là chưa đủ.
# CHỈ áp dụng cho dạng câu nói tự do (opinion, describe, respond), KHÔNG áp
# dụng cho Read Aloud: đọc nhanh nhưng đọc hết đoạn vẫn là hoàn thành đầy đủ.
SHORT_RESPONSE_RATIO = 0.4
# Read Aloud: độ phủ script (hits/reference_word_count) dưới ngưỡng này nghĩa
# là thí sinh đọc nhầm đoạn / thiếu phần lớn → coi như chưa hoàn thành bài.
READ_ALOUD_FAIL_COVERAGE = 0.50


@dataclass
class GatingResult:
    is_empty: bool                 # không nhận ra lời nào → nên dừng, không gọi AI
    task_completion_floor: str | None  # mức trần thấp gợi ý cho Claude (hoặc None)
    reasons: list[str]
    # Read Aloud: độ phủ script và cờ "đọc sai/không khớp đoạn được giao".
    reference_coverage: float | None = None
    fail_reference_match: bool = False

    @property
    def should_skip_ai(self) -> bool:
        return self.is_empty


def evaluate(
    transcription: Transcription,
    features: Features,
    expected_duration_sec: float | None = None,
    question_type: "QuestionType | None" = None,
) -> GatingResult:
    reasons: list[str] = []

    if transcription.word_count == 0:
        return GatingResult(
            is_empty=True,
            task_completion_floor="very_low",
            reasons=["Không nhận ra lời nói nào trong audio."],
        )

    floor: str | None = None
    is_read_aloud = bool(question_type and question_type.uses_reference_script)

    if features.speaking_duration_sec < MIN_DURATION_SEC:
        floor = "very_low"
        reasons.append(
            f"Thời lượng nói {features.speaking_duration_sec:.1f}s < "
            f"{MIN_DURATION_SEC}s tối thiểu."
        )

    if features.word_count < MIN_WORD_COUNT:
        floor = "very_low"
        reasons.append(f"Chỉ có {features.word_count} từ, quá ít.")

    # Read Aloud: chấm hoàn thành theo độ phủ script, KHÔNG theo thời lượng.
    reference_coverage: float | None = None
    fail_reference_match = False
    if is_read_aloud and features.accuracy_metrics is not None:
        reference_coverage = features.accuracy_metrics.coverage
        if reference_coverage < READ_ALOUD_FAIL_COVERAGE:
            floor = "very_low"
            fail_reference_match = True
            reasons.append(
                f"Read Aloud: chỉ đọc trúng {reference_coverage:.0%} script "
                f"(coverage < {READ_ALOUD_FAIL_COVERAGE:.0%}) — đọc nhầm đoạn "
                f"hoặc thiếu phần lớn, không phải bài được giao."
            )
    elif expected_duration_sec:
        # Chỉ phạt nói-quá-ngắn cho các dạng nói tự do, không cho Read Aloud.
        ratio = features.speaking_duration_sec / max(1e-6, expected_duration_sec)
        if ratio < SHORT_RESPONSE_RATIO:
            # Không ghi đè very_low nếu đã có
            floor = floor or "low"
            reasons.append(
                f"Nói {features.speaking_duration_sec:.1f}s, ngắn hơn nhiều so với "
                f"kỳ vọng {expected_duration_sec:.0f}s (tỉ lệ {ratio:.0%})."
            )

    return GatingResult(
        is_empty=False,
        task_completion_floor=floor,
        reasons=reasons,
        reference_coverage=reference_coverage,
        fail_reference_match=fail_reference_match,
    )

"""Data classes cho kết quả phoneme analysis.

Tất cả data classes đều immutable (frozen=True) để an toàn khi truyền giữa
các backend và scoring modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PhonemeErrorType(str, Enum):
    """Loại lỗi phoneme khi so với reference."""
    SUBSTITUTION = "substitution"   # phoneme được thay bằng phoneme khác
    DELETION = "deletion"           # phoneme trong reference bị bỏ qua
    INSERTION = "insertion"         # phoneme thừa so với reference


@dataclass(frozen=True)
class PhonemeSegment:
    """Một phoneme được phát hiện trong audio.

    Attributes:
        phoneme: ký hiệu IPA (vd /æ/, /θ/, /ʃ/)
        start: mốc thời gian bắt đầu (giây)
        end: mốc thời gian kết thúc (giây)
        confidence: độ tin cậy 0.0–1.0 (từ wav2vec probability)
        backend: tên backend sinh ra segment này ("wav2vec" | "mfa")
    """
    phoneme: str
    start: float
    end: float
    confidence: float = 0.0
    backend: str = "wav2vec"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phoneme": self.phoneme,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
        }


@dataclass(frozen=True)
class PhonemeError:
    """Một lỗi phoneme so với reference IPA sequence.

    Attributes:
        error_type: substitution | deletion | insertion
        expected: phoneme trong reference (None nếu insertion)
        predicted: phoneme từ audio (None nếu deletion)
        position: chỉ số trong reference sequence
        severity: "high" | "medium" | "low" — dựa trên phoneme similarity
    """
    error_type: PhonemeErrorType
    expected: str | None
    predicted: str | None
    position: int
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "expected": self.expected,
            "predicted": self.predicted,
            "position": self.position,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class PhonemeScore:
    """Điểm phoneme accuracy tổng hợp.

    Attributes:
        overall_accuracy: tỉ lệ phonemes đúng (0.0–1.0)
        substitution_count: số phonemes bị thay
        deletion_count: số phonemes bị thiếu
        insertion_count: số phonemes thừa
        reference_count: số phonemes trong reference
        predicted_count: số phonemes từ audio
        avg_confidence: độ tin cậy trung bình của predicted phonemes
        errors: chi tiết từng lỗi (top-N)
    """
    overall_accuracy: float
    substitution_count: int
    deletion_count: int
    insertion_count: int
    reference_count: int
    predicted_count: int
    avg_confidence: float
    errors: list[PhonemeError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_accuracy": round(self.overall_accuracy, 4),
            "substitution_count": self.substitution_count,
            "deletion_count": self.deletion_count,
            "insertion_count": self.insertion_count,
            "reference_count": self.reference_count,
            "predicted_count": self.predicted_count,
            "avg_confidence": round(self.avg_confidence, 4),
            "errors": [e.to_dict() for e in self.errors[:20]],
        }


@dataclass(frozen=True)
class PhonemeResult:
    """Kết quả phoneme analysis cho 1 đoạn audio.

    Attributes:
        audio_path: đường dẫn file audio
        segments: danh sách phoneme segments từ audio
        reference_phonemes: danh sách phonemes tham chiếu (từ reference script)
        score: điểm phoneme accuracy (None nếu không có reference)
        backend_used: tên backend ("wav2vec" | "mfa" | "hybrid")
        backend_available: backend có sẵn sàng không
        warning: cảnh báo nếu backend không sẵn sàng
    """
    audio_path: str
    segments: list[PhonemeSegment]
    reference_phonemes: list[str]
    score: PhonemeScore | None
    backend_used: str
    backend_available: bool = True
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "segments": [s.to_dict() for s in self.segments],
            "reference_phonemes": self.reference_phonemes,
            "score": self.score.to_dict() if self.score else None,
            "backend_used": self.backend_used,
            "backend_available": self.backend_available,
            "warning": self.warning,
        }
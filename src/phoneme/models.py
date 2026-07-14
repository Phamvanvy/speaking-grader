"""Data classes cho kết quả phoneme analysis.

Tất cả data classes đều immutable (frozen=True) để an toàn khi truyền giữa
các backend và scoring modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple


# Phiên bản công thức deletion-evidence (shadow probe). Bump khi đổi công thức /
# margin frame / cách nhóm token — telemetry các đợt không trộn lẫn khi phân tích.
EVIDENCE_VERSION: str = "v1"


@dataclass(frozen=True)
class EvidenceStats:
    """Thống kê bằng chứng âm học của 1 âm bị THIẾU (deletion) — SHADOW ONLY.

    Tính từ frame posteriors của wav2vec trong cửa sổ thời gian của từ: mass mỗi
    frame = tổng probability các token cùng nhóm IPA (sau normalize) với âm bị thiếu.
    CHỈ telemetry/hiển thị chẩn đoán — KHÔNG bao giờ tham gia penalty/điểm.

    Attributes:
        max_mass: mass lớn nhất qua các frame trong cửa sổ (0.0 nếu cửa sổ rỗng)
        top_k_mean: trung bình k=3 frame mass cao nhất (ít hơn 3 frame → trung bình có gì)
        p90: percentile 90 của mass qua các frame
        n_frames: số frame trong cửa sổ (0 = cửa sổ rỗng/ngoài biên)
        argmax_token: token đang "thắng" (argmax) tại frame có mass cao nhất — cho
            biết wav2vec nghe ra âm gì ở chỗ lẽ ra có âm bị thiếu ("" nếu cửa sổ rỗng)
        argmax_prob: probability của argmax_token tại frame đó
        argmax_is_silence: argmax_token là token BLANK/SILENCE của model (<pad>, |,
            sil...) → wav2vec nhả "khoảng lặng" ở chỗ này. Kết hợp với max_mass cao =
            chữ ký CTC blank-collapse (âm CÓ trong audio nhưng thua token blank), khác
            với "nghe ra âm KHÁC" (argmax là một IPA) hay "âm vắng thật" (mass ~0).
    """
    max_mass: float
    top_k_mean: float
    p90: float
    n_frames: int
    argmax_token: str = ""
    argmax_prob: float = 0.0
    argmax_is_silence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_mass": round(self.max_mass, 4),
            "top_k_mean": round(self.top_k_mean, 4),
            "p90": round(self.p90, 4),
            "n_frames": self.n_frames,
            "argmax_token": self.argmax_token,
            "argmax_prob": round(self.argmax_prob, 4),
            "argmax_is_silence": self.argmax_is_silence,
        }


class PhonemeErrorType(str, Enum):
    """Loại lỗi phoneme khi so với reference."""
    SUBSTITUTION = "substitution"   # phoneme được thay bằng phoneme khác
    DELETION = "deletion"           # phoneme trong reference bị bỏ qua
    INSERTION = "insertion"         # phoneme thừa so với reference


class WordSpan(NamedTuple):
    """Một từ trong reference text + khoảng index của nó trong reference phoneme list.

    Dùng để map ngược 1 lỗi phoneme (theo position trong reference sequence) về
    đúng từ đã sinh ra phoneme đó. Đặt ở models.py (leaf module, không import gì
    trong package) để cả ipa.py lẫn scoring.py dùng chung mà không tạo vòng import.

    Attributes:
        word: từ như xuất hiện trong text (giữ nguyên hoa/thường để hiển thị)
        start_idx: index bắt đầu (inclusive) trong reference phoneme list
        end_idx: index kết thúc (exclusive)
        source: nguồn IPA của từ ("override" | "cmudict" | "espeak") — eSpeak là
            G2P đoán cho từ ngoài từ điển (OOV/tên riêng) nên IPA kém tin cậy;
            scoring dùng cờ này để nới lỏng (skip ở free-speech, cap severity khi
            có script). Default "cmudict" để mọi chỗ dựng 3-positional cũ vẫn chạy.
    """
    word: str
    start_idx: int
    end_idx: int
    source: str = "cmudict"


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
        word: từ chứa phoneme này (None nếu không map được — vd insertion, hoặc
            không có reference spans). Substitution/deletion mới có word.
    """
    error_type: PhonemeErrorType
    expected: str | None
    predicted: str | None
    position: int
    severity: str = "medium"
    word: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "expected": self.expected,
            "predicted": self.predicted,
            "position": self.position,
            "severity": self.severity,
            "word": self.word,
        }


@dataclass(frozen=True)
class PhonemePoint:
    """Một phoneme trong reference của 1 từ + trạng thái phát âm của nó.

    Dùng để hiển thị IPA full-từ kiểu ELSA: từng âm 1, highlight âm đọc sai.

    Attributes:
        symbol: ký hiệu IPA tham chiếu (KHÔNG kèm dấu / / — frontend tự bọc)
        status: "ok" (đúng) | "sub" (đọc sai thành âm khác) | "del" (thiếu âm)
        heard: âm nghe được (chỉ có với "sub"; None với "ok"/"del")
        severity: "high" | "medium" | "low" cho sub/del; None với "ok"
        stress: "primary" | "secondary" cho nguyên âm được nhấn; None nếu không
            nhấn hoặc từ đơn âm tiết. CHỈ để hiển thị — không tham gia alignment.
            Đặt TRÊN nguyên âm (nguồn cho severity/nhân chính phía scoring).
        display_stress: như `stress` nhưng dấu nhấn ĐÃ dời về đầu âm tiết (onset) để
            render `/ˈledʒənd/` thay vì `/lˈedʒənd/`. CHỈ để hiển thị; None nếu không có
            (payload cũ) — UI fallback về `stress`.
        penalty_reason: lý do điều chỉnh penalty (L1-aware layer): "l1_final_deletion"
            (nuốt phụ âm cuối kiểu L1 → "accent note"), "low_confidence_neutralized",
            "hard_error", hoặc None (âm đúng / layer tắt). UI dùng để gắn nhãn accent.
        penalty_adjustment: hệ số ĐÃ áp lên penalty gốc (1.0 = không đổi, <1 = giảm,
            0.0 = trung hoà). TÁCH khỏi `penalty_reason` (why vs how-much).
        evidence: bằng chứng âm học của âm bị THIẾU (deletion evidence probe, SHADOW —
            chỉ telemetry, không tham gia điểm). None với ok/sub/skipped hoặc probe tắt.
        evidence_source: nguồn cửa sổ thời gian dùng để probe: "wav2vec_window" |
            "whisper_window" | "none" (từ không có cửa sổ → evidence=None). None = probe tắt.
        evidence_version: phiên bản công thức evidence (EVIDENCE_VERSION) — chỉ set khi
            probe chạy trên point này.
    """
    symbol: str
    status: str
    heard: str | None = None
    severity: str | None = None
    stress: str | None = None
    display_stress: str | None = None
    penalty_reason: str | None = None
    penalty_adjustment: float = 1.0
    evidence: EvidenceStats | None = None
    evidence_source: str | None = None
    evidence_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "status": self.status,
            "heard": self.heard,
            "severity": self.severity,
            "stress": self.stress,
            "display_stress": self.display_stress,
            "penalty_reason": self.penalty_reason,
            "penalty_adjustment": round(self.penalty_adjustment, 4),
        }
        # Shadow evidence: CHỈ thêm key khi probe đã chạy — payload các point khác
        # (và toàn bộ payload khi probe tắt) giữ nguyên byte-for-byte như cũ.
        if self.evidence_source is not None:
            d["evidence"] = self.evidence.to_dict() if self.evidence else None
            d["evidence_source"] = self.evidence_source
            d["evidence_version"] = self.evidence_version
        return d


@dataclass(frozen=True)
class WordPronunciation:
    """Phát âm chi tiết của 1 từ trong reference (IPA full + từng âm).

    Attributes:
        word: từ như xuất hiện trong text (giữ nguyên hoa/thường)
        ipa: phiên âm IPA đầy đủ của từ (ghép các symbol, KHÔNG kèm / /)
        phonemes: danh sách PhonemePoint theo thứ tự reference
        accuracy: tỉ lệ âm đúng trong từ (ok_count / len(phonemes))
        skip_reason: nếu từ bị Recognition Reliability bỏ qua (không chấm phoneme),
            đây là lý do (vd "whisper_mismatch"); None nếu từ được chấm bình thường.
        start, end: cửa sổ thời gian (giây) của từ trong audio, lấy từ Whisper WORD
            timestamp (xem map_reference_words_to_windows; đã đệm ~50–100ms mỗi phía +
            clamp theo từ kề — _pad_and_clamp_windows), fallback wav2vec segment khi từ
            không có Whisper window. Cho UI phát lại đoạn audio của RIÊNG từ này. None
            nếu không có nguồn timing nào (vd từ bị skip / thí sinh không đọc) hoặc
            payload cũ — UI khi đó không hiện nút nghe lại.
    """
    word: str
    ipa: str
    phonemes: list[PhonemePoint] = field(default_factory=list)
    accuracy: float = 0.0
    skip_reason: str | None = None
    start: float | None = None
    end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "ipa": self.ipa,
            "phonemes": [p.to_dict() for p in self.phonemes],
            "accuracy": round(self.accuracy, 4),
            "skip_reason": self.skip_reason,
            "start": self.start,
            "end": self.end,
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
        words: phát âm chi tiết từng từ (IPA full + từng âm, cho UI kiểu ELSA)
        words_truncated: True nếu danh sách words bị cắt bớt do quá dài
        words_total: tổng số từ trong reference (luôn set, kể cả khi không cắt)
    """
    overall_accuracy: float
    substitution_count: int
    deletion_count: int
    insertion_count: int
    reference_count: int
    predicted_count: int
    avg_confidence: float
    errors: list[PhonemeError] = field(default_factory=list)
    words: list[WordPronunciation] = field(default_factory=list)
    words_truncated: bool = False
    words_total: int = 0
    # L1-aware layer metadata (PRD §8) — diagnostic/explainability, KHÔNG đổi math điểm.
    raw_penalty: float = 0.0               # tổng penalty TRƯỚC L1 + neutralization
    adjusted_penalty: float = 0.0          # tổng penalty SAU điều chỉnh (== nguồn accuracy)
    l1_adjusted_count: int = 0             # số âm được L1 giảm penalty
    low_conf_neutralized_count: int = 0    # số sub bị trung hoà do confidence rất thấp
    recognizer_noise_count: int = 0        # số sub bị gate thành recognizer-noise (hallucinate)
    l1_adjustment_ratio: float = 0.0       # (raw - adjusted) / raw (tỉ lệ penalty được giảm)
    coverage_collapse_count: int = 0       # số del bị coverage gate cap (từ collapse, Track A)
    drift_capped_count: int = 0            # số sub bị drift cap (segment ngoài window, Track B)

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
            "words": [w.to_dict() for w in self.words],
            "words_truncated": self.words_truncated,
            "words_total": self.words_total,
            "raw_penalty": round(self.raw_penalty, 4),
            "adjusted_penalty": round(self.adjusted_penalty, 4),
            "l1_adjusted_count": self.l1_adjusted_count,
            "low_conf_neutralized_count": self.low_conf_neutralized_count,
            "recognizer_noise_count": self.recognizer_noise_count,
            "l1_adjustment_ratio": round(self.l1_adjustment_ratio, 4),
            "coverage_collapse_count": self.coverage_collapse_count,
            "drift_capped_count": self.drift_capped_count,
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
"""L1-aware phoneme scoring rules — Vietnamese (deterministic, no I/O, no model).

Tầng tolerance "tiếng mẹ đẻ" (L1): GIẢM penalty cho lỗi phù hợp chuyển di âm vị học
tiếng Việt (vd nuốt phụ âm cuối) — KHÔNG bỏ qua, vẫn hiển thị như "accent note". Thuần
rule/threshold → deterministic + giải thích được. Recognition Reliability vẫn là NƠI DUY
NHẤT quyết định skip; module này CHỈ điều biến penalty.

v1: chỉ word-final consonant deletion. Mở rộng (substitution clusters / ngôn ngữ khác) qua
register_l1_pattern(). KHÔNG có logic model/ngẫu nhiên.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ipa import normalize_ipa

# Guardrail (PRD §11): multiplier KHÔNG bao giờ > 0.6 (chống over-tolerance → lạm điểm).
L1_MULTIPLIER_CAP: float = 0.6


class PenaltyReason(str, Enum):
    """Lý do (WHY) của penalty — TÁCH khỏi mức điều chỉnh (penalty_adjustment = HOW-MUCH)."""

    NONE = "none"                              # âm đúng (ok) hoặc không chấm (skipped)
    HARD_ERROR = "hard_error"                  # lỗi thật, penalty đầy đủ
    L1_FINAL_DELETION = "l1_final_deletion"    # nuốt phụ âm cuối kiểu L1 → giảm penalty
    LOW_CONFIDENCE_NEUTRALIZED = "low_confidence_neutralized"  # conf rất thấp → trung hoà


@dataclass(frozen=True)
class L1Match:
    """1 rule L1 đã khớp — trả về thay cho float để telemetry biết CHÍNH XÁC rule nào kích hoạt."""

    rule_id: str        # vd "vi.final_stop.t"
    category: str       # vd "final_stop"
    phoneme: str        # phoneme đã chuẩn hoá, đã khớp
    multiplier: float   # hệ số nhân penalty (đã clamp ≤ L1_MULTIPLIER_CAP)


def _clamp(multiplier: float) -> float:
    return min(max(multiplier, 0.0), L1_MULTIPLIER_CAP)


# Phụ âm CUỐI TỪ mà người Việt thường nuốt/giảm. Tiếng Việt cho phép coda /p t k m n ŋ/
# (stops không bật hơi); fricative/affricate/liquid KHÔNG có ở coda tiếng Việt → hay bị nuốt.
# Final stops English thường unreleased với người Việt → wav2vec espeak dễ "không thấy"
# (R) ⇒ giảm penalty đúng cho cả R lẫn S. NASAL /m n ŋ/ KHÔNG vào đây (tiếng Việt có nasal
# cuối, nuốt nasal nhiều khả năng là lỗi thật → giữ penalty đầy đủ).
_FINAL_DELETION_CATEGORIES: dict[str, tuple[float, frozenset[str]]] = {
    "final_stop":      (0.35, frozenset({"p", "t", "k", "b", "d", "ɡ"})),
    "final_fricative": (0.50, frozenset({"s", "z", "f", "v", "θ", "ð", "ʃ", "ʒ"})),
    "final_affricate": (0.50, frozenset({"tʃ", "dʒ"})),
    "final_liquid":    (0.50, frozenset({"l", "r"})),
}


def _build_map() -> dict[str, L1Match]:
    m: dict[str, L1Match] = {}
    for category, (mult, phonemes) in _FINAL_DELETION_CATEGORIES.items():
        for ph in phonemes:
            m[ph] = L1Match(
                rule_id=f"vi.{category}.{ph}", category=category,
                phoneme=ph, multiplier=_clamp(mult),
            )
    return m


_L1_FINAL_DELETION: dict[str, L1Match] = _build_map()


def match_l1_final_deletion(phoneme: str) -> L1Match | None:
    """L1Match nếu `phoneme` là phụ âm cuối được L1 dung sai (sau chuẩn hoá), else None.

    Người gọi (scorer) CHỈ hỏi hàm này khi âm thực sự ở vị trí coda/cuối từ (ref_is_coda).
    """
    return _L1_FINAL_DELETION.get(normalize_ipa(phoneme))


def register_l1_pattern(
    category: str, phoneme: str, multiplier: float, *, language: str = "vi"
) -> None:
    """Hook mở rộng (PRD §10): thêm 1 rule final-deletion. multiplier bị clamp ≤ cap.

    v1 chỉ hỗ trợ language='vi' + final deletion; substitution/ngôn ngữ khác mở rộng sau.
    """
    if language != "vi":
        raise NotImplementedError(f"L1 language {language!r} chưa hỗ trợ (v1: 'vi').")
    ph = normalize_ipa(phoneme)
    _L1_FINAL_DELETION[ph] = L1Match(
        rule_id=f"{language}.{category}.{ph}", category=category,
        phoneme=ph, multiplier=_clamp(multiplier),
    )

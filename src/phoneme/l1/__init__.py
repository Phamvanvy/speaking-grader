"""L1-aware scoring profiles — registry key theo cặp (l1, target) [M5, plan D9].

Generalize tầng L1 tolerance: cùng MỘT người học (L1 = tiếng Việt) nhưng lỗi
chuyển di khác nhau theo NGÔN NGỮ ĐÍCH đang chấm (nuốt phụ âm cuối khi nói tiếng
Anh ≠ tense→plain khi nói tiếng Hàn). `L1Profile` bundle 2 matcher thuần hàm
(deterministic, no I/O):

  - `match_final_deletion(phoneme)`: âm CUỐI TỪ bị thiếu có được L1 dung sai không.
  - `match_substitution(ref, heard, is_coda)`: cặp sub có khớp bảng chuyển di L1 không.

Bảng ("vi","en") wrap NGUYÊN VẸN module l1_vietnamese.py (không đổi bit nào —
PenaltyReason/L1Match vẫn sống ở đó vì được import rộng khắp scoring). Bảng
("vi","ko") seed theo plan D9, flag TOEIC_PHONEME_L1_KO_ENABLED default OFF,
multiplier tune sau bằng telemetry học viên thật.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..l1_vietnamese import L1Match, match_l1_final_deletion


class _SubMatcher(Protocol):
    def __call__(self, ref: str, heard: str, *, is_coda: bool) -> L1Match | None: ...


@dataclass(frozen=True)
class L1Profile:
    """Bộ rule L1 cho một cặp (tiếng mẹ đẻ, ngôn ngữ đích)."""

    l1: str
    target: str
    match_final_deletion: Callable[[str], L1Match | None]
    match_substitution: _SubMatcher


def _no_substitution(ref: str, heard: str, *, is_coda: bool) -> L1Match | None:
    """("vi","en") v1 không có bảng sub — sub leniency của EN là low-conf
    neutralization (generic, xử lý trong _align_points), không phải bảng cặp âm."""
    return None


# ("vi","en") — hành vi L1 hiện hành, wrap đúng hàm cũ (bit-for-bit).
VI_EN = L1Profile(
    l1="vi", target="en",
    match_final_deletion=match_l1_final_deletion,
    match_substitution=_no_substitution,
)


def get_l1_profile(l1: str, target: str) -> L1Profile:
    """Trả L1Profile cho cặp (l1, target). Cặp lạ → KeyError (chặn sớm, không đoán)."""
    if (l1, target) == ("vi", "en"):
        return VI_EN
    if (l1, target) == ("vi", "ko"):
        from .vi_ko import VI_KO  # import trễ: chỉ nạp bảng ko khi thực sự chấm ko

        return VI_KO
    raise KeyError(f"Chưa có L1 profile cho cặp ({l1!r}, {target!r}).")

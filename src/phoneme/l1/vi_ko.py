"""L1 rules: người VIỆT học tiếng HÀN — seed theo plan D9 (M5), tune bằng telemetry.

Nguyên tắc chọn seed (chỉ những chuyển di CÓ CƠ SỞ âm vị học tiếng Việt, mirror
cách bảng EN chỉ chứa coda tiếng Việt không có):

  - Tense → plain (ㄲ→ㄱ, ㄸ→ㄷ, ㅃ→ㅂ, ㅆ→ㅅ, ㅉ→ㅈ): tiếng Việt KHÔNG có phụ âm
    căng — phân biệt 3 bậc lenis/aspirated/tense là khó kinh điển của học viên Việt
    → giảm penalty (vẫn hiển thị), multiplier 0.5.
  - ʌ ↔ o (ㅓ/ㅗ): cặp nguyên âm sau tròn/không tròn — ơ/ô tiếng Việt gần nhưng
    không trùng, nhầm hai chiều rất phổ biến → 0.5.
  - Coda l → n, và coda l bị NUỐT: tiếng Việt không có /l/ cuối âm tiết (học viên
    thay bằng n hoặc bỏ) → 0.5.

CỐ Ý KHÔNG dung sai (để nguyên penalty đầy đủ):
  - ɯ (ㅡ): tiếng Việt CÓ ư → học viên Việt phát âm được, nhầm ɯ→u là lỗi thật.
  - aspirated → plain (ㅋ→ㄱ...): pʰ/kʰ tiếng Việt map sang f/x (phụ âm xát) chứ
    không phải bật hơi yếu — chưa đủ bằng chứng chuyển di, chờ telemetry.

Mọi multiplier bị clamp ≤ L1_MULTIPLIER_CAP (0.6) như bảng EN — L1 chỉ GIẢM
penalty, không bao giờ xoá trắng (khác allophone/phonemes_match).
"""
from __future__ import annotations

from ..ipa.ko.phoneme_set_ko import normalize_ipa_ko
from ..l1_vietnamese import L1Match, _clamp
from . import L1Profile

# ── final deletion: coda ㄹ /l/ ───────────────────────────────────────────────
# Coda stop (p t k) KHÔNG cần L1: deletion_severity_ko đã cho "low" theo nguyên
# tắc unreleased [p̚ t̚ k̚] (recognizer-prone) — áp cho MỌI học viên, không riêng vi.
_FINAL_DELETION_KO: dict[str, float] = {"l": 0.5}

# ── substitution: (ref → heard, sau normalize_ipa_ko) → (category, multiplier) ──
# MỌI VỊ TRÍ (onset/coda):
_SUB_ANY_KO: dict[tuple[str, str], tuple[str, float]] = {
    ("k͈", "k"): ("tense_plain", 0.5),
    ("t͈", "t"): ("tense_plain", 0.5),
    ("p͈", "p"): ("tense_plain", 0.5),
    ("s͈", "s"): ("tense_plain", 0.5),
    ("t͈ɕ", "tɕ"): ("tense_plain", 0.5),
    ("ʌ", "o"): ("vowel_round", 0.5),
    ("o", "ʌ"): ("vowel_round", 0.5),
}
# CHỈ CODA:
_SUB_CODA_KO: dict[tuple[str, str], tuple[str, float]] = {
    ("l", "n"): ("final_l", 0.5),
}


def _match(table: dict[tuple[str, str], tuple[str, float]],
           ref: str, heard: str) -> L1Match | None:
    entry = table.get((normalize_ipa_ko(ref), normalize_ipa_ko(heard)))
    if entry is None:
        return None
    category, mult = entry
    return L1Match(
        rule_id=f"vi-ko.{category}.{normalize_ipa_ko(ref)}",
        category=category, phoneme=normalize_ipa_ko(ref),
        multiplier=_clamp(mult),
    )


def match_final_deletion_ko(phoneme: str) -> L1Match | None:
    """L1Match nếu âm coda bị thiếu nằm trong bảng dung sai vi→ko, else None."""
    p = normalize_ipa_ko(phoneme)
    mult = _FINAL_DELETION_KO.get(p)
    if mult is None:
        return None
    return L1Match(
        rule_id=f"vi-ko.final_deletion.{p}", category="final_deletion",
        phoneme=p, multiplier=_clamp(mult),
    )


def match_substitution_ko(ref: str, heard: str, *, is_coda: bool) -> L1Match | None:
    """L1Match nếu cặp sub khớp bảng chuyển di vi→ko (bảng coda ưu tiên hơn), else None."""
    if is_coda:
        m = _match(_SUB_CODA_KO, ref, heard)
        if m is not None:
            return m
    return _match(_SUB_ANY_KO, ref, heard)


VI_KO = L1Profile(
    l1="vi", target="ko",
    match_final_deletion=match_final_deletion_ko,
    match_substitution=match_substitution_ko,
)

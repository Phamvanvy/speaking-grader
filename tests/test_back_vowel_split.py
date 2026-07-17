"""TOEIC_PHONEME_BACK_VOWEL_SPLIT — tách ɑ khỏi nhóm gộp back-vowel (star/store).

Flag là IMPORT-TIME (normalize_ipa + các bảng keyed theo nó build 1 lần lúc
import, phoneme_similarity có lru_cache) nên nhánh ON phải kiểm qua SUBPROCESS
với env bật; nhánh OFF (default) kiểm in-process như test thường.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.phoneme.ipa.phoneme_set import BACK_VOWEL_SPLIT_ENABLED, normalize_ipa
from src.phoneme.ipa.profile import get_profile
from src.phoneme.ipa.similarity import error_severity, phoneme_similarity, phonemes_match
from src.phoneme.scoring.alignment import _back_vowel_merger_ok

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Flag OFF (default) — hành vi cũ bit-for-bit ─────────────────────────

def test_default_off_in_this_process():
    """Bộ test chạy với env mặc định — flag phải OFF (default OFF cho tới khi bench)."""
    assert BACK_VOWEL_SPLIT_ENABLED is False


def test_off_merges_back_vowels():
    """OFF: ɑ/ɒ/o đều gộp về ɔ (hành vi hiện hành — star ≡ store)."""
    assert normalize_ipa("ɑː") == "ɔ"
    assert normalize_ipa("ɒ") == "ɔ"
    assert normalize_ipa("o") == "ɔ"
    assert phoneme_similarity("ɑː", "ɔː") == 1.0
    assert phonemes_match("ɔː", "ɑː")


# ── Rule ngữ cảnh (hàm thuần — độc lập flag) ────────────────────────────

def test_merger_context_rule():
    """ɑ↔ɔ chấp nhận NGOẠI TRỪ trước /r/ cùng từ ở content word (store/star)."""
    en = get_profile("en")
    # store /s t ɔː r/: ɔː (idx 2) đứng trước /r/ cùng từ → KHÔNG chấp nhận.
    ref = ["s", "t", "ɔː", "r"]
    ref_word = ["store"] * 4
    assert _back_vowel_merger_ok(ref, ref_word, 2, "store", en) is False
    # call /k ɔː l/: không trước /r/ → cot-caught merger, chấp nhận.
    assert _back_vowel_merger_ok(["k", "ɔː", "l"], ["call"] * 3, 1, "call", en) is True
    # or /ɔː r/: trước /r/ NHƯNG function word (dạng yếu) → chấp nhận.
    assert _back_vowel_merger_ok(["ɔː", "r"], ["or"] * 2, 0, "or", en) is True
    # ɔː cuối từ, từ kế bắt đầu bằng /r/ ("saw red"): khác từ → chấp nhận.
    ref = ["s", "ɔː", "r", "e", "d"]
    ref_word = ["saw", "saw", "red", "red", "red"]
    assert _back_vowel_merger_ok(ref, ref_word, 1, "saw", en) is True


# ── Flag ON — kiểm qua subprocess (import-time flag) ────────────────────

_ON_PROBE = r"""
import json
from src.phoneme.ipa.phoneme_set import BACK_VOWEL_SPLIT_ENABLED, is_vowel, normalize_ipa
from src.phoneme.ipa.similarity import error_severity, phoneme_similarity, phonemes_match

print(json.dumps({
    "enabled": BACK_VOWEL_SPLIT_ENABLED,
    "norm_a": normalize_ipa("ɑː"),          # ɑː
    "norm_turned_a": normalize_ipa("ɒ"),          # ɒ (vẫn gộp ɔ)
    "norm_o": normalize_ipa("o"),
    "sim": phoneme_similarity("ɑː", "ɔː"),   # ɑː vs ɔː
    "severity": error_severity(phoneme_similarity("ɑ", "ɔ")),
    "match": phonemes_match("ɔː", "ɑː"),     # store vowel vs star vowel
    "match_stressed": phonemes_match("ɔː", "ɑː", reducible=False),
    "is_vowel_a": is_vowel("ɑː"),
    # Các cặp khác không đổi (guard chống flag lan sang phần còn lại của bảng)
    "sim_i": phoneme_similarity("ɪ", "iː"),  # ɪ↔iː 0.85
    "sim_o_ou": phoneme_similarity("ɔ", "əʊ"),    # ɔ↔əʊ 0.55
}))
"""


def test_on_splits_alpha_via_subprocess():
    env = dict(os.environ)
    env["TOEIC_PHONEME_BACK_VOWEL_SPLIT"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    out = subprocess.run(
        [sys.executable, "-c", _ON_PROBE],
        capture_output=True, text=True, encoding="utf-8", env=env,
        cwd=str(REPO_ROOT), timeout=120,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["enabled"] is True
    assert got["norm_a"] == "ɑ"          # ɑ giữ riêng, không còn về ɔ
    assert got["norm_turned_a"] == "ɔ"   # ɒ vẫn gộp ɔ (inventory recognizer)
    assert got["norm_o"] == "ɔ"
    assert got["sim"] == 0.60                 # near-pair mới
    assert got["severity"] == "medium"        # hiện lỗi với người học, penalty vừa
    assert got["match"] is False              # star KHÔNG còn khớp store
    assert got["match_stressed"] is False
    assert got["is_vowel_a"] is True          # ɑ vẫn nhận diện là nguyên âm
    # Phần còn lại của bảng không đổi
    assert got["sim_i"] == 0.85
    assert got["sim_o_ou"] == 0.55

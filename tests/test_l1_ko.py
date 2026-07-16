"""M5 — L1 vi→ko leniency: registry (l1, target), bảng seed D9, và tính bất biến.

Ba lớp đảm bảo:
1. Registry/bảng thuần hàm trả đúng rule (tense→plain, ʌ↔o, coda l; ɯ CỐ Ý không).
2. compute_phoneme_score: l1_profile=None (mặc định) giữ EN bit-for-bit kể cả khi
   l1_enabled=True (prod đang bật TOEIC_PHONEME_L1_ENABLED); profile vi→ko chỉ
   GIẢM penalty cho cặp trong bảng, không đổi status.
3. Flag config default OFF.
"""
from __future__ import annotations

import pytest

from src.phoneme.ipa.profile import get_profile
from src.phoneme.l1 import VI_EN, get_l1_profile
from src.phoneme.l1_vietnamese import (
    L1_MULTIPLIER_CAP,
    PenaltyReason,
    match_l1_final_deletion,
)
from src.phoneme.models import PhonemeSegment, WordSpan
from src.phoneme.scoring import compute_phoneme_score


# ── 1. Registry + bảng ─────────────────────────────────────────────────────────

def test_vi_en_wraps_legacy_exactly():
    assert VI_EN.match_final_deletion is match_l1_final_deletion
    # v1 EN không có bảng sub — matcher luôn None (nhánh sub bit-for-bit).
    assert VI_EN.match_substitution("t", "d", is_coda=False) is None
    assert get_l1_profile("vi", "en") is VI_EN


def test_unknown_pair_raises():
    with pytest.raises(KeyError):
        get_l1_profile("vi", "fr")
    with pytest.raises(KeyError):
        get_l1_profile("ko", "en")


@pytest.fixture()
def vi_ko():
    return get_l1_profile("vi", "ko")


@pytest.mark.parametrize("tense,plain", [
    ("k͈", "k"), ("t͈", "t"), ("p͈", "p"), ("s͈", "s"), ("t͈ɕ", "tɕ"),
])
def test_tense_to_plain_lenient(vi_ko, tense, plain):
    m = vi_ko.match_substitution(tense, plain, is_coda=False)
    assert m is not None and m.category == "tense_plain"
    assert 0.0 < m.multiplier <= L1_MULTIPLIER_CAP
    # Chiều ngược (nói plain thành tense / hypercorrection) KHÔNG dung sai.
    assert vi_ko.match_substitution(plain, tense, is_coda=False) is None


def test_vowel_round_both_directions(vi_ko):
    assert vi_ko.match_substitution("ʌ", "o", is_coda=False) is not None
    assert vi_ko.match_substitution("o", "ʌ", is_coda=False) is not None


def test_eu_not_lenient(vi_ko):
    # Tiếng Việt CÓ ư → nhầm ɯ là lỗi thật, giữ penalty đầy đủ (D9).
    assert vi_ko.match_substitution("ɯ", "u", is_coda=False) is None
    assert vi_ko.match_substitution("u", "ɯ", is_coda=False) is None


def test_coda_l_rules(vi_ko):
    # l→n CHỈ ở coda; nuốt coda l được dung sai; coda nasal thì không.
    assert vi_ko.match_substitution("l", "n", is_coda=True) is not None
    assert vi_ko.match_substitution("l", "n", is_coda=False) is None
    assert vi_ko.match_final_deletion("l") is not None
    assert vi_ko.match_final_deletion("n") is None
    assert vi_ko.match_final_deletion("ŋ") is None


# ── 2. Tích hợp compute_phoneme_score ──────────────────────────────────────────

def _segments(phs: list[str], conf: float = 0.95) -> list[PhonemeSegment]:
    return [
        PhonemeSegment(phoneme=p, start=i * 0.1, end=(i + 1) * 0.1, confidence=conf)
        for i, p in enumerate(phs)
    ]


def _score_ko(predicted, reference, spans, l1_profile=None, l1_enabled=True):
    return compute_phoneme_score(
        _segments(predicted), reference, spans,
        l1_enabled=l1_enabled,
        l1_profile=l1_profile,
        # Tắt noise gate để test nhìn thẳng vào penalty L1 (không bị gate che).
        recognizer_noise_conf=0.0, recognizer_noise_conf_vowel=0.0,
        profile=get_profile("ko"),
    )


def test_tense_sub_penalty_halved_with_vi_ko():
    # 까 /k͈ a/ đọc thành /k a/ — tense→plain.
    reference = ["k͈", "a"]
    spans = [WordSpan(word="까", start_idx=0, end_idx=2)]
    base = _score_ko(["k", "a"], reference, spans, l1_profile=None)
    lenient = _score_ko(["k", "a"], reference, spans,
                        l1_profile=get_l1_profile("vi", "ko"))
    p_base = base.words[0].phonemes[0]
    p_len = lenient.words[0].phonemes[0]
    assert p_base.status == p_len.status == "sub"          # vẫn hiển thị lỗi
    assert p_base.penalty_reason == PenaltyReason.HARD_ERROR.value
    assert p_len.penalty_reason == PenaltyReason.L1_SUBSTITUTION.value
    assert p_len.penalty_adjustment == pytest.approx(0.5)
    # Penalty giảm → accuracy tăng, và đúng bằng nửa phần phạt của sub đó.
    assert lenient.overall_accuracy > base.overall_accuracy
    assert lenient.l1_adjusted_count == 1


def test_non_table_sub_unchanged_by_vi_ko():
    # ɯ→u KHÔNG có trong bảng → hai bên giống hệt (không tolerance ké).
    reference = ["k", "ɯ"]
    spans = [WordSpan(word="그", start_idx=0, end_idx=2)]
    base = _score_ko(["k", "u"], reference, spans, l1_profile=None)
    lenient = _score_ko(["k", "u"], reference, spans,
                        l1_profile=get_l1_profile("vi", "ko"))
    assert base.overall_accuracy == pytest.approx(lenient.overall_accuracy)
    assert lenient.l1_adjusted_count == 0


def test_coda_l_deletion_lenient_with_vi_ko():
    # 물 /m u l/ thiếu /l/ cuối. So với l1_enabled=False (penalty đầy đủ) — KHÔNG
    # so với profile None: bảng EN cũ tình cờ cũng có "l" (final_liquid) nên
    # fallback None cho cùng multiplier; điểm chốt là vi_ko tự khớp + gắn đúng rule.
    reference = ["m", "u", "l"]
    spans = [WordSpan(word="물", start_idx=0, end_idx=3)]
    raw = _score_ko(["m", "u"], reference, spans, l1_enabled=False)
    lenient = _score_ko(["m", "u"], reference, spans,
                        l1_profile=get_l1_profile("vi", "ko"))
    d_raw = raw.words[0].phonemes[2]
    d_len = lenient.words[0].phonemes[2]
    assert d_raw.status == d_len.status == "del"
    assert d_len.penalty_reason == PenaltyReason.L1_FINAL_DELETION.value
    assert d_len.penalty_adjustment == pytest.approx(0.5)
    assert lenient.overall_accuracy > raw.overall_accuracy
    # Coda nasal bị nuốt KHÔNG được vi_ko dung sai (bảng chỉ có l).
    ref_n = ["m", "u", "n"]
    spans_n = [WordSpan(word="문", start_idx=0, end_idx=3)]
    raw_n = _score_ko(["m", "u"], ref_n, spans_n, l1_enabled=False)
    len_n = _score_ko(["m", "u"], ref_n, spans_n,
                      l1_profile=get_l1_profile("vi", "ko"))
    assert len_n.overall_accuracy == pytest.approx(raw_n.overall_accuracy)


def test_en_bit_for_bit_when_profile_none():
    # EN + l1_enabled=True (như prod): thêm tham số l1_profile không đổi GÌ khi None
    # — so cả accuracy lẫn từng point (status/penalty_reason/adjustment).
    reference = ["h", "ə", "l", "oʊ"]
    spans = [WordSpan(word="hello", start_idx=0, end_idx=4)]
    kwargs = dict(l1_enabled=True, recognizer_noise_conf=0.0,
                  recognizer_noise_conf_vowel=0.0)
    a = compute_phoneme_score(_segments(["h", "ə", "l"]), reference, spans, **kwargs)
    b = compute_phoneme_score(_segments(["h", "ə", "l"]), reference, spans,
                              l1_profile=None, **kwargs)
    c = compute_phoneme_score(_segments(["h", "ə", "l"]), reference, spans,
                              l1_profile=VI_EN, **kwargs)
    for other in (b, c):
        assert other.overall_accuracy == pytest.approx(a.overall_accuracy)
        for pa, po in zip(a.words[0].phonemes, other.words[0].phonemes):
            assert (pa.status, pa.penalty_reason, pa.penalty_adjustment) == (
                po.status, po.penalty_reason, po.penalty_adjustment)


# ── 3. Config flag ──────────────────────────────────────────────────────────────

def test_config_flag_default_off(monkeypatch):
    monkeypatch.delenv("TOEIC_PHONEME_L1_KO_ENABLED", raising=False)
    import dataclasses as _dc

    from src.config import Config
    field = {f.name: f for f in _dc.fields(Config)}["phoneme_l1_ko_enabled"]
    assert field.default is False

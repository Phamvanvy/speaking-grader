"""Golden tests cho G2P tiếng Hàn (hangul → 표준발음법 → IPA) + Korean LangProfile.

Golden strings lấy từ ví dụ chuẩn trong 표준 발음법 (chuẩn phát âm quốc ngữ Hàn) —
deterministic: đổi rule engine mà lệch bảng này là regression thật, không phải noise.
"""

from __future__ import annotations

import pytest

from src.phoneme.ipa.ko.hangul import (
    CHOSEONG,
    HANGUL_BASE,
    JONGSEONG,
    JUNGSEONG,
    decompose_syllable,
    decompose_word,
)
from src.phoneme.ipa.ko.phonology import apply_phonology
from src.phoneme.ipa.ko import (
    text_to_ipa_sequence_with_spans_ko,
    word_to_ipa_ko,
)
from src.phoneme.ipa.ko.similarity_ko import (
    deletion_severity_ko,
    phoneme_similarity_ko,
    phonemes_match_ko,
)
from src.phoneme.ipa.profile import get_profile


def _recompose(sylls: list[list[str]]) -> str:
    """Ghép jamo về Hangul để so golden dễ đọc (chỉ dùng trong test)."""
    out = ""
    for cho, jung, jong in sylls:
        out += chr(
            HANGUL_BASE
            + (CHOSEONG.index(cho) * 21 + JUNGSEONG.index(jung)) * 28
            + JONGSEONG.index(jong)
        )
    return out


# ── decompose ────────────────────────────────────────────────────────────────

def test_decompose_syllable():
    assert decompose_syllable("한") == ("ㅎ", "ㅏ", "ㄴ")
    assert decompose_syllable("가") == ("ㄱ", "ㅏ", "")
    assert decompose_syllable("값") == ("ㄱ", "ㅏ", "ㅄ")


def test_decompose_rejects_non_hangul():
    with pytest.raises(ValueError):
        decompose_syllable("a")


# ── 표준발음법 golden (surface form Hangul) ───────────────────────────────────

GOLDEN_PHONOLOGY = {
    # điều 17 구개음화
    "같이": "가치",
    "굳이": "구지",
    # điều 12 ㅎ (bật hơi 2 chiều + ㅎ tan)
    "좋다": "조타",
    "좋아요": "조아요",
    "많아요": "마나요",
    "입학": "이팍",
    "못하다": "모타다",
    "놓는": "논는",
    # điều 13/14 liaison (kể cả coda đôi + ㅆ)
    "옷이": "오시",
    "있어요": "이써요",
    "앉아요": "안자요",
    "읽어요": "일거요",
    "값이": "갑씨",
    "먹었어요": "머거써요",
    # điều 9/10/11 coda neutralization
    "꽃": "꼳",
    "앞": "압",
    "부엌": "부억",
    "넋": "넉",
    # điều 18/19 비음화
    "국물": "궁물",
    "입니다": "임니다",
    "감사합니다": "감사함니다",
    "닫는": "단는",
    "독립": "동닙",
    "침략": "침냑",
    # điều 20 유음화
    "신라": "실라",
    "칼날": "칼랄",
    # điều 23 경음화
    "학교": "학꾜",
    "국밥": "국빱",
    "옆집": "엽찝",
}


@pytest.mark.parametrize(("src", "want"), sorted(GOLDEN_PHONOLOGY.items()))
def test_phonology_golden(src: str, want: str):
    assert _recompose(apply_phonology(decompose_word(src))) == want


def test_word_overrides():
    # Ngoại lệ morphology (điều 10/11) đi qua bảng override, không qua rule engine.
    assert word_to_ipa_ko("밟다") == word_to_ipa_ko("밥따")
    assert word_to_ipa_ko("맑게") == word_to_ipa_ko("말께")


# ── IPA output ───────────────────────────────────────────────────────────────

def test_word_to_ipa_basic():
    assert word_to_ipa_ko("안녕하세요") == [
        "a", "n", "n", "j", "ʌ", "ŋ", "h", "a", "s", "e", "j", "o",
    ]
    # tensification giữ tense symbol; coda giữ 7 âm chuẩn
    assert word_to_ipa_ko("학교") == ["h", "a", "k", "k͈", "j", "o"]


def test_text_to_ipa_spans_structure():
    phs, spans, stress, disp = text_to_ipa_sequence_with_spans_ko(
        "저는 한국어를 공부해요"
    )
    assert [s.word for s in spans] == ["저는", "한국어를", "공부해요"]
    assert all(s.source == "ko_g2p" for s in spans)
    # spans phủ kín + không chồng lấn
    assert spans[0].start_idx == 0
    for a, b in zip(spans, spans[1:]):
        assert a.end_idx == b.start_idx
    assert spans[-1].end_idx == len(phs)
    # tiếng Hàn không có stress
    assert stress == [None] * len(phs)
    assert disp == [None] * len(phs)


def test_tokenizer_drops_non_hangul():
    phs, spans, _st, _ds = text_to_ipa_sequence_with_spans_ko("3시에 meeting 있어요")
    assert [s.word for s in spans] == ["시에", "있어요"]


# ── similarity / tolerance ───────────────────────────────────────────────────

def test_similarity_laryngeal_triads():
    assert phoneme_similarity_ko("k", "k͈") == 0.75
    assert phoneme_similarity_ko("kʰ", "k͈") == 0.65
    assert phoneme_similarity_ko("s", "s͈") == 0.75
    # identical sau normalize (ㅐ/ㅔ merger)
    assert phoneme_similarity_ko("ɛ", "e") == 1.0


def test_match_allophones():
    # lenis voiced giữa 2 âm hữu thanh — recognizer emit voiced = phát âm ĐÚNG
    assert phonemes_match_ko("k", "ɡ")
    assert phonemes_match_ko("p", "b")
    assert phonemes_match_ko("ɾ", "l")
    # tense/aspirated KHÔNG match lenis — contrast có nghĩa
    assert not phonemes_match_ko("k", "k͈")
    assert not phonemes_match_ko("t", "tʰ")


def test_deletion_severity_batchim():
    # coda stop unreleased → low (recognizer-prone theo nguyên tắc)
    assert deletion_severity_ko("k", is_onset=False) == "low"
    # coda mũi thiếu là lỗi thật đáng nhắc
    assert deletion_severity_ko("n", is_onset=False) == "medium"
    # onset + nguyên âm nặng
    assert deletion_severity_ko("k", is_onset=True) == "high"
    assert deletion_severity_ko("a") == "high"


# ── LangProfile ko ───────────────────────────────────────────────────────────

def test_ko_profile_wiring():
    p = get_profile("ko")
    assert p.lang == "ko"
    assert not p.english_rules_enabled
    assert p.function_words == frozenset()
    phs, spans, _st, _ds = p.text_to_ipa_with_spans("감사합니다")
    assert [s.word for s in spans] == ["감사합니다"]
    assert phs == ["k", "a", "m", "s", "a", "h", "a", "m", "n", "i", "t", "a"]


def test_ko_profile_scoring_end_to_end():
    """compute_phoneme_score chạy trọn với profile ko — perfect read = 1.0."""
    from src.phoneme.models import PhonemeSegment
    from src.phoneme.scoring import compute_phoneme_score

    p = get_profile("ko")
    phs, spans, stress, disp = p.text_to_ipa_with_spans("저는 학교에 가요")
    segs = [
        PhonemeSegment(phoneme=x, start=i * 0.1, end=i * 0.1 + 0.05, confidence=0.9)
        for i, x in enumerate(phs)
    ]
    score = compute_phoneme_score(
        segs, phs, spans, stress, reference_display_stress=disp, profile=p
    )
    assert score is not None
    assert score.overall_accuracy == 1.0
    # voiced allophone thay lenis vẫn 1.0 (부부: ㅂ thứ 2 nghe ra b)
    segs2 = list(segs)
    k_idx = phs.index("k")
    segs2[k_idx] = PhonemeSegment(
        phoneme="ɡ", start=k_idx * 0.1, end=k_idx * 0.1 + 0.05, confidence=0.9
    )
    score2 = compute_phoneme_score(
        segs2, phs, spans, stress, reference_display_stress=disp, profile=p
    )
    assert score2.overall_accuracy == 1.0
    # tense thay lenis là lỗi thật (k → k͈ sub, similarity 0.75 → penalty nhẹ)
    segs3 = list(segs)
    segs3[k_idx] = PhonemeSegment(
        phoneme="k͈", start=k_idx * 0.1, end=k_idx * 0.1 + 0.05, confidence=0.9
    )
    score3 = compute_phoneme_score(
        segs3, phs, spans, stress, reference_display_stress=disp, profile=p
    )
    assert score3.substitution_count == 1
    assert 0.0 < score3.overall_accuracy < 1.0


def test_exam_language_registry():
    from src.rubrics.base import exam_language, exam_score_field, exam_score_max

    assert exam_language("topik") == "ko"
    assert exam_language("toeic") == "en"
    assert exam_score_max("topik") == 200
    assert exam_score_field("topik") == "estimated_topik_score"

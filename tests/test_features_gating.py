"""Test offline cho features + gating (không cần audio thật hay API key).

Dựng Transcription giả lập để kiểm tra logic tính toán.
"""

from __future__ import annotations

from src.asr import Transcription, Word
from src import features as features_mod
from src import gating


def _make_words(words_with_times):
    return [
        Word(text=t, start=s, end=e, probability=p)
        for (t, s, e, p) in words_with_times
    ]


def test_features_basic():
    # 5 từ trong 2 giây → 150 wpm; có 1 quãng ngắt 0.5s
    words = _make_words([
        ("I", 0.0, 0.2, 0.95),
        ("like", 0.2, 0.5, 0.90),
        ("hot", 1.0, 1.3, 0.80),   # gap 0.5s trước "hot" → 1 pause
        ("black", 1.3, 1.6, 0.85),
        ("coffee", 1.6, 2.0, 0.92),
    ])
    tr = Transcription(text="I like hot black coffee", words=words, duration=2.5)

    feats = features_mod.extract_features(tr)
    assert feats.word_count == 5
    assert feats.pause_count == 1
    assert feats.longest_pause_sec == 0.5
    assert feats.speech_rate_wpm > 0
    assert feats.features_version == "v1"


def test_read_aloud_accuracy_deletion():
    # Script có "coffee" nhưng người đọc bỏ → phải có ít nhất 1 deletion
    words = _make_words([
        ("I", 0.0, 0.3, 0.95),
        ("like", 0.3, 0.8, 0.90),
    ])
    tr = Transcription(text="I like", words=words, duration=1.0)

    feats = features_mod.extract_features(tr, reference_script="I like coffee")
    assert feats.accuracy_metrics is not None
    assert feats.accuracy_metrics.deletions >= 1
    assert feats.accuracy_metrics.wer > 0


def test_word_issues_substitution():
    # Script "good morning" nhưng ASR nghe "good warning" → 1 substitution.
    words = _make_words([("good", 0.0, 0.3, 0.9), ("warning", 0.3, 0.8, 0.7)])
    tr = Transcription(text="good warning", words=words, duration=1.0)
    feats = features_mod.extract_features(tr, reference_script="good morning")
    assert feats.accuracy_metrics is not None
    issues = feats.accuracy_metrics.word_issues
    subs = [i for i in issues if i.issue_type == "substitution"]
    assert any(i.expected == "morning" and i.recognized == "warning" for i in subs)


def test_word_issues_deletion_and_insertion():
    # Script "i like coffee"; đọc "i really like" → thiếu 'coffee' (deletion),
    # thừa 'really' (insertion).
    words = _make_words(
        [("i", 0.0, 0.2, 0.9), ("really", 0.2, 0.5, 0.9), ("like", 0.5, 0.9, 0.9)]
    )
    tr = Transcription(text="i really like", words=words, duration=1.0)
    feats = features_mod.extract_features(tr, reference_script="i like coffee")
    issues = feats.accuracy_metrics.word_issues
    assert any(i.issue_type == "deletion" and i.expected == "coffee" for i in issues)
    assert any(i.issue_type == "insertion" and i.recognized == "really" for i in issues)


def test_word_issues_empty_when_perfect():
    ref = "i like coffee very much"
    words = _make_words(
        [(w, i * 0.5, i * 0.5 + 0.3, 0.9) for i, w in enumerate(ref.split())]
    )
    tr = Transcription(text=ref, words=words, duration=3.0)
    feats = features_mod.extract_features(tr, reference_script=ref)
    assert feats.accuracy_metrics.word_issues == []


def test_gating_too_short():
    words = _make_words([("Yes", 0.0, 0.5, 0.9)])
    tr = Transcription(text="Yes", words=words, duration=0.5)
    feats = features_mod.extract_features(tr)

    gate = gating.evaluate(tr, feats, expected_duration_sec=60)
    assert gate.task_completion_floor == "very_low"
    assert not gate.should_skip_ai  # có 1 từ → vẫn gọi được AI, nhưng floor thấp


def test_gating_empty_audio():
    tr = Transcription(text="", words=[], duration=3.0)
    feats = features_mod.extract_features(tr)
    gate = gating.evaluate(tr, feats)
    assert gate.is_empty
    assert gate.should_skip_ai


def test_gating_short_vs_expected():
    # Nói 20s nhưng kỳ vọng 60s → tỉ lệ 0.33 < 0.4 → floor 'low'
    words = _make_words(
        [(f"w{i}", float(i), float(i) + 0.4, 0.9) for i in range(20)]
    )
    tr = Transcription(text=" ".join(f"w{i}" for i in range(20)), words=words, duration=20.0)
    feats = features_mod.extract_features(tr)
    gate = gating.evaluate(tr, feats, expected_duration_sec=60)
    assert gate.task_completion_floor == "low"


def test_coverage_metric():
    # Đọc trúng "i like", thiếu "coffee" → coverage = 2/3
    words = _make_words([("I", 0.0, 0.3, 0.95), ("like", 0.3, 0.8, 0.90)])
    tr = Transcription(text="I like", words=words, duration=1.0)
    feats = features_mod.extract_features(tr, reference_script="I like coffee")
    assert feats.accuracy_metrics is not None
    assert abs(feats.accuracy_metrics.coverage - 2 / 3) < 1e-3


def test_reading_pace_computed_not_gating():
    # pace_ratio tính từ expected nhưng KHÔNG ảnh hưởng gating của Read Aloud.
    from src.rubrics.toeic import get_question_type

    ref = "I like coffee very much"
    words = _make_words(
        [(w, i * 1.5, i * 1.5 + 1.0, 0.9) for i, w in enumerate(ref.split())]
    )
    tr = Transcription(text=ref, words=words, duration=16.0)
    feats = features_mod.extract_features(
        tr, reference_script=ref, expected_duration_sec=45
    )
    assert feats.reading_pace is not None
    assert feats.reading_pace.expected_duration_sec == 45
    # actual ~ 6.0s (5 từ, mốc cuối 6.0+1.0? -> end của từ cuối = 4*1.5+1=7.0)
    assert 0.0 < feats.reading_pace.pace_ratio < 1.0
    # Không có expected → không có reading_pace
    assert features_mod.extract_features(tr, reference_script=ref).reading_pace is None


def test_read_aloud_fast_but_complete_not_penalized():
    # Read Aloud: đọc đủ script nhưng nhanh (16s, kỳ vọng 45s) → KHÔNG bị phạt.
    from src.rubrics.toeic import get_question_type

    ref = "I like coffee very much"
    # Trải đều >5s để không dính rule thời lượng tối thiểu (đọc đủ nhưng nhanh).
    words = _make_words(
        [(w, i * 1.5, i * 1.5 + 1.0, 0.9) for i, w in enumerate(ref.split())]
    )
    tr = Transcription(text=ref, words=words, duration=16.0)
    feats = features_mod.extract_features(tr, reference_script=ref)
    gate = gating.evaluate(
        tr, feats, expected_duration_sec=45,
        question_type=get_question_type("read_aloud"),
    )
    assert gate.task_completion_floor is None  # đọc đủ → không cap
    assert gate.fail_reference_match is False
    assert gate.reference_coverage == 1.0


def test_read_aloud_wrong_passage_flagged():
    # Read Aloud: đọc đoạn hoàn toàn khác → coverage thấp → very_low + cờ fail.
    from src.rubrics.toeic import get_question_type

    ref = "Thank you for calling Sunrise Electronics our store hours are nine"
    spoken = "Are you a global company needing to deal with many foreign currencies"
    words = _make_words(
        [(w, float(i), float(i) + 0.3, 0.9) for i, w in enumerate(spoken.split())]
    )
    tr = Transcription(text=spoken, words=words, duration=18.0)
    feats = features_mod.extract_features(tr, reference_script=ref)
    gate = gating.evaluate(
        tr, feats, expected_duration_sec=45,
        question_type=get_question_type("read_aloud"),
    )
    assert gate.fail_reference_match is True
    assert gate.task_completion_floor == "very_low"
    assert gate.reference_coverage < 0.5

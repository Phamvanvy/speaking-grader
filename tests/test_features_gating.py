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

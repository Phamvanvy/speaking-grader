"""Tests cho audio chunking TRƯỚC wav2vec (fix IPA "lem" trên audio dài).

Ba lớp:
  - compute_chunk_spans: pause/hybrid đúng thứ tự ưu tiên (pause → ranh giới câu
    → hard-cut), bất biến span (phủ mọi từ, không overlap, tăng dần, ≤ max trừ
    từ đơn quá dài), pad clamp về trung điểm gap, determinism, edge cases.
  - ChunkedFramePosteriors.evidence_stats: routing đúng chunk theo thời gian
    tuyệt đối, cửa sổ vắt 2 chunk lấy max_mass lớn hơn, cửa sổ trong gap im lặng
    → stats 0/n_frames=0, vocab thiếu token → None (cùng ngữ nghĩa bản đơn).
  - Predictor chunked: segment times cộng đúng offset + thứ tự thời gian
    (mock forward pass); chunk_spans=None → đường single-pass cũ.
"""

import numpy as np
import pytest

from src.phoneme.chunking import compute_chunk_spans
from src.phoneme.models import PhonemeSegment
from src.phoneme.wav2vec_backend import (
    ChunkedFramePosteriors,
    FramePosteriors,
    Wav2VecPhonemePredictor,
)

FRAME = 0.02


def _posteriors(id_to_token, n_frames, model_id, cells=None):
    """FramePosteriors tổng hợp: nền <pad>≈1.0, ghi đè từng ô (frame, token_id, prob)."""
    vocab = len(id_to_token)
    probs = np.zeros((n_frames, vocab), dtype=np.float32)
    probs[:, 0] = 1.0
    for frame, token_id, p in cells or []:
        probs[frame, token_id] = p
        probs[frame, 0] = max(0.0, 1.0 - p)
    return FramePosteriors(
        probs=probs, frame_duration=FRAME, id_to_token=id_to_token,
        model_id=model_id,
    )


# ── compute_chunk_spans ───────────────────────────────────────────────────────

def _spans_cover_all_words(spans, words, eps=1e-6):
    # eps: span đã round 3 chữ số, word timestamps có nhiễu float (4.3000000004).
    return all(
        any(lo <= s + eps and e <= hi + eps for lo, hi in spans)
        for _t, s, e in words
    )


def _non_overlapping_sorted(spans):
    return all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


class TestComputeChunkSpans:
    def test_empty_words(self):
        assert compute_chunk_spans([], 10.0, "pause") == []

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            compute_chunk_spans([("a", 0.0, 0.5)], 10.0, "sentence")

    def test_single_word(self):
        spans = compute_chunk_spans([("hello", 1.0, 1.5)], 10.0, "pause",
                                    pad_sec=0.25)
        assert spans == [(0.75, 1.75)]

    def test_pause_splits_at_gap(self):
        words = [("a", 0.0, 0.4), ("b", 0.5, 0.9),      # gap 0.1 < 0.5 — dính
                 ("c", 2.0, 2.4), ("d", 2.5, 2.9)]      # gap 1.1 ≥ 0.5 — cắt
        spans = compute_chunk_spans(words, 10.0, "pause", pad_sec=0.0)
        assert spans == [(0.0, 0.9), (2.0, 2.9)]

    def test_pad_clamped_to_gap_midpoint_no_overlap(self):
        # Gap [0.9, 1.5], pad 0.5 > nửa gap → clamp về trung điểm 1.2.
        words = [("a", 0.0, 0.9), ("b", 1.5, 2.0)]
        spans = compute_chunk_spans(words, 10.0, "pause", min_pause_sec=0.5,
                                    pad_sec=0.5)
        assert spans == [(0.0, 1.2), (1.2, 2.5)]
        assert _non_overlapping_sorted(spans)

    def test_pad_clamped_to_zero_and_duration(self):
        words = [("a", 0.1, 0.5)]
        spans = compute_chunk_spans(words, 0.6, "pause", pad_sec=1.0)
        assert spans == [(0.0, 0.6)]

    def test_pause_hard_cuts_long_chunk_at_word_boundary(self):
        # Không có gap ≥ 0.5, tổng 4.3s > max 2.5 → hard-cut tại biên từ:
        # 2 từ đầu (0.0→2.1 = 2.1s ≤ 2.5), 2 từ sau (2.2→4.3).
        words = [(f"w{i}", i * 1.1, i * 1.1 + 1.0) for i in range(4)]
        spans = compute_chunk_spans(words, 10.0, "pause", max_chunk_sec=2.5,
                                    pad_sec=0.0)
        assert spans == [(0.0, 2.1), (2.2, 4.3)]
        assert _spans_cover_all_words(spans, words)

    def test_hybrid_prefers_sentence_boundary_over_hard_cut(self):
        # Nhóm 1 hơi dài hơn max; ranh giới câu ("two.") nằm TRƯỚC điểm hard-cut
        # → hybrid cắt tại câu, pause hard-cut tại từ xa nhất còn vừa max.
        words = [("one", 0.0, 0.5), ("two.", 0.6, 1.0),
                 ("three", 1.1, 1.6), ("four", 1.7, 2.6)]
        hybrid = compute_chunk_spans(words, 10.0, "hybrid", max_chunk_sec=2.0,
                                     pad_sec=0.0)
        pause = compute_chunk_spans(words, 10.0, "pause", max_chunk_sec=2.0,
                                    pad_sec=0.0)
        assert hybrid == [(0.0, 1.0), (1.1, 2.6)]   # cắt sau "two."
        assert pause == [(0.0, 1.6), (1.7, 2.6)]    # hard-cut sau "three"

    def test_hybrid_hard_cuts_when_no_punctuation(self):
        words = [("one", 0.0, 0.5), ("two", 0.6, 1.0),
                 ("three", 1.1, 1.6), ("four", 1.7, 2.6)]
        hybrid = compute_chunk_spans(words, 10.0, "hybrid", max_chunk_sec=2.0,
                                     pad_sec=0.0)
        assert hybrid == [(0.0, 1.6), (1.7, 2.6)]

    def test_sentence_boundary_strips_closing_quotes(self):
        words = [("said.\"", 0.0, 0.5), ("Then", 0.6, 1.2), ("more", 1.3, 2.4)]
        spans = compute_chunk_spans(words, 10.0, "hybrid", max_chunk_sec=1.5,
                                    pad_sec=0.0)
        assert spans[0] == (0.0, 0.5)  # cắt sau said." (bỏ dấu bao đóng)

    def test_single_word_longer_than_max_kept_whole(self):
        words = [("looong", 0.0, 5.0), ("next", 5.1, 5.4)]
        spans = compute_chunk_spans(words, 10.0, "hybrid", max_chunk_sec=2.0,
                                    pad_sec=0.0)
        assert spans == [(0.0, 5.0), (5.1, 5.4)]

    def test_huge_gap_like_intro_music(self):
        # Gap 15s (nhạc intro 9.0.mp4) — im lặng KHÔNG thuộc chunk nào.
        words = [("intro.", 0.0, 1.0), ("Speech", 16.0, 16.5)]
        spans = compute_chunk_spans(words, 20.0, "hybrid", pad_sec=0.25)
        assert spans == [(0.0, 1.25), (15.75, 16.75)]

    def test_deterministic(self):
        words = [(f"w{i}.", i * 0.7, i * 0.7 + 0.5) for i in range(50)]
        a = compute_chunk_spans(words, 60.0, "hybrid", max_chunk_sec=5.0)
        b = compute_chunk_spans(words, 60.0, "hybrid", max_chunk_sec=5.0)
        assert a == b
        assert _non_overlapping_sorted(a)
        assert _spans_cover_all_words(a, words)

    def test_max_enforced_all_strategies(self):
        words = [(f"w{i}", i * 0.6, i * 0.6 + 0.5) for i in range(100)]
        for strategy in ("pause", "hybrid"):
            spans = compute_chunk_spans(words, 70.0, strategy,
                                        max_chunk_sec=10.0, pad_sec=0.25)
            assert all(hi - lo <= 10.0 + 2 * 0.25 + 1e-9 for lo, hi in spans)
            assert _non_overlapping_sorted(spans)
            assert _spans_cover_all_words(spans, words)


# ── ChunkedFramePosteriors ────────────────────────────────────────────────────

class TestChunkedFramePosteriors:
    VOCAB = {0: "<pad>", 1: "l", 2: "k"}

    def _chunked(self):
        # Chunk A bắt đầu 0.0s (50 frame = 1.0s): /l/ mass 0.5 tại frame 10 (~0.2s).
        # Chunk B bắt đầu 2.0s (50 frame): /l/ mass 0.8 tại frame 5 (~2.1s tuyệt đối).
        a = _posteriors(self.VOCAB, 50, "t-chunked", cells=[(10, 1, 0.5)])
        b = _posteriors(self.VOCAB, 50, "t-chunked", cells=[(5, 1, 0.8)])
        return ChunkedFramePosteriors(chunks=((0.0, a), (2.0, b)))

    def test_routes_to_correct_chunk_with_offset(self):
        post = self._chunked()
        # Cửa sổ tuyệt đối quanh 2.1s → chunk B, local [0.06, 0.14].
        stats = post.evidence_stats("l", 2.06, 2.14)
        assert stats is not None
        assert abs(stats.max_mass - 0.8) < 1e-6
        # So với query trực tiếp chunk B bằng thời gian local: kết quả y hệt.
        direct = post.chunks[1][1].evidence_stats("l", 0.06, 0.14)
        assert stats == direct

    def test_window_spanning_two_chunks_takes_larger_mass(self):
        post = self._chunked()
        stats = post.evidence_stats("l", 0.1, 2.2)  # phủ cả A (0.5) lẫn B (0.8)
        assert stats is not None
        assert abs(stats.max_mass - 0.8) < 1e-6

    def test_window_in_silence_gap_returns_zero_stats(self):
        post = self._chunked()
        stats = post.evidence_stats("l", 1.2, 1.8)  # gap giữa A (hết 1.0s) và B (2.0s)
        assert stats is not None
        assert stats.max_mass == 0.0
        assert stats.n_frames == 0

    def test_unknown_ipa_returns_none(self):
        post = self._chunked()
        assert post.evidence_stats("θ", 0.1, 0.3) is None
        assert post.evidence_stats("θ", 1.2, 1.8) is None  # cả trong gap


# ── Predictor chunked: offset + thứ tự + fallback single-pass ────────────────

class TestPredictChunked:
    VOCAB = {0: "<pad>", 1: "l", 2: "k"}

    def _predictor_with_mock(self, monkeypatch):
        predictor = Wav2VecPhonemePredictor(model_id="mock", device="cpu")

        def fake_forward(waveform, feature_extractor, model, id_to_label, torch):
            # 1 segment /l/ ở [0.1, 0.2] LOCAL cho mỗi chunk; posteriors 1 frame
            # mỗi 0.02s để ChunkedFramePosteriors tính đúng chunk_end.
            n_frames = max(1, int(len(waveform) / 16000 / FRAME))
            segs = [PhonemeSegment(phoneme="l", start=0.1, end=0.2,
                                   confidence=0.9, backend="wav2vec")]
            return segs, _posteriors(self.VOCAB, n_frames, "mock")

        monkeypatch.setattr(predictor, "_forward_decode", fake_forward)
        return predictor

    def test_segments_offset_and_ordered(self, monkeypatch):
        predictor = self._predictor_with_mock(monkeypatch)
        waveform = np.zeros(16000 * 6, dtype=np.float32)  # 6s
        spans = [(0.0, 2.0), (3.0, 5.0)]
        segments, posteriors = predictor._predict_chunked(
            waveform, spans, None, None, self.VOCAB, None
        )
        assert [(s.start, s.end) for s in segments] == [(0.1, 0.2), (3.1, 3.2)]
        assert [t for t, _p in posteriors.chunks] == [0.0, 3.0]

    def test_tiny_and_out_of_range_spans_skipped(self, monkeypatch):
        predictor = self._predictor_with_mock(monkeypatch)
        waveform = np.zeros(16000 * 2, dtype=np.float32)  # 2s
        spans = [(0.5, 0.51),    # 10ms < _MIN_CHUNK_SEC → bỏ
                 (5.0, 6.0),     # ngoài audio → clamp rỗng → bỏ
                 (1.0, 1.5)]     # hợp lệ
        segments, posteriors = predictor._predict_chunked(
            waveform, spans, None, None, self.VOCAB, None
        )
        assert len(posteriors.chunks) == 1
        assert posteriors.chunks[0][0] == 1.0
        assert [(s.start, s.end) for s in segments] == [(1.1, 1.2)]

    def test_none_chunk_spans_uses_single_pass(self, monkeypatch):
        """chunk_spans=None → KHÔNG đi qua _predict_chunked (đường cũ nguyên vẹn)."""
        predictor = Wav2VecPhonemePredictor(model_id="mock", device="cpu")
        called = {"chunked": False}
        monkeypatch.setattr(
            predictor, "_predict_chunked",
            lambda *a, **k: called.__setitem__("chunked", True),
        )
        # Không có file + backend mock không available → return sớm, nhưng đủ để
        # khẳng định nhánh chunked không được gọi.
        segments, warning, post = predictor.predict_with_posteriors(
            "nonexistent_file.wav", chunk_spans=None
        )
        assert called["chunked"] is False
        assert segments == [] and warning is not None


# ── Parallel chunk trên nhiều device (TOEIC_PHONEME_DEVICES) ─────────────────

class TestPredictChunkedParallel:
    VOCAB = {0: "<pad>", 1: "l", 2: "k"}

    def _mk_predictor(self, monkeypatch, devices=("cpu", "cuda:9"),
                      fail_device=None):
        """Predictor với _forward_decode mock: ghi lại (giây_chunk, device) mỗi
        call; fail_device != None → raise RuntimeError khi worker đó chạy."""
        predictor = Wav2VecPhonemePredictor(
            model_id="mock", device="cpu", devices=list(devices)
        )
        calls: list[tuple[int, str | None]] = []

        def fake_forward(waveform, feature_extractor, model, id_to_label,
                         torch, device=None):
            if fail_device is not None and device == fail_device:
                raise RuntimeError("boom")
            calls.append((round(len(waveform) / 16000), device))
            n_frames = max(1, int(len(waveform) / 16000 / FRAME))
            segs = [PhonemeSegment(phoneme="l", start=0.1, end=0.2,
                                   confidence=0.9, backend="wav2vec")]
            return segs, _posteriors(self.VOCAB, n_frames, "mock")

        monkeypatch.setattr(predictor, "_forward_decode", fake_forward)
        # Worker lấy model per-device qua module-level _get_wav2vec_model — stub
        # để không tải model thật.
        monkeypatch.setattr(
            "src.phoneme.wav2vec_backend._get_wav2vec_model",
            lambda model_id, device="cpu": (None, None, self.VOCAB),
        )
        return predictor, calls

    # 5 spans: span thứ 2 quá ngắn (< _MIN_CHUNK_SEC) → 4 job hợp lệ, mỗi job
    # một độ dài riêng để nhận diện job qua độ dài waveform trong mock.
    SPANS = [(0.0, 1.0), (1.2, 1.21), (2.0, 4.0), (5.0, 8.0), (9.0, 13.0)]

    def test_parallel_merge_matches_sequential(self, monkeypatch):
        """Output parallel == sequential từng segment + từng chunk offset."""
        predictor, _calls = self._mk_predictor(monkeypatch)
        waveform = np.zeros(16000 * 14, dtype=np.float32)
        seq_segments, seq_post = predictor._predict_chunked(
            waveform, self.SPANS, None, None, self.VOCAB, None
        )
        jobs = predictor._chunk_jobs(waveform, self.SPANS)
        par_segments, par_post = predictor._predict_chunked_parallel(
            waveform, jobs, self.VOCAB, None, ["cpu", "cuda:9"]
        )
        assert (
            [(s.phoneme, s.start, s.end) for s in par_segments]
            == [(s.phoneme, s.start, s.end) for s in seq_segments]
        )
        assert (
            [t for t, _p in par_post.chunks]
            == [t for t, _p in seq_post.chunks]
        )

    def test_round_robin_assignment(self, monkeypatch):
        """Chunk i → devices[i % n], tất định (nhận diện job qua độ dài)."""
        predictor, calls = self._mk_predictor(monkeypatch)
        waveform = np.zeros(16000 * 14, dtype=np.float32)
        jobs = predictor._chunk_jobs(waveform, self.SPANS)
        predictor._predict_chunked_parallel(
            waveform, jobs, self.VOCAB, None, ["cpu", "cuda:9"]
        )
        # Job hợp lệ dài 1s, 2s, 3s, 4s → round-robin cpu, cuda:9, cpu, cuda:9.
        assert sorted(calls) == [
            (1, "cpu"), (2, "cuda:9"), (3, "cpu"), (4, "cuda:9"),
        ]

    def test_worker_error_reraises(self, monkeypatch):
        """Worker raise → _predict_chunked_parallel re-raise (caller fallback)."""
        predictor, _calls = self._mk_predictor(monkeypatch, fail_device="cuda:9")
        waveform = np.zeros(16000 * 14, dtype=np.float32)
        jobs = predictor._chunk_jobs(waveform, self.SPANS)
        with pytest.raises(RuntimeError):
            predictor._predict_chunked_parallel(
                waveform, jobs, self.VOCAB, None, ["cpu", "cuda:9"]
            )

    # ── dispatch qua predict_with_posteriors ─────────────────────────────────

    def _run_predict(self, monkeypatch, tmp_path, predictor, spans):
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"fake")
        monkeypatch.setattr(
            "src.phoneme.wav2vec_backend._load_audio",
            lambda p, sr: np.zeros(16000 * 14, dtype=np.float32),
        )
        predictor._available = True
        return predictor.predict_with_posteriors(str(audio), chunk_spans=spans)

    def test_no_devices_never_calls_parallel(self, monkeypatch, tmp_path):
        """Flag OFF (devices rỗng) → không đụng đường parallel (bit-for-bit)."""
        predictor, _calls = self._mk_predictor(monkeypatch, devices=())
        called = {"parallel": False}
        monkeypatch.setattr(
            predictor, "_predict_chunked_parallel",
            lambda *a, **k: called.__setitem__("parallel", True),
        )
        segments, warning, post = self._run_predict(
            monkeypatch, tmp_path, predictor, self.SPANS
        )
        assert called["parallel"] is False
        assert warning is None and len(post.chunks) == 4

    def test_single_chunk_stays_sequential(self, monkeypatch, tmp_path):
        """1 chunk hợp lệ → không parallel (không có gì để chia)."""
        predictor, _calls = self._mk_predictor(monkeypatch)
        called = {"parallel": False}
        monkeypatch.setattr(
            predictor, "_predict_chunked_parallel",
            lambda *a, **k: called.__setitem__("parallel", True),
        )
        segments, warning, post = self._run_predict(
            monkeypatch, tmp_path, predictor, [(0.0, 2.0)]
        )
        assert called["parallel"] is False
        assert warning is None and len(post.chunks) == 1

    def test_parallel_failure_falls_back_sequential(self, monkeypatch, tmp_path):
        """Parallel nổ giữa chừng → kết quả VẪN trả về từ đường tuần tự."""
        predictor, _calls = self._mk_predictor(monkeypatch)

        def boom(*a, **k):
            raise RuntimeError("gpu mất điện")

        monkeypatch.setattr(predictor, "_predict_chunked_parallel", boom)
        segments, warning, post = self._run_predict(
            monkeypatch, tmp_path, predictor, self.SPANS
        )
        assert warning is None
        assert len(post.chunks) == 4
        assert [(s.start, s.end) for s in segments] == [
            (0.1, 0.2), (2.1, 2.2), (5.1, 5.2), (9.1, 9.2),
        ]

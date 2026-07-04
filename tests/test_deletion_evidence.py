"""Tests cho Deletion-Evidence Probe (SHADOW MODE).

Probe soi frame posteriors của wav2vec trong cửa sổ thời gian của từ để đo "âm bị
thiếu có bằng chứng âm học không" — CHỈ telemetry, tuyệt đối không đổi điểm:

  - EvidenceStats: đúng bộ thống kê (max/top_k_mean/p90/n_frames + argmax context)
    trên ma trận posterior tổng hợp; deterministic (bit-giống-nhau giữa 2 lần gọi).
  - Nhóm token: normalize_ipa gộp biến thể (oʊ↔əʊ); âm không có token → None.
  - Cửa sổ: refinement theo âm khớp liền kề trong cùng từ; GUARD cửa sổ đảo
    (start >= end) → fallback cửa sổ nền; fallback Whisper window; source="none".
  - Shadow bất biến: điểm/counts/severity Y HỆT khi có/không posteriors.
"""

import numpy as np

from src.phoneme.models import (
    EVIDENCE_VERSION,
    EvidenceStats,
    PhonemePoint,
    PhonemeSegment,
    WordSpan,
)
from src.phoneme.scoring import compute_phoneme_score
from src.phoneme.scoring.word_details import _attach_deletion_evidence
from src.phoneme.wav2vec_backend import (
    EVIDENCE_WINDOW_MARGIN_FRAMES,
    FramePosteriors,
    _ipa_token_groups,
)

# Frame rate wav2vec ~50Hz.
FRAME = 0.02


def _posteriors(id_to_token, n_frames, model_id, cells=None):
    """FramePosteriors tổng hợp: nền <pad>≈1.0, ghi đè từng ô (frame, token_id, prob)."""
    vocab = len(id_to_token)
    probs = np.zeros((n_frames, vocab), dtype=np.float32)
    probs[:, 0] = 1.0  # id 0 = <pad> (blank) chiếm nền
    for frame, token_id, p in cells or []:
        probs[frame, token_id] = p
        probs[frame, 0] = max(0.0, 1.0 - p)
    return FramePosteriors(
        probs=probs, frame_duration=FRAME, id_to_token=id_to_token,
        model_id=model_id,
    )


def _segs(items):
    """[(phoneme, start, end)] → segments."""
    return [
        PhonemeSegment(phoneme=p, start=s, end=e, confidence=0.9)
        for p, s, e in items
    ]


# ── EvidenceStats: thống kê trên ma trận tổng hợp ────────────────────────────

class TestEvidenceStats:
    VOCAB = {0: "<pad>", 1: "l", 2: "k", 3: "s", 4: "a"}

    def test_stats_match_synthetic_matrix(self):
        # /l/ mass tại frame 10..12 = 0.5, 0.3, 0.2; cửa sổ [0.16, 0.30] phủ hết
        # (margin ±2 frame). max=0.5; top3 mean=(0.5+0.3+0.2)/3.
        post = _posteriors(
            self.VOCAB, 50, "t-stats",
            cells=[(10, 1, 0.5), (11, 1, 0.3), (12, 1, 0.2)],
        )
        stats = post.evidence_stats("l", 0.16, 0.30)
        assert stats is not None
        assert abs(stats.max_mass - 0.5) < 1e-6
        assert abs(stats.top_k_mean - (0.5 + 0.3 + 0.2) / 3) < 1e-6
        assert stats.n_frames > 0
        # Frame mass cao nhất (10): argmax toàn cục vẫn là <pad> (0.5 vs 0.5 → tie
        # về id nhỏ hơn = <pad>)... tránh tie: prob <pad> tại frame 10 là 0.5, bằng
        # /l/ → dùng frame có mass thắng rõ.
        post2 = _posteriors(self.VOCAB, 50, "t-stats2", cells=[(10, 1, 0.8)])
        s2 = post2.evidence_stats("l", 0.16, 0.30)
        assert s2.argmax_token == "l"
        assert abs(s2.argmax_prob - 0.8) < 1e-6

    def test_argmax_context_shows_competing_token(self):
        # /l/ có mass 0.3 nhưng frame đó token "a" thắng argmax (0.6) → probe lộ
        # "wav2vec nghe ra âm gì ở chỗ lẽ ra có /l/".
        post = _posteriors(
            self.VOCAB, 50, "t-argmax", cells=[(10, 1, 0.3)],
        )
        post.probs[10, 4] = 0.6
        post.probs[10, 0] = 0.1
        stats = post.evidence_stats("l", 0.16, 0.26)
        assert abs(stats.max_mass - 0.3) < 1e-6
        assert stats.argmax_token == "a"
        assert abs(stats.argmax_prob - 0.6) < 1e-6

    def test_true_deletion_has_no_mass(self):
        # Không ô nào có /l/ → thiếu âm thật: mọi thống kê ~0 (khác None!).
        post = _posteriors(self.VOCAB, 50, "t-nomass")
        stats = post.evidence_stats("l", 0.10, 0.40)
        assert stats is not None
        assert stats.max_mass == 0.0
        assert stats.top_k_mean == 0.0
        assert stats.p90 == 0.0

    def test_unknown_phoneme_returns_none(self):
        # Vocab không có token nào normalize trùng /θ/ → không đo được → None.
        post = _posteriors(self.VOCAB, 50, "t-unknown")
        assert post.evidence_stats("θ", 0.10, 0.40) is None

    def test_empty_or_out_of_range_window_gives_zero_frames(self):
        post = _posteriors(self.VOCAB, 10, "t-empty", cells=[(5, 1, 0.9)])
        for t0, t1 in [(0.5, 0.4), (5.0, 6.0)]:  # đảo | ngoài biên audio
            stats = post.evidence_stats("l", t0, t1)
            assert stats == EvidenceStats(0.0, 0.0, 0.0, 0)

    def test_deterministic_bit_identical(self):
        post = _posteriors(
            self.VOCAB, 50, "t-deter",
            cells=[(8, 1, 0.11), (9, 1, 0.47), (20, 1, 0.05)],
        )
        a = post.evidence_stats("l", 0.10, 0.45)
        b = post.evidence_stats("l", 0.10, 0.45)
        assert a == b  # equality tuyệt đối (dataclass eq), không approx

    def test_margin_expands_window(self):
        # Mass nằm NGOÀI [t0,t1] đúng 1 frame → margin ±2 frame vẫn bắt được.
        post = _posteriors(self.VOCAB, 50, "t-margin", cells=[(9, 1, 0.4)])
        t0 = (9 + 2) * FRAME  # lo = 11 - margin(2) = 9 → frame 9 nằm trong
        stats = post.evidence_stats("l", t0, t0 + 2 * FRAME)
        assert EVIDENCE_WINDOW_MARGIN_FRAMES >= 2
        assert abs(stats.max_mass - 0.4) < 1e-6


# ── Nhóm token theo normalize_ipa ────────────────────────────────────────────

class TestTokenGroups:
    def test_variants_merge_into_one_group(self):
        # normalize_ipa gộp oʊ↔əʊ → 2 token cùng 1 nhóm; mass = TỔNG 2 cột.
        vocab = {0: "<pad>", 1: "oʊ", 2: "əʊ"}
        groups = _ipa_token_groups("t-groups", vocab)
        merged = {g for g in groups.values() if g == frozenset({1, 2})}
        assert merged, f"oʊ/əʊ phải chung nhóm, got {groups}"

        post = _posteriors(vocab, 20, "t-groups-mass")
        post.probs[5, 1] = 0.2
        post.probs[5, 2] = 0.25
        stats = post.evidence_stats("oʊ", 0.06, 0.14)
        assert abs(stats.max_mass - 0.45) < 1e-6

    def test_length_mark_stripped(self):
        # Token "iː" (espeak) khớp reference "i" sau normalize (bỏ ː).
        vocab = {0: "<pad>", 1: "iː"}
        post = _posteriors(vocab, 20, "t-length", cells=[(5, 1, 0.6)])
        stats = post.evidence_stats("i", 0.06, 0.14)
        assert stats is not None and abs(stats.max_mass - 0.6) < 1e-6


# ── Cửa sổ refinement + guard (gọi trực tiếp _attach_deletion_evidence) ─────

class TestWindowRefinement:
    VOCAB = {0: "<pad>", 1: "l", 2: "k", 3: "s"}

    def _run(self, segments, path, seg_times, posteriors, word_windows=None):
        reference = ["k", "l", "s"]
        spans = [WordSpan("kls", 0, 3)]
        result = {
            0: (PhonemePoint(symbol="k", status="ok"), 0.0),
            1: (PhonemePoint(symbol="l", status="del", severity="high"), 0.9),
            2: (PhonemePoint(symbol="s", status="ok"), 0.0),
        }
        _attach_deletion_evidence(
            result, reference, spans, [False] * 3, path, segments,
            seg_times, word_windows, posteriors,
        )
        return result

    def test_refinement_narrows_to_between_neighbors(self):
        # k: 0.10-0.20, s: 0.30-0.40 → cửa sổ /l/ tinh chỉnh = [0.20, 0.30].
        # Mass /l/ đặt tại 0.36s: TRONG cửa sổ nền [0.10,0.40] nhưng NGOÀI cửa sổ
        # tinh chỉnh (kể cả margin) → max_mass = 0 chứng minh refinement có chạy.
        segments = _segs([("k", 0.10, 0.20), ("s", 0.30, 0.40)])
        path = [(0, 0), (-1, 1), (1, 2)]
        post = _posteriors(self.VOCAB, 50, "t-refine", cells=[(18, 1, 0.5)])
        result = self._run(segments, path, {0: (0.10, 0.40)}, post)
        point, pen = result[1]
        assert pen == 0.9  # penalty KHÔNG đổi (shadow)
        assert point.evidence_source == "wav2vec_window"
        assert point.evidence_version == EVIDENCE_VERSION
        assert point.evidence.max_mass == 0.0

    def test_inverted_refined_window_falls_back_to_base(self):
        # Segment hai bên "đảo" (prev.end=0.40 > next.start=0.10) → lo >= hi →
        # GUARD: fallback cửa sổ nền [0.10, 0.40]; mass tại 0.24s phải bắt được.
        segments = _segs([("k", 0.30, 0.40), ("s", 0.10, 0.20)])
        path = [(0, 0), (-1, 1), (1, 2)]
        post = _posteriors(self.VOCAB, 50, "t-guard", cells=[(12, 1, 0.5)])
        result = self._run(segments, path, {0: (0.10, 0.40)}, post)
        point, _pen = result[1]
        assert point.evidence is not None
        assert abs(point.evidence.max_mass - 0.5) < 1e-6

    def test_whisper_fallback_when_no_wav2vec_window(self):
        # Từ toàn deletion → seg_times rỗng → dùng Whisper window.
        post = _posteriors(self.VOCAB, 50, "t-whisper", cells=[(12, 1, 0.5)])
        result = self._run([], [], {}, post, word_windows={0: (0.20, 0.30)})
        point, _pen = result[1]
        assert point.evidence_source == "whisper_window"
        assert abs(point.evidence.max_mass - 0.5) < 1e-6

    def test_no_window_at_all_gives_source_none(self):
        post = _posteriors(self.VOCAB, 50, "t-nowin")
        result = self._run([], [], {}, post)
        point, _pen = result[1]
        assert point.evidence_source == "none"
        assert point.evidence is None
        assert point.evidence_version == EVIDENCE_VERSION


# ── Shadow bất biến qua compute_phoneme_score ────────────────────────────────

class TestShadowInvariance:
    VOCAB = {0: "<pad>", 1: "l", 2: "k", 3: "oʊ", 4: "s"}

    def _score(self, posteriors):
        # "close" /k l oʊ s/ đọc thiếu /l/ (case kinh điển của sprint).
        segments = _segs([("k", 0.10, 0.20), ("oʊ", 0.30, 0.40), ("s", 0.40, 0.50)])
        return compute_phoneme_score(
            segments, ["k", "l", "oʊ", "s"],
            reference_spans=[WordSpan("close", 0, 4)],
            posteriors=posteriors,
        )

    def test_score_identical_with_and_without_probe(self):
        post = _posteriors(self.VOCAB, 50, "t-shadow", cells=[(12, 1, 0.4)])
        with_probe = self._score(post)
        without = self._score(None)
        assert with_probe.overall_accuracy == without.overall_accuracy
        assert with_probe.adjusted_penalty == without.adjusted_penalty
        assert with_probe.raw_penalty == without.raw_penalty
        assert with_probe.deletion_count == without.deletion_count == 1
        assert [e.to_dict() for e in with_probe.errors] == [
            e.to_dict() for e in without.errors
        ]
        # Cùng severity/status từng âm.
        pw = with_probe.words[0].phonemes
        po = without.words[0].phonemes
        assert [(p.status, p.severity) for p in pw] == [
            (p.status, p.severity) for p in po
        ]

    def test_del_point_carries_evidence_others_do_not(self):
        # /l/ hallucinated-deletion: mass 0.4 giữa /k/ và /oʊ/ (0.24s) → probe thấy.
        post = _posteriors(self.VOCAB, 50, "t-carry", cells=[(12, 1, 0.4)])
        score = self._score(post)
        points = score.words[0].phonemes
        del_point = points[1]
        assert del_point.status == "del"
        assert del_point.evidence_source == "wav2vec_window"
        assert abs(del_point.evidence.max_mass - 0.4) < 1e-6
        # Point ok không có evidence; payload to_dict không thêm key.
        assert points[0].evidence_source is None
        assert "evidence" not in points[0].to_dict()
        assert "evidence" in del_point.to_dict()

    def test_probe_off_payload_unchanged(self):
        # posteriors=None → to_dict từng point KHÔNG có key evidence (byte-for-byte cũ).
        score = self._score(None)
        for p in score.words[0].phonemes:
            d = p.to_dict()
            assert "evidence" not in d
            assert "evidence_source" not in d

    def test_deterministic_across_runs(self):
        post_a = _posteriors(self.VOCAB, 50, "t-deter-a", cells=[(12, 1, 0.4)])
        post_b = _posteriors(self.VOCAB, 50, "t-deter-b", cells=[(12, 1, 0.4)])
        a = self._score(post_a).words[0].phonemes[1].evidence
        b = self._score(post_b).words[0].phonemes[1].evidence
        assert a == b

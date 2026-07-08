"""Tests cho boundary refinement (TOEIC_PHONEME_BOUNDARY_REFINE — fix bleed biên từ).

Case flagship "our eyes" (diag test-1 câu 8, 2026-07-08): pred `aʊ aɪ z z` vs ref
our=`aʊ r` eyes=`aɪ z` → DTW gán pred aɪ vào slot /r/ của "our" (path bleed và path
đúng HOÀ cost DTW thô 2.0 = 2.0, bleed thắng vì tie-break diagonal) → "eyes" hiển thị
/z z/ + sub aɪ→z oan. Refinement re-pair aɪ về "eyes" theo thang SCORING cost —
test flagship pin luôn cả vụ tie (đổi tiêu chí về cost DTW thô sẽ làm test này đỏ).
"""

from src.phoneme.l1_vietnamese import PenaltyReason
from src.phoneme.models import PhonemeSegment, WordSpan
from src.phoneme.scoring import compute_phoneme_score
from src.phoneme.scoring.alignment import _refine_boundary_bleed


def _segs(phs, conf=0.9):
    return [
        PhonemeSegment(phoneme=p, start=float(i), end=float(i + 1), confidence=conf)
        for i, p in enumerate(phs)
    ]


# Flagship "our eyes": ref [aʊ r | aɪ z], pred [aʊ aɪ z z].
REF = ["aʊ", "r", "aɪ", "z"]
SPANS = [WordSpan("our", 0, 2), WordSpan("eyes", 2, 4)]
PRED = ["aʊ", "aɪ", "z", "z"]


def _refine_direct(path, predicted, reference, spans, *, r_droppable=None,
                   skipped=None, accent=True, **kw):
    """Gọi thẳng _refine_boundary_bleed với metadata tối giản (test unit-level)."""
    n = len(reference)
    ref_word: list[str | None] = [None] * n
    ref_is_onset = [False] * n
    ref_is_coda = [False] * n
    for s in spans:
        for i in range(s.start_idx, min(s.end_idx, n)):
            ref_word[i] = s.word
        if s.start_idx < n:
            ref_is_onset[s.start_idx] = True
        if 0 <= min(s.end_idx, n) - 1 < n:
            ref_is_coda[min(s.end_idx, n) - 1] = True
    return _refine_boundary_bleed(
        path, predicted, reference, spans,
        ref_word, ref_is_onset, ref_is_coda,
        [None] * n,                      # ref_stress
        [False] * n,                     # ref_reducible
        skipped or [False] * n,          # ref_skipped
        r_droppable or [False] * n,      # ref_r_droppable
        [False] * n,                     # ref_g2p_uncertain
        accept_accent_variants=accent,
        **kw,
    )


class TestFlagshipOurEyes:
    """End-to-end qua compute_phoneme_score — hành vi flag ON/OFF."""

    def test_flag_on_fixes_bleed(self):
        score = compute_phoneme_score(
            _segs(PRED), REF, SPANS,
            accept_accent_variants=True, boundary_refine_enabled=True,
        )
        eyes = score.words[1]
        assert [p.status for p in eyes.phonemes] == ["ok", "ok"]
        assert [p.symbol for p in eyes.phonemes] == ["aɪ", "z"]
        our = score.words[0]
        # /r/ bỏ trống → deletion được accent-accept (non-rhotic), không lỗi.
        assert our.phonemes[0].status == "ok"
        assert our.phonemes[1].status == "ok"
        assert our.phonemes[1].penalty_reason == PenaltyReason.ACCENT_VARIANT.value
        assert score.overall_accuracy == 1.0

    def test_flag_off_keeps_old_behavior(self):
        # Pin hành vi cũ: bleed giữ nguyên — "eyes" có sub aɪ→z (heard=z).
        score = compute_phoneme_score(
            _segs(PRED), REF, SPANS, accept_accent_variants=True,
        )
        eyes = score.words[1]
        assert eyes.phonemes[0].status == "sub"
        assert eyes.phonemes[0].heard == "z"
        assert score.overall_accuracy < 1.0

    def test_default_param_is_off(self):
        a = compute_phoneme_score(_segs(PRED), REF, SPANS,
                                  accept_accent_variants=True)
        b = compute_phoneme_score(_segs(PRED), REF, SPANS,
                                  accept_accent_variants=True,
                                  boundary_refine_enabled=False)
        assert a.overall_accuracy == b.overall_accuracy
        assert [p.status for w in a.words for p in w.phonemes] == \
               [p.status for w in b.words for p in w.phonemes]

    def test_fires_without_accent_variants_too(self):
        # Accent gb/us: trước = 2 sub thô (2.0), sau = deletion /r/ thật (<2.0)
        # → vẫn move; "eyes" sạch, "our" còn deletion /r/ (đúng bản chất rhotic).
        score = compute_phoneme_score(
            _segs(PRED), REF, SPANS,
            accept_accent_variants=False, boundary_refine_enabled=True,
        )
        eyes = score.words[1]
        assert [p.status for p in eyes.phonemes] == ["ok", "ok"]
        assert score.words[0].phonemes[1].status == "del"


class TestRefineUnit:
    """Unit-level trên path hand-built — mechanics + guards."""

    BLEED_PATH = [(0, 0), (1, 1), (2, 2), (3, 3)]  # aʊ→aʊ, aɪ→r, z→aɪ, z→z

    def test_flagship_move(self):
        new_path, moves = _refine_direct(
            self.BLEED_PATH, PRED, REF, SPANS,
            r_droppable=[False, True, False, False],
        )
        assert len(moves) == 1
        mv = moves[0]
        assert (mv["from_ref"], mv["to_ref"], mv["pred_idx"]) == (1, 2, 1)
        assert mv["displaced_pred_idx"] == 2
        # Path mới: aɪ re-pair vào ref2, pred z cũ của ref2 thành insertion,
        # slot /r/ (ref1) rơi khỏi path.
        assert (1, 2) in new_path and (2, -1) in new_path
        assert all(r != 1 for _, r in new_path)

    def test_noop_on_correct_alignment(self):
        # Đọc rhotic chuẩn: pred khớp 1-1 → không move, trả path gốc nguyên vẹn.
        path = [(0, 0), (1, 1), (2, 2), (3, 3)]
        new_path, moves = _refine_direct(
            path, ["aʊ", "r", "aɪ", "z"], REF, SPANS,
            r_droppable=[False, True, False, False],
        )
        assert moves == []
        assert new_path is path

    def test_mirror_direction_b(self):
        # Mirror flagship: từ PHẢI ăn coda... nguồn ở slot đầu từ phải, đích ở
        # slot cuối từ trái. ref [z aɪ | r aʊ], pred [z z aɪ aʊ], bleed: aɪ→r.
        ref = ["z", "aɪ", "r", "aʊ"]
        spans = [WordSpan("eyes_rev", 0, 2), WordSpan("our_rev", 2, 4)]
        pred = ["z", "z", "aɪ", "aʊ"]
        path = [(0, 0), (1, 1), (2, 2), (3, 3)]  # z→z, z→aɪ(sub), aɪ→r, aʊ→aʊ
        new_path, moves = _refine_direct(
            path, pred, ref, spans, r_droppable=[False, False, True, False],
        )
        assert len(moves) == 1
        assert (moves[0]["from_ref"], moves[0]["to_ref"]) == (2, 1)
        assert (2, 1) in new_path and (1, -1) in new_path

    def test_refuse_dest_not_matching(self):
        # Đích không phonemes_match với segment → guard cứng chặn dù đích mismatched.
        ref = ["aʊ", "r", "m", "z"]  # "eyes" giả có onset /m/ — aɪ không khớp
        spans = [WordSpan("our", 0, 2), WordSpan("mz", 2, 4)]
        _, moves = _refine_direct(
            self.BLEED_PATH, PRED, ref, spans,
            r_droppable=[False, True, False, False],
        )
        assert moves == []

    def test_refuse_on_cost_tie(self):
        # Geminate nguyên âm ("see eat"): nguồn ok, vacate = deletion cùng loại
        # → after == before, KHÔNG giảm chặt → refuse (không move lung tung).
        ref = ["s", "iː", "iː", "t"]
        spans = [WordSpan("see", 0, 2), WordSpan("eat", 2, 4)]
        pred = ["s", "iː", "t"]
        path = [(0, 0), (1, 1), (-1, 2), (2, 3)]
        new_path, moves = _refine_direct(path, pred, ref, spans)
        assert moves == []
        assert new_path is path

    def test_window_veto_blocks_move(self):
        # Từ đích có cửa sổ Whisper KHÔNG chứa segment → veto.
        _, moves = _refine_direct(
            self.BLEED_PATH, PRED, REF, SPANS,
            r_droppable=[False, True, False, False],
            predicted_times=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
            word_windows={0: (0.0, 2.0), 1: (5.0, 6.0)},  # eyes ở 5-6s, aɪ ở 1-2s
        )
        assert moves == []

    def test_window_veto_skipped_when_locked(self):
        # Từ đích locked (cửa sổ sub-token không đáng tin) → bỏ veto, cost tự quyết.
        _, moves = _refine_direct(
            self.BLEED_PATH, PRED, REF, SPANS,
            r_droppable=[False, True, False, False],
            predicted_times=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
            word_windows={0: (0.0, 2.0), 1: (5.0, 6.0)},
            word_windows_locked={1},
        )
        assert len(moves) == 1

    def test_window_ok_allows_move(self):
        _, moves = _refine_direct(
            self.BLEED_PATH, PRED, REF, SPANS,
            r_droppable=[False, True, False, False],
            predicted_times=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
            word_windows={0: (0.0, 1.5), 1: (1.2, 4.0)},  # aɪ nằm trong window eyes
        )
        assert len(moves) == 1
        assert moves[0]["window_ok"] is True

    def test_skipped_word_boundary_untouched(self):
        _, moves = _refine_direct(
            self.BLEED_PATH, PRED, REF, SPANS,
            r_droppable=[False, True, False, False],
            skipped=[True, True, False, False],  # "our" bị Reliability skip
        )
        assert moves == []

    def test_deterministic(self):
        r1, m1 = _refine_direct(self.BLEED_PATH, PRED, REF, SPANS,
                                r_droppable=[False, True, False, False])
        r2, m2 = _refine_direct(self.BLEED_PATH, PRED, REF, SPANS,
                                r_droppable=[False, True, False, False])
        assert r1 == r2 and m1 == m2

    def test_moved_pred_frozen_across_boundaries(self):
        # 3 từ: pred aɪ move từ /r/ của "our" sang "eye"; biên kế tiếp không
        # được move lại chính segment đó (frozen) → đúng 1 move tổng.
        ref = ["aʊ", "r", "aɪ", "aɪ"]
        spans = [WordSpan("our", 0, 2), WordSpan("eye", 2, 3), WordSpan("i", 3, 4)]
        pred = ["aʊ", "aɪ"]
        path = [(0, 0), (1, 1), (-1, 2), (-1, 3)]
        new_path, moves = _refine_direct(
            path, pred, ref, spans, r_droppable=[False, True, False, False],
        )
        assert len(moves) == 1
        assert (1, 2) in new_path
        assert sum(1 for p, r in new_path if p == 1) == 1

"""Tests cho Recognizer-Collapse Gate (mở rộng coverage gate cho collapse TỪNG PHẦN).

Bối cảnh (case "line to" 2026-07-14): wav2vec CTC nhả token BLANK (<pad>) đè lên âm
VẪN CÓ trong audio → 1-2 âm giữa từ thành del/sub dù từ đọc rõ (Whisper tự tin). Gate
cap các âm này về COVERAGE_COLLAPSE khi:
  - âm THAM CHIẾU (KHÔNG phải onset) có max_mass posterior ≥ floor (được nói), VÀ
  - argmax tại frame mass cao nhất là token SILENCE/blank (argmax_is_silence).
Hai điều kiện tách CTC-collapse khỏi (a) nghe ra âm KHÁC = lỗi thật và (b) âm vắng thật.

Guard chính (không được giấu lỗi thật): (1) mass cao nhưng argmax là 1 IPA khác → KHÔNG
cap; (2) phụ âm ONSET (v→b, r→t kiểu L1) → KHÔNG cap (lỗi có giá trị sư phạm nhất).
"""

import numpy as np

from src.phoneme.l1_vietnamese import PenaltyReason
from src.phoneme.models import PhonemePoint, PhonemeSegment, WordSpan
from src.phoneme.scoring import compute_phoneme_score
from src.phoneme.scoring.word_details import _apply_recognizer_collapse_gate
from src.phoneme.wav2vec_backend import FramePosteriors

FRAME = 0.02


def _posteriors(id_to_token, n_frames, model_id, cells=None):
    """FramePosteriors tổng hợp: nền <pad>(id 0)≈1.0, ghi đè từng ô (frame, token_id, prob)."""
    vocab = len(id_to_token)
    probs = np.zeros((n_frames, vocab), dtype=np.float32)
    probs[:, 0] = 1.0
    for frame, token_id, p in cells or []:
        probs[frame, token_id] = p
        probs[frame, 0] = max(0.0, 1.0 - p)
    return FramePosteriors(
        probs=probs, frame_duration=FRAME, id_to_token=id_to_token, model_id=model_id,
    )


# Vocab: 0=<pad> (blank), 1=l, 2=s, 3=t, 4=k
VOCAB = {0: "<pad>", 1: "l", 2: "s", 3: "t", 4: "k"}


def _kls_case(point, penalty, cells, model_id, *, prob=0.99, window=(0.10, 0.40),
              mass_floor=0.10, onset=(False, False, False)):
    """Chạy gate trên "kls" với target (index 1) = `point`; trả (point, penalty) sau.

    `onset` = ref_is_onset của 3 âm (mặc định target KHÔNG phải onset → gate được xét).
    """
    reference = ["k", "l", "s"]
    spans = [WordSpan("kls", 0, 3)]
    result = {
        0: (PhonemePoint(symbol="k", status="ok"), 0.0),
        1: (point, penalty),
        2: (PhonemePoint(symbol="s", status="ok"), 0.0),
    }
    post = _posteriors(VOCAB, 40, model_id, cells=cells)
    _apply_recognizer_collapse_gate(
        result, {1: penalty}, reference, spans, [False] * 3, list(onset),
        {0: window}, {0: prob}, post, mass_floor=mass_floor,
    )
    return result[1]


def _del(sev="high"):
    return PhonemePoint(symbol="l", status="del", severity=sev,
                        penalty_reason=PenaltyReason.HARD_ERROR.value)


def _sub(sev="high"):
    return PhonemePoint(symbol="l", status="sub", heard="t", severity=sev,
                        penalty_reason=PenaltyReason.HARD_ERROR.value)


# ── Bắt đúng blank-collapse ──────────────────────────────────────────────────

def test_del_blank_collapse_capped():
    # /l/ mass 0.4 tại 0.24s nhưng <pad> thắng (0.6) → blank-collapse → cap về 0.2 low.
    point, pen = _kls_case(_del(), 0.9, [(12, 1, 0.4)], "del-collapse")
    assert pen == 0.2
    assert point.status == "del"
    assert point.severity == "low"
    assert point.penalty_reason == PenaltyReason.COVERAGE_COLLAPSE.value


def test_sub_blank_collapse_capped_preserves_heard():
    # Case "line to": /n/→/t/ sub do DTW mượn âm drift, nhưng /l/(giả /n/) có mass +
    # argmax blank → cap, GIỮ status sub + heard.
    point, pen = _kls_case(_sub(), 0.9, [(12, 1, 0.4)], "sub-collapse")
    assert pen == 0.2
    assert point.status == "sub"
    assert point.heard == "t"
    assert point.severity == "low"
    assert point.penalty_reason == PenaltyReason.COVERAGE_COLLAPSE.value


# ── KHÔNG giấu lỗi thật ───────────────────────────────────────────────────────

def test_onset_consonant_protected():
    # /l/ ĐẦU TỪ (onset) dù có mass + argmax blank → KHÔNG cap: lỗi onset (v→b, r→t
    # kiểu L1) là lỗi giá trị nhất, không được giấu (bench "very" v→b 2026-07-14).
    point, pen = _kls_case(_sub(), 0.9, [(12, 1, 0.4)], "onset",
                           onset=(False, True, False))
    assert pen == 0.9
    assert point.penalty_reason == PenaltyReason.HARD_ERROR.value


def test_true_deletion_not_capped():
    # Không ô nào có /l/ → âm vắng thật (mass 0) → giữ nguyên penalty đầy đủ.
    point, pen = _kls_case(_del(), 0.9, [], "true-del")
    assert pen == 0.9
    assert point.penalty_reason == PenaltyReason.HARD_ERROR.value


def test_real_substitution_high_mass_but_argmax_real_phoneme_not_capped():
    # /l/ có mass 0.35 (≥ floor) NHƯNG argmax tại frame đó là /s/ (âm THẬT, không blank)
    # → người đọc nói /s/ thay /l/ = lỗi thật → KHÔNG cap.
    reference = ["k", "l", "s"]
    spans = [WordSpan("kls", 0, 3)]
    result = {
        0: (PhonemePoint(symbol="k", status="ok"), 0.0),
        1: (_sub(), 0.9),
        2: (PhonemePoint(symbol="s", status="ok"), 0.0),
    }
    post = _posteriors(VOCAB, 40, "real-sub", cells=[(12, 1, 0.35)])
    post.probs[12, 2] = 0.5   # /s/ thắng argmax
    post.probs[12, 0] = 0.15  # <pad> không còn thắng
    _apply_recognizer_collapse_gate(
        result, {1: 0.9}, reference, spans, [False] * 3, [False] * 3,
        {0: (0.10, 0.40)}, {0: 0.99}, post,
    )
    point, pen = result[1]
    assert pen == 0.9
    assert point.penalty_reason == PenaltyReason.HARD_ERROR.value


def test_mass_below_floor_not_capped():
    # /l/ mass 0.05 < floor 0.10 (dù argmax blank) → bằng chứng "được nói" yếu → giữ.
    _point, pen = _kls_case(_del(), 0.9, [(12, 1, 0.05)], "low-mass")
    assert pen == 0.9


def test_low_asr_prob_not_capped():
    # Whisper không tự tin (prob 0.3 < 0.6) → transcript không đủ tin → không cap.
    _point, pen = _kls_case(_del(), 0.9, [(12, 1, 0.4)], "low-prob", prob=0.3)
    assert pen == 0.9


def test_already_low_penalty_untouched():
    # penalty ≤ cap (đã bị gate khác hạ) → không đụng (precedence: chỉ hạ).
    p = PhonemePoint(symbol="l", status="del", severity="low",
                     penalty_reason=PenaltyReason.RECOGNIZER_NOISE.value)
    point, pen = _kls_case(p, 0.1, [(12, 1, 0.4)], "already-low")
    assert pen == 0.1
    assert point.penalty_reason == PenaltyReason.RECOGNIZER_NOISE.value


def test_ok_point_never_capped():
    # Âm "ok" không bao giờ bị gate đụng (chỉ del/sub penalty > cap).
    reference = ["k", "l", "s"]
    spans = [WordSpan("kls", 0, 3)]
    ok = PhonemePoint(symbol="l", status="ok")
    result = {1: (ok, 0.0)}
    post = _posteriors(VOCAB, 40, "ok-pt", cells=[(12, 1, 0.4)])
    _apply_recognizer_collapse_gate(
        result, {}, reference, spans, [False] * 3, [False] * 3,
        {0: (0.10, 0.40)}, {0: 0.99}, post,
    )
    assert result[1][0].status == "ok" and result[1][1] == 0.0


def test_no_posteriors_is_noop():
    reference = ["k", "l", "s"]
    spans = [WordSpan("kls", 0, 3)]
    result = {1: (_del(), 0.9)}
    _apply_recognizer_collapse_gate(
        result, {1: 0.9}, reference, spans, [False] * 3, [False] * 3,
        {0: (0.10, 0.40)}, {0: 0.99}, None,
    )
    assert result[1][1] == 0.9  # no-op


# ── Tích hợp qua compute_phoneme_score (flag on/off) ─────────────────────────

class TestIntegration:
    # "sun" /s ʌ n/: /n/ CODA (không phải onset) bị blank-collapse.
    VOCAB = {0: "<pad>", 1: "n", 2: "s", 3: "ʌ"}

    def _score(self, collapse_on):
        # /s/ onset ok, /ʌ/ nucleus ok, /n/ coda del + mass 0.4 @0.30s argmax <pad>.
        segments = [
            PhonemeSegment(phoneme="s", start=0.10, end=0.20, confidence=0.9),
            PhonemeSegment(phoneme="ʌ", start=0.20, end=0.30, confidence=0.9),
        ]
        post = _posteriors(self.VOCAB, 40, "integ", cells=[(15, 1, 0.4)])
        return compute_phoneme_score(
            segments, ["s", "ʌ", "n"],
            reference_spans=[WordSpan("sun", 0, 3)],
            word_windows={0: (0.10, 0.40)}, word_probs={0: 0.99},
            posteriors=post, collapse_gate_enabled=collapse_on,
        )

    def test_flag_off_leaves_del_high(self):
        score = self._score(False)
        npt = score.words[0].phonemes[2]
        assert npt.status == "del" and npt.severity != "low"
        assert score.coverage_collapse_count == 0

    def test_flag_on_caps_collapsed_coda_del(self):
        score = self._score(True)
        npt = score.words[0].phonemes[2]
        assert npt.status == "del"
        assert npt.severity == "low"
        assert npt.penalty_reason == PenaltyReason.COVERAGE_COLLAPSE.value
        assert score.coverage_collapse_count == 1
        assert self._score(True).overall_accuracy > self._score(False).overall_accuracy

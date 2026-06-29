"""Phoneme scoring — so sánh predicted phonemes với reference IPA sequence.

Dùng Dynamic Time Warping (DTW) để align 2 sequences (predicted vs reference),
từ đó detect substitution / deletion / insertion + tính overall accuracy.

Tổ chức (package):
  - constants.py: hằng số tinh chỉnh (ngưỡng confidence/noise, giới hạn payload)
  - word_details.py: metadata, skip, deletion scoring, per-word detail + timing
  - alignment.py: _dtw_align (DTW) + _align_points (path → point/penalty)
  - __init__.py (đây): compute_phoneme_score (orchestrator) + weighted_accuracy
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from ..diagnostics import WordDiagnostic, build_word_diagnostics
from ..l1_vietnamese import PenaltyReason
from ..models import (
    PhonemeError,
    PhonemeErrorType,
    PhonemePoint,
    PhonemeScore,
    PhonemeSegment,
    WordSpan,
)
from ..reliability import SkipDecision
from .alignment import _align_points, _dtw_align
from .constants import (
    MAX_ERRORS_RETURNED,
    MAX_WORDS_RETURNED,
    PHONEME_CONFIDENCE_KNEE,
    PHONEME_L1_MIN_CONFIDENCE,
    PHONEME_LOW_CONF_FLOOR,
    PHONEME_RECOGNIZER_NOISE_CONF,
    PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    PHONEME_RECOGNIZER_NOISE_SIM,
)
from .word_details import (
    _WORD_PLAY_LEAD,
    _WORD_PLAY_TRAIL,
    _build_word_details,
    _pad_and_clamp_windows,
    _ref_metadata,
    _resolve_skips,
    _score_deletion,
    _severity_from_penalty,
    _word_at,
    _word_segment_times,
)

logger = logging.getLogger("toeic.phoneme.scoring")


def compute_phoneme_score(
    segments: list[PhonemeSegment],
    reference_phonemes: list[str],
    reference_spans: list[WordSpan] | None = None,
    reference_stress: list[str | None] | None = None,
    reference_display_stress: list[str | None] | None = None,
    max_words: int = MAX_WORDS_RETURNED,
    skips: Mapping[int, SkipDecision] | None = None,
    confidence_knee: float = PHONEME_CONFIDENCE_KNEE,
    diagnostics_sink: Callable[[list[WordDiagnostic]], None] | None = None,
    word_windows: Mapping[int, tuple[float, float]] | None = None,
    l1_enabled: bool = False,
    l1_min_confidence: float = PHONEME_L1_MIN_CONFIDENCE,
    low_conf_floor: float = PHONEME_LOW_CONF_FLOOR,
    recognizer_noise_sim: float = PHONEME_RECOGNIZER_NOISE_SIM,
    recognizer_noise_conf: float = PHONEME_RECOGNIZER_NOISE_CONF,
    recognizer_noise_conf_vowel: float = PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    accept_accent_variants: bool = False,
) -> PhonemeScore | None:
    """Tính phoneme accuracy score từ predicted segments + reference.

    Pipeline de-noise: normalize tối thiểu → phoneme_similarity (liên tục) →
    phonemes_match (allophone/reduction) → confidence weighting → điểm có trọng số.

    Args:
        reference_spans: map reference index → từ (WordSpan) để gắn `word` cho lỗi
            và xác định onset; None thì lỗi giữ word=None (tương thích ngược).
        reference_stress: nhấn âm song song 1-1 với reference_phonemes (TRÊN nguyên âm —
            scoring đọc để xác định nhân chính/severity).
        reference_display_stress: nhấn âm ĐÃ dời về onset (chỉ HIỂN THỊ) song song 1-1 với
            reference_phonemes. CHỈ gắn vào PhonemePoint.display_stress cho UI — KHÔNG tham gia
            alignment/severity/điểm. None = không có (PhonemePoint.display_stress = None).
        skips: quyết định bỏ qua từ Recognition Reliability (tầng TRÊN), keyed theo
            CHỈ SỐ SPAN chuẩn (vị trí trong reference_spans). Scorer CHỈ tiêu thụ —
            KHÔNG tự quyết định reliability (không suy từ match-ratio/similarity/penalty).
            Từ bị skip: mọi âm thành "skipped", loại khỏi cả tử số lẫn mẫu số accuracy.
        confidence_knee: ngưỡng confidence để hạ penalty lỗi sub (xem PHONEME_CONFIDENCE_KNEE).
        diagnostics_sink: optional — nhận list[WordDiagnostic] để ghi telemetry (PR2).
            CHỈ để quan sát; KHÔNG ảnh hưởng điểm. None = không tính telemetry (zero overhead).
        word_windows: optional — cửa sổ thời gian Whisper theo CHỈ SỐ TỪ chuẩn (khớp
            reference_spans). Dùng cho (a) telemetry drift-vs-hallucination (PR3-0) và
            (b) FALLBACK timestamp phát lại từng từ cho từ toàn deletion (wav2vec không
            nghe ra segment). Nguồn CHÍNH cho start/end phát lại là wav2vec segment
            (_word_segment_times); Whisper chỉ bù chỗ thiếu. KHÔNG ảnh hưởng điểm.
            Telemetry drift vẫn cần thêm diagnostics_sink.
        l1_enabled: bật L1-aware layer (Vietnamese). False (mặc định) = hành vi y hệt trước
            (bit-for-bit). True = giảm penalty cho nuốt phụ âm cuối kiểu L1 + trung hoà sub
            confidence rất thấp; vẫn hiển thị (accent note). KHÔNG skip (Reliability mới được skip).
        l1_min_confidence: ngưỡng confidence để áp L1 *substitution* tolerance (chưa dùng ở v1).
        low_conf_floor: sub có confidence < ngưỡng → penalty trung hoà về 0 (chỉ khi l1_enabled).
        recognizer_noise_sim: ngưỡng similarity BẢO VỆ — sub có sim ≥ ngưỡng (hoặc nằm trong
            _REAL_ERROR_SUBS) KHÔNG bao giờ bị gate noise (xem PHONEME_RECOGNIZER_NOISE_SIM).
        recognizer_noise_conf / recognizer_noise_conf_vowel: ngưỡng confidence (phụ âm / nguyên âm)
            của recognizer-noise gate. Sub bất khả thi + conf dưới ngưỡng → recognizer hallucinate →
            penalty 0 + severity low. ĐỘC LẬP với l1_enabled. Đặt 0 để tắt gate (bit-for-bit như cũ).
        accept_accent_variants: chế độ accent "default" (caller map từ accent=="default"). Chấp nhận
            coda /r/ non-rhotic (Anh-Anh) — nuốt /r/ hoặc nguyên âm align lên /r/ → "ok" (ACCENT_VARIANT),
            KHÔNG trừ điểm. False (mặc định) = bit-for-bit như cũ. CHƯA phải union GB/US đầy đủ: các khác
            biệt còn lại đã được normalize_ipa() gộp; BATH (æ↔ɑ) cố ý không gộp (xem _score_deletion).

    Returns None nếu reference_phonemes rỗng.
    """
    if not reference_phonemes:
        return None

    predicted_phonemes = [s.phoneme for s in segments]
    predicted_conf = [s.confidence for s in segments]
    avg_confidence = (
        sum(predicted_conf) / len(predicted_conf) if predicted_conf else 0.0
    )
    ref_word, ref_is_onset, ref_stress, ref_reducible, ref_is_coda = _ref_metadata(
        reference_phonemes, reference_spans, reference_stress
    )
    # Nhấn-hiển-thị (dời về onset) song song reference — CHỈ để gắn PhonemePoint.display_stress.
    # Pad/cắt về đúng len(reference) như ref_stress; không tham gia bất kỳ tính toán điểm nào.
    n_ref = len(reference_phonemes)
    ref_display_stress: list[str | None] = (
        list(reference_display_stress) if reference_display_stress else [None] * n_ref
    )
    if len(ref_display_stress) < n_ref:
        ref_display_stress += [None] * (n_ref - len(ref_display_stress))
    # Map quyết định skip (theo chỉ số span chuẩn) → cờ per-phoneme + lý do per-span.
    ref_skipped, span_skip_reason = _resolve_skips(
        reference_phonemes, reference_spans, skips
    )

    # "Said nothing" (không có phoneme nào nhận diện được) → 0% (path rỗng).
    if not predicted_phonemes:
        path: list[tuple[int, int]] = []
        result: dict[int, tuple[PhonemePoint, float | None]] = {}
        insertion_count = 0
        raw_by_ref: dict[int, float] = {}
        empty_prediction = True
    else:
        path = _dtw_align(predicted_phonemes, reference_phonemes)
        result, insertion_count, raw_by_ref = _align_points(
            path, predicted_phonemes, predicted_conf, reference_phonemes,
            ref_word, ref_is_onset, ref_stress, ref_reducible, ref_skipped,
            ref_is_coda, confidence_knee, ref_display_stress,
            l1_enabled=l1_enabled, low_conf_floor=low_conf_floor,
            recognizer_noise_sim=recognizer_noise_sim,
            recognizer_noise_conf=recognizer_noise_conf,
            recognizer_noise_conf_vowel=recognizer_noise_conf_vowel,
            accept_accent_variants=accept_accent_variants,
        )
        empty_prediction = False

    # Bổ sung mọi reference index chưa được path chạm → deletion (hoặc skipped).
    for i in range(len(reference_phonemes)):
        if i in result:
            continue
        stress = ref_stress[i]
        disp = ref_display_stress[i]
        if ref_skipped[i]:
            result[i] = (PhonemePoint(symbol=reference_phonemes[i], status="skipped",
                                      stress=stress, display_stress=disp), None)
        else:
            point, pen, raw = _score_deletion(
                reference_phonemes[i], is_onset=ref_is_onset[i],
                is_coda=ref_is_coda[i], stress=stress, display_stress=disp,
                l1_enabled=l1_enabled,
                accept_accent_variants=accept_accent_variants,
            )
            result[i] = (point, pen)
            raw_by_ref[i] = raw

    # Tổng hợp: errors + counts + penalty (một nguồn → không lệch nhau).
    errors: list[PhonemeError] = []
    matches = substitutions = deletions = 0
    total_penalty = 0.0
    raw_penalty_sum = 0.0
    l1_adjusted_count = low_conf_neutralized_count = recognizer_noise_count = 0
    scored = 0
    point_by_ref: dict[int, PhonemePoint] = {}
    for i in range(len(reference_phonemes)):
        point, penalty = result[i]
        point_by_ref[i] = point
        if point.status == "ok":
            matches += 1
        elif point.status == "sub":
            substitutions += 1
            errors.append(PhonemeError(
                error_type=PhonemeErrorType.SUBSTITUTION,
                expected=reference_phonemes[i], predicted=point.heard,
                position=i, severity=point.severity or "medium", word=ref_word[i],
            ))
        elif point.status == "del":
            deletions += 1
            errors.append(PhonemeError(
                error_type=PhonemeErrorType.DELETION,
                expected=reference_phonemes[i], predicted=None,
                position=i, severity=point.severity or "medium", word=ref_word[i],
            ))
        # skipped → không phải lỗi
        if penalty is not None:
            total_penalty += penalty
            scored += 1
            # L1 metadata: penalty GỐC (trước L1/neutralization) để tính tỉ lệ giảm.
            raw_penalty_sum += raw_by_ref.get(i, penalty)
        if point.penalty_reason == PenaltyReason.L1_FINAL_DELETION.value:
            l1_adjusted_count += 1
        elif point.penalty_reason == PenaltyReason.LOW_CONFIDENCE_NEUTRALIZED.value:
            low_conf_neutralized_count += 1
        elif point.penalty_reason == PenaltyReason.RECOGNIZER_NOISE.value:
            recognizer_noise_count += 1

    if empty_prediction:
        accuracy = 0.0
    elif scored > 0:
        accuracy = max(0.0, 1.0 - total_penalty / scored)
    else:
        accuracy = 1.0  # mọi âm đều skip (ASR nghe nhầm cả) → không có gì để chấm

    # Cửa sổ thời gian phát lại từng từ: ưu tiên wav2vec segment (chính xác ~20ms, đúng
    # "cái wav2vec nghe"), fallback Whisper window cho từ toàn deletion. word_windows
    # (Whisper) VẪN giữ riêng cho telemetry drift bên dưới — KHÔNG trộn vào đây.
    seg_times = _word_segment_times(path, segments, reference_spans)
    if word_windows:
        playback_times: dict[int, tuple[float, float]] = {**word_windows, **seg_times}
    else:
        playback_times = seg_times
    # Đệm + clamp theo từ liền kề → cửa sổ phát lại không lẹm sang từ khác (frontend phát
    # verbatim). Backend là nơi DUY NHẤT biết ranh giới từ kề nên đệm phải nằm ở đây.
    playback_times = _pad_and_clamp_windows(playback_times)
    words, words_truncated, words_total = _build_word_details(
        point_by_ref, reference_phonemes, reference_spans, max_words, span_skip_reason,
        word_times=playback_times,
    )

    # Telemetry (PR2/PR3-0) — DIAGNOSTIC ONLY, chỉ tính khi có sink (zero overhead khi tắt).
    if diagnostics_sink is not None:
        predicted_times = [(s.start, s.end) for s in segments]
        diagnostics_sink(build_word_diagnostics(
            path, predicted_phonemes, predicted_conf, reference_phonemes,
            reference_spans, result, span_skip_reason,
            predicted_times=predicted_times, word_windows=word_windows,
        ))

    # Sort errors by severity (high → medium → low) rồi cap.
    severity_order = {"high": 0, "medium": 1, "low": 2}
    errors.sort(key=lambda e: severity_order.get(e.severity, 3))

    l1_adjustment_ratio = (
        (raw_penalty_sum - total_penalty) / raw_penalty_sum
        if raw_penalty_sum > 0 else 0.0
    )

    logger.info(
        "Phoneme score: accuracy=%.2f | matches=%d | subs=%d | del=%d | ins=%d | "
        "skipped_words=%d | ref=%d | pred=%d | l1_adj=%d | low_conf_neut=%d | recog_noise=%d",
        accuracy, matches, substitutions, deletions, insertion_count,
        len(span_skip_reason), len(reference_phonemes), len(predicted_phonemes),
        l1_adjusted_count, low_conf_neutralized_count, recognizer_noise_count,
    )

    return PhonemeScore(
        overall_accuracy=round(accuracy, 4),
        substitution_count=substitutions,
        deletion_count=deletions,
        insertion_count=insertion_count,
        reference_count=len(reference_phonemes),
        predicted_count=len(predicted_phonemes),
        avg_confidence=round(avg_confidence, 4),
        errors=errors[:MAX_ERRORS_RETURNED],
        words=words,
        words_truncated=words_truncated,
        words_total=words_total,
        raw_penalty=raw_penalty_sum,
        adjusted_penalty=total_penalty,
        l1_adjusted_count=l1_adjusted_count,
        low_conf_neutralized_count=low_conf_neutralized_count,
        recognizer_noise_count=recognizer_noise_count,
        l1_adjustment_ratio=l1_adjustment_ratio,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: weighted accuracy (có tính severity)
# ──────────────────────────────────────────────────────────────────────────────

def weighted_accuracy(score: PhonemeScore) -> float:
    """Tính weighted accuracy: mỗi lỗi penalty theo severity.

    - high severity: penalty = 1.0
    - medium severity: penalty = 0.5
    - low severity: penalty = 0.25

    Formula: weighted = 1.0 - (weighted_penalty_sum / reference_count)
    """
    if score.reference_count == 0:
        return 0.0

    penalty_weights = {"high": 1.0, "medium": 0.5, "low": 0.25}
    total_penalty = sum(
        penalty_weights.get(e.severity, 0.5)
        for e in score.errors
    )
    weighted = 1.0 - (total_penalty / score.reference_count)
    return round(max(0.0, min(1.0, weighted)), 4)


__all__ = [
    # orchestrator + convenience
    "compute_phoneme_score",
    "weighted_accuracy",
    # tuning constants (analyzer import qua facade)
    "MAX_ERRORS_RETURNED",
    "MAX_WORDS_RETURNED",
    "PHONEME_CONFIDENCE_KNEE",
    "PHONEME_L1_MIN_CONFIDENCE",
    "PHONEME_LOW_CONF_FLOOR",
    "PHONEME_RECOGNIZER_NOISE_SIM",
    "PHONEME_RECOGNIZER_NOISE_CONF",
    "PHONEME_RECOGNIZER_NOISE_CONF_VOWEL",
    # helpers tham chiếu bởi tests
    "_word_at",
    "_ref_metadata",
    "_WORD_PLAY_LEAD",
    "_WORD_PLAY_TRAIL",
]

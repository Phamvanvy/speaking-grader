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
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING

from ..diagnostics import (
    DRIFT_WINDOW_PAD_SEC,
    WordDiagnostic,
    build_word_diagnostics,
)
from ..ipa import is_elidable_stop, is_vowel, normalize_ipa
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
from .alignment import (
    _align_points,
    _apply_drift_cap,
    _dtw_align,
    _refine_boundary_bleed,
)
from .homograph import select_homograph_references
from .constants import (
    MAX_ERRORS_RETURNED,
    MAX_WORDS_RETURNED,
    PHONEME_CONFIDENCE_KNEE,
    PHONEME_COVERAGE_GATE_CAP,
    PHONEME_COVERAGE_GATE_MAX_LEN,
    PHONEME_COVERAGE_GATE_MIN_ASR_PROB,
    PHONEME_DRIFT_SUB_CAP,
    PHONEME_L1_MIN_CONFIDENCE,
    PHONEME_LOW_CONF_FLOOR,
    PHONEME_RECOGNIZER_NOISE_CONF,
    PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    PHONEME_RECOGNIZER_NOISE_SIM,
)
from .word_details import (
    _WORD_PLAY_LEAD,
    _WORD_PLAY_TRAIL,
    _apply_coverage_gate,
    _attach_deletion_evidence,
    _build_word_details,
    _merge_playback_windows,
    _pad_and_clamp_windows,
    _reanchor_locked_windows,
    _ref_metadata,
    _resolve_skips,
    _score_deletion,
    _severity_from_penalty,
    _word_at,
    _word_segment_counts,
    _word_segment_times,
)

if TYPE_CHECKING:
    from ..wav2vec_backend import FramePosteriors

logger = logging.getLogger("toeic.phoneme.scoring")


def _connected_ok(point: PhonemePoint) -> PhonemePoint:
    """Bản sao point với status "ok" + tag connected_speech (giữ symbol/stress)."""
    return PhonemePoint(
        symbol=point.symbol, status="ok", stress=point.stress,
        display_stress=point.display_stress,
        penalty_reason=PenaltyReason.CONNECTED_SPEECH.value,
        penalty_adjustment=0.0,
    )


def _apply_connected_speech(
    result: dict[int, tuple[PhonemePoint, float | None]],
    raw_by_ref: dict[int, float],
    reference: list[str],
    spans: list[WordSpan],
    ref_skipped: list[bool],
) -> None:
    """Chấp nhận nuốt STOP cuối từ khi nối từ (connected speech) — sửa `result` in-place.

    Người bản xứ nuốt stop cuối từ trước phụ âm đầu từ kế ("test preparation" →
    /tes-prep/). DTW khi đó tạo lỗi ảo theo 2 dạng — xử lý cả hai trên CẶP TỪ KỀ (a, b),
    i = âm CUỐI của a, j = âm ĐẦU của b, điều kiện chung: reference[i] là stop elidable,
    reference[j] là PHỤ ÂM, cả hai không bị skip:

      - C1: result[i] là deletion → âm bị nuốt hợp lệ → flip "ok"/connected_speech.
      - C2: result[i] là sub mà âm NGHE ĐƯỢC chính là onset của từ kế (test /tesp/:
        t→p vì DTW gán /p/ của "preparation" lệch sang "test") → flip "ok"; và C3
        (CHỈ đi kèm C2): nếu result[j] là deletion (âm /p/ đã bị "mượn" mất) → trả
        lại "ok" cho onset từ kế luôn. C3 không bao giờ đứng một mình.

    Guard giữ lỗi thật của học viên: CHỈ âm cuối từ (nuốt giữa từ vẫn bắt); CHỈ stop
    (/s z l n/... cuối — lỗi L1 VN kinh điển — vẫn bắt); CHỈ khi từ kế mở đầu bằng
    phụ âm (trước nguyên âm phải nối âm, nuốt là lỗi → vẫn bắt).
    """
    n = len(reference)
    for a, b in zip(spans, spans[1:]):
        i = a.end_idx - 1
        j = b.start_idx
        if i < a.start_idx or i >= n or not (b.start_idx <= j < min(b.end_idx, n)):
            continue
        if ref_skipped[i] or i not in result or j not in result:
            continue
        if is_vowel(reference[j]) or not is_elidable_stop(reference[i]):
            continue
        point, _pen = result[i]
        if point.status == "del":
            result[i] = (_connected_ok(point), 0.0)
            raw_by_ref.pop(i, None)
        elif (
            point.status == "sub"
            and point.heard is not None
            and normalize_ipa(point.heard) == normalize_ipa(reference[j])
        ):
            result[i] = (_connected_ok(point), 0.0)
            raw_by_ref.pop(i, None)
            j_point, _jpen = result[j]
            if j_point.status == "del" and not ref_skipped[j]:
                result[j] = (_connected_ok(j_point), 0.0)
                raw_by_ref.pop(j, None)


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
    word_windows_locked: Collection[int] | None = None,
    l1_enabled: bool = False,
    l1_min_confidence: float = PHONEME_L1_MIN_CONFIDENCE,
    low_conf_floor: float = PHONEME_LOW_CONF_FLOOR,
    recognizer_noise_sim: float = PHONEME_RECOGNIZER_NOISE_SIM,
    recognizer_noise_conf: float = PHONEME_RECOGNIZER_NOISE_CONF,
    recognizer_noise_conf_vowel: float = PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    accept_accent_variants: bool = False,
    connected_speech_enabled: bool = True,
    word_probs: Mapping[int, float] | None = None,
    coverage_gate_enabled: bool = False,
    coverage_gate_cap: float = PHONEME_COVERAGE_GATE_CAP,
    coverage_gate_max_len: int = PHONEME_COVERAGE_GATE_MAX_LEN,
    coverage_gate_min_asr_prob: float = PHONEME_COVERAGE_GATE_MIN_ASR_PROB,
    drift_cap_enabled: bool = False,
    drift_sub_cap: float = PHONEME_DRIFT_SUB_CAP,
    drift_window_pad: float = DRIFT_WINDOW_PAD_SEC,
    posteriors: FramePosteriors | None = None,
    homograph_selection_enabled: bool = False,
    boundary_refine_enabled: bool = False,
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
            reference_spans). Dùng cho (a) telemetry drift-vs-hallucination (PR3-0),
            (b) FALLBACK timestamp phát lại từng từ cho từ toàn deletion (wav2vec không
            nghe ra segment), và (c) evidence cho coverage gate + drift cap KHI các flag
            tương ứng bật. Nguồn CHÍNH cho start/end phát lại là wav2vec segment
            (_word_segment_times); Whisper chỉ bù chỗ thiếu. Với flags mặc định (OFF),
            KHÔNG ảnh hưởng điểm — bật coverage_gate_enabled/drift_cap_enabled thì CÓ
            (thay đổi có chủ đích, xem 2 gate bên dưới). Telemetry drift vẫn cần thêm
            diagnostics_sink.
        word_windows_locked: optional — CHỈ SỐ TỪ có cửa sổ Whisper đã bị CẮT sub-token
            từ upstream (diagnostics.subtoken_window — token alphanumeric "9am" mà
            reference chỉ còn "am"). Với các từ này DTW attribution KHÔNG đáng tin (âm
            của phần số bị rơi không có trong reference, bị "hút" vào từ) → playback
            BỎ QUA bước siết theo seg_times, dùng nguyên cửa sổ đã cắt (đảm bảo chứa
            từ thật). CHỈ ảnh hưởng playback — scoring/telemetry không đổi.
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
        connected_speech_enabled: chấp nhận nuốt STOP cuối từ khi từ kế bắt đầu bằng phụ âm
            (elision bản xứ, vd "test preparation" → /tes-prep/) — xem _apply_connected_speech.
            False = hành vi cũ bit-for-bit.
        word_probs: optional — Whisper word probability theo CHỈ SỐ TỪ chuẩn (khớp
            reference_spans, cùng nguồn word_windows). CHỈ dùng làm guard cho coverage
            gate (không coi transcript là ground truth tuyệt đối).
        coverage_gate_enabled: bật coverage gate (Track A) — từ bị "del" 100% + wav2vec
            im lặng trong window + Whisper prob đủ cao → cap penalty về coverage_gate_cap
            (severity "low", COVERAGE_COLLAPSE). False (mặc định) = bit-for-bit như cũ.
            Xem _apply_coverage_gate (word_details.py).
        drift_cap_enabled: bật drift cap (Track B) — sub có predicted segment NGOÀI
            window Whisper của từ (±drift_window_pad) → cap penalty về drift_sub_cap
            (severity "low", DRIFT_SUSPECTED). False (mặc định) = bit-for-bit như cũ.
            Xem _apply_drift_cap (alignment.py). An toàn khi word_windows thiếu (no-op).
        posteriors: optional — FramePosteriors từ wav2vec (predict_with_posteriors) cho
            deletion-evidence probe (SHADOW): gắn EvidenceStats vào mỗi point "del" để
            telemetry/phân tích. KHÔNG BAO GIỜ ảnh hưởng điểm — chỉ thêm metadata.
            None = không probe (payload y hệt cũ). Xem _attach_deletion_evidence.
        homograph_selection_enabled: multi-reference homograph — với từ đa-entry
            CMUdict có cửa sổ Whisper, chọn LẠI entry khớp acoustic nhất (align
            segments trong window với từng entry, cost = 1 − phoneme_similarity)
            TRƯỚC khi DTW, thay vì entry ranking context-free (case "project"
            2026-07-05: luôn bị so với dạng động từ). Cần word_windows + spans.
            False (mặc định) = bit-for-bit như cũ. Xem scoring/homograph.py.
        boundary_refine_enabled: boundary refinement — sửa bleed cục bộ quanh ranh
            giới từ TRÊN DTW path (SAU _dtw_align, TRƯỚC _align_points): segment bị
            gán nhầm sang từ kề (case "our eyes" → "eyes" /z z/) được re-pair về đúng
            từ khi tổng scoring cost cục bộ giảm chặt + đích match thật. Cần
            reference_spans; word_windows (nếu có) làm time-veto phụ. False (mặc
            định) = bit-for-bit như cũ. Xem _refine_boundary_bleed (alignment.py).

    Precedence giữa các gate: mọi gate chỉ HẠ penalty, không nâng; post-pass chạy tuần
    tự connected_speech → coverage_gate (chỉ del) → drift_cap (chỉ sub); penalty_reason
    = gate cuối cùng THỰC SỰ thay đổi penalty (point đã 0 hoặc đã cap ≤ mức mới giữ
    nguyên reason cũ).

    Returns None nếu reference_phonemes rỗng.
    """
    if not reference_phonemes:
        return None

    # Multi-reference homograph (flag OFF = no-op): đổi lát reference của từ
    # đa-entry sang entry khớp acoustic nhất TRƯỚC DTW. Số span/chỉ số span không
    # đổi → skips/word_windows/word_probs (keyed span index) vẫn đúng.
    if homograph_selection_enabled and reference_spans and word_windows and segments:
        (reference_phonemes, reference_spans, reference_stress,
         reference_display_stress) = select_homograph_references(
            reference_phonemes, reference_spans, reference_stress,
            reference_display_stress, segments, word_windows, skips=skips,
        )

    predicted_phonemes = [s.phoneme for s in segments]
    predicted_conf = [s.confidence for s in segments]
    # Dùng chung cho coverage gate / drift cap / telemetry (hoist khỏi nhánh diagnostics).
    predicted_times = [(s.start, s.end) for s in segments]
    avg_confidence = (
        sum(predicted_conf) / len(predicted_conf) if predicted_conf else 0.0
    )
    (
        ref_word, ref_is_onset, ref_stress, ref_reducible, ref_is_coda,
        ref_g2p_uncertain, ref_r_droppable,
    ) = _ref_metadata(reference_phonemes, reference_spans, reference_stress)
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

    boundary_move_count = 0
    # "Said nothing" (không có phoneme nào nhận diện được) → 0% (path rỗng).
    if not predicted_phonemes:
        path: list[tuple[int, int]] = []
        result: dict[int, tuple[PhonemePoint, float | None]] = {}
        insertion_count = 0
        raw_by_ref: dict[int, float] = {}
        empty_prediction = True
    else:
        path = _dtw_align(predicted_phonemes, reference_phonemes)
        # Boundary refinement (flag OFF = no-op): sửa segment bị DTW gán nhầm
        # sang từ kề TRƯỚC khi chấm — statuses/penalty do _align_points tính lại
        # trên path đã sửa, slot bị bỏ trống rơi vào vòng bổ sung deletion dưới.
        if boundary_refine_enabled and reference_spans:
            path, boundary_moves = _refine_boundary_bleed(
                path, predicted_phonemes, reference_phonemes, reference_spans,
                ref_word, ref_is_onset, ref_is_coda, ref_stress, ref_reducible,
                ref_skipped, ref_r_droppable, ref_g2p_uncertain,
                accept_accent_variants=accept_accent_variants,
                l1_enabled=l1_enabled,
                predicted_times=predicted_times,
                word_windows=word_windows,
                word_windows_locked=word_windows_locked,
                pad=drift_window_pad,
            )
            for mv in boundary_moves:
                logger.info(
                    "Boundary refine: %r->%r | moved pred[%d]=%s t=%s "
                    "from ref[%d]=%s to ref[%d]=%s | displaced pred[%s] | "
                    "cost %.2f->%.2f | window_ok=%s",
                    mv["left_word"], mv["right_word"], mv["pred_idx"],
                    mv["pred_ph"], mv["pred_time"], mv["from_ref"],
                    mv["from_ph"], mv["to_ref"], mv["to_ph"],
                    mv["displaced_pred_idx"], mv["cost_before"],
                    mv["cost_after"], mv["window_ok"],
                )
            boundary_move_count = len(boundary_moves)
        result, insertion_count, raw_by_ref = _align_points(
            path, predicted_phonemes, predicted_conf, reference_phonemes,
            ref_word, ref_is_onset, ref_stress, ref_reducible, ref_skipped,
            ref_is_coda, confidence_knee, ref_display_stress,
            ref_g2p_uncertain=ref_g2p_uncertain,
            ref_r_droppable=ref_r_droppable,
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
                g2p_uncertain=ref_g2p_uncertain[i],
                r_droppable=ref_r_droppable[i],
            )
            result[i] = (point, pen)
            raw_by_ref[i] = raw

    # Post-pass connected speech (SAU khi mọi ref index đã resolve — cần nhìn điểm
    # của âm lân cận nên không làm được trong _align_points).
    if connected_speech_enabled and reference_spans and not empty_prediction:
        _apply_connected_speech(
            result, raw_by_ref, reference_phonemes, reference_spans, ref_skipped
        )

    # Post-pass coverage gate (Track A, chỉ del) rồi drift cap (Track B, chỉ sub) —
    # disjoint theo status nên thứ tự giao hoán; cả hai chỉ HẠ penalty (precedence).
    if coverage_gate_enabled and reference_spans and not empty_prediction:
        # Segment đã có chủ (cặp diagonal với ref BẤT KỲ) không tính là acoustic
        # evidence — chỉ insertion trong window mới chặn cap (xem _apply_coverage_gate).
        claimed_preds = frozenset(p for p, r in path if p >= 0 and r >= 0)
        _apply_coverage_gate(
            result, raw_by_ref, reference_phonemes, reference_spans, ref_skipped,
            predicted_times, word_windows, word_probs, claimed_preds,
            max_len=coverage_gate_max_len, cap=coverage_gate_cap,
            min_asr_prob=coverage_gate_min_asr_prob, pad=drift_window_pad,
        )
    if drift_cap_enabled and reference_spans and not empty_prediction:
        _apply_drift_cap(
            result, raw_by_ref, path, reference_phonemes, reference_spans,
            predicted_times, word_windows,
            pad=drift_window_pad, cap=drift_sub_cap,
        )

    # Cửa sổ phát lại từng từ theo wav2vec segment — tính MỘT lần, dùng cho cả
    # deletion-evidence probe (dưới) lẫn playback windows (cuối hàm).
    seg_times = _word_segment_times(path, segments, reference_spans)

    # Deletion-evidence probe (SHADOW) — SAU mọi gate để gắn lên point cuối cùng,
    # TRƯỚC vòng tổng hợp để point_by_ref/word details/diagnostics cùng thấy.
    # Chỉ telemetry: không chạm penalty/status — điểm bất biến bit-for-bit.
    if posteriors is not None and reference_spans:
        _attach_deletion_evidence(
            result, reference_phonemes, reference_spans, ref_skipped,
            path, segments, seg_times, word_windows, posteriors,
        )

    # Tổng hợp: errors + counts + penalty (một nguồn → không lệch nhau).
    errors: list[PhonemeError] = []
    matches = substitutions = deletions = 0
    total_penalty = 0.0
    raw_penalty_sum = 0.0
    l1_adjusted_count = low_conf_neutralized_count = recognizer_noise_count = 0
    coverage_collapse_count = drift_capped_count = 0
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
        elif point.penalty_reason == PenaltyReason.COVERAGE_COLLAPSE.value:
            coverage_collapse_count += 1
        elif point.penalty_reason == PenaltyReason.DRIFT_SUSPECTED.value:
            drift_capped_count += 1

    if empty_prediction:
        accuracy = 0.0
    elif scored > 0:
        accuracy = max(0.0, 1.0 - total_penalty / scored)
    else:
        accuracy = 1.0  # mọi âm đều skip (ASR nghe nhầm cả) → không có gì để chấm

    # Cửa sổ thời gian phát lại từng từ: Whisper WORD window là nguồn ranh giới TỪ, nhưng
    # bị SIẾT theo cửa sổ wav2vec segment (âm vị thực của từ) khi cả hai chồng nhau —
    # chặn cả 2 chiều "lem": Whisper gộp token/lem sang từ kế (bug "am" phát "9 am",
    # "helps" phát "helps me") LẪN seg phình do DTW mượn âm (bug "discount" phát "20
    # percent discount"). Giao rỗng → giữ Whisper; từ chỉ 1 nguồn → dùng nguồn đó (xem
    # _merge_playback_windows). Scoring/telemetry vẫn dùng word_windows/seg_times như
    # cũ — merge này CHỈ cho playback.
    # seg_counts + ref_lens: cho merge bypass sàn thời lượng khi attribution ĐỦ
    # (từ ngắn — segment spike ~20ms không bao giờ đạt sàn, xem _word_segment_counts).
    playback_times = _merge_playback_windows(
        seg_times, word_windows or {}, locked=word_windows_locked,
        seg_counts=_word_segment_counts(path, reference_spans),
        ref_lens={
            k: s.end_idx - s.start_idx
            for k, s in enumerate(reference_spans or [])
        },
    )
    # Từ locked (cửa sổ cắt sub-token "9am"→"am"): cắt theo tỉ lệ ký tự chỉ XẤP XỈ
    # (ngập ngừng/kéo dài phần số → vẫn dính "nine"), DTW attribution thì nhiễm →
    # neo lại theo acoustic: fitting-align âm vị của từ vào segments trong cửa sổ,
    # lấy đúng khoảng đoạn khớp (xem _reanchor_locked_windows). CHỈ playback.
    if word_windows_locked and reference_spans:
        _reanchor_locked_windows(
            playback_times, word_windows_locked, segments,
            reference_phonemes, reference_spans,
        )
    # Đệm + clamp theo từ liền kề → cửa sổ phát lại không lẹm sang từ khác (frontend phát
    # verbatim). Backend là nơi DUY NHẤT biết ranh giới từ kề nên đệm phải nằm ở đây.
    playback_times = _pad_and_clamp_windows(playback_times)
    words, words_truncated, words_total = _build_word_details(
        point_by_ref, reference_phonemes, reference_spans, max_words, span_skip_reason,
        word_times=playback_times,
    )

    # Telemetry (PR2/PR3-0) — DIAGNOSTIC ONLY, chỉ tính khi có sink (zero overhead khi tắt).
    if diagnostics_sink is not None:
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
        "skipped_words=%d | ref=%d | pred=%d | l1_adj=%d | low_conf_neut=%d | "
        "recog_noise=%d | cov_collapse=%d | drift_capped=%d | boundary_moves=%d",
        accuracy, matches, substitutions, deletions, insertion_count,
        len(span_skip_reason), len(reference_phonemes), len(predicted_phonemes),
        l1_adjusted_count, low_conf_neutralized_count, recognizer_noise_count,
        coverage_collapse_count, drift_capped_count, boundary_move_count,
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
        coverage_collapse_count=coverage_collapse_count,
        drift_capped_count=drift_capped_count,
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

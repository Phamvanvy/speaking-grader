"""Phoneme scoring — so sánh predicted phonemes với reference IPA sequence.

Dùng Dynamic Time Warping (DTW) để align 2 sequences (predicted vs reference),
từ đó detect substitution / deletion / insertion + tính overall accuracy.

Architecture:
  - compute_phoneme_score(): main entry point → PhonemeScore
  - _dtw_align(): align 2 phoneme sequences bằng DTW
  - _classify_errors(): phân loại lỗi từ alignment path
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import replace
from typing import Final

from .ipa import error_severity, normalize_ipa, phoneme_similarity
from .models import (
    PhonemeError,
    PhonemeErrorType,
    PhonemePoint,
    PhonemeScore,
    PhonemeSegment,
    WordPronunciation,
    WordSpan,
)

# Lỗi mang `position` là index trong reference sequence → map được về từ.
# Insertion mang `position` là index predicted → KHÔNG map (để word=None).
_WORD_MAPPABLE: Final = frozenset(
    {PhonemeErrorType.SUBSTITUTION, PhonemeErrorType.DELETION}
)

logger = logging.getLogger("toeic.phoneme.scoring")

# Số lỗi tối đa trả về trong results (tránh payload quá lớn)
MAX_ERRORS_RETURNED: Final[int] = 30

# Số từ tối đa trả về trong word details (cắt theo ranh giới từ, không giữa từ)
MAX_WORDS_RETURNED: Final[int] = 80

# Xếp hạng để chọn status "tốt hơn" khi 1 reference index bị chạm nhiều lần
_STATUS_RANK: Final[dict[str, int]] = {"ok": 0, "sub": 1, "del": 2}
_SEVERITY_RANK: Final[dict[str, int]] = {"low": 0, "medium": 1, "high": 2}


# ──────────────────────────────────────────────────────────────────────────────
# DTW alignment (simplified, pure Python — không cần numpy dependency)
# ──────────────────────────────────────────────────────────────────────────────

def _dtw_align(
    predicted: list[str],
    reference: list[str],
) -> list[tuple[int, int]]:
    """Dynamic Time Warping: align predicted → reference phoneme sequences.

    Returns list of (pred_idx, ref_idx) pairs theo optimal path.
    """
    if not predicted or not reference:
        return []

    n, m = len(predicted), len(reference)

    # Cost matrix: 0 = match, 1-m similarity = mismatch cost
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        cost[i][0] = float("inf")
    for j in range(1, m + 1):
        cost[0][j] = float("inf")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = phoneme_similarity(predicted[i - 1], reference[j - 1])
            match_cost = 1.0 - sim  # 0.0 = match, 1.0 = hoàn toàn khác
            cost[i][j] = match_cost + min(
                cost[i - 1][j - 1],   # match/substitution (diagonal)
                cost[i - 1][j],       # insertion in predicted (vertical)
                cost[i][j - 1],       # deletion from reference (horizontal)
            )

    # Backtrace
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            path.append((-1, j))  # deletion
        elif j == 0:
            i -= 1
            path.append((i, -1))  # insertion
        else:
            # Find which predecessor gave the min cost (excluding match_cost)
            cum_diag = cost[i - 1][j - 1]
            cum_up = cost[i - 1][j]
            cum_left = cost[i][j - 1]
            minimum = min(cum_diag, cum_up, cum_left)
            if cum_diag == minimum:
                i -= 1
                j -= 1
                path.append((i, j))  # match or substitution
            elif cum_up == minimum:
                i -= 1
                path.append((i, -1))  # insertion (predicted i, không có reference)
            else:
                j -= 1
                path.append((-1, j))  # deletion

    path.reverse()
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Error classification from DTW path
# ──────────────────────────────────────────────────────────────────────────────

def _classify_errors(
    path: list[tuple[int, int]],
    predicted: list[str],
    reference: list[str],
) -> list[PhonemeError]:
    """Từ DTW alignment path → danh sách PhonemeError."""
    errors: list[PhonemeError] = []

    for pred_idx, ref_idx in path:
        if pred_idx >= 0 and ref_idx >= 0:
            # Match or substitution
            pred_ph = predicted[pred_idx]
            ref_ph = reference[ref_idx]
            if normalize_ipa(pred_ph) != normalize_ipa(ref_ph):
                sim = phoneme_similarity(pred_ph, ref_ph)
                errors.append(PhonemeError(
                    error_type=PhonemeErrorType.SUBSTITUTION,
                    expected=ref_ph,
                    predicted=pred_ph,
                    position=ref_idx,
                    severity=error_severity(sim),
                ))
            # else: exact match, no error
        elif pred_idx < 0 and ref_idx >= 0:
            # Deletion: reference phoneme missing
            errors.append(PhonemeError(
                error_type=PhonemeErrorType.DELETION,
                expected=reference[ref_idx],
                predicted=None,
                position=ref_idx,
                severity="high",
            ))
        elif pred_idx >= 0 and ref_idx < 0:
            # Insertion: extra phoneme in prediction
            errors.append(PhonemeError(
                error_type=PhonemeErrorType.INSERTION,
                expected=None,
                predicted=predicted[pred_idx],
                position=pred_idx,
                severity="low",
            ))

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# Word mapping: gắn từ vào lỗi phoneme theo reference position
# ──────────────────────────────────────────────────────────────────────────────

def _word_at(position: int, spans: list[WordSpan], starts: list[int]) -> str | None:
    """Tìm từ chứa reference-index `position` bằng binary search.

    `spans` sắp xếp tăng dần, không chồng lấn; `starts` là [s.start_idx] đã tính
    sẵn (truyền vào để không tạo lại mỗi lần gọi). bisect_right + idx-1 cho span
    có start_idx lớn nhất mà ≤ position. Chỉ trả word nếu position thực sự nằm
    trong [start_idx, end_idx); nếu rơi vào gap (từ bị drop) hoặc position ==
    end_idx của span trước thì trả None (KHÔNG mượn từ liền trước).
    """
    idx = bisect.bisect_right(starts, position) - 1
    if idx < 0:
        return None
    span = spans[idx]
    if span.start_idx <= position < span.end_idx:
        return span.word
    return None


def _annotate_words(
    errors: list[PhonemeError],
    reference_spans: list[WordSpan] | None,
) -> list[PhonemeError]:
    """Gắn `word` vào các lỗi substitution/deletion theo reference_spans.

    Insertion giữ word=None (position là predicted index, không map được).
    PhonemeError là frozen → tạo bản mới bằng dataclasses.replace.
    """
    if not reference_spans:
        return errors
    starts = [s.start_idx for s in reference_spans]
    annotated: list[PhonemeError] = []
    for e in errors:
        if e.error_type in _WORD_MAPPABLE:
            annotated.append(
                replace(e, word=_word_at(e.position, reference_spans, starts))
            )
        else:
            annotated.append(e)
    return annotated


# ──────────────────────────────────────────────────────────────────────────────
# Per-word phoneme detail: IPA full từng từ + trạng thái từng âm (UI kiểu ELSA)
# ──────────────────────────────────────────────────────────────────────────────

def _better_point(a: PhonemePoint, b: PhonemePoint) -> PhonemePoint:
    """Chọn point 'tốt hơn' khi 1 reference index bị chạm nhiều lần trong path.

    Ưu tiên status ok > sub > del; nếu cùng status (vd 2 sub) thì ưu tiên
    severity nhẹ hơn (similarity cao hơn).
    """
    ra, rb = _STATUS_RANK.get(a.status, 9), _STATUS_RANK.get(b.status, 9)
    if ra != rb:
        return a if ra < rb else b
    sa = _SEVERITY_RANK.get(a.severity or "", -1)
    sb = _SEVERITY_RANK.get(b.severity or "", -1)
    return a if sa <= sb else b


def _build_word_details(
    path: list[tuple[int, int]],
    predicted: list[str],
    reference: list[str],
    spans: list[WordSpan] | None,
) -> tuple[list[WordPronunciation], bool, int]:
    """Từ DTW path + reference_spans → phát âm chi tiết từng từ.

    Mỗi reference index → đúng 1 PhonemePoint (ok/sub/del). KHÔNG giả định path
    cho đúng 1 status/index: nếu 1 ri bị chạm nhiều lần thì giữ status tốt nhất
    (_better_point); ri không xuất hiện trong path mặc định "del" severity high.
    Insertion (ref_idx < 0) bỏ qua (không gắn vào reference position).

    Cắt theo ranh giới từ — giữ nguyên cả WordSpan, không bao giờ cắt giữa từ.
    Path rỗng (vd nhánh không có predicted) → mọi âm thành "del".

    Returns (words, truncated, total_words).
    """
    if not spans:
        return [], False, 0

    status_by_ref: dict[int, PhonemePoint] = {}
    for pred_idx, ref_idx in path:
        if ref_idx < 0:
            continue
        ref_ph = reference[ref_idx]
        if pred_idx >= 0:
            pred_ph = predicted[pred_idx]
            if normalize_ipa(pred_ph) == normalize_ipa(ref_ph):
                point = PhonemePoint(symbol=ref_ph, status="ok")
            else:
                sim = phoneme_similarity(pred_ph, ref_ph)
                point = PhonemePoint(
                    symbol=ref_ph,
                    status="sub",
                    heard=pred_ph,
                    severity=error_severity(sim),
                )
        else:
            point = PhonemePoint(symbol=ref_ph, status="del", severity="high")

        existing = status_by_ref.get(ref_idx)
        status_by_ref[ref_idx] = (
            point if existing is None else _better_point(existing, point)
        )

    total = len(spans)
    kept = spans[:MAX_WORDS_RETURNED]
    truncated = total > len(kept)

    words: list[WordPronunciation] = []
    for span in kept:
        points = [
            status_by_ref.get(
                i, PhonemePoint(symbol=reference[i], status="del", severity="high")
            )
            for i in range(span.start_idx, span.end_idx)
        ]
        if not points:
            continue
        ok = sum(1 for p in points if p.status == "ok")
        words.append(WordPronunciation(
            word=span.word,
            ipa="".join(p.symbol for p in points),
            phonemes=points,
            accuracy=ok / len(points),
        ))

    if truncated:
        logger.info(
            "Word details truncated: kept %d / %d words", len(words), total
        )
    return words, truncated, total


# ──────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ──────────────────────────────────────────────────────────────────────────────

def compute_phoneme_score(
    segments: list[PhonemeSegment],
    reference_phonemes: list[str],
    reference_spans: list[WordSpan] | None = None,
) -> PhonemeScore | None:
    """Tính phoneme accuracy score từ predicted segments + reference.

    Algorithm:
      1. Trích predicted phoneme list từ segments
      2. DTW alignment với reference
      3. Classification: match / substitution / deletion / insertion
      4. Gắn `word` cho substitution/deletion nếu có reference_spans
      5. Tính accuracy = matches / reference_count

    Args:
        reference_spans: optional — map từ reference index → từ (xem WordSpan).
            Có thì mỗi lỗi substitution/deletion được gắn `word`; None thì các
            lỗi giữ word=None (giữ tương thích ngược).

    Returns None nếu reference_phonemes rỗng.
    """
    if not reference_phonemes:
        return None

    predicted_phonemes = [s.phoneme for s in segments]
    avg_confidence = (
        sum(s.confidence for s in segments) / len(segments)
        if segments
        else 0.0
    )

    if not predicted_phonemes:
        # Không có predicted phonemes → tất cả là deletions
        errors = [
            PhonemeError(
                error_type=PhonemeErrorType.DELETION,
                expected=ref_ph,
                predicted=None,
                position=i,
                severity="high",
            )
            for i, ref_ph in enumerate(reference_phonemes)
        ]
        errors = _annotate_words(errors, reference_spans)
        # Path rỗng → _build_word_details gán mọi âm thành "del"
        words, words_truncated, words_total = _build_word_details(
            [], [], reference_phonemes, reference_spans
        )
        return PhonemeScore(
            overall_accuracy=0.0,
            substitution_count=0,
            deletion_count=len(reference_phonemes),
            insertion_count=0,
            reference_count=len(reference_phonemes),
            predicted_count=0,
            avg_confidence=avg_confidence,
            errors=errors[:MAX_ERRORS_RETURNED],
            words=words,
            words_truncated=words_truncated,
            words_total=words_total,
        )

    # DTW alignment
    path = _dtw_align(predicted_phonemes, reference_phonemes)

    # Classify errors
    errors = _classify_errors(path, predicted_phonemes, reference_phonemes)

    # Gắn từ cho substitution/deletion (nếu có spans) — trước khi sort/cap.
    errors = _annotate_words(errors, reference_spans)

    # Phát âm chi tiết từng từ (IPA full + từng âm) cho UI kiểu ELSA.
    words, words_truncated, words_total = _build_word_details(
        path, predicted_phonemes, reference_phonemes, reference_spans
    )

    # Count by type
    substitutions = sum(1 for e in errors if e.error_type == PhonemeErrorType.SUBSTITUTION)
    deletions = sum(1 for e in errors if e.error_type == PhonemeErrorType.DELETION)
    insertions = sum(1 for e in errors if e.error_type == PhonemeErrorType.INSERTION)

    # Count matches from path
    matches = sum(
        1 for pi, ri in path
        if pi >= 0 and ri >= 0
        and normalize_ipa(predicted_phonemes[pi]) == normalize_ipa(reference_phonemes[ri])
    )

    # Accuracy = matches / reference_count
    accuracy = matches / len(reference_phonemes) if reference_phonemes else 0.0

    # Sort errors by severity (high → medium → low)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    errors.sort(key=lambda e: severity_order.get(e.severity, 3))

    logger.info(
        "Phoneme score: accuracy=%.2f | matches=%d | subs=%d | del=%d | ins=%d | ref=%d | pred=%d",
        accuracy, matches, substitutions, deletions, insertions,
        len(reference_phonemes), len(predicted_phonemes),
    )

    return PhonemeScore(
        overall_accuracy=round(accuracy, 4),
        substitution_count=substitutions,
        deletion_count=deletions,
        insertion_count=insertions,
        reference_count=len(reference_phonemes),
        predicted_count=len(predicted_phonemes),
        avg_confidence=round(avg_confidence, 4),
        errors=errors[:MAX_ERRORS_RETURNED],
        words=words,
        words_truncated=words_truncated,
        words_total=words_total,
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
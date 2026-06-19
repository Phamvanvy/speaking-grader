"""Phoneme scoring — so sánh predicted phonemes với reference IPA sequence.

Dùng Dynamic Time Warping (DTW) để align 2 sequences (predicted vs reference),
từ đó detect substitution / deletion / insertion + tính overall accuracy.

Architecture:
  - compute_phoneme_score(): main entry point → PhonemeScore
  - _dtw_align(): align 2 phoneme sequences bằng DTW
  - _classify_errors(): phân loại lỗi từ alignment path
"""

from __future__ import annotations

import logging
from typing import Final

from .ipa import error_severity, phoneme_similarity
from .models import PhonemeError, PhonemeErrorType, PhonemeSegment, PhonemeScore

logger = logging.getLogger("toeic.phoneme.scoring")

# Số lỗi tối đa trả về trong results (tránh payload quá lớn)
MAX_ERRORS_RETURNED: Final[int] = 30


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
                path.append((i, j))  # insertion
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
            if pred_ph != ref_ph:
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
# Main scoring function
# ──────────────────────────────────────────────────────────────────────────────

def compute_phoneme_score(
    segments: list[PhonemeSegment],
    reference_phonemes: list[str],
) -> PhonemeScore | None:
    """Tính phoneme accuracy score từ predicted segments + reference.

    Algorithm:
      1. Trích predicted phoneme list từ segments
      2. DTW alignment với reference
      3. Classification: match / substitution / deletion / insertion
      4. Tính accuracy = matches / reference_count

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
        return PhonemeScore(
            overall_accuracy=0.0,
            substitution_count=0,
            deletion_count=len(reference_phonemes),
            insertion_count=0,
            reference_count=len(reference_phonemes),
            predicted_count=0,
            avg_confidence=avg_confidence,
            errors=errors[:MAX_ERRORS_RETURNED],
        )

    # DTW alignment
    path = _dtw_align(predicted_phonemes, reference_phonemes)

    # Classify errors
    errors = _classify_errors(path, predicted_phonemes, reference_phonemes)

    # Count by type
    substitutions = sum(1 for e in errors if e.error_type == PhonemeErrorType.SUBSTITUTION)
    deletions = sum(1 for e in errors if e.error_type == PhonemeErrorType.DELETION)
    insertions = sum(1 for e in errors if e.error_type == PhonemeErrorType.INSERTION)

    # Count matches from path
    matches = sum(
        1 for pi, ri in path
        if pi >= 0 and ri >= 0 and predicted_phonemes[pi] == reference_phonemes[ri]
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
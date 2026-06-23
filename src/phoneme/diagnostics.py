"""Phoneme telemetry (PR2) — per-word diagnostic records cho calibration.

DIAGNOSTIC ONLY: dữ liệu ở đây KHÔNG bao giờ quay lại quyết định scoring/reliability.
Nó được tính từ DTW alignment (nơi DUY NHẤT có attribution predicted→word) nhưng tách
khỏi scorer: scorer chỉ gọi build_word_diagnostics() (THUẦN) rồi đưa kết quả cho một
`sink` được inject; module này giữ ids + ghi JSONL. Khi telemetry tắt, scorer không gọi
gì → zero overhead.

Mỗi audio sinh nhiều dòng JSON `type:"word"` + 1 dòng `type:"summary"`. `schema_version`
bắt buộc để đổi format sau này không vỡ parser. Greppable bằng jq/grep.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import PhonemePoint, WordSpan

logger = logging.getLogger("toeic.phoneme.diagnostics")

# Tăng khi đổi cấu trúc record. Parser downstream đọc field này để biết schema.
TELEMETRY_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class DiagnosticsContext:
    """Định danh để aggregate telemetry sau này (qua nhiều audio/phiên)."""

    session_id: str
    audio_id: str
    utterance_id: str


@dataclass(frozen=True)
class WordDiagnostic:
    """Số liệu chẩn đoán của 1 TỪ tham chiếu (theo chỉ số chuẩn)."""

    word: str
    index: int
    reference_ipa: str
    predicted_ipa: str       # lát predicted được DTW gán cho từ (attribution vị trí)
    coverage: float          # predicted_count / reference_len (chẩn đoán, KHÔNG quyết định)
    avg_conf: float
    p20_conf: float          # 20th-percentile confidence (lộ collapse cục bộ)
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    penalty: float
    skip_reason: str | None


def percentile(values: list[float], p: float) -> float:
    """Phân vị p (0..100) với nội suy tuyến tính; 0.0 nếu rỗng."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def build_word_diagnostics(
    path: list[tuple[int, int]],
    predicted: list[str],
    predicted_conf: list[float],
    reference: list[str],
    spans: list[WordSpan] | None,
    result: dict[int, tuple[PhonemePoint, float | None]],
    span_skip_reason: dict[int, str],
) -> list[WordDiagnostic]:
    """THUẦN: dựng WordDiagnostic cho mỗi span từ alignment đã có.

    Attribution predicted→từ dùng VỊ TRÍ trong DTW path (không phải match-ratio):
    phoneme predicted khớp ref index trong span → thuộc từ đó; insertion (ref_idx<0)
    gán cho từ liền trước. coverage/confidence chỉ là CHẨN ĐOÁN.
    """
    if not spans:
        return []

    n = len(reference)
    ref_to_span = [-1] * n
    for k, span in enumerate(spans):
        for i in range(span.start_idx, min(span.end_idx, n)):
            ref_to_span[i] = k

    word_pred: dict[int, list[int]] = defaultdict(list)  # span k → predicted indices
    word_ins: dict[int, int] = defaultdict(int)
    last_k = -1
    for pred_idx, ref_idx in path:
        if ref_idx >= 0:
            k = ref_to_span[ref_idx] if ref_idx < n else -1
            if k >= 0:
                last_k = k
                if pred_idx >= 0:
                    word_pred[k].append(pred_idx)
        elif pred_idx >= 0 and last_k >= 0:
            word_pred[last_k].append(pred_idx)  # insertion → từ liền trước
            word_ins[last_k] += 1

    diags: list[WordDiagnostic] = []
    for k, span in enumerate(spans):
        idxs = list(range(span.start_idx, min(span.end_idx, n)))
        if not idxs:
            continue
        preds = sorted(set(word_pred.get(k, [])))
        confs = [predicted_conf[pi] for pi in preds if pi < len(predicted_conf)]
        matches = subs = dels = 0
        penalty = 0.0
        for i in idxs:
            point, pen = result[i]
            if point.status == "ok":
                matches += 1
            elif point.status == "sub":
                subs += 1
            elif point.status == "del":
                dels += 1
            if pen is not None:
                penalty += pen
        diags.append(WordDiagnostic(
            word=span.word,
            index=k,
            reference_ipa="".join(reference[i] for i in idxs),
            predicted_ipa="".join(predicted[pi] for pi in preds),
            coverage=round(len(preds) / len(idxs), 3),
            avg_conf=round(sum(confs) / len(confs), 4) if confs else 0.0,
            p20_conf=round(percentile(confs, 20), 4),
            matches=matches,
            substitutions=subs,
            deletions=dels,
            insertions=word_ins.get(k, 0),
            penalty=round(penalty, 4),
            skip_reason=span_skip_reason.get(k),
        ))
    return diags


class TelemetryWriter:
    """Sink ghi WordDiagnostic ra JSONL (mỗi từ 1 dòng + 1 dòng summary/utterance)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def emit(
        self, context: DiagnosticsContext, diagnostics: Iterable[WordDiagnostic]
    ) -> None:
        diags = list(diagnostics)
        ids = {
            "session_id": context.session_id,
            "audio_id": context.audio_id,
            "utterance_id": context.utterance_id,
        }
        lines: list[str] = []
        for d in diags:
            lines.append(json.dumps(
                {"schema_version": TELEMETRY_SCHEMA_VERSION, "type": "word", **ids,
                 **asdict(d)},
                ensure_ascii=False,
            ))
        skipped = sum(1 for d in diags if d.skip_reason)
        reasons: dict[str, int] = defaultdict(int)
        for d in diags:
            if d.skip_reason:
                reasons[d.skip_reason] += 1
        lines.append(json.dumps(
            {"schema_version": TELEMETRY_SCHEMA_VERSION, "type": "summary", **ids,
             "words_total": len(diags), "words_skipped": skipped,
             "skip_reasons": dict(reasons)},
            ensure_ascii=False,
        ))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:  # telemetry là phụ trợ — KHÔNG được làm hỏng chấm điểm
            logger.warning("Không ghi được phoneme telemetry %s: %s", self.path, e)
            return
        logger.info(
            "Phoneme telemetry: %d words (%d skipped) → %s",
            len(diags), skipped, self.path,
        )

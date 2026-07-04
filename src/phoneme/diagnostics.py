"""Phoneme telemetry (PR2/PR3-0) — per-word diagnostic records cho calibration.

DIAGNOSTIC ONLY: dữ liệu ở đây KHÔNG bao giờ quay lại quyết định scoring/reliability.
Nó được tính từ DTW alignment (nơi DUY NHẤT có attribution predicted→word) nhưng tách
khỏi scorer: scorer chỉ gọi build_word_diagnostics() (THUẦN) rồi đưa kết quả cho một
`sink` được inject; module này giữ ids + ghi JSONL. Khi telemetry tắt, scorer không gọi
gì → zero overhead.

Mỗi audio sinh nhiều dòng JSON `type:"word"` + 1 dòng `type:"summary"`. `schema_version`
bắt buộc để đổi format sau này không vỡ parser. Greppable bằng jq/grep.

PR3-0 (drift-vs-hallucination): để KIỂM CHỨNG giả thuyết "false positive đến từ DTW gán
phoneme LỆCH ranh giới từ", mỗi substitution được phân loại theo VỊ TRÍ THỜI GIAN của
predicted segment so với CỬA SỔ THỜI GIAN Whisper của từ đó:
  - predicted segment NẰM TRONG window (±pad)  → hallucination (wav2vec nhả rác trong từ).
  - predicted segment NGOÀI window              → drift (DTW mượn âm của từ kế bên).
Cửa sổ từ map qua `map_reference_words_to_windows` (cùng kỹ thuật difflib với Recognition
Reliability — KHÔNG phải alignment thứ hai độc lập). Vẫn DIAGNOSTIC ONLY.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .l1_vietnamese import PenaltyReason, match_l1_final_deletion
from .models import PhonemePoint, WordSpan

logger = logging.getLogger("toeic.phoneme.diagnostics")

# Tăng khi đổi cấu trúc record. Parser downstream đọc field này để biết schema.
# v2: thêm window_start/end + sub_inside_window/sub_outside_window (PR3-0 drift telemetry).
# v3: thêm correspondences[] (ref↔pred từng phoneme + confidence + status + is_final) cho
#     phân tích R-vs-S (hallucination vs L1 error) chi tiết theo từng âm.
# v4: thêm penalty_reason / penalty_adjustment / l1_rule_id vào mỗi correspondence (L1-aware
#     scoring layer) → đo l1_adjustment_ratio + đếm rule kích hoạt từ telemetry.
# v5: thêm evidence / evidence_source / evidence_version vào correspondence "del" (deletion-
#     evidence probe, SHADOW) — dữ liệu gán nhãn cho sprint Word Reliability Gate. Các key
#     này CHỈ xuất hiện khi probe chạy (evidence_source != None) — record cũ không đổi.
TELEMETRY_SCHEMA_VERSION: int = 5

# Đệm cửa sổ thời gian khi phân loại drift: predicted segment chạm mép window trong
# khoảng này vẫn coi là "trong từ". Whisper word timestamps lệch ~±100–300ms; pad nhỏ
# ở đây CHỈ nuốt sai số biên, KHÔNG nới rộng tới từ kế bên. CHỈ dùng cho telemetry.
DRIFT_WINDOW_PAD_SEC: float = 0.08

# Tách transcript/từ thành token chuẩn (lowercase, bỏ dấu câu) — KHỚP với
# RecognizerEvidence.from_transcript trong reliability để mapping nhất quán với skip.
_WORD_TOKEN_RE = re.compile(r"[a-z0-9']+")


def is_within_word_window(
    pred_start: float,
    pred_end: float,
    window: tuple[float, float],
    pad: float = DRIFT_WINDOW_PAD_SEC,
) -> bool:
    """True nếu predicted segment [pred_start, pred_end] overlap cửa sổ từ (±pad).

    Một nguồn DUY NHẤT cho phép so overlap segment↔window: dùng bởi telemetry drift
    (build_word_diagnostics) VÀ các gate scoring (coverage/drift cap) để hai bên
    không bao giờ lệch định nghĩa "trong từ".
    """
    return pred_end >= window[0] - pad and pred_start <= window[1] + pad


def _normalize_word(text: str) -> str:
    """Chuẩn hoá 1 từ → 1 token (ghép các mảnh [a-z0-9']); '' nếu toàn dấu câu."""
    return "".join(_WORD_TOKEN_RE.findall((text or "").lower()))


def map_reference_words_to_indices(
    reference_words: list[str],
    transcript_texts: list[str],
) -> dict[int, int]:
    """Map CHỈ SỐ TỪ THAM CHIẾU (occurrence) → CHỈ SỐ Whisper word đã khớp.

    Lõi dùng chung cho map_reference_words_to_windows (timestamp) và word_probs
    (probability, guard coverage gate) — MỘT alignment duy nhất, hai bên đọc field
    khác nhau của cùng transcript word, không bao giờ lệch nhau. Dùng CÙNG kỹ thuật
    difflib.SequenceMatcher trên 2 danh sách từ như Recognition Reliability
    (reliability.assess_reliability) — KHÔNG tạo alignment thứ hai độc lập:
      - 'equal'  : ref[i] khớp hyp[j] theo vị trí → index j.
      - 'replace': mỗi ref[i] lấy hyp[j] có ratio cao nhất trong block.
      - 'delete' : từ script không có trong transcript → KHÔNG có key (unalignable;
        cũng chính là từ Recognition Reliability skip).
      - 'insert' : transcript thừa từ → không có ref tương ứng → bỏ qua.
    """
    if not reference_words or not transcript_texts:
        return {}
    ref_norm = [w.lower() for w in reference_words]
    hyp_norm = [_normalize_word(t) for t in transcript_texts]
    indices: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(a=ref_norm, b=hyp_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                indices[i1 + off] = j1 + off
        elif tag == "replace":
            for i in range(i1, i2):
                best_j, best_ratio = -1, -1.0
                for j in range(j1, j2):
                    r = difflib.SequenceMatcher(None, ref_norm[i], hyp_norm[j]).ratio()
                    if r > best_ratio:
                        best_j, best_ratio = j, r
                if best_j >= 0:
                    indices[i] = best_j
        # 'delete' / 'insert' → không có key
    return indices


def map_reference_words_to_windows(
    reference_words: list[str],
    transcript_words: list[tuple[str, float, float]],
) -> dict[int, tuple[float, float]]:
    """Map CHỈ SỐ TỪ THAM CHIẾU (occurrence) → cửa sổ thời gian Whisper (start, end).

    Wrapper trên map_reference_words_to_indices (xem docstring đó cho quy tắc khớp).
    `transcript_words`: list (text, start_s, end_s) lấy thẳng từ Whisper word timestamps.
    Trả {word_index: (start_s, end_s)}; chỉ số khớp WordSpan của scorer (cùng nguồn từ).
    """
    indices = map_reference_words_to_indices(
        reference_words, [t for (t, _s, _e) in transcript_words]
    )
    return {
        i: (float(transcript_words[j][1]), float(transcript_words[j][2]))
        for i, j in indices.items()
    }


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
    # PR3-0 drift telemetry — cửa sổ thời gian Whisper của từ + phân loại substitution
    # theo vị trí predicted segment (mặc định để tương thích ngược: không có window).
    window_start: float | None = None   # None nếu từ không map được sang transcript word
    window_end: float | None = None
    sub_inside_window: int = 0          # sub có predicted segment TRONG window → hallucination
    sub_outside_window: int = 0         # sub có predicted segment NGOÀI window → drift
    # PR3-0/pivot: 1 phần tử cho MỖI phoneme tham chiếu của từ (ok/sub/del/skipped). Mỗi phần
    # tử: {ref_symbol, pred_symbol, confidence, status, is_final, sub_outside_window}. Cho phép
    # phân tích R-vs-S theo từng âm (confidence của lỗi, deletion âm cuối). KHÔNG gồm insertion.
    correspondences: list[dict] = field(default_factory=list)


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
    predicted_times: list[tuple[float, float]] | None = None,
    word_windows: Mapping[int, tuple[float, float]] | None = None,
    pad: float = DRIFT_WINDOW_PAD_SEC,
) -> list[WordDiagnostic]:
    """THUẦN: dựng WordDiagnostic cho mỗi span từ alignment đã có.

    Attribution predicted→từ dùng VỊ TRÍ trong DTW path (không phải match-ratio):
    phoneme predicted khớp ref index trong span → thuộc từ đó; insertion (ref_idx<0)
    gán cho từ liền trước. coverage/confidence chỉ là CHẨN ĐOÁN.

    PR3-0: nếu có `predicted_times` (start,end mỗi predicted segment) + `word_windows`
    (cửa sổ Whisper theo chỉ số từ), mỗi substitution được phân loại drift-vs-hallucination
    theo vị trí segment so với window (±pad). Thiếu một trong hai → 0/0 (không phân loại).
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
    pred_for_ref: dict[int, int] = {}  # ref_idx → predicted_idx (cặp diagonal, cho drift)
    last_k = -1
    for pred_idx, ref_idx in path:
        if ref_idx >= 0:
            k = ref_to_span[ref_idx] if ref_idx < n else -1
            if k >= 0:
                last_k = k
                if pred_idx >= 0:
                    word_pred[k].append(pred_idx)
            if pred_idx >= 0:
                pred_for_ref[ref_idx] = pred_idx
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
        win = word_windows.get(k) if word_windows else None
        matches = subs = dels = 0
        penalty = 0.0
        sub_in = sub_out = 0
        correspondences: list[dict] = []
        final_idx = min(span.end_idx, n) - 1  # index âm cuối của từ (cho is_final)
        # 1 lượt qua reference index của từ: đếm + drift + dựng correspondence từng âm.
        for i in idxs:
            point, pen = result[i]
            status = point.status
            if status == "ok":
                matches += 1
            elif status == "sub":
                subs += 1
            elif status == "del":
                dels += 1
            if pen is not None:
                penalty += pen

            # predicted gắn với âm i (cặp diagonal); del/không gắn → None.
            pi = pred_for_ref.get(i)
            has_pred = status != "del" and pi is not None and pi < len(predicted)
            pred_symbol = predicted[pi] if has_pred else None
            conf = (
                predicted_conf[pi]
                if has_pred and pi is not None and pi < len(predicted_conf)
                else None
            )
            outside = False
            # PR3-0 drift: sub có window + times → trong/ngoài window.
            if (status == "sub" and win is not None and predicted_times is not None
                    and pi is not None and pi < len(predicted_times)):
                ps, pe = predicted_times[pi]
                if is_within_word_window(ps, pe, win, pad):  # overlap → trong từ
                    sub_in += 1
                else:
                    sub_out += 1
                    outside = True
            # L1-aware layer: lý do + mức điều chỉnh penalty (point đã ghi sẵn); rule_id
            # suy từ phoneme khi reason là L1 (thuần hàm → deterministic).
            reason = point.penalty_reason
            l1_rule_id = None
            if reason == PenaltyReason.L1_FINAL_DELETION.value:
                m = match_l1_final_deletion(reference[i])
                l1_rule_id = m.rule_id if m is not None else None
            entry = {
                "ref_symbol": reference[i],
                "pred_symbol": pred_symbol,
                "confidence": round(conf, 4) if conf is not None else None,
                "status": status,
                "is_final": i == final_idx,
                "sub_outside_window": outside,
                "penalty_reason": reason,
                "penalty_adjustment": round(point.penalty_adjustment, 4),
                "l1_rule_id": l1_rule_id,
            }
            # Deletion-evidence probe (SHADOW, v5): chỉ thêm key khi probe đã chạy
            # trên point này — record khi probe tắt giữ nguyên format cũ.
            if point.evidence_source is not None:
                entry["evidence"] = (
                    point.evidence.to_dict() if point.evidence else None
                )
                entry["evidence_source"] = point.evidence_source
                entry["evidence_version"] = point.evidence_version
            correspondences.append(entry)

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
            window_start=round(win[0], 3) if win is not None else None,
            window_end=round(win[1], 3) if win is not None else None,
            sub_inside_window=sub_in,
            sub_outside_window=sub_out,
            correspondences=correspondences,
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
        # PR3-0: tổng hợp drift-vs-hallucination cho utterance này. drift_fraction =
        # sub ngoài window / tổng sub đã phân loại; None nếu chưa có window nào (telemetry
        # bật nhưng thiếu Whisper timestamps / không có script). Gate kill PR3-0 cộng dồn
        # các con số này qua cả corpus, KHÔNG đọc 1 utterance lẻ.
        subs_inside = sum(d.sub_inside_window for d in diags)
        subs_outside = sum(d.sub_outside_window for d in diags)
        subs_classified = subs_inside + subs_outside
        drift_fraction = (
            round(subs_outside / subs_classified, 4) if subs_classified else None
        )
        lines.append(json.dumps(
            {"schema_version": TELEMETRY_SCHEMA_VERSION, "type": "summary", **ids,
             "words_total": len(diags), "words_skipped": skipped,
             "skip_reasons": dict(reasons),
             "subs_inside_window": subs_inside, "subs_outside_window": subs_outside,
             "drift_fraction": drift_fraction},
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

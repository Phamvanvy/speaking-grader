#!/usr/bin/env python3
"""Helper dùng chung cho các script chẩn đoán/benchmark phoneme pipeline.

Tách từ debug_full_vs_sentence.py (2026-07-04) để bench_chunking.py dùng lại:
audio (ffmpeg), DTW stats, so raw segments, scoring wrapper mirror production,
sentence↔span mapping, dựng reference context free-speech. DIAGNOSTIC ONLY —
không import ngược vào src/.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phoneme.diagnostics import map_reference_words_to_indices  # noqa: E402
from src.phoneme.ipa import (  # noqa: E402
    normalize_ipa,
    phoneme_similarity,
    text_to_ipa_sequence_with_spans,
)
from src.phoneme.reliability import (  # noqa: E402
    SkipDecision,
    SkipReason,
    assess_asr_confidence,
)
from src.phoneme.scoring import compute_phoneme_score  # noqa: E402
from src.phoneme.scoring.alignment import _dtw_align  # noqa: E402

SLICE_PAD_SEC = 0.3       # đệm khi cắt câu khỏi audio
SEG_WINDOW_PAD_SEC = 0.2  # đệm khi so raw segments quanh window của từ

_TOKEN_RE = re.compile(r"[a-zA-Z'-]+")  # KHỚP text_to_ipa_sequence_with_spans


def norm_word(w: str) -> str:
    return (w or "").lower().strip(".,;:!?\"()[]{}")


# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers (ffmpeg)
# ──────────────────────────────────────────────────────────────────────────────

def extract_wav(video: Path, out_wav: Path, fresh: bool = False) -> None:
    if out_wav.exists() and not fresh:
        return
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True, capture_output=True)


def slice_wav(full_wav: Path, out_wav: Path, start: float, end: float) -> None:
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
           "-i", str(full_wav), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
           str(out_wav)]
    subprocess.run(cmd, check=True, capture_output=True)


# ──────────────────────────────────────────────────────────────────────────────
# DTW stats + edit distance (chỉ dùng trong script)
# ──────────────────────────────────────────────────────────────────────────────

def dtw_stats(predicted: list[str], reference: list[str]) -> dict:
    """Chạy lại _dtw_align (đúng hàm production) rồi đo path: length, số bước
    diagonal/ins/del, tổng cost diagonal = sum(1 - similarity)."""
    t0 = time.perf_counter()
    path = _dtw_align(predicted, reference)
    elapsed = time.perf_counter() - t0
    diag = ins = dele = 0
    diag_cost = 0.0
    for pi, ri in path:
        if pi >= 0 and ri >= 0:
            diag += 1
            diag_cost += 1.0 - phoneme_similarity(predicted[pi], reference[ri])
        elif pi >= 0:
            ins += 1
        else:
            dele += 1
    return {
        "ref_len": len(reference), "pred_len": len(predicted),
        "path_len": len(path), "diagonal_steps": diag,
        "insertion_steps": ins, "deletion_steps": dele,
        "diagonal_cost": round(diag_cost, 3),
        "cost_per_ref": round(diag_cost / len(reference), 4) if reference else None,
        "dtw_runtime_sec": round(elapsed, 2),
    }


def edit_align(a: list[str], b: list[str]) -> tuple[int, list[tuple[int, int]]]:
    """Levenshtein distance + danh sách cặp index khớp nhau (equal ops)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + c, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return dp[n][m], pairs


def segment_comparison(full_segs: list[dict], slice_segs: list[dict],
                       t0: float, t1: float) -> dict:
    """So raw segments của 2 run trong cửa sổ thời gian tuyệt đối [t0, t1]."""
    fw = [s for s in full_segs if s["end"] >= t0 and s["start"] <= t1]
    sw = [s for s in slice_segs if s["end"] >= t0 and s["start"] <= t1]
    fa = [normalize_ipa(s["phoneme"]) for s in fw]
    sa = [normalize_ipa(s["phoneme"]) for s in sw]
    if not fa and not sa:
        agreement = None
        pairs: list[tuple[int, int]] = []
    else:
        dist, pairs = edit_align(fa, sa)
        agreement = round(1.0 - dist / max(len(fa), len(sa)), 3)
    shifts = [sw[j]["start"] - fw[i]["start"] for i, j in pairs]
    fconf = [s["conf"] for s in fw]
    sconf = [s["conf"] for s in sw]
    return {
        "window": [round(t0, 3), round(t1, 3)],
        "full_phonemes": "".join(s["phoneme"] for s in fw),
        "slice_phonemes": "".join(s["phoneme"] for s in sw),
        "segment_agreement": agreement,
        "n_segments_full": len(fw), "n_segments_slice": len(sw),
        "mean_conf_full": round(sum(fconf) / len(fconf), 4) if fconf else None,
        "mean_conf_slice": round(sum(sconf) / len(sconf), 4) if sconf else None,
        "mean_conf_delta": (
            round(sum(fconf) / len(fconf) - sum(sconf) / len(sconf), 4)
            if fconf and sconf else None
        ),
        "time_shift_median": (
            round(statistics.median(shifts), 3) if shifts else None
        ),
        "full_segments": fw,
        "slice_segments": sw,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring wrapper — mirror analyzer.analyze + grade_response free-speech path
# ──────────────────────────────────────────────────────────────────────────────

def run_scoring(config, segments, posteriors, reference_text: str,
                skips: dict, word_windows: dict | None, word_probs: dict | None,
                gates_on: bool) -> dict:
    """Gọi compute_phoneme_score với đúng tham số analyzer truyền (accent default),
    capture diagnostics in-memory. Trả {score, diags}."""
    phonemes, spans, stress, disp = text_to_ipa_sequence_with_spans(reference_text)
    captured: list = []
    score = compute_phoneme_score(
        segments, phonemes, spans, stress,
        reference_display_stress=disp,
        max_words=10**9,  # diagnostic: không cắt danh sách từ
        skips=skips,
        confidence_knee=config.phoneme_confidence_knee,
        diagnostics_sink=captured.extend,
        word_windows=word_windows,
        l1_enabled=config.phoneme_l1_enabled,
        l1_min_confidence=config.phoneme_l1_min_confidence,
        low_conf_floor=config.phoneme_l1_low_conf_floor,
        recognizer_noise_sim=config.phoneme_recognizer_noise_sim,
        recognizer_noise_conf=config.phoneme_recognizer_noise_conf,
        recognizer_noise_conf_vowel=config.phoneme_recognizer_noise_conf_vowel,
        accept_accent_variants=True,  # accent="default" như UI
        connected_speech_enabled=config.phoneme_connected_speech_enabled,
        word_probs=word_probs,
        coverage_gate_enabled=gates_on,
        coverage_gate_cap=config.phoneme_coverage_gate_cap,
        coverage_gate_max_len=config.phoneme_coverage_gate_max_len,
        coverage_gate_min_asr_prob=config.phoneme_coverage_gate_min_asr_prob,
        drift_cap_enabled=gates_on,
        drift_sub_cap=config.phoneme_drift_sub_cap,
        drift_window_pad=config.phoneme_drift_window_pad,
        posteriors=posteriors,
    )
    return {"score": score, "diags": [asdict(d) for d in captured]}


def score_summary(score, diags: list[dict]) -> dict:
    return {
        "overall_accuracy": score.overall_accuracy,
        "matches": sum(d["matches"] for d in diags),
        "substitutions": score.substitution_count,
        "deletions": score.deletion_count,
        "insertions": score.insertion_count,
        "sub_inside_window": sum(d["sub_inside_window"] for d in diags),
        "sub_outside_window": sum(d["sub_outside_window"] for d in diags),
        "coverage_collapse_count": score.coverage_collapse_count,
        "drift_capped_count": score.drift_capped_count,
        "recognizer_noise_count": score.recognizer_noise_count,
    }


def word_block(d: dict) -> dict:
    """Rút gọn 1 WordDiagnostic dict cho báo cáo + kèm deletion evidence."""
    evidence = [
        {"ref": c["ref_symbol"], **(c.get("evidence") or {}),
         "argmax": (c.get("evidence") or {}).get("argmax_token")}
        for c in d["correspondences"]
        if c["status"] == "del" and c.get("evidence") is not None
    ]
    return {
        "word": d["word"], "index": d["index"],
        "reference_ipa": d["reference_ipa"], "predicted_ipa": d["predicted_ipa"],
        "coverage": d["coverage"], "avg_conf": d["avg_conf"],
        "matches": d["matches"], "substitutions": d["substitutions"],
        "deletions": d["deletions"], "insertions": d["insertions"],
        "penalty": d["penalty"], "skip_reason": d["skip_reason"],
        "window": [d["window_start"], d["window_end"]],
        "sub_inside_window": d["sub_inside_window"],
        "sub_outside_window": d["sub_outside_window"],
        "statuses": "".join(
            {"ok": ".", "sub": "S", "del": "D", "skipped": "_"}.get(
                c["status"], "?") for c in d["correspondences"]
        ),
        "deletion_evidence": evidence or None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sentence mapping
# ──────────────────────────────────────────────────────────────────────────────

def split_sentences(transcript: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", transcript.strip())
    return [p for p in parts if p.strip()]


def map_sentences_to_spans(sentences: list[str], spans) -> list[list[int]]:
    """Mỗi câu → danh sách GLOBAL span index. Đi song song: token nào tạo span
    (word khớp) thì tiêu thụ span kế tiếp; token bị G2P drop thì không."""
    out: list[list[int]] = []
    ptr = 0
    for sent in sentences:
        idxs: list[int] = []
        for tok in _TOKEN_RE.findall(sent):
            if ptr < len(spans) and spans[ptr].word == tok:
                idxs.append(ptr)
                ptr += 1
            # else: token bị drop khi build reference (không có span)
        out.append(idxs)
    if ptr != len(spans):
        print(f"[WARN] sentence/span walk lệch: consumed {ptr}/{len(spans)} spans")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reference context — mirror grade_response free-speech path
# ──────────────────────────────────────────────────────────────────────────────

def build_reference_context(config, asr_data: dict) -> dict:
    """Dựng reference/spans/skips/word_windows/word_probs từ ASR data cached
    ({'text', 'words': [{'text','start','end','probability'}]}) — đúng free-speech
    path của grade_response (faster_whisper → phoneme_asr_conf_min)."""
    transcript = asr_data["text"]
    phonemes, spans, _stress, _disp = text_to_ipa_sequence_with_spans(transcript)
    reference_words = [s.word for s in spans]
    skips = dict(assess_asr_confidence(
        reference_words,
        [(w["text"], w["probability"]) for w in asr_data["words"]],
        min_probability=config.phoneme_asr_conf_min,
        transcript_text=transcript,
    ))
    for k, s in enumerate(spans):
        if s.source == "espeak":
            skips.setdefault(k, SkipDecision(k, SkipReason.OOV_ESPEAK))
    widx = map_reference_words_to_indices(
        reference_words, [w["text"] for w in asr_data["words"]])
    word_windows = {
        i: (float(asr_data["words"][j]["start"]), float(asr_data["words"][j]["end"]))
        for i, j in widx.items()
    }
    word_probs = {
        i: float(asr_data["words"][j]["probability"] or 0.0)
        for i, j in widx.items()
    }
    return {
        "transcript": transcript, "phonemes": phonemes, "spans": spans,
        "skips": skips, "word_windows": word_windows, "word_probs": word_probs,
    }


def dump_segments(segments) -> list[dict]:
    """PhonemeSegment list → dicts {phoneme,start,end,conf} cho JSONL/so sánh."""
    return [
        {"phoneme": s.phoneme, "start": s.start, "end": s.end,
         "conf": s.confidence} for s in segments
    ]


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

#!/usr/bin/env python3
"""Parity bench cho parallel chunk phoneme (TOEIC_PHONEME_DEVICES).

Chạy trên bộ wav + ASR đã cache của scripts/trace_word_case.py
(outputs/case_project/wav + asr): với CÙNG chunk_spans (tính như core.py từ
Whisper word timestamps), predict 2 lần mỗi clip:

  A) tuần tự  — devices=None (đường production hiện tại)
  B) parallel — devices=--devices (chunk chia round-robin lên các GPU)

rồi so từng clip:
  - segments: bằng tuyệt đối từng (phoneme, start, end, confidence) — GATE
    chính (scoring đọc segments).
  - posteriors: bitwise (np.array_equal) từng chunk + max|Δ| nếu lệch — GPU
    cùng kiến trúc kỳ vọng bitwise; lệch epsilon mà segments vẫn bằng thì báo
    để cân nhắc.
  - wall-clock A vs B → speedup.

Chạy:  python scripts/bench_phoneme_parallel.py [--devices cuda:0,cuda:1]
       (nên chạy KHI llama-server đang chiếm GPU1 để test đúng envelope VRAM)
Ghi outputs/bench_phoneme_parallel.json. Exit 1 nếu có clip lệch segments.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench_common import REPO_ROOT  # noqa: E402  (thêm repo root vào sys.path)

from src.config import load_config  # noqa: E402
from src.phoneme.chunking import compute_chunk_spans  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402


def _spans_for(config, asr_data) -> list[tuple[float, float]] | None:
    """chunk_spans đúng như core.py tính từ Whisper word timestamps."""
    words = asr_data.get("words") or []
    if not words:
        return None
    strategy = config.phoneme_chunking_strategy
    if strategy == "off":
        strategy = "hybrid"  # bench luôn cần chunking để có gì mà song song
    return compute_chunk_spans(
        [(w["text"], float(w["start"]), float(w["end"])) for w in words],
        float(asr_data.get("duration") or 0.0),
        strategy=strategy,
        max_chunk_sec=config.phoneme_chunk_max_sec,
        min_pause_sec=config.phoneme_chunk_min_pause_sec,
        pad_sec=config.phoneme_chunk_pad_sec,
    ) or None


def _segments_key(segments):
    return [(s.phoneme, s.start, s.end, round(s.confidence, 6)) for s in segments]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/case_project",
                    help="thư mục có wav/ + asr/ cache từ trace_word_case.py")
    ap.add_argument("--devices", default="cuda:0,cuda:1",
                    help="danh sách device cho đường parallel (comma)")
    args = ap.parse_args()

    config = load_config()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    cache = REPO_ROOT / args.outdir
    wavs = sorted((cache / "wav").glob("*.wav"),
                  key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, 0))
    if not wavs:
        print(f"[!] Không có wav cache tại {cache/'wav'} — chạy trace_word_case trước.")
        return 1

    common = dict(
        model_id=config.phoneme_wav2vec_model,
        device=config.phoneme_device,
        min_phoneme_duration=config.phoneme_min_duration_sec,
        confidence_threshold=config.phoneme_confidence_threshold,
    )
    seq_pred = Wav2VecPhonemePredictor(**common)
    par_pred = Wav2VecPhonemePredictor(**common, devices=devices)

    rows = []
    seq_total = par_total = 0.0
    mismatches = 0
    for wav in wavs:
        asr_path = cache / "asr" / f"{wav.stem}.json"
        if not asr_path.exists():
            print(f"[!] thiếu ASR cache {asr_path.name} — bỏ qua")
            continue
        asr_data = json.loads(asr_path.read_text(encoding="utf-8"))
        spans = _spans_for(config, asr_data)
        if not spans or len(spans) < 2:
            print(f"[-] {wav.name}: <2 chunks — bỏ qua (không có gì để song song)")
            continue

        t0 = time.perf_counter()
        seq_segs, seq_warn, seq_post = seq_pred.predict_with_posteriors(
            str(wav), chunk_spans=spans)
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        par_segs, par_warn, par_post = par_pred.predict_with_posteriors(
            str(wav), chunk_spans=spans)
        t_par = time.perf_counter() - t0

        if seq_warn or par_warn:
            print(f"[!] {wav.name}: warning seq={seq_warn} par={par_warn} — bỏ qua")
            continue

        segs_equal = _segments_key(seq_segs) == _segments_key(par_segs)
        n_chunks = len(seq_post.chunks)
        post_bitwise = (
            len(par_post.chunks) == n_chunks
            and all(
                a[0] == b[0] and np.array_equal(a[1].probs, b[1].probs)
                for a, b in zip(seq_post.chunks, par_post.chunks)
            )
        )
        max_diff = 0.0
        if not post_bitwise and len(par_post.chunks) == n_chunks:
            max_diff = max(
                float(np.max(np.abs(
                    a[1].probs.astype(np.float64) - b[1].probs.astype(np.float64)
                )))
                for a, b in zip(seq_post.chunks, par_post.chunks)
                if a[1].probs.shape == b[1].probs.shape
            )

        if not segs_equal:
            mismatches += 1
        seq_total += t_seq
        par_total += t_par
        rows.append({
            "wav": wav.name, "chunks": n_chunks,
            "segments_equal": segs_equal,
            "posteriors_bitwise": post_bitwise,
            "posteriors_max_abs_diff": max_diff,
            "t_seq_sec": round(t_seq, 2), "t_par_sec": round(t_par, 2),
        })
        flag = "OK " if segs_equal else "***MISMATCH***"
        print(f"[{flag}] {wav.name}: chunks={n_chunks} "
              f"seq={t_seq:.2f}s par={t_par:.2f}s "
              f"bitwise={post_bitwise} maxΔ={max_diff:.2e}")

    speedup = (seq_total / par_total) if par_total > 0 else 0.0
    summary = {
        "devices": devices, "clips": len(rows), "segment_mismatches": mismatches,
        "t_seq_total_sec": round(seq_total, 2),
        "t_par_total_sec": round(par_total, 2),
        "speedup": round(speedup, 2),
    }
    out = REPO_ROOT / "outputs" / "bench_phoneme_parallel.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTổng: {len(rows)} clip | mismatch={mismatches} | "
          f"seq={seq_total:.1f}s par={par_total:.1f}s → speedup {speedup:.2f}× | "
          f"kết quả: {out}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())

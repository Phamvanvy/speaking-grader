#!/usr/bin/env python3
"""Benchmark A/B multi-reference homograph selection (TOEIC_PHONEME_MULTIREF).

Chạy trên bộ wav + ASR đã cache của scripts/trace_word_case.py
(outputs/case_project/wav + asr — 12 câu trả lời full TOEIC test 1): wav2vec
single-pass 1 lần/clip, scoring 2 lần (homograph OFF vs ON, gates OFF như
production default), rồi so:

  - overall_accuracy off/on từng clip (delta).
  - Danh sách từ được SWAP entry (diff reference_ipa giữa 2 run) + sub+del
    off/on của riêng từ đó.
  - Regression guard: từ nào penalty TĂNG khi bật flag (kỳ vọng: không có —
    swap chỉ xảy ra khi entry mới khớp acoustic hơn hẳn).

Chạy:  python scripts/bench_multiref.py [--outdir outputs/case_project]
Ghi outputs/homographs/bench_multiref.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bench_common import REPO_ROOT, build_reference_context, run_scoring

from src.config import load_config  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/case_project",
                    help="thư mục có wav/ + asr/ cache từ trace_word_case.py")
    args = ap.parse_args()

    config = load_config()
    cache = REPO_ROOT / args.outdir
    wavs = sorted((cache / "wav").glob("*.wav"),
                  key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, 0))
    if not wavs:
        print(f"[!] Không có wav cache tại {cache/'wav'} — chạy trace_word_case trước.")
        return 1

    predictor = Wav2VecPhonemePredictor(
        model_id=config.phoneme_wav2vec_model, device=config.phoneme_device,
        min_phoneme_duration=config.phoneme_min_duration_sec,
        confidence_threshold=config.phoneme_confidence_threshold,
    )

    rows, all_swaps, regressions = [], [], []
    for wav in wavs:
        asr_path = cache / "asr" / f"{wav.stem}.json"
        if not asr_path.exists():
            print(f"[!] thiếu ASR cache {asr_path.name} — bỏ qua")
            continue
        asr_data = json.loads(asr_path.read_text(encoding="utf-8"))
        ctx = build_reference_context(config, asr_data)
        t0 = time.perf_counter()
        segs, warn, posts = predictor.predict_with_posteriors(str(wav))
        if warn:
            print(f"[!] {wav.name}: wav2vec warning {warn} — bỏ qua")
            continue
        off = run_scoring(config, segs, posts, asr_data["text"], ctx["skips"],
                          ctx["word_windows"], ctx["word_probs"],
                          gates_on=False, homograph_on=False)
        on = run_scoring(config, segs, posts, asr_data["text"], ctx["skips"],
                         ctx["word_windows"], ctx["word_probs"],
                         gates_on=False, homograph_on=True)
        dt = time.perf_counter() - t0

        off_by_idx = {d["index"]: d for d in off["diags"]}
        swaps, regs = [], []
        for d_on in on["diags"]:
            d_off = off_by_idx.get(d_on["index"])
            if d_off is None:
                continue
            if d_on["reference_ipa"] != d_off["reference_ipa"]:
                swaps.append({
                    "clip": wav.stem, "word": d_on["word"], "index": d_on["index"],
                    "ref_off": d_off["reference_ipa"], "ref_on": d_on["reference_ipa"],
                    "predicted": d_on["predicted_ipa"],
                    "sub_del_off": d_off["substitutions"] + d_off["deletions"],
                    "sub_del_on": d_on["substitutions"] + d_on["deletions"],
                    "penalty_off": d_off["penalty"], "penalty_on": d_on["penalty"],
                })
            if d_on["penalty"] > d_off["penalty"] + 1e-9:
                regs.append({
                    "clip": wav.stem, "word": d_on["word"], "index": d_on["index"],
                    "penalty_off": d_off["penalty"], "penalty_on": d_on["penalty"],
                    "swapped": d_on["reference_ipa"] != d_off["reference_ipa"],
                })

        acc_off = off["score"].overall_accuracy
        acc_on = on["score"].overall_accuracy
        rows.append({
            "clip": wav.stem,
            "acc_off": acc_off, "acc_on": acc_on,
            "delta": round(acc_on - acc_off, 4),
            "sub_del_off": off["score"].substitution_count
            + off["score"].deletion_count,
            "sub_del_on": on["score"].substitution_count
            + on["score"].deletion_count,
            "n_swaps": len(swaps), "n_regressions": len(regs),
        })
        all_swaps.extend(swaps)
        regressions.extend(regs)
        print(f"[{wav.stem:>2s}] acc {acc_off:.3f}→{acc_on:.3f} "
              f"(Δ{acc_on - acc_off:+.4f}) | sub+del "
              f"{rows[-1]['sub_del_off']}→{rows[-1]['sub_del_on']} | "
              f"swaps={len(swaps)} reg={len(regs)} | {dt:.1f}s")
        for s in swaps:
            print(f"      swap {s['word']!r}: /{s['ref_off']}/ → /{s['ref_on']}/ "
                  f"(nghe /{s['predicted']}/) sub+del {s['sub_del_off']}→"
                  f"{s['sub_del_on']} pen {s['penalty_off']:.2f}→{s['penalty_on']:.2f}")

    mean_delta = sum(r["delta"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nTổng: {len(rows)} clip | mean Δacc = {mean_delta:+.4f} | "
          f"swaps = {len(all_swaps)} | từ penalty tăng = {len(regressions)}")
    if regressions:
        print("REGRESSIONS:")
        for r in regressions:
            print(f"  {r}")

    outdir = REPO_ROOT / "outputs" / "homographs"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "bench_multiref.json").write_text(
        json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clips": rows, "swaps": all_swaps, "regressions": regressions,
            "mean_delta": round(mean_delta, 4),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Outputs: {outdir / 'bench_multiref.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Benchmark A/B s-cluster leniency (TOEIC_PHONEME_S_CLUSTER).

Chạy trên bộ wav + ASR đã cache của scripts/trace_word_case.py
(outputs/case_project/wav + asr): wav2vec 1 lần/clip, scoring 2 lần
(s_cluster OFF vs ON, các flag khác theo config/.env như production), rồi so:

  - overall_accuracy off/on từng clip (delta).
  - Danh sách từ THAY ĐỔI penalty (kỳ vọng: chỉ từ có onset /sp st sk/, chỉ GIẢM).
  - Regression guard: từ nào penalty TĂNG khi bật flag (kỳ vọng: 0 — rule chỉ hạ).

Chạy:  python scripts/bench_s_cluster.py [--outdir outputs/case_project]
Ghi outputs/s_cluster/bench_s_cluster.json.
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

    rows, all_changes, regressions = [], [], []
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
        # Gates (coverage/drift) theo production hiện tại (.env đã bật) — chỉ A/B
        # đúng 1 biến s_cluster để delta quy được cho rule mới.
        gates_on = config.phoneme_coverage_gate_enabled or config.phoneme_drift_cap_enabled
        off = run_scoring(config, segs, posts, asr_data["text"], ctx["skips"],
                          ctx["word_windows"], ctx["word_probs"],
                          gates_on=gates_on,
                          word_windows_locked=ctx["word_windows_locked"],
                          s_cluster_on=False)
        on = run_scoring(config, segs, posts, asr_data["text"], ctx["skips"],
                         ctx["word_windows"], ctx["word_probs"],
                         gates_on=gates_on,
                         word_windows_locked=ctx["word_windows_locked"],
                         s_cluster_on=True)
        dt = time.perf_counter() - t0

        off_by_idx = {d["index"]: d for d in off["diags"]}
        changes, regs = [], []
        for d_on in on["diags"]:
            d_off = off_by_idx.get(d_on["index"])
            if d_off is None:
                continue
            if abs(d_on["penalty"] - d_off["penalty"]) > 1e-9:
                changes.append({
                    "clip": wav.stem, "word": d_on["word"], "index": d_on["index"],
                    "reference_ipa": d_on["reference_ipa"],
                    "predicted": d_on["predicted_ipa"],
                    "sub_del_off": d_off["substitutions"] + d_off["deletions"],
                    "sub_del_on": d_on["substitutions"] + d_on["deletions"],
                    "penalty_off": d_off["penalty"], "penalty_on": d_on["penalty"],
                })
            if d_on["penalty"] > d_off["penalty"] + 1e-9:
                regs.append({
                    "clip": wav.stem, "word": d_on["word"], "index": d_on["index"],
                    "penalty_off": d_off["penalty"], "penalty_on": d_on["penalty"],
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
            "n_changes": len(changes), "n_regressions": len(regs),
        })
        all_changes.extend(changes)
        regressions.extend(regs)
        print(f"[{wav.stem:>2s}] acc {acc_off:.3f}→{acc_on:.3f} "
              f"(Δ{acc_on - acc_off:+.4f}) | sub+del "
              f"{rows[-1]['sub_del_off']}→{rows[-1]['sub_del_on']} | "
              f"changes={len(changes)} reg={len(regs)} | {dt:.1f}s")
        for c in changes:
            print(f"      {c['word']!r}: /{c['reference_ipa']}/ nghe "
                  f"/{c['predicted']}/ sub+del {c['sub_del_off']}→{c['sub_del_on']} "
                  f"pen {c['penalty_off']:.2f}→{c['penalty_on']:.2f}")

    mean_delta = sum(r["delta"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nTổng: {len(rows)} clip | mean Δacc = {mean_delta:+.4f} | "
          f"từ đổi penalty = {len(all_changes)} | từ penalty tăng = {len(regressions)}")
    if regressions:
        print("REGRESSIONS:")
        for r in regressions:
            print(f"  {r}")

    outdir = REPO_ROOT / "outputs" / "s_cluster"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "bench_s_cluster.json").write_text(
        json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clips": rows, "changes": all_changes, "regressions": regressions,
            "mean_delta": round(mean_delta, 4),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Outputs: {outdir / 'bench_s_cluster.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

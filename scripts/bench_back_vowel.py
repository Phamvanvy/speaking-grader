#!/usr/bin/env python3
"""Benchmark A/B tách back-vowel (TOEIC_PHONEME_BACK_VOWEL_SPLIT).

Khác bench_s_cluster/bench_multiref (toggle tham số scoring trong 1 process):
flag này là IMPORT-TIME (normalize_ipa + bảng similarity build lúc import, có
lru_cache) nên phải A/B bằng 2 SUBPROCESS — script tự spawn chính nó 2 lần với
env 0/1 (mode --worker), mỗi worker chấm cả bộ clip rồi dump JSON; process cha
so sánh:

  - overall_accuracy off/on từng clip (delta).
  - Từ THAY ĐỔI penalty, tách 2 nhóm:
      * detections MỚI liên quan ɑ↔ɔ (chủ đích của flag — cần eyeball xem là
        lỗi thật hay giọng Mỹ cot-caught merged);
      * regression KHÔNG liên quan ɑ↔ɔ (kỳ vọng: 0 — flag không được lan).

Corpus: outputs/case_project (12 câu learner TOEIC test 1, wav + ASR cache của
scripts/trace_word_case.py) — cùng bộ với bench_multiref/bench_s_cluster.

Chạy:  python scripts/bench_back_vowel.py [--outdir outputs/case_project]
Ghi outputs/back_vowel/bench_back_vowel.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FLAG = "TOEIC_PHONEME_BACK_VOWEL_SPLIT"


# ──────────────────────────────────────────────────────────────────────────────
# Worker: chấm cả bộ clip với flag theo env hiện tại → dump JSON
# ──────────────────────────────────────────────────────────────────────────────

def worker(outdir: str, dump_path: str) -> int:
    from bench_common import build_reference_context, run_scoring

    from src.config import load_config
    from src.phoneme.ipa.phoneme_set import BACK_VOWEL_SPLIT_ENABLED
    from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor

    config = load_config()
    cache = REPO_ROOT / outdir
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
    gates_on = config.phoneme_coverage_gate_enabled or config.phoneme_drift_cap_enabled

    clips: dict[str, dict] = {}
    for wav in wavs:
        asr_path = cache / "asr" / f"{wav.stem}.json"
        if not asr_path.exists():
            print(f"[!] thiếu ASR cache {asr_path.name} — bỏ qua")
            continue
        asr_data = json.loads(asr_path.read_text(encoding="utf-8"))
        ctx = build_reference_context(config, asr_data)
        segs, warn, posts = predictor.predict_with_posteriors(str(wav))
        if warn:
            print(f"[!] {wav.name}: wav2vec warning {warn} — bỏ qua")
            continue
        res = run_scoring(config, segs, posts, asr_data["text"], ctx["skips"],
                          ctx["word_windows"], ctx["word_probs"],
                          gates_on=gates_on,
                          word_windows_locked=ctx["word_windows_locked"])
        words = {}
        for d in res["diags"]:
            words[str(d["index"])] = {
                "word": d["word"], "penalty": d["penalty"],
                "reference_ipa": d["reference_ipa"],
                "predicted_ipa": d["predicted_ipa"],
                "substitutions": d["substitutions"], "deletions": d["deletions"],
                # cặp (ref, heard) của từng sub — để parent quy delta cho ɑ↔ɔ
                "sub_pairs": [
                    [c["ref_symbol"], c.get("pred_symbol") or ""]
                    for c in d["correspondences"] if c["status"] == "sub"
                ],
            }
        clips[wav.stem] = {
            "accuracy": res["score"].overall_accuracy,
            "sub_del": res["score"].substitution_count + res["score"].deletion_count,
            "words": words,
        }
        print(f"  [{wav.stem:>2s}] acc={clips[wav.stem]['accuracy']:.4f} "
              f"sub+del={clips[wav.stem]['sub_del']}")

    Path(dump_path).write_text(
        json.dumps({"flag_on": BACK_VOWEL_SPLIT_ENABLED, "clips": clips},
                   ensure_ascii=False), encoding="utf-8")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Parent: chạy worker 2 lần (flag 0/1) rồi diff
# ──────────────────────────────────────────────────────────────────────────────

def _is_back_vowel_pair(ref: str, heard: str) -> bool:
    """Cặp sub có phải ɑ↔ɔ (so trên symbol thô, bỏ ː) — nhóm detection chủ đích."""
    strip = {"ɑː": "ɑ", "ɔː": "ɔ"}
    r = strip.get(ref, ref.replace("ː", ""))
    h = strip.get(heard, heard.replace("ː", ""))
    return {r, h} == {"ɑ", "ɔ"} or {r, h} == {"ɑ", "ɒ"} or {r, h} == {"ɑ", "o"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/case_project")
    ap.add_argument("--worker", metavar="DUMP_PATH", default=None,
                    help="(nội bộ) chạy 1 lượt chấm với flag theo env, dump JSON")
    args = ap.parse_args()

    if args.worker:
        return worker(args.outdir, args.worker)

    bench_dir = REPO_ROOT / "outputs" / "back_vowel"
    bench_dir.mkdir(parents=True, exist_ok=True)
    dumps: dict[str, Path] = {}
    for mode, flag_val in (("off", "0"), ("on", "1")):
        dump = bench_dir / f"run_{mode}.json"
        env = dict(os.environ)
        env[FLAG] = flag_val
        env["PYTHONIOENCODING"] = "utf-8"
        print(f"── worker flag={mode} ──")
        rc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--outdir", args.outdir,
             "--worker", str(dump)],
            env=env, cwd=str(REPO_ROOT),
        ).returncode
        if rc != 0:
            print(f"[!] worker {mode} lỗi (rc={rc})")
            return rc
        dumps[mode] = dump

    off = json.loads(dumps["off"].read_text(encoding="utf-8"))
    on = json.loads(dumps["on"].read_text(encoding="utf-8"))
    assert off["flag_on"] is False and on["flag_on"] is True, "env flag không ăn vào worker"

    rows, target_changes, side_effects = [], [], []
    for clip, c_off in off["clips"].items():
        c_on = on["clips"].get(clip)
        if c_on is None:
            continue
        for idx, w_on in c_on["words"].items():
            w_off = c_off["words"].get(idx)
            if w_off is None or abs(w_on["penalty"] - w_off["penalty"]) <= 1e-9:
                continue
            new_pairs = [p for p in w_on["sub_pairs"] if p not in w_off["sub_pairs"]]
            entry = {
                "clip": clip, "word": w_on["word"], "index": int(idx),
                "reference_ipa": w_on["reference_ipa"],
                "predicted": w_on["predicted_ipa"],
                "penalty_off": w_off["penalty"], "penalty_on": w_on["penalty"],
                "new_sub_pairs": new_pairs,
            }
            # Delta quy được cho ɑ↔ɔ (chủ đích) hay không (side effect — kỳ vọng 0)?
            if new_pairs and all(_is_back_vowel_pair(r, h) for r, h in new_pairs):
                target_changes.append(entry)
            else:
                side_effects.append(entry)
        rows.append({
            "clip": clip,
            "acc_off": c_off["accuracy"], "acc_on": c_on["accuracy"],
            "delta": round(c_on["accuracy"] - c_off["accuracy"], 4),
            "sub_del_off": c_off["sub_del"], "sub_del_on": c_on["sub_del"],
        })

    print(f"\n{'clip':>4s} {'acc_off':>8s} {'acc_on':>8s} {'delta':>8s} {'sub+del':>9s}")
    for r in rows:
        print(f"{r['clip']:>4s} {r['acc_off']:8.4f} {r['acc_on']:8.4f} "
              f"{r['delta']:+8.4f} {r['sub_del_off']:>4d}→{r['sub_del_on']:<4d}")
    mean_delta = sum(r["delta"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nTổng {len(rows)} clip | mean Δacc = {mean_delta:+.4f}")
    print(f"Detection ɑ↔ɔ mới (chủ đích — cần eyeball): {len(target_changes)}")
    for c in target_changes:
        print(f"  {c['clip']}/{c['word']!r}: /{c['reference_ipa']}/ nghe "
              f"/{c['predicted']}/ pen {c['penalty_off']:.2f}→{c['penalty_on']:.2f} "
              f"pairs={c['new_sub_pairs']}")
    print(f"Side effect KHÔNG liên quan ɑ↔ɔ (kỳ vọng 0): {len(side_effects)}")
    for c in side_effects:
        print(f"  {c['clip']}/{c['word']!r}: /{c['reference_ipa']}/ nghe "
              f"/{c['predicted']}/ pen {c['penalty_off']:.2f}→{c['penalty_on']:.2f} "
              f"pairs={c['new_sub_pairs']}")

    (bench_dir / "bench_back_vowel.json").write_text(
        json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clips": rows, "mean_delta": round(mean_delta, 4),
            "target_changes": target_changes, "side_effects": side_effects,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Outputs: {bench_dir / 'bench_back_vowel.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

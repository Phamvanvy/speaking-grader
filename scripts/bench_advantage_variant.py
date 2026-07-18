#!/usr/bin/env python3
"""Bench A/B biến thể "advantage" æ↔ə (_WORD_REDUCIBLE_VARIANT_PAIRS).

Khác bench_back_vowel (flag import-time → phải 2 subprocess): bảng biến thể được
phonemes_match đọc lúc GỌI và phonemes_match không có lru_cache → A/B trong 1
process bằng cách tạm RỖNG bảng (bảng rỗng ≡ hành vi code cũ) rồi khôi phục.
Mỗi clip wav2vec chạy 1 lần, scoring chấm 2 lượt trên cùng segs/posts → diff
tất định thuần score-path.

Nhóm kết quả:
  - target: từ trong họ advantage có penalty GIẢM (chủ đích — học viên đọc /əd-/).
  - side_effects: MỌI thay đổi ở từ ngoài họ advantage (kỳ vọng 0 — guard word-scoped).

Corpus: outputs/case_project (12 câu learner TOEIC test 1, wav + ASR cache) —
cùng bộ với bench_multiref/bench_s_cluster/bench_back_vowel.

Chạy:  python scripts/bench_advantage_variant.py [--outdir outputs/case_project]
Ghi outputs/advantage_variant/bench_advantage_variant.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TARGET_WORDS = {"advantage", "advantages", "advantaged"}


def _word_key(word: str) -> str:
    return word.lower().strip(".,;:!?\"'()[]{}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/case_project")
    args = ap.parse_args()

    from bench_common import build_reference_context, run_scoring

    import src.phoneme.ipa.similarity as similarity
    from src.config import load_config
    from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor

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
    gates_on = config.phoneme_coverage_gate_enabled or config.phoneme_drift_cap_enabled

    table = similarity._WORD_REDUCIBLE_VARIANT_PAIRS
    saved = dict(table)

    def score_words(segs, posts, asr_data, ctx) -> tuple[float, int, dict]:
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
                "sub_pairs": [
                    [c["ref_symbol"], c.get("pred_symbol") or ""]
                    for c in d["correspondences"] if c["status"] == "sub"
                ],
            }
        score = res["score"]
        return (score.overall_accuracy,
                score.substitution_count + score.deletion_count, words)

    rows, target_changes, side_effects = [], [], []
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

        table.clear()                       # bảng rỗng ≡ code cũ (baseline)
        try:
            acc_off, sd_off, words_off = score_words(segs, posts, asr_data, ctx)
        finally:
            table.update(saved)             # khôi phục → hành vi mới
        acc_on, sd_on, words_on = score_words(segs, posts, asr_data, ctx)

        for idx, w_on in words_on.items():
            w_off = words_off.get(idx)
            if w_off is None or abs(w_on["penalty"] - w_off["penalty"]) <= 1e-9:
                continue
            entry = {
                "clip": wav.stem, "word": w_on["word"], "index": int(idx),
                "reference_ipa": w_on["reference_ipa"],
                "predicted": w_on["predicted_ipa"],
                "penalty_off": w_off["penalty"], "penalty_on": w_on["penalty"],
                "gone_sub_pairs": [p for p in w_off["sub_pairs"]
                                   if p not in w_on["sub_pairs"]],
            }
            if (_word_key(w_on["word"]) in _TARGET_WORDS
                    and w_on["penalty"] < w_off["penalty"]):
                target_changes.append(entry)
            else:
                side_effects.append(entry)
        rows.append({
            "clip": wav.stem, "acc_off": acc_off, "acc_on": acc_on,
            "delta": round(acc_on - acc_off, 4),
            "sub_del_off": sd_off, "sub_del_on": sd_on,
        })
        print(f"  [{wav.stem:>2s}] acc {acc_off:.4f}→{acc_on:.4f} "
              f"sub+del {sd_off}→{sd_on}")

    print(f"\n{'clip':>4s} {'acc_off':>8s} {'acc_on':>8s} {'delta':>8s} {'sub+del':>9s}")
    for r in rows:
        print(f"{r['clip']:>4s} {r['acc_off']:8.4f} {r['acc_on']:8.4f} "
              f"{r['delta']:+8.4f} {r['sub_del_off']:>4d}→{r['sub_del_on']:<4d}")
    mean_delta = sum(r["delta"] for r in rows) / len(rows) if rows else 0.0
    print(f"\nTổng {len(rows)} clip | mean Δacc = {mean_delta:+.4f}")
    print(f"Cải thiện họ advantage (chủ đích): {len(target_changes)}")
    for c in target_changes:
        print(f"  {c['clip']}/{c['word']!r}: /{c['reference_ipa']}/ nghe "
              f"/{c['predicted']}/ pen {c['penalty_off']:.2f}→{c['penalty_on']:.2f} "
              f"hết-sub={c['gone_sub_pairs']}")
    print(f"Side effect ngoài họ advantage (kỳ vọng 0): {len(side_effects)}")
    for c in side_effects:
        print(f"  {c['clip']}/{c['word']!r}: /{c['reference_ipa']}/ nghe "
              f"/{c['predicted']}/ pen {c['penalty_off']:.2f}→{c['penalty_on']:.2f}")

    bench_dir = REPO_ROOT / "outputs" / "advantage_variant"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "bench_advantage_variant.json").write_text(
        json.dumps({
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clips": rows, "mean_delta": round(mean_delta, 4),
            "target_changes": target_changes, "side_effects": side_effects,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Outputs: {bench_dir / 'bench_advantage_variant.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

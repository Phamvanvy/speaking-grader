#!/usr/bin/env python3
"""Thống kê residual phoneme substitutions SAU multi-reference (homograph ON).

Mục tiêu: trước khi thiết kế policy pronunciation-variant, đo trên TOÀN BỘ regression
set hiện có (outputs/case_project/wav+asr — 12 câu trả lời full TOEIC test 1, cache của
scripts/trace_word_case.py) xem các substitution CÒN LẠI (sau khi multi-ref homograph
selection đã bật) rơi vào nhóm nào:

  - "recognizer": is_real_error_substitution(...) == False → khả năng cao là
    wav2vec hallucination/artifact, KHÔNG phải lỗi phát âm.
  - "real_error_known": plausible + nằm trong _REAL_ERROR_SUBS (bảng lỗi L1-VN đã biết,
    vd th-stopping ð→d) → lỗi đọc thật đã có bằng chứng.
  - "variant_candidate": plausible + nằm trong _NEAR_PAIRS (bảng near-pair dùng cho
    phoneme_similarity nhưng CHƯA có trong phonemes_match tolerance) → ứng viên biến thể
    phát âm hợp lệ (accent) chưa được whitelist.
  - "uncertain_plausible": plausible nhưng không khớp 2 bảng trên (chỉ same-class/place
    heuristic) → cần xem tay, có thể là lỗi thật hoặc biến thể chưa biết.

DIAGNOSTIC ONLY — không sửa điểm, không đổi production. Chạy:

    python scripts/analyze_residual_subs.py [--outdir outputs/case_project] [--top N]

Ghi outputs/homographs/residual_subs.json + in top N cặp ra console.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from bench_common import REPO_ROOT, build_reference_context, run_scoring

from src.config import load_config  # noqa: E402
from src.phoneme.ipa.phoneme_set import normalize_ipa  # noqa: E402
from src.phoneme.ipa.similarity import (  # noqa: E402
    _NEAR_PAIRS,
    _REAL_ERROR_SUBS,
    is_real_error_substitution,
)
from src.phoneme.scoring import PHONEME_RECOGNIZER_NOISE_SIM  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

DELETED = "∅"


def classify(ref: str, pred: str) -> str:
    if not is_real_error_substitution(ref, pred, sim_floor=PHONEME_RECOGNIZER_NOISE_SIM):
        return "recognizer"
    pair = frozenset({normalize_ipa(ref), normalize_ipa(pred)})
    if pair in _REAL_ERROR_SUBS:
        return "real_error_known"
    if pair in _NEAR_PAIRS:
        return "variant_candidate"
    return "uncertain_plausible"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/case_project",
                    help="thư mục có wav/ + asr/ cache từ trace_word_case.py")
    ap.add_argument("--top", type=int, default=50, help="số cặp top hiển thị")
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

    all_subs: list[dict] = []
    n_clips = 0
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
        result = run_scoring(
            config, segs, posts, asr_data["text"], ctx["skips"],
            ctx["word_windows"], ctx["word_probs"],
            gates_on=False, homograph_on=True,
        )
        n_clips += 1
        for d in result["diags"]:
            for c in d["correspondences"]:
                if c["status"] != "sub" or c.get("pred_symbol") is None:
                    continue
                ref, pred = c["ref_symbol"], c["pred_symbol"]
                all_subs.append({
                    "clip": wav.stem, "word": d["word"], "index": d["index"],
                    "ref": ref, "pred": pred,
                    "confidence": c.get("confidence"),
                    "category": classify(ref, pred),
                })
        print(f"[{wav.stem:>2s}] {wav.name}: {sum(1 for s in all_subs if s['clip'] == wav.stem)} subs")

    if not all_subs:
        print("Không có substitution nào — không có gì để phân tích.")
        return 0

    pair_counts: collections.Counter = collections.Counter(
        (s["ref"], s["pred"]) for s in all_subs)
    pair_examples: dict = collections.defaultdict(list)
    pair_category: dict = {}
    for s in all_subs:
        key = (s["ref"], s["pred"])
        pair_category[key] = s["category"]
        if len(pair_examples[key]) < 5:
            pair_examples[key].append(f"{s['clip']}:{s['word']}")

    top_pairs = pair_counts.most_common(args.top)
    rows = []
    for (ref, pred), cnt in top_pairs:
        rows.append({
            "ref": ref, "pred": pred, "count": cnt,
            "category": pair_category[(ref, pred)],
            "examples": pair_examples[(ref, pred)],
        })

    cat_totals: collections.Counter = collections.Counter(s["category"] for s in all_subs)
    total = len(all_subs)

    print(f"\nTổng: {n_clips} clip | {total} substitution | {len(pair_counts)} cặp distinct")
    print("\nPhân loại (toàn bộ substitution, không chỉ top N):")
    for cat, cnt in cat_totals.most_common():
        print(f"  {cat:<20} {cnt:>4}  ({100*cnt/total:.1f}%)")

    print(f"\nTop {len(rows)} cặp (ref → pred):")
    for r in rows:
        ex = ", ".join(r["examples"])
        print(f"  {r['ref']} → {r['pred']}: {r['count']:>3}  [{r['category']}]  vd: {ex}")

    outdir = REPO_ROOT / "outputs" / "homographs"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "residual_subs.json"
    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_clips": n_clips, "total_substitutions": total,
        "distinct_pairs": len(pair_counts),
        "category_totals": dict(cat_totals),
        "top_pairs": rows,
        "all_substitutions": all_subs,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nOutputs: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

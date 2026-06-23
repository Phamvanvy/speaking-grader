#!/usr/bin/env python3
"""PR3-0 drift-vs-hallucination report over phoneme telemetry JSONL.

Đọc `outputs/phoneme_telemetry.jsonl` (schema_version >= 2) do scorer phát ra khi
TOEIC_PHONEME_TELEMETRY=1, rồi tổng hợp:

  - drift_fraction toàn corpus = sub NGOÀI window / tổng sub ĐÃ PHÂN LOẠI.
    (sub TRONG window = hallucination; sub NGOÀI = drift attribution.)
  - deletion share — deletion KHÔNG BAO GIỜ là drift (không có predicted phoneme để
    quy thời gian) → trần lý thuyết của drift = 1 - deletion_share.
  - per-named-word breakdown cho các ca PR3 quan tâm.
  - so với ngưỡng kill (mặc định 0.40): < ngưỡng → giả thuyết drift bị bác → pivot
    sang hallucination (theo RFC PR3-0).

DIAGNOSTIC ONLY. Không sửa điểm. Chạy:

    python scripts/pr3_drift_report.py outputs/phoneme_telemetry.jsonl --kill 0.40
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
from collections import defaultdict

NAMED_CASES = ("traditional", "vietnam", "vietnamese", "folktales", "blood")


def load(path: str) -> tuple[list[dict], list[dict]]:
    words, summaries = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (summaries if r.get("type") == "summary" else words).append(r)
    return [w for w in words if w.get("type") == "word"], summaries


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="phoneme telemetry JSONL (schema >= 2)")
    ap.add_argument("--kill", type=float, default=0.40,
                    help="kill threshold cho drift_fraction (mặc định 0.40)")
    args = ap.parse_args()

    words, summaries = load(args.jsonl)
    if not words:
        print("Không có word record nào.")
        return 1

    schema = words[0].get("schema_version")
    sub_in = sum(w.get("sub_inside_window", 0) for w in words)
    sub_out = sum(w.get("sub_outside_window", 0) for w in words)
    classified = sub_in + sub_out
    with_window = sum(1 for w in words if w.get("window_start") is not None)

    tot_sub = sum(w["substitutions"] for w in words)
    tot_del = sum(w["deletions"] for w in words)
    tot_match = sum(w["matches"] for w in words)
    del_share = tot_del / (tot_sub + tot_del) if (tot_sub + tot_del) else 0.0

    print(f"file: {args.jsonl}  schema_version={schema}")
    print(f"utterances={len(summaries)}  word_rows={len(words)}  "
          f"words_with_window={with_window} ({with_window / len(words):.0%})")
    print(f"matches={tot_match}  substitutions={tot_sub}  deletions={tot_del}")
    print(f"deletion share of errors = {del_share:.1%}  "
          f"→ drift trần lý thuyết ≤ {1 - del_share:.1%} (deletion không thể là drift)")
    print("-" * 64)

    if classified == 0:
        print("CHƯA PHÂN LOẠI ĐƯỢC substitution nào (không có window).")
        print("→ Cần chạy lại với word_windows (telemetry bật + Whisper word timestamps).")
        return 2

    drift = sub_out / classified
    print(f"sub TRONG window (hallucination) = {sub_in}")
    print(f"sub NGOÀI window (drift)         = {sub_out}")
    print(f"substitutions đã phân loại        = {classified} / {tot_sub}")
    print(f"DRIFT FRACTION = {drift:.1%}")
    verdict = ("DRIFT chiếm ưu thế → tiếp tục PR3 (window-constrained alignment)"
               if drift >= args.kill else
               "DRIFT KHÔNG chiếm ưu thế → KILL PR3, pivot sang hallucination")
    print(f"ngưỡng kill = {args.kill:.0%}  ⇒  {verdict}")
    print("-" * 64)

    # Per-named-word breakdown.
    by_word: dict[str, list[dict]] = defaultdict(list)
    for w in words:
        by_word[w["word"].lower()].append(w)
    print("named cases (PR3):")
    for name in NAMED_CASES:
        ws = by_word.get(name, [])
        if not ws:
            continue
        i = sum(x.get("sub_inside_window", 0) for x in ws)
        o = sum(x.get("sub_outside_window", 0) for x in ws)
        d = sum(x["deletions"] for x in ws)
        ex = ws[0]
        print(f"  {name:12s} n={len(ws)}  sub_in={i} sub_out={o} del={d}  "
              f"ref={ex['reference_ipa']!r} pred={ex['predicted_ipa']!r}")

    covs = [w["coverage"] for w in words]
    print("-" * 64)
    print(f"coverage mean={st.mean(covs):.2f} median={st.median(covs):.2f} "
          f"frac<0.8={sum(c < 0.8 for c in covs) / len(covs):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

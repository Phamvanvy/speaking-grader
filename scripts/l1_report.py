#!/usr/bin/env python3
"""L1-aware scoring acceptance report over phoneme telemetry JSONL (schema >= 4).

Đọc telemetry do scorer phát ra khi TOEIC_PHONEME_TELEMETRY=1 (và thường
TOEIC_PHONEME_L1_ENABLED=1), tổng hợp từ correspondences[]:

  - đếm theo penalty_reason (hard_error / l1_final_deletion / low_confidence_neutralized).
  - L1 final-deletion: số lần kích hoạt, theo rule_id, và avg penalty reduction = mean(1 -
    penalty_adjustment) → kiểm acceptance "≥ 30%" (PRD §12).
  - low-confidence neutralization: số sub bị trung hoà.
  - (tùy chọn) so với baseline file (L1 tắt) để thấy delta.

DIAGNOSTIC ONLY. Chạy:

    python scripts/l1_report.py outputs/phoneme_telemetry.jsonl
    python scripts/l1_report.py l1_on.jsonl --baseline l1_off.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
import sys
from collections import Counter


def load_correspondences(path: str) -> list[dict]:
    cs: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") != "word":
                continue
            cs.extend(obj.get("correspondences", []))
    return cs


def summarize(cs: list[dict]) -> dict:
    reasons = Counter(c.get("penalty_reason") for c in cs)
    l1 = [c for c in cs if c.get("penalty_reason") == "l1_final_deletion"]
    neut = [c for c in cs if c.get("penalty_reason") == "low_confidence_neutralized"]
    dels = [c for c in cs if c.get("status") == "del"]
    l1_reductions = [1.0 - c.get("penalty_adjustment", 1.0) for c in l1]
    return {
        "n": len(cs),
        "reasons": reasons,
        "l1": l1,
        "neut": neut,
        "dels": dels,
        "rule_counts": Counter(c.get("l1_rule_id") for c in l1),
        "avg_l1_reduction": st.mean(l1_reductions) if l1_reductions else 0.0,
    }


def print_report(tag: str, s: dict) -> None:
    print(f"=== {tag} ===")
    print(f"correspondences: {s['n']}")
    print(f"by penalty_reason: {dict(s['reasons'])}")
    nd = len(s["dels"])
    print(f"deletions: {nd}  | L1-tolerated final deletions: {len(s['l1'])} "
          f"({len(s['l1']) / nd:.0%} of deletions)" if nd else "deletions: 0")
    print(f"avg penalty reduction on L1 final deletions: {s['avg_l1_reduction']:.1%} "
          f"(acceptance ≥ 30% → {'PASS' if s['avg_l1_reduction'] >= 0.30 else 'FAIL'})")
    print(f"low-confidence neutralized subs: {len(s['neut'])}")
    if s["rule_counts"]:
        top = ", ".join(f"{r}={n}" for r, n in s["rule_counts"].most_common(10))
        print(f"top L1 rules fired: {top}")


def main() -> int:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="L1-on telemetry JSONL (schema >= 4)")
    ap.add_argument("--baseline", help="optional L1-off telemetry to diff against")
    args = ap.parse_args()

    cs = load_correspondences(args.jsonl)
    if not cs:
        print("Không có correspondence nào (cần schema >= 4 + telemetry bật).")
        return 1
    if not any("penalty_reason" in c for c in cs):
        print("Telemetry thiếu penalty_reason → schema < 4. Chạy lại với scorer mới.")
        return 2

    s = summarize(cs)
    print_report(args.jsonl, s)

    if args.baseline:
        b = summarize(load_correspondences(args.baseline))
        print()
        print_report(f"{args.baseline} (baseline)", b)
        print("\n=== delta (L1 on − off) ===")
        print(f"hard_error: {s['reasons']['hard_error'] - b['reasons']['hard_error']:+d}")
        print(f"l1_final_deletion: +{len(s['l1'])} (baseline {len(b['l1'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

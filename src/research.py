#!/usr/bin/env python3
import json
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# LOAD TELEMETRY
# -----------------------------
def load_records(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "summary":
                    continue

                words = obj.get("words", []) if "words" in obj else [obj]

                for w in words:
                    if "correspondences" not in w:
                        continue

                    for c in w["correspondences"]:
                        rows.append({
                            "ref": c.get("ref_symbol"),
                            "pred": c.get("pred_symbol"),
                            "conf": c.get("confidence"),
                            "status": c.get("status"),
                            "outside": c.get("sub_outside_window", False),
                            "is_final": c.get("is_final", False)
                        })

            except Exception:
                continue

    return pd.DataFrame(rows)


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def build_features(df: pd.DataFrame):
    df = df.copy()

    # Dùng `status` của scorer (ok/sub/del/skipped) — KHÔNG so ref==pred (sai với
    # allophone/vowel-reduction được scorer chấm "ok" dù symbol khác nhau).
    df["is_correct"] = df["status"] == "ok"
    df["is_deletion"] = df["status"] == "del"
    # Lỗi để phân tích R-vs-S = sub/del NẰM TRONG window (loại drift = sub ngoài window).
    df["is_error"] = df["status"].isin(["sub", "del"]) & (~df["outside"])

    # final consonant proxy: âm cuối từ bị xoá (lean S/L1 nếu chiếm đa số).
    df["final_deletion"] = df["is_deletion"] & df["is_final"]

    return df


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(df):
    correct = df[df["is_correct"] & df["conf"].notna()]
    error = df[df["is_error"] & df["conf"].notna()]

    def mean_ci(x):
        if len(x) == 0:
            return 0, (0, 0)

        arr = np.array(x)
        mean = arr.mean()
        ci = 1.96 * arr.std() / math.sqrt(len(arr))
        return mean, (mean - ci, mean + ci)

    c_mean, c_ci = mean_ci(correct["conf"])
    e_mean, e_ci = mean_ci(error["conf"])

    return {
        "correct_mean": c_mean,
        "error_mean": e_mean,
        "gap": c_mean - e_mean,
        "correct_ci": c_ci,
        "error_ci": e_ci
    }


# -----------------------------
# CLASSIFICATION SIGNALS
# -----------------------------
def classify(df, metrics):
    gap = metrics["gap"]

    deletion_rate = df["final_deletion"].sum() / max(df["is_deletion"].sum(), 1)

    if gap > 0.25 and deletion_rate < 0.5:
        return "R_LIKELY"
    elif deletion_rate >= 0.5 and gap < 0.25:
        return "S_LIKELY"
    else:
        return "UNKNOWN"


# -----------------------------
# VISUALIZATION
# -----------------------------
def plot_dashboard(df, metrics):
    plt.figure()

    # confidence distribution
    df["conf"].dropna().hist(bins=30)
    plt.title("Confidence Distribution")
    plt.show()

    # correct vs error
    correct = df[df["is_correct"]]["conf"].dropna()
    error = df[df["is_error"]]["conf"].dropna()

    plt.figure()
    plt.hist(correct, bins=30, alpha=0.6, label="correct")
    plt.hist(error, bins=30, alpha=0.6, label="error")
    plt.legend()
    plt.title("Confidence Separation")
    plt.show()


# -----------------------------
# EXPORT UNKNOWN CASES
# -----------------------------
def export_unknown(df, out="unknown_cases.jsonl"):
    unknown = df[df["is_error"] & df["conf"].notna()]

    with open(out, "w", encoding="utf-8") as f:
        for _, r in unknown.iterrows():
            f.write(json.dumps(r.to_dict()) + "\n")


# -----------------------------
# MAIN DASHBOARD
# -----------------------------
def run(file_path="tel2.jsonl"):
    df = load_records(file_path)
    df = build_features(df)

    metrics = compute_metrics(df)
    verdict = classify(df, metrics)

    print("\n==============================")
    print("📊 RESEARCH DASHBOARD v1")
    print("==============================")
    print(f"Correct mean conf: {metrics['correct_mean']:.4f}")
    print(f"Error mean conf:   {metrics['error_mean']:.4f}")
    print(f"Gap:               {metrics['gap']:.4f}")
    print(f"Decision:          {verdict}")
    print("==============================\n")

    export_unknown(df)

    plot_dashboard(df, metrics)


if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "tel2.jsonl"
    run(file)
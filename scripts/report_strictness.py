#!/usr/bin/env python3
"""Báo cáo độ chặt chấm phoneme — DIAGNOSTIC ONLY, không đổi điểm.

Mine outputs/phoneme_telemetry.jsonl (per-word, per-correspondence) để trả lời
"chấm có đang quá chặt không?":

  1. Phân bố status/severity-ước-tính trên toàn bộ âm ĐƯỢC CHẤM (không skip).
  2. Riêng nhóm band-9 (utterance_id bắt đầu "band9"): % âm bị medium/high —
     speaker gần bản xứ mà tỉ lệ này > ~3% nghĩa là pipeline đang tô đỏ oan.
  3. Top cặp sub medium/high lặp lại nhiều nhất (ứng viên rule/variant mới).
  4. Share của s-cluster (/p t k/ sau /s/ đầu từ) trong nhóm high — đo tác động
     kỳ vọng của TOEIC_PHONEME_S_CLUSTER.
  5. Cụm sub rơi ĐÚNG penalty 0.6 (sim 0.4, ranh giới ">= 0.6 = high") — nhạy
     với lựa chọn ngưỡng, đổi 1 nấc similarity là đổi màu label.

Severity ở đây là ƯỚC TÍNH lại từ (1 − phoneme_similarity) × min(1, conf/knee),
neutralize khi correspondence có penalty_reason đã biết — đúng công thức
production cho sub thường; các gate word-level (coverage/drift) không mô phỏng
lại nên con số là CHẶN TRÊN của độ chặt thực tế.

Chạy:  python scripts/report_strictness.py [--telemetry outputs/phoneme_telemetry.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.phoneme.ipa import normalize_ipa, phoneme_similarity  # noqa: E402
from src.phoneme.scoring.constants import PHONEME_CONFIDENCE_KNEE  # noqa: E402

# penalty_reason đã trung hoà/cap penalty → KHÔNG tính là lỗi hiển thị đỏ.
_NEUTRALIZED_REASONS = {
    "recognizer_noise", "low_confidence_neutralized", "accent_variant",
    "linking_variant", "connected_speech", "coverage_collapse",
    "s_cluster_variant",
}
_CAPPED_LOW_REASONS = {"g2p_uncertain", "drift_suspected", "s_cluster_unaspirated"}

_S_CLUSTER_STOPS = {"p", "t", "k"}


def _severity(penalty: float) -> str:
    if penalty >= 0.6:
        return "high"
    if penalty >= 0.3:
        return "medium"
    return "low"


def _estimate(corr: dict, knee: float) -> tuple[str, float]:
    """(severity, penalty ước tính) cho 1 correspondence sub/del."""
    reason = corr.get("penalty_reason")
    if reason in _NEUTRALIZED_REASONS:
        return "low", 0.0
    if reason in _CAPPED_LOW_REASONS:
        return "low", 0.2
    if corr["status"] == "del":
        # Deletion penalty phụ thuộc onset/coda (không có trong telemetry corr) —
        # dùng adjustment đã ghi nếu có, mặc định medium 0.5.
        adj = corr.get("penalty_adjustment")
        pen = 0.5 * (adj if isinstance(adj, (int, float)) else 1.0)
        return _severity(pen), pen
    sim = phoneme_similarity(corr.get("pred_symbol") or "", corr["ref_symbol"])
    conf = float(corr.get("confidence") or 1.0)
    pen = (1.0 - sim) * (min(1.0, conf / knee) if knee > 0 else 1.0)
    return _severity(pen), pen


def _is_s_cluster(word_rec: dict, idx: int) -> bool:
    """corr[idx] là /p t k/ ngay sau /s/ ở ĐẦU chuỗi reference của từ."""
    corrs = word_rec["correspondences"]
    if idx == 0:
        return False
    ref_now = normalize_ipa(corrs[idx]["ref_symbol"])
    ref_prev = normalize_ipa(corrs[idx - 1]["ref_symbol"])
    # onset đầu từ: mọi corr trước idx-1 cũng phải là phụ âm đầu — xấp xỉ bằng
    # điều kiện idx-1 == 0 (cụm /sC/ đứng đầu từ, đúng cho speak/stay/school).
    return ref_now in _S_CLUSTER_STOPS and ref_prev == "s" and idx - 1 == 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--telemetry", default="outputs/phoneme_telemetry.jsonl")
    args = ap.parse_args()

    path = REPO_ROOT / args.telemetry
    if not path.exists():
        print(f"[!] Không thấy {path}")
        return 1

    knee = PHONEME_CONFIDENCE_KNEE
    total = Counter()          # status toàn cục
    sev_all = Counter()        # severity ước tính (sub+del) toàn cục
    sev_band9 = Counter()
    scored_all = scored_band9 = 0
    pair_high: Counter = Counter()   # cặp sub medium/high
    s_cluster_high = 0
    high_total = 0
    boundary_06 = 0            # sub có penalty ước tính đúng [0.58, 0.62]

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") != "word" or rec.get("skip_reason"):
            continue
        band9 = str(rec.get("utterance_id") or "").startswith("band9")
        for i, c in enumerate(rec["correspondences"]):
            st = c["status"]
            if st == "skipped":
                continue
            total[st] += 1
            scored_all += 1
            if band9:
                scored_band9 += 1
            if st not in ("sub", "del"):
                continue
            sev, pen = _estimate(c, knee)
            sev_all[sev] += 1
            if band9:
                sev_band9[sev] += 1
            if st == "sub" and sev in ("medium", "high"):
                pair_high[f"{c['ref_symbol']}→{c.get('pred_symbol')}"] += 1
            if sev == "high":
                high_total += 1
                if st == "sub" and _is_s_cluster(rec, i):
                    s_cluster_high += 1
            if st == "sub" and 0.58 <= pen <= 0.62:
                boundary_06 += 1

    def pct(n, d):
        return f"{100.0 * n / d:.2f}%" if d else "n/a"

    print(f"Telemetry: {path.name} | âm được chấm: {scored_all}")
    print(f"\n1. Status toàn cục: {dict(total)}")
    print(f"   Severity (sub+del, ước tính): {dict(sev_all)}")
    print(f"   → medium/high trên tổng âm chấm: "
          f"{pct(sev_all['medium'] + sev_all['high'], scored_all)}")
    print(f"\n2. Band-9 speakers: {scored_band9} âm | medium/high = "
          f"{pct(sev_band9['medium'] + sev_band9['high'], scored_band9)} "
          f"(mục tiêu ≈ 0–3%; cao hơn = đang chấm oan speaker gần bản xứ)")
    print(f"   chi tiết band9: {dict(sev_band9)}")
    print("\n3. Top 15 cặp sub medium/high lặp lại:")
    for pair, n in pair_high.most_common(15):
        print(f"   {pair:>10s}  ×{n}")
    print(f"\n4. S-cluster (/p t k/ sau /s/ đầu từ) trong nhóm high: "
          f"{s_cluster_high}/{high_total} ({pct(s_cluster_high, high_total)}) "
          f"— phần TOEIC_PHONEME_S_CLUSTER sẽ hạ về low")
    print(f"\n5. Sub rơi sát ranh 0.6 (sim 0.4 × conf cao — '>=0.6' = high): "
          f"{boundary_06} — đổi ngưỡng/similarity 1 nấc là đổi màu các lỗi này")
    return 0


if __name__ == "__main__":
    sys.exit(main())

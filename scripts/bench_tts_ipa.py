#!/usr/bin/env python3
"""Bench âm học cho nhánh đọc-IPA của TTS (src/tts.py:synthesize(ipa=...)).

Câu hỏi: audio mẫu tổng hợp TỪ IPA hiển thị có phát âm ĐÚNG chuỗi tham chiếu bằng
hoặc hơn audio tổng hợp từ chữ viết (Piper G2P) không? Đây là cổng rollout: chỉ bật
TTS_IPA_SYNTH khi bench này KHÔNG regression (xem CACHE_VERSION v6 trong src/tts.py).

Cách đo (dùng CHÍNH bộ nhận diện phoneme của app — wav2vec):
  với mỗi từ w:
    ref   = text_to_ipa_sequence(w)               # chuỗi phoneme tham chiếu app chấm
    A_txt = predict(synthesize(text=w))            # audio "đọc chữ"  → phoneme nghe được
    A_ipa = predict(synthesize(ipa=word_ipa_display(w)))  # audio "đọc IPA" → nghe được
    agree_* = 1 - editdistance(pred_*, ref)/max_len   # càng cao càng khớp tham chiếu
  Δ = agree_ipa - agree_txt  (>0: đọc-IPA tốt hơn)

Từ vựng gồm 3 nhóm: thường, homograph/dạng-trích-dẫn, và viết tắt/OOV (nơi đọc chữ
hay đánh vần) — để lộ cả regression lẫn thắng lợi. In per-word + tổng hợp; exit code
1 nếu regression trung bình vượt ngưỡng (dùng cho CI/rollout gate).

Chạy:  python scripts/bench_tts_ipa.py [--limit N] [--json out.json]
Cần voice TTS (TTS_VOICE_US) + wav2vec (torch/transformers). KHÔNG cần server chạy.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_common import edit_align  # noqa: E402
from src import tts  # noqa: E402
from src.config import load_config  # noqa: E402
from src.phoneme.ipa import (  # noqa: E402
    normalize_ipa,
    text_to_ipa_sequence,
    word_ipa_display,
)
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

# Từ thường (đọc chữ vốn đã tốt — kiểm tra KHÔNG regression).
WORDS_COMMON = [
    "advantage", "store", "student", "important", "beautiful", "company",
    "develop", "question", "example", "customer", "restaurant", "necessary",
    "jerry", "prepared", "measured", "founder", "internship", "government",
]
# Homograph / dạng trích dẫn — nơi Piper G2P có thể chọn cách đọc khác IPA hiển thị.
WORDS_HOMOGRAPH = [
    "read", "lead", "the", "project", "record", "present", "live",
]
# Viết tắt / chuỗi ngắn — đọc chữ hay bị đánh vần; đọc IPA nên thắng.
WORDS_ABBREV = [
    "sh", "oct", "av", "mon",
]


def phonemes_of(word: str, cfg, predictor, kind: str, tmpdir: Path) -> list[str] | None:
    """Tổng hợp `word` theo kind ('text'|'ipa') rồi nhận diện → list phoneme chuẩn hoá.
    None nếu tổng hợp thất bại (vd IPA không map được)."""
    try:
        if kind == "ipa":
            disp = word_ipa_display(word)
            if not disp:
                return None
            wav = tts.synthesize(ipa=disp, accent="us", config=cfg)
        else:
            wav = tts.synthesize(text=word, accent="us", config=cfg)
    except ValueError:
        return None
    wav_path = tmpdir / f"{kind}_{word}.wav"
    wav_path.write_bytes(wav)
    segments, warning = predictor.predict(str(wav_path))
    if warning:
        raise RuntimeError(f"wav2vec không sẵn sàng: {warning}")
    pred = predictor.get_predicted_phoneme_list(segments)
    return [normalize_ipa(p) for p in pred if normalize_ipa(p)]


def agreement(pred: list[str], ref: list[str]) -> float:
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    dist, _ = edit_align(pred, ref)
    return round(1.0 - dist / max(len(pred), len(ref)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số từ mỗi nhóm")
    ap.add_argument("--json", type=str, default="", help="ghi kết quả JSON ra file")
    ap.add_argument("--regression-thresh", type=float, default=-0.03,
                    help="Δ trung bình nhóm thường dưới ngưỡng này → exit 1")
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.tts_voice_us:
        print("[LỖI] Chưa cấu hình TTS_VOICE_US — không tổng hợp được.")
        return 2
    predictor = Wav2VecPhonemePredictor()
    if not predictor.is_available:
        print("[LỖI] wav2vec backend không khả dụng (cần torch/transformers).")
        return 2

    groups = {
        "common": WORDS_COMMON,
        "homograph": WORDS_HOMOGRAPH,
        "abbrev": WORDS_ABBREV,
    }
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for group, words in groups.items():
            wl = words[: args.limit] if args.limit else words
            for w in wl:
                ref = [normalize_ipa(p) for p in text_to_ipa_sequence(w)]
                ref = [p for p in ref if p]
                pred_txt = phonemes_of(w, cfg, predictor, "text", tmpdir)
                pred_ipa = phonemes_of(w, cfg, predictor, "ipa", tmpdir)
                a_txt = agreement(pred_txt, ref) if pred_txt is not None else None
                a_ipa = agreement(pred_ipa, ref) if pred_ipa is not None else None
                delta = (
                    round(a_ipa - a_txt, 3)
                    if (a_txt is not None and a_ipa is not None) else None
                )
                rows.append({
                    "group": group, "word": w,
                    "ref": "".join(ref),
                    "disp_ipa": word_ipa_display(w),
                    "pred_text": "".join(pred_txt or []),
                    "pred_ipa": "".join(pred_ipa or []),
                    "agree_text": a_txt, "agree_ipa": a_ipa, "delta": delta,
                })

    # ── Report ──
    print(f"\n{'grp':9s} {'word':11s} {'ref':16s} {'a_txt':6s} {'a_ipa':6s} {'Δ':6s}")
    print("-" * 62)
    for r in rows:
        d = r["delta"]
        mark = "" if d is None else ("  ↑" if d > 0.02 else ("  ↓" if d < -0.02 else ""))
        print(f"{r['group']:9s} {r['word']:11s} {r['ref']:16s} "
              f"{str(r['agree_text']):6s} {str(r['agree_ipa']):6s} "
              f"{str(d):6s}{mark}")

    def mean_delta(grp: str | None) -> float | None:
        ds = [r["delta"] for r in rows
              if r["delta"] is not None and (grp is None or r["group"] == grp)]
        return round(sum(ds) / len(ds), 4) if ds else None

    print("\n── Δ trung bình (agree_ipa − agree_text) ──")
    for grp in ("common", "homograph", "abbrev", None):
        label = grp or "TẤT CẢ"
        print(f"  {label:10s}: {mean_delta(grp)}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"rows": rows,
                        "mean_delta": {g: mean_delta(g)
                                       for g in ("common", "homograph", "abbrev", None)}},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[ghi] {args.json}")

    common_delta = mean_delta("common")
    if common_delta is not None and common_delta < args.regression_thresh:
        print(f"\n[REGRESSION] Δ nhóm thường {common_delta} < {args.regression_thresh} "
              "→ KHÔNG bật TTS_IPA_SYNTH.")
        return 1
    print(f"\n[OK] Δ nhóm thường {common_delta} ≥ ngưỡng {args.regression_thresh}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

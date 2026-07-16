#!/usr/bin/env python3
"""Bench acoustic tiếng Hàn trên audio NGƯỜI THẬT (giáo trình Tiếng Hàn Tổng Hợp 1).

Khác bench_ko_acoustic.py (edge-tts): đây là giọng bản xứ THU STUDIO, hội thoại,
connected speech — gần production hơn TTS. Vẫn KHÔNG phải audio học viên (đó là
điểm mù còn lại của gate M2).

Pipeline 2 pha:
  1. --extract: Whisper large-v3 (ko) cắt track dài thành câu 2–10s; giữ segment
     có text thuần Hangul + avg word prob >= --min-prob (mặc định 0.90 — reference
     lấy từ ASR nên phải lọc chặt để không bench trên transcript sai). Xuất wav
     16k mono + manifest.json vào data/bench/ko_real/.
  2. (mặc định) chấm native-metrics trên corpus đã cắt với --model bất kỳ,
     cùng thang đo bench_ko_acoustic: acc mean/median/min + false visible
     errors (med/high) per clip.

Usage:
    python scripts/bench_ko_real_audio.py --extract            # pha 1 (một lần)
    python scripts/bench_ko_real_audio.py                      # model theo config
    python scripts/bench_ko_real_audio.py --model slplab/...   # ứng viên
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_DIR = REPO_ROOT / "data" / "초급 1" / "tieng Han tong hop 1 - nghe"
BENCH_DIR = REPO_ROOT / "data" / "bench" / "ko_real"

# Chỉ nhận reference thuần Hangul (+ khoảng trắng): số/Latin/ký tự lạ làm G2P
# lệch khỏi những gì thật sự được đọc → loại từ gốc.
_HANGUL_ONLY = re.compile(r"^[가-힣\s]+$")


def extract(args) -> None:
    import numpy as np
    import soundfile as sf

    from src.asr import _get_model  # tái dùng cache faster-whisper của repo
    from src.phoneme.wav2vec_backend import _load_audio

    model = _get_model(args.whisper_model, args.device)
    tracks = sorted(SOURCE_DIR.glob("*.mp3"))[: args.max_tracks]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    kept = skipped = 0
    for track in tracks:
        segments, _info = model.transcribe(
            str(track), language="ko", word_timestamps=True, vad_filter=True,
        )
        # Materialize để nhìn được segment KẾ TIẾP: Whisper seg.end hay cắt cụt
        # âm tiết cuối (verify 2026-07-16: 14/14 k͈-del ở 까 cuối câu biến mất khi
        # pad đuôi) → pad TAIL_PAD_SEC, clamp vào start segment kế (không lem
        # tiếng người nói sau).
        segments = list(segments)
        wav = None  # lazy: chỉ load khi track có segment đạt lọc
        for si, seg in enumerate(segments):
            text = seg.text.strip().rstrip(".!?").strip()
            # Bỏ dấu câu giữa chừng rồi kiểm thuần Hangul
            clean = re.sub(r"[,.!?~…‥·\"'“”‘’]", "", text).strip()
            dur = seg.end - seg.start
            words = seg.words or []
            avg_prob = (
                sum(w.probability for w in words) / len(words) if words else 0.0
            )
            if (
                not _HANGUL_ONLY.match(clean)
                or not (args.min_dur <= dur <= args.max_dur)
                or avg_prob < args.min_prob
                or len(clean.split()) < 2
            ):
                skipped += 1
                continue
            if wav is None:
                wav = _load_audio(str(track), 16000)
            tail = seg.end + args.tail_pad
            if si + 1 < len(segments):
                tail = min(tail, segments[si + 1].start)
            s0 = max(0, int(seg.start * 16000))
            s1 = min(len(wav), int(tail * 16000))
            if s1 - s0 < int(args.min_dur * 16000):
                skipped += 1
                continue
            clip_id = f"{track.stem}_s{si:02d}"
            sf.write(BENCH_DIR / f"{clip_id}.wav", wav[s0:s1], 16000)
            manifest.append({
                "id": clip_id,
                "kind": "native",
                "reference": clean,
                "source": track.name,
                "span": [round(seg.start, 2), round(tail, 2)],
                "asr_avg_prob": round(avg_prob, 3),
                "wav": f"{clip_id}.wav",
            })
            kept += 1
        print(f"{track.name}: giữ {kept} / bỏ {skipped} (luỹ kế)")

    (BENCH_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nXong: {kept} clip → {BENCH_DIR}")


def score(args) -> None:
    from src.config import load_config
    from src.phoneme.ipa.profile import get_profile
    from src.phoneme.scoring import compute_phoneme_score
    from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor

    manifest_path = BENCH_DIR / "manifest.json"
    if not manifest_path.exists():
        sys.exit("Chưa có corpus — chạy --extract trước.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    config = load_config()
    model_id = args.model or config.phoneme_wav2vec_model_ko
    device = args.device or config.phoneme_device
    profile = get_profile("ko")
    predictor = Wav2VecPhonemePredictor(model_id=model_id, device=device)

    accs: list[float] = []
    false_total = 0
    per_clip: list[dict] = []
    for entry in manifest:
        wav = BENCH_DIR / entry["wav"]
        phs, spans, stress, disp = profile.text_to_ipa_with_spans(entry["reference"])
        segments, warning = predictor.predict(str(wav))
        if warning and not segments:
            sys.exit(f"wav2vec unavailable: {warning}")
        sc = compute_phoneme_score(
            segments, phs, spans, stress,
            reference_display_stress=disp, profile=profile,
        )
        visible = [
            {"type": e.error_type.value, "expected": e.expected,
             "predicted": e.predicted, "word": e.word}
            for e in sc.errors if e.severity in ("medium", "high")
        ]
        accs.append(sc.overall_accuracy)
        false_total += len(visible)
        per_clip.append({
            "id": entry["id"], "reference": entry["reference"],
            "accuracy": sc.overall_accuracy, "visible_errors": visible,
        })
        print(f"{entry['id']:12s} acc={sc.overall_accuracy:.3f} "
              f"visible={len(visible)}")

    report = {
        "model": model_id,
        "corpus": "ko_real (Tiếng Hàn Tổng Hợp 1 — native studio, ASR-referenced)",
        "n": len(accs),
        "acc_mean": round(statistics.mean(accs), 4),
        "acc_median": round(statistics.median(accs), 4),
        "acc_min": round(min(accs), 4),
        "visible_false_errors_per_clip": round(false_total / len(accs), 3),
        "clips": per_clip,
    }
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "outputs" / f"bench_ko_real_{model_id.split('/')[-1]}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n══ TÓM TẮT (native người thật) ══")
    print(f"model: {model_id}")
    print(f"n={report['n']} acc mean={report['acc_mean']} "
          f"median={report['acc_median']} min={report['acc_min']} | "
          f"false visible errors/clip={report['visible_false_errors_per_clip']}")
    print(f"report: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Pha 1: cắt corpus")
    parser.add_argument("--model", default=None, help="HF model id (pha 2)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--max-tracks", type=int, default=60)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=10.0)
    parser.add_argument("--min-prob", type=float, default=0.90)
    parser.add_argument("--tail-pad", type=float, default=0.35,
                        help="Pad đuôi mỗi segment (Whisper hay cắt cụt âm cuối)")
    args = parser.parse_args()
    if args.extract:
        extract(args)
    else:
        score(args)


if __name__ == "__main__":
    main()

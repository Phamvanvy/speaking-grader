#!/usr/bin/env python3
"""Bench accent dual-reference (TOEIC_PHONEME_ACCENT_DUALREF) — BATH-split UK/US.

Câu hỏi: học viên đọc giọng ANH các từ BATH (dance /dɑːns/, path, class, ask…) có bị
tính SAI nguyên âm không, và cờ dualref có sửa mà KHÔNG nới lỏng cho từ khác / lỗi thật?

Cách đo (dùng chính TTS + wav2vec + scoring của app, KHÔNG cần audio người thật):
  - Audio UK  = synthesize(text=word, accent=gb)  → voice en_GB (espeak en-gb tự áp
    BATH → phát /ɑː/). Đây là "học viên nói giọng Anh".
  - Audio US  = synthesize(text=word, accent=us)  → đối chứng, phải KHÔNG đổi ON/OFF.
  - Chấm mỗi clip với accent_dualref ON và OFF (reference = US CMUdict). So
    overall_accuracy.
  Kỳ vọng: UK-audio nhóm BATH: accuracy ON > OFF (hết phạt æ→ɑː). US-audio + nhóm
  control (không thuộc BATH) KHÔNG đổi (learner guard: không nới cho từ/âm khác).

Chạy: python scripts/bench_accent_dualref.py [--json out.json]
Cần TTS voice US+GB + wav2vec.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench_common import run_scoring, score_summary  # noqa: E402
from src import tts  # noqa: E402
from src.config import load_config  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

# Từ BATH (giọng Anh /ɑː/, giọng Mỹ /æ/) — kỳ vọng dualref GIÚP khi đọc giọng Anh.
BATH_WORDS = ["dance", "path", "class", "ask", "fast", "last", "half", "bath",
              "grass", "example", "answer", "command", "plant", "master"]
# Control: /æ/ giống nhau ở CẢ hai giọng (TRAP set) — dualref KHÔNG được đụng.
CONTROL_WORDS = ["cat", "map", "bad", "hand", "flag", "match", "trap", "panic"]


def clip_duration(wav_bytes: bytes) -> float:
    import io
    import wave
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        return wf.getnframes() / wf.getframerate()


def score_word(word, accent, cfg, predictor, tmpdir, dualref: bool) -> dict:
    wav = tts.synthesize(text=word, accent=accent, config=cfg)
    p = Path(tmpdir) / f"{accent}_{word}.wav"
    p.write_bytes(wav)
    segments, warn, posteriors = predictor.predict_with_posteriors(str(p))
    if warn:
        raise RuntimeError(warn)
    dur = clip_duration(wav)
    res = run_scoring(
        cfg, segments, None, word,
        skips={}, word_windows={0: (0.0, dur)}, word_probs={0: 0.99},
        gates_on=False, homograph_on=False, collapse_on=False,
        boundary_refine_on=False, s_cluster_on=False,
        accent_dualref_on=dualref,
    )
    summ = score_summary(res["score"], res["diags"])
    return {"acc": summ["overall_accuracy"], "sub": summ["substitutions"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    cfg = load_config()
    if not cfg.tts_voice_us or not cfg.tts_voice_gb:
        print("[LỖI] cần TTS_VOICE_US và TTS_VOICE_GB.")
        return 2
    predictor = Wav2VecPhonemePredictor()
    if not predictor.is_available:
        print("[LỖI] wav2vec không khả dụng.")
        return 2

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for group, words in (("BATH", BATH_WORDS), ("CONTROL", CONTROL_WORDS)):
            for w in words:
                r = {"group": group, "word": w}
                for accent in ("gb", "us"):
                    off = score_word(w, accent, cfg, predictor, td, dualref=False)
                    on = score_word(w, accent, cfg, predictor, td, dualref=True)
                    r[accent] = {
                        "acc_off": off["acc"], "acc_on": on["acc"],
                        "d_acc": round((on["acc"] or 0) - (off["acc"] or 0), 3),
                        "sub_off": off["sub"], "sub_on": on["sub"],
                    }
                rows.append(r)

    print(f"\n{'grp':8s} {'word':10s} | UK acc off→on  Δ  sub | US acc off→on  Δ  sub")
    print("-" * 74)
    for r in rows:
        gb, us = r["gb"], r["us"]
        print(f"{r['group']:8s} {r['word']:10s} | "
              f"{str(gb['acc_off']):5s}→{str(gb['acc_on']):5s} {gb['d_acc']:+.2f} "
              f"{gb['sub_off']}/{gb['sub_on']} | "
              f"{str(us['acc_off']):5s}→{str(us['acc_on']):5s} {us['d_acc']:+.2f} "
              f"{us['sub_off']}/{us['sub_on']}")

    def mean(group, accent, key):
        xs = [r[accent][key] for r in rows if r["group"] == group]
        return round(sum(xs) / len(xs), 4) if xs else None

    print("\n── Δaccuracy trung bình (dualref ON − OFF) ──")
    print(f"  BATH    · UK audio: {mean('BATH','gb','d_acc')}   · US audio: {mean('BATH','us','d_acc')}")
    print(f"  CONTROL · UK audio: {mean('CONTROL','gb','d_acc')}   · US audio: {mean('CONTROL','us','d_acc')}")
    print("\nKỳ vọng: BATH/UK > 0 (sửa được); CONTROL & US ≈ 0 (không nới oan).")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"[ghi] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A/B hallucination benchmark giữa 2 checkpoint phoneme wav2vec2.

So sánh model HIỆN TẠI với một sibling cùng họ (cùng vocab eSpeak-IPA → cắm thẳng
qua model_id, không sửa backend) trên các audio thật trong data/audio/. Không có
ground-truth phiên âm tay → reference text lấy từ Whisper MỘT LẦN/ audio, dùng CHUNG
cho cả 2 model để so apples-to-apples.

DIAGNOSTIC ONLY — không sửa điểm, không đổi model production. Chạy:

    .venv/Scripts/python.exe scripts/bench_hallucination.py

Metric (tái dùng định nghĩa production, KHÔNG tự chế):
  1. implausible_sub_rate (PRIMARY, độc lập gate): sub mà is_real_error_substitution(...)
     == False ÷ tổng phoneme chấm. Proxy hallucination thuần acoustic.
  2. recognizer_noise_rate (SECONDARY, theo gate): correspondence có
     penalty_reason == "recognizer_noise" ÷ tổng phoneme chấm.
  3. cross-model disagreement: cùng reference index → so trực tiếp pred_symbol 2 model.
  4. context: overall_accuracy, #sub/#del, mean confidence của sub.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

# Console Windows mặc định cp1258 → ép stdout/stderr UTF-8 để in IPA + tiếng Việt.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# Cho phép `from src...` khi chạy `python scripts/bench_hallucination.py` từ repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asr import transcribe  # noqa: E402
from src.phoneme.analyzer import HybridPhonemeAnalyzer  # noqa: E402
from src.phoneme.diagnostics import WordDiagnostic  # noqa: E402
from src.phoneme.ipa import is_real_error_substitution  # noqa: E402
from src.phoneme.l1_vietnamese import PenaltyReason  # noqa: E402
from src.phoneme.scoring import PHONEME_RECOGNIZER_NOISE_SIM  # noqa: E402

# ── Cấu hình A/B ────────────────────────────────────────────────────────────────
MODEL_A = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"   # hiện tại (production default)
MODEL_B = "facebook/wav2vec2-lv-60-espeak-cv-ft"     # sibling, cùng vocab eSpeak-IPA
MODELS = {"A_xlsr53": MODEL_A, "B_lv60": MODEL_B}

AUDIO_DIR = _REPO_ROOT / "data" / "audio"
AUDIO_EXTS = ("*.m4a", "*.wav", "*.mp3", "*.flac")
WHISPER_MODEL = "base"   # khớp default production (config.whisper_model)
DEVICE = "cpu"

_NOISE_REASON = PenaltyReason.RECOGNIZER_NOISE.value


def _collect_correspondences(analyzer: HybridPhonemeAnalyzer, audio: str, ref: str):
    """Chạy 1 model trên 1 audio; trả (score, correspondences_phẳng).

    correspondences phẳng = list dict theo ĐÚNG thứ tự reference (span → âm trong span),
    mỗi phần tử: {ref_symbol, pred_symbol, confidence, status, penalty_reason}.
    """
    sink_rows: list[WordDiagnostic] = []

    def sink(diags: list[WordDiagnostic]) -> None:
        sink_rows.extend(diags)

    result = analyzer.analyze(
        audio, reference_text=ref, skips=None, diagnostics_sink=sink
    )
    flat: list[dict] = []
    for d in sink_rows:          # WordDiagnostic theo thứ tự span chuẩn
        flat.extend(d.correspondences)   # correspondences đã theo thứ tự âm trong từ
    return result, flat


def _metrics(flat: list[dict]) -> dict:
    n_ref = len(flat)
    subs = [c for c in flat if c["status"] == "sub"]
    dels = [c for c in flat if c["status"] == "del"]
    implausible = [
        c for c in subs
        if c["pred_symbol"] is not None
        and not is_real_error_substitution(
            c["ref_symbol"], c["pred_symbol"], sim_floor=PHONEME_RECOGNIZER_NOISE_SIM
        )
    ]
    noise = [c for c in flat if c.get("penalty_reason") == _NOISE_REASON]
    sub_confs = [c["confidence"] for c in subs if c["confidence"] is not None]
    return {
        "n_ref": n_ref,
        "n_sub": len(subs),
        "n_del": len(dels),
        "n_implausible_sub": len(implausible),
        "n_recognizer_noise": len(noise),
        "implausible_sub_rate": (len(implausible) / n_ref) if n_ref else 0.0,
        "recognizer_noise_rate": (len(noise) / n_ref) if n_ref else 0.0,
        "mean_sub_conf": (round(st.mean(sub_confs), 4) if sub_confs else None),
        "implausible_examples": [
            f'{c["ref_symbol"]}→{c["pred_symbol"]}@{c["confidence"]}'
            for c in implausible[:8]
        ],
    }


def _disagreement(flat_a: list[dict], flat_b: list[dict]) -> dict:
    """Cùng reference → so pred_symbol theo từng ref index (zip theo độ dài chung)."""
    n = min(len(flat_a), len(flat_b))
    if n == 0:
        return {"comparable": 0, "disagree": 0, "disagree_rate": 0.0, "len_mismatch": True}
    disagree = sum(
        1 for i in range(n)
        if flat_a[i]["pred_symbol"] != flat_b[i]["pred_symbol"]
    )
    return {
        "comparable": n,
        "disagree": disagree,
        "disagree_rate": round(disagree / n, 4),
        "len_mismatch": len(flat_a) != len(flat_b),
    }


def main() -> int:
    audios: list[str] = []
    for ext in AUDIO_EXTS:
        audios.extend(sorted(glob.glob(str(AUDIO_DIR / ext))))
    if not audios:
        print(f"Không tìm thấy audio trong {AUDIO_DIR}", file=sys.stderr)
        return 1

    # Khởi tạo 2 analyzer (mặc định khớp production: l1_enabled=False + ngưỡng noise default).
    analyzers = {
        name: HybridPhonemeAnalyzer(wav2vec_model=mid, device=DEVICE)
        for name, mid in MODELS.items()
    }
    for name, az in analyzers.items():
        if not az.wav2vec_available:
            print(f"[CẢNH BÁO] backend {name} ({MODELS[name]}) KHÔNG khả dụng — sẽ rỗng.",
                  file=sys.stderr)

    results: dict = {"config": {"models": MODELS, "whisper": WHISPER_MODEL,
                                "noise_sim_floor": PHONEME_RECOGNIZER_NOISE_SIM},
                     "audios": {}}
    # Cộng dồn toàn corpus theo model.
    agg = {name: {"n_ref": 0, "n_sub": 0, "n_del": 0, "n_implausible_sub": 0,
                  "n_recognizer_noise": 0} for name in MODELS}

    for audio in audios:
        name = os.path.basename(audio)
        print(f"\n=== {name} ===", flush=True)
        try:
            tr = transcribe(audio, model_size=WHISPER_MODEL, device=DEVICE)
        except Exception as e:  # noqa: BLE001
            print(f"  [LỖI Whisper] {e}", file=sys.stderr)
            continue
        ref = tr.text.strip()
        print(f"  ref (whisper): {ref!r}")
        if not ref:
            print("  (transcript rỗng — bỏ qua)")
            continue

        per_model: dict = {}
        flats: dict = {}
        for mname, az in analyzers.items():
            result, flat = _collect_correspondences(az, audio, ref)
            flats[mname] = flat
            m = _metrics(flat)
            acc = result.score.overall_accuracy if result.score else None
            m["overall_accuracy"] = round(acc, 4) if acc is not None else None
            per_model[mname] = m
            for k in agg[mname]:
                agg[mname][k] += m[k]
            print(f"  [{mname}] acc={m['overall_accuracy']} "
                  f"impl_sub={m['n_implausible_sub']}/{m['n_ref']} "
                  f"({m['implausible_sub_rate']:.3f}) "
                  f"noise={m['n_recognizer_noise']} ({m['recognizer_noise_rate']:.3f}) "
                  f"sub={m['n_sub']} del={m['n_del']} mean_sub_conf={m['mean_sub_conf']}")

        dis = _disagreement(flats.get("A_xlsr53", []), flats.get("B_lv60", []))
        print(f"  disagreement A vs B: {dis['disagree']}/{dis['comparable']} "
              f"({dis['disagree_rate']:.3f})"
              + ("  [LEN MISMATCH!]" if dis["len_mismatch"] else ""))
        results["audios"][name] = {"ref": ref, "models": per_model,
                                   "disagreement": dis}

    # ── Aggregate ───────────────────────────────────────────────────────────────
    print("\n\n===== AGGREGATE (toàn corpus) =====")
    print(f"{'model':<12} {'n_ref':>7} {'impl_sub':>9} {'impl_rate':>10} "
          f"{'noise':>7} {'noise_rate':>11} {'sub':>5} {'del':>5}")
    agg_out: dict = {}
    for mname, a in agg.items():
        nref = a["n_ref"] or 1
        impl_rate = a["n_implausible_sub"] / nref
        noise_rate = a["n_recognizer_noise"] / nref
        agg_out[mname] = {**a, "implausible_sub_rate": round(impl_rate, 4),
                          "recognizer_noise_rate": round(noise_rate, 4)}
        print(f"{mname:<12} {a['n_ref']:>7} {a['n_implausible_sub']:>9} "
              f"{impl_rate:>10.4f} {a['n_recognizer_noise']:>7} {noise_rate:>11.4f} "
              f"{a['n_sub']:>5} {a['n_del']:>5}")
    results["aggregate"] = agg_out

    out_dir = Path(os.environ.get(
        "CLAUDE_SCRATCHPAD",
        r"C:\Users\ADMIN\AppData\Local\Temp\claude\e--repos-speaking-grader"
        r"\299e6f41-64f0-4a98-a6ee-35775f794d11\scratchpad",
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bench_hallucination_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nĐã ghi số liệu thô → {out_path}")
    print("LƯU Ý: chỉ 5 clip → tín hiệu định hướng, chưa phải kết luận thống kê.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

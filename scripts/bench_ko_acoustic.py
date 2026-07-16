#!/usr/bin/env python3
"""Bench acoustic tiếng Hàn (M2) — GATE trước khi bật TOEIC_LANG_KO_ENABLED.

Đo trên corpus data/bench/ko/ (sinh bằng gen_bench_ko_corpus.py):
  1. NATIVE (audio đúng + reference đúng): phân bố accuracy + false-error rate
     (lỗi severity medium/high báo ra trên audio chuẩn = lỗi HỆ THỐNG).
  2. ERROR (audio mô phỏng lỗi, chấm против reference gốc):
     - separation: accuracy twin native − accuracy clip lỗi (phải > 0 rõ ràng)
     - detection: lỗi kỳ vọng (expect_ref_phoneme → expect_heard_phoneme/del)
       có xuất hiện trong danh sách sub/del của từ tương ứng không.

So sánh nhiều model: chạy lại với --model <hf_id> (audio giữ nguyên) rồi so report.

Usage:
    python scripts/bench_ko_acoustic.py                       # model theo config
    python scripts/bench_ko_acoustic.py --model slplab/...    # ứng viên khác
    python scripts/bench_ko_acoustic.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.phoneme.ipa.profile import get_profile  # noqa: E402
from src.phoneme.scoring import compute_phoneme_score  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

BENCH_DIR = REPO_ROOT / "data" / "bench" / "ko"


def _score_clip(predictor: Wav2VecPhonemePredictor, profile, wav: Path, ref: str):
    phs, spans, stress, disp = profile.text_to_ipa_with_spans(ref)
    segments, warning = predictor.predict(str(wav))
    if warning and not segments:
        raise RuntimeError(f"wav2vec unavailable: {warning}")
    score = compute_phoneme_score(
        segments, phs, spans, stress,
        reference_display_stress=disp, profile=profile,
    )
    return score, phs, spans


def _all_errors(score) -> list[dict]:
    return [
        {"type": e.error_type.value, "expected": e.expected,
         "predicted": e.predicted, "word": e.word, "severity": e.severity}
        for e in score.errors
    ]


def _visible_errors(errors: list[dict]) -> list[dict]:
    """Lỗi 'nhìn thấy' (med/high) — low coi như noise-tolerated, không tính false."""
    return [e for e in errors if e["severity"] in ("medium", "high")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model id (default: config ko model)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="Đường dẫn report JSON")
    args = parser.parse_args()

    config = load_config()
    model_id = args.model or config.phoneme_wav2vec_model_ko
    device = args.device or config.phoneme_device

    manifest_path = BENCH_DIR / "manifest.json"
    if not manifest_path.exists():
        sys.exit("Chưa có corpus — chạy scripts/gen_bench_ko_corpus.py trước.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    profile = get_profile("ko")
    predictor = Wav2VecPhonemePredictor(model_id=model_id, device=device)

    results: dict[str, dict] = {}
    for entry in manifest:
        wav = BENCH_DIR / entry["wav"]
        if not wav.exists():
            print(f"MISSING {entry['id']} — bỏ qua")
            continue
        score, phs, spans = _score_clip(predictor, profile, wav, entry["reference"])
        errors = _all_errors(score)
        results[entry["id"]] = {
            "entry": entry,
            "accuracy": score.overall_accuracy,
            "sub": score.substitution_count,
            "del": score.deletion_count,
            "ins": score.insertion_count,
            "errors": errors,
            "visible_errors": _visible_errors(errors),
        }
        print(f"{entry['id']:8s} {entry['kind']:6s} acc={score.overall_accuracy:.3f} "
              f"sub={score.substitution_count} del={score.deletion_count}")

    # ── Tổng hợp ─────────────────────────────────────────────────────────────
    native = [r for r in results.values() if r["entry"]["kind"] == "native"]
    errors = [r for r in results.values() if r["entry"]["kind"] == "error"]

    native_acc = [r["accuracy"] for r in native]
    false_visible = sum(len(r["visible_errors"]) for r in native)
    report: dict = {
        "model": model_id,
        "clips": len(results),
        "native": {
            "n": len(native),
            "acc_mean": round(statistics.mean(native_acc), 4) if native_acc else None,
            "acc_median": round(statistics.median(native_acc), 4) if native_acc else None,
            "acc_min": round(min(native_acc), 4) if native_acc else None,
            # false-error: lỗi med/high trên audio chuẩn / tổng clip native
            "visible_false_errors_total": false_visible,
            "visible_false_errors_per_clip": (
                round(false_visible / len(native), 3) if native else None
            ),
        },
        "error_cases": [],
    }

    detected = 0
    separated = 0
    for r in errors:
        e = r["entry"]
        twin = results.get(e["twin_id"])
        twin_acc = twin["accuracy"] if twin else None
        delta = round(twin_acc - r["accuracy"], 4) if twin_acc is not None else None
        # Detection = âm reference kỳ vọng bị flag SUB/DEL ở BẤT KỲ severity nào
        # (triad laryngeal/near-pair là lỗi NHẸ theo thiết kế — low vẫn hiển thị
        # cho user, vẫn tính là "bắt được"). Heard-khớp-chính-xác chỉ là stat phụ
        # (model emit nhãn riêng: nói o nghe ra u/ɔ).
        want_ref = profile.normalize_ipa(e["expect_ref_phoneme"])
        want_heard = (
            profile.normalize_ipa(e["expect_heard_phoneme"])
            if e["expect_heard_phoneme"] else None
        )
        hit = False
        hit_exact = False
        for err in r["errors"]:
            if profile.normalize_ipa(err["expected"]) != want_ref:
                continue
            hit = True
            if want_heard is None and err["type"] == "deletion":
                hit_exact = True
            elif err["predicted"] and want_heard and (
                profile.normalize_ipa(err["predicted"]) == want_heard
            ):
                hit_exact = True
        detected += hit
        if delta is not None and delta > 0.01:
            separated += 1
        report["error_cases"].append({
            "id": e["id"], "error_type": e["error_type"], "note": e["note"],
            "accuracy": r["accuracy"], "twin_accuracy": twin_acc, "delta": delta,
            "expected_error_detected": hit,
            "expected_error_detected_exact": hit_exact,
            "errors": r["errors"],
        })

    report["detection_rate"] = round(detected / len(errors), 3) if errors else None
    report["separation_rate"] = round(separated / len(errors), 3) if errors else None

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "outputs" / f"bench_ko_{model_id.split('/')[-1]}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print("\n══ TÓM TẮT ══")
    print(f"model: {model_id}")
    n = report["native"]
    print(f"native ({n['n']} clip): acc mean={n['acc_mean']} median={n['acc_median']} "
          f"min={n['acc_min']} | false visible errors/clip={n['visible_false_errors_per_clip']}")
    print(f"error cases: detection={report['detection_rate']} "
          f"separation={report['separation_rate']}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bench so sánh model CHẤM ĐIỂM LLM (local Qwen vs OpenRouter candidates).

DIAGNOSTIC ONLY — không import ngược vào src/. Gate bắt buộc trước khi flip
.env sang TOEIC_BACKEND=openrouter (đổi model chấm = điểm thay đổi).

2 bước (vì prompt_logs bị sanitize/cắt 5000 ký tự → KHÔNG replay từ đó được):

1) capture — chạy pipeline chấm THẬT (cần ASR/GPU + llama.cpp local đang chạy)
   trên corpus clip, hook tầng scoring.generate để lưu FULL system/user prompt
   + json_schema + kết quả baseline vào outputs/bench_llm/corpus/*.json:

     python scripts/bench_llm_scoring.py capture --manifest bench_manifest.json

   manifest = JSON list, mỗi item:
     {"id": "clip01", "audio": "path/to/clip.wav", "exam": "toeic",
      "question_type": "read_aloud",            # optional (tự đoán như API)
      "text": "reference script...",            # optional (Read Aloud)
      "prompt": "đề bài...", "provided_info": null,
      "image": "path/to/image.jpg"}             # optional (Describe Picture)

2) run — replay corpus lên từng target (KHÔNG cần ASR/GPU, chỉ gọi LLM):

     python scripts/bench_llm_scoring.py run \
         --target local \
         --target openrouter:anthropic/claude-haiku-4.5 \
         --target openrouter:qwen/qwen3-235b-a22b

   In bảng: per-criterion delta vs baseline (local), delta điểm tổng,
   tỉ lệ validation-fail, số correction bị drop, latency p50/p95.

Cần OPENROUTER_API_KEY trong .env cho target openrouter (không cần đổi
TOEIC_BACKEND — script tự replace config).
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORPUS_DIR = REPO_ROOT / "outputs" / "bench_llm" / "corpus"
RESULTS_DIR = REPO_ROOT / "outputs" / "bench_llm"


# ──────────────────────────────────────────────────────────────────────────────
# capture
# ──────────────────────────────────────────────────────────────────────────────

def cmd_capture(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    import src.scoring as scoring_pkg
    from src import core
    from src.config import load_config
    from src.rubrics import EXAM_REGISTRIES
    from src.scoring import backends

    config = load_config()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    captured: dict = {}

    real_generate = backends.generate

    def _recording_generate(cfg, system_prompt, user_prompt, output_model, **kw):
        result, meta = real_generate(cfg, system_prompt, user_prompt, output_model, **kw)
        # Chỉ giữ call CHẤM ĐIỂM (SpeakingResult) — bỏ qua call phụ nếu có.
        if output_model.__name__ == "SpeakingResult":
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["json_schema"] = kw.get("json_schema")
            captured["schema_name"] = kw.get("schema_name")
            captured["image_b64"] = kw.get("image_b64")
            captured["image_media_type"] = kw.get("image_media_type")
            captured["meta"] = meta
        return result, meta

    # score() gọi tên `generate` đã import vào namespace src.scoring → patch ở đó.
    scoring_pkg.generate = _recording_generate

    n_ok = 0
    for item in manifest:
        item_id = item["id"]
        captured.clear()
        exam = (item.get("exam") or config.default_exam).lower()
        registry = EXAM_REGISTRIES[exam]
        qt_key = item.get("question_type")
        if not qt_key:
            print(f"[{item_id}] SKIP: manifest cần question_type rõ ràng.")
            continue
        qt = registry[qt_key]

        image_b64 = image_media_type = None
        if item.get("image"):
            p = Path(item["image"])
            image_b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            image_media_type = {
                ".png": "image/png", ".webp": "image/webp",
            }.get(p.suffix.lower(), "image/jpeg")

        print(f"[{item_id}] chấm baseline ({config.backend})…", flush=True)
        try:
            output = core.grade_response(
                item["audio"],
                config,
                qt,
                reference_script=item.get("text"),
                image_b64=image_b64,
                image_media_type=image_media_type,
                prompt_text=item.get("prompt", "") or "",
                provided_info=item.get("provided_info"),
                save=False,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{item_id}] LỖI capture: {e}")
            continue
        if not captured.get("user_prompt"):
            print(f"[{item_id}] SKIP: không bắt được call chấm điểm (bị gating?).")
            continue

        corpus_item = {
            "id": item_id,
            "exam": exam,
            "question_type": qt.key,
            "system_prompt": captured["system_prompt"],
            "user_prompt": captured["user_prompt"],
            "json_schema": captured["json_schema"],
            "schema_name": captured["schema_name"],
            "image_b64": captured["image_b64"],
            "image_media_type": captured["image_media_type"],
            "transcript": output.get("transcript"),
            "baseline": {
                "backend": captured["meta"].get("backend_used"),
                "model": captured["meta"].get("model"),
                "scores": output.get("scores"),
                "latency_ms": captured["meta"].get("latency_ms"),
            },
        }
        out = CORPUS_DIR / f"{item_id}.json"
        out.write_text(
            json.dumps(corpus_item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_ok += 1
        print(f"[{item_id}] OK → {out}")

    print(f"\nCapture xong: {n_ok}/{len(manifest)} item → {CORPUS_DIR}")
    return 0 if n_ok else 1


# ──────────────────────────────────────────────────────────────────────────────
# run
# ──────────────────────────────────────────────────────────────────────────────

def _score_one(config, target_spec: str, item: dict):
    """Chấm 1 corpus item bằng 1 target → dict kết quả (hoặc error)."""
    from src.rubrics import EXAM_REGISTRIES
    from src.schema import SpeakingResult
    from src.scoring import backends
    from src.scoring.compute import _compute_ielts_band, _compute_toeic_score
    from src.scoring.validation import _drop_invalid_corrections, _validate_result

    if target_spec == "local":
        cfg = dataclasses.replace(config, backend="local")
        target = backends._local_target(cfg)
    elif target_spec.startswith("openrouter:"):
        model = target_spec.split(":", 1)[1]
        if not config.openrouter_api_key:
            raise SystemExit("Thiếu OPENROUTER_API_KEY trong .env cho target openrouter.")
        cfg = dataclasses.replace(
            config, backend="openrouter", openrouter_model=model,
            openrouter_fallback_local=False,
        )
        target = backends._openrouter_target(cfg)
    else:
        raise SystemExit(f"Target không hợp lệ: {target_spec} (local | openrouter:<model>)")

    qt = EXAM_REGISTRIES[item["exam"]][item["question_type"]]
    t0 = time.monotonic()
    try:
        result = backends._generate_openai_compat(
            cfg, target,
            item["system_prompt"], item["user_prompt"],
            SpeakingResult, item["json_schema"], item["schema_name"],
            item.get("image_b64"), item.get("image_media_type"),
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}",
                "latency_ms": int((time.monotonic() - t0) * 1000)}
    latency_ms = int((time.monotonic() - t0) * 1000)

    problems = _validate_result(result, qt)
    n_corr_before = len(result.corrections or [])
    _drop_invalid_corrections(result, item.get("transcript") or "")
    n_corr_dropped = n_corr_before - len(result.corrections or [])
    # Điểm tổng tất định như score() production.
    if item["exam"] == "ielts":
        overall = _compute_ielts_band(result)
    elif item["exam"] == "topik":
        overall = None
    else:
        overall = _compute_toeic_score(result)

    return {
        "criteria": {c.criterion: c.score for c in (result.criteria or [])},
        "overall": overall,
        "validation_problems": problems,
        "corrections_dropped": n_corr_dropped,
        "latency_ms": latency_ms,
    }


def _baseline_overall(item: dict):
    scores = (item.get("baseline") or {}).get("scores") or {}
    return scores.get("estimated_toeic_score") or scores.get("estimated_ielts_band")


def _baseline_criteria(item: dict) -> dict:
    scores = (item.get("baseline") or {}).get("scores") or {}
    return {c.get("criterion"): c.get("score") for c in scores.get("criteria") or []}


def cmd_run(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    from src.config import load_config

    # load_config validate backend=openrouter cần model — bench tự replace nên
    # nạp với backend hiện tại của .env là đủ.
    config = load_config()

    items = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(CORPUS_DIR.glob("*.json"))
    ]
    if not items:
        raise SystemExit(f"Corpus rỗng — chạy capture trước ({CORPUS_DIR}).")
    if args.limit:
        items = items[: args.limit]

    all_results: dict[str, dict] = {}
    for target_spec in args.target:
        print(f"\n=== Target: {target_spec} ({len(items)} bài) ===")
        per_item: dict[str, dict] = {}
        for item in items:
            r = _score_one(config, target_spec, item)
            per_item[item["id"]] = r
            if "error" in r:
                print(f"  [{item['id']}] LỖI: {r['error']}")
            else:
                base = _baseline_overall(item)
                delta = (
                    (r["overall"] - base)
                    if (r["overall"] is not None and base is not None)
                    else None
                )
                print(
                    f"  [{item['id']}] overall={r['overall']} "
                    f"(baseline={base}, Δ={delta}) | val_fail={bool(r['validation_problems'])} "
                    f"| corr_dropped={r['corrections_dropped']} | {r['latency_ms']}ms"
                )
        all_results[target_spec] = per_item

        # Tóm tắt target
        oks = [r for r in per_item.values() if "error" not in r]
        deltas = []
        crit_deltas: list[float] = []
        for item in items:
            r = per_item[item["id"]]
            if "error" in r:
                continue
            base = _baseline_overall(item)
            if r["overall"] is not None and base is not None:
                deltas.append(r["overall"] - base)
            bc = _baseline_criteria(item)
            for k, v in (r["criteria"] or {}).items():
                if k in bc and bc[k] is not None and v is not None:
                    crit_deltas.append(v - bc[k])
        lat = sorted(r["latency_ms"] for r in oks) or [0]
        print(
            f"  → ok={len(oks)}/{len(items)} | val_fail="
            f"{sum(1 for r in oks if r['validation_problems'])} | "
            f"Δoverall mean={statistics.mean(deltas):+.2f} "
            f"max|Δ|={max((abs(d) for d in deltas), default=0):.2f} | "
            f"Δcriterion mean={statistics.mean(crit_deltas):+.3f} | "
            f"latency p50={lat[len(lat) // 2]}ms p95={lat[int(len(lat) * 0.95) - 1]}ms"
            if deltas
            else f"  → ok={len(oks)}/{len(items)} (không có delta so sánh được)"
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"results_{ts}.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKết quả chi tiết → {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_cap = sub.add_parser("capture", help="chạy pipeline thật, lưu full prompt + baseline")
    ap_cap.add_argument("--manifest", required=True, help="JSON list các clip")
    ap_cap.set_defaults(fn=cmd_capture)

    ap_run = sub.add_parser("run", help="replay corpus lên các target LLM")
    ap_run.add_argument(
        "--target", action="append", required=True,
        help="local | openrouter:<model> (lặp lại được)",
    )
    ap_run.add_argument("--limit", type=int, default=0)
    ap_run.set_defaults(fn=cmd_run)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

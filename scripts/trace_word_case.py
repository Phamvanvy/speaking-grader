#!/usr/bin/env python3
"""Trace 1 TỪ qua từng tầng pipeline phoneme — xác định tầng gây lỗi chấm.

Case gốc (2026-07-05): từ "project" bị báo `ɑː → ɒ`, `ɪ → e` trên UI production.
Script dump theo đúng thứ tự pipeline cho MỌI occurrence của --word:

  L0  Whisper      : transcript + word window (start/end) + probability.
  L1  Reference    : word_to_ipa_with_stress_source + TẤT CẢ entry CMUdict kèm
                     _entry_score (bằng chứng chọn homograph noun/verb).
  L2  Raw wav2vec  : segments trong window ± pad (phoneme/start/end/conf) —
                     chạy CẢ single-pass lẫn chunked (hybrid, như bản ship).
  L3  IPA decode   : chuỗi segments trong window (đã CTC decode) — chính L2 ghép chuỗi.
  L4  DTW/scoring  : correspondences (expected↔predicted↔status) của từ, mỗi cặp
                     re-eval phonemes_match/phoneme_similarity + dạng normalize;
                     kèm PhonemeError đúng shape UI (score.errors).

DIAGNOSTIC ONLY — không sửa điểm, không đổi production. Chạy:

    python scripts/trace_word_case.py --audio "data/audio/answer for test 1" \
        --word project [--outdir outputs/case_project] [--fresh]

--audio nhận 1 file hoặc 1 thư mục (quét mọi file audio, ASR có cache, chỉ trace
file nào transcript chứa --word).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bench_common import (
    REPO_ROOT,
    SEG_WINDOW_PAD_SEC,
    build_reference_context,
    dump_segments,
    extract_wav,
    norm_word,
    run_scoring,
)

from src import asr  # noqa: E402  (bench_common đã thêm REPO_ROOT vào sys.path)
from src.config import load_config  # noqa: E402
from src.phoneme.chunking import compute_chunk_spans  # noqa: E402
from src.phoneme.ipa import (  # noqa: E402
    normalize_ipa,
    phoneme_similarity,
    word_to_ipa_with_stress_source,
)
from src.phoneme.ipa.g2p import _entry_score, _get_cmudict  # noqa: E402
from src.phoneme.ipa.phoneme_set import ARPABET_TO_IPA  # noqa: E402
from src.phoneme.ipa.similarity import phonemes_match  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

AUDIO_EXTS = {".weba", ".webm", ".wav", ".m4a", ".mp3", ".mp4", ".ogg", ".flac"}


def _numeric_key(p: Path):
    """Sort '2.weba' trước '10.weba' khi stem là số."""
    return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)


def collect_audio_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(
        (p for p in target.iterdir() if p.suffix.lower() in AUDIO_EXTS),
        key=_numeric_key,
    )


def run_asr_cached(config, wav: Path, cache: Path, fresh: bool) -> dict:
    if cache.exists() and not fresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    run = asr.transcribe_with_backend(
        str(wav), backend=config.asr_engine_practice,
        model_size=config.asr_model_practice, device=config.whisper_device,
    )
    tr = run.transcription
    data = {
        "text": tr.text, "duration": tr.duration,
        "backend": run.backend_used, "elapsed_ms": run.elapsed_ms,
        "words": [
            {"text": w.text, "start": w.start, "end": w.end,
             "probability": w.probability} for w in tr.words
        ],
    }
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def arpabet_to_ipa_str(tokens: list[str]) -> str:
    """Hiển thị entry CMUdict dạng IPA thô (không stress) để đọc nhanh."""
    return "".join(ARPABET_TO_IPA.get(t.rstrip("012"), "?") for t in tokens)


def reference_layer(word: str) -> dict:
    """L1: IPA reference runtime + toàn bộ entry CMUdict kèm điểm ranking."""
    symbols, stresses, source = word_to_ipa_with_stress_source(word)
    entries = _get_cmudict().get(word.lower()) or []
    ranked = [
        {
            "arpabet": e,
            "ipa": arpabet_to_ipa_str(e),
            "entry_score": round(_entry_score(e, is_function_word=False), 4),
        }
        for e in entries
    ]
    return {
        "word": word,
        "runtime_symbols": symbols,
        "runtime_ipa": "".join(symbols),
        "runtime_stress": stresses,
        "source": source,
        "cmudict_entries": ranked,
    }


def segments_in_window(segs_dump: list[dict], t0: float, t1: float) -> list[dict]:
    return [s for s in segs_dump if s["end"] >= t0 and s["start"] <= t1]


def annotate_correspondences(diag: dict, word: str) -> list[dict]:
    """L4: mỗi correspondence kèm dạng normalize + re-eval match/similarity."""
    out = []
    for c in diag["correspondences"]:
        ref, pred = c["ref_symbol"], c.get("pred_symbol")
        entry = {
            "ref": ref, "pred": pred, "status": c["status"],
            "confidence": c.get("confidence"),
            "penalty_reason": c.get("penalty_reason"),
            "penalty_adjustment": c.get("penalty_adjustment"),
            "ref_norm": normalize_ipa(ref) if ref else None,
            "pred_norm": normalize_ipa(pred) if pred else None,
        }
        if ref and pred:
            entry["similarity"] = round(phoneme_similarity(ref, pred), 3)
            entry["match_default"] = phonemes_match(ref, pred, word=word)
            entry["match_strict"] = phonemes_match(
                ref, pred, word=word, reducible=False)
        out.append(entry)
    return out


def trace_occurrence(k: int, ctx: dict, modes: dict, word: str) -> dict:
    """Gom L0/L2/L3/L4 cho 1 occurrence (global span index k)."""
    win = ctx["word_windows"].get(k)
    occ: dict = {
        "span_index": k,
        "word_as_spoken": ctx["spans"][k].word,
        "skip": str(ctx["skips"][k].reason) if k in ctx["skips"] else None,
        "L0_whisper": {
            "window": list(win) if win else None,
            "asr_probability": ctx["word_probs"].get(k),
        },
        "modes": {},
    }
    for mode_name, m in modes.items():
        mode_entry: dict = {}
        if win:
            w0, w1 = win[0] - SEG_WINDOW_PAD_SEC, win[1] + SEG_WINDOW_PAD_SEC
            raw = segments_in_window(m["segments_dump"], w0, w1)
            mode_entry["L2_raw_segments"] = raw
            mode_entry["L3_decoded_ipa_in_window"] = "".join(
                s["phoneme"] for s in raw)
        for gates, sc in (("gates_off", m["score_off"]), ("gates_on", m["score_on"])):
            diag = next((d for d in sc["diags"] if d["index"] == k), None)
            if diag is None:
                mode_entry[f"L4_{gates}"] = None
                continue
            mode_entry[f"L4_{gates}"] = {
                "reference_ipa": diag["reference_ipa"],
                "predicted_ipa": diag["predicted_ipa"],
                "coverage": diag["coverage"],
                "matches": diag["matches"],
                "substitutions": diag["substitutions"],
                "deletions": diag["deletions"],
                "penalty": diag["penalty"],
                "skip_reason": diag["skip_reason"],
                "correspondences": annotate_correspondences(diag, word),
            }
        # PhonemeError đúng shape UI (score.errors) — chỉ của từ này.
        ui_errors = [
            e for e in m["score_off"]["score"].to_dict()["errors"]
            if norm_word(e.get("word") or "") == norm_word(word)
        ]
        mode_entry["ui_errors_gates_off"] = ui_errors
        occ["modes"][mode_name] = mode_entry
    return occ


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default="data/audio/answer for test 1")
    ap.add_argument("--word", default="project")
    ap.add_argument("--outdir", default="outputs/case_project")
    ap.add_argument("--fresh", action="store_true", help="bỏ cache wav/ASR")
    args = ap.parse_args()

    config = load_config()
    target = norm_word(args.word)
    outdir = REPO_ROOT / args.outdir
    (outdir / "wav").mkdir(parents=True, exist_ok=True)
    (outdir / "asr").mkdir(parents=True, exist_ok=True)

    files = collect_audio_files(REPO_ROOT / args.audio)
    if not files:
        print(f"[!] Không thấy file audio nào tại {args.audio}")
        return 1
    print(f"[0] {len(files)} file audio | từ mục tiêu: {target!r}")

    # ── Stage 0: wav + ASR từng file, lọc file chứa từ mục tiêu ─────────────
    hits: list[tuple[Path, Path, dict]] = []
    for f in files:
        wav = outdir / "wav" / f"{f.stem}.wav"
        extract_wav(f, wav, args.fresh)
        asr_data = run_asr_cached(
            config, wav, outdir / "asr" / f"{f.stem}.json", args.fresh)
        words_norm = [norm_word(w["text"]) for w in asr_data["words"]]
        n = words_norm.count(target)
        flag = f"  ← {n} occurrence(s)" if n else ""
        print(f"[0] {f.name:12s} {asr_data['duration']:6.1f}s "
              f"{len(asr_data['words']):4d} words{flag}")
        if n:
            hits.append((f, wav, asr_data))
    if not hits:
        print(f"[!] Không file nào chứa {target!r} trong transcript.")
        return 1

    # ── L1: reference layer (per-word, không phụ thuộc audio) ───────────────
    ref_layer = reference_layer(target)
    print(f"\n[L1] runtime IPA: /{ref_layer['runtime_ipa']}/ "
          f"(source={ref_layer['source']}, stress={ref_layer['runtime_stress']})")
    for e in ref_layer["cmudict_entries"]:
        chosen = " ← RANKER CHỌN" if e["ipa"] == ref_layer["runtime_ipa"] else ""
        print(f"[L1] cmudict {' '.join(e['arpabet']):28s} /{e['ipa']}/ "
              f"score={e['entry_score']}{chosen}")

    predictor = Wav2VecPhonemePredictor(
        model_id=config.phoneme_wav2vec_model, device=config.phoneme_device,
        min_phoneme_duration=config.phoneme_min_duration_sec,
        confidence_threshold=config.phoneme_confidence_threshold,
    )

    file_reports = []
    for f, wav, asr_data in hits:
        print(f"\n{'=' * 78}\n[{f.name}] trace\n{'=' * 78}")
        ctx = build_reference_context(config, asr_data)
        occurrences = [
            k for k, s in enumerate(ctx["spans"]) if norm_word(s.word) == target
        ]
        print(f"[L0] occurrences (span index): {occurrences}")

        # ── L2: wav2vec 2 chế độ — single-pass + chunked (hybrid như ship) ──
        modes: dict[str, dict] = {}
        chunk_spans = compute_chunk_spans(
            [(w["text"], float(w["start"]), float(w["end"]))
             for w in asr_data["words"]],
            float(asr_data["duration"] or 0.0),
            strategy="hybrid",
            max_chunk_sec=config.phoneme_chunk_max_sec,
            min_pause_sec=config.phoneme_chunk_min_pause_sec,
            pad_sec=config.phoneme_chunk_pad_sec,
        ) or None
        for mode_name, spans_arg in (
            ("single_pass", None),
            ("chunked_hybrid", chunk_spans),
        ):
            t0 = time.perf_counter()
            segs, warn, posts = predictor.predict_with_posteriors(
                str(wav), chunk_spans=spans_arg)
            dt = time.perf_counter() - t0
            if warn:
                print(f"[L2] {mode_name}: wav2vec warning: {warn} — bỏ qua mode")
                continue
            segs_dump = dump_segments(segs)
            (outdir / f"{f.stem}_segments_{mode_name}.jsonl").write_text(
                "\n".join(json.dumps(s, ensure_ascii=False) for s in segs_dump),
                encoding="utf-8")
            print(f"[L2] {mode_name}: {len(segs)} segments trong {dt:.1f}s"
                  + (f" ({len(spans_arg)} chunks)" if spans_arg else ""))
            modes[mode_name] = {
                "segments_dump": segs_dump,
                "n_chunks": len(spans_arg) if spans_arg else None,
                "score_off": run_scoring(
                    config, segs, posts, asr_data["text"], ctx["skips"],
                    ctx["word_windows"], ctx["word_probs"], gates_on=False),
                "score_on": run_scoring(
                    config, segs, posts, asr_data["text"], ctx["skips"],
                    ctx["word_windows"], ctx["word_probs"], gates_on=True),
            }

        occ_reports = [trace_occurrence(k, ctx, modes, target) for k in occurrences]
        for occ in occ_reports:
            k = occ["span_index"]
            print(f"\n[L0] #{k} window={occ['L0_whisper']['window']} "
                  f"prob={occ['L0_whisper']['asr_probability']} skip={occ['skip']}")
            for mode_name, me in occ["modes"].items():
                raw = me.get("L3_decoded_ipa_in_window", "—")
                print(f"[L3] {mode_name:14s} raw trong window: /{raw}/")
                l4 = me.get("L4_gates_off")
                if l4:
                    pairs = " ".join(
                        f"{c['ref']}→{c['pred'] or '∅'}[{c['status']}"
                        + (f",sim={c.get('similarity')}" if c["status"] == "sub"
                           else "") + "]"
                        for c in l4["correspondences"])
                    print(f"[L4] {mode_name:14s} ref=/{l4['reference_ipa']}/ "
                          f"pred=/{l4['predicted_ipa']}/ pen={l4['penalty']}")
                    print(f"[L4] {mode_name:14s} {pairs}")
                    for e in me["ui_errors_gates_off"]:
                        print(f"[UI] {mode_name:14s} {e['error_type']}: "
                              f"{e['expected']} → {e['predicted']} "
                              f"(sev={e['severity']})")

        file_reports.append({
            "file": str(f), "duration": asr_data["duration"],
            "transcript_excerpt": asr_data["text"][:200],
            "chunk_spans": chunk_spans,
            "occurrences": occ_reports,
        })

    trace = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_word": target,
        "config": {
            "asr": f"{config.asr_engine_practice}/{config.asr_model_practice}",
            "wav2vec": config.phoneme_wav2vec_model,
            "chunking_strategy_traced": "hybrid",
            "chunk_max_sec": config.phoneme_chunk_max_sec,
        },
        "L1_reference": ref_layer,
        "files": file_reports,
    }
    (outdir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nOutputs: {outdir / 'trace.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

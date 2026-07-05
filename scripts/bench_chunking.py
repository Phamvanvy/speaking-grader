#!/usr/bin/env python3
"""Benchmark chiến lược chunk audio trước wav2vec (sprint 2026-07-05).

Ma trận 5 cấu hình trên data/videos/9.0.mp4:
  off (baseline single-pass) | pause(0.5, max=30) | hybrid(0.5) × max ∈ {15, 20, 30}

Deliverable KHÔNG phải tăng điểm — 4 tiêu chí:
  (a) IPA validity tăng (metric chính): từ không-skip có matches ≥ 50% phoneme
      reference VÀ coverage ≥ 0.5; đếm riêng "lem nặng" (coverage < 0.5 hoặc
      matches = 0 với từ ≥ 3 phoneme — kiểu zbɡŋ/ŋwt/knðnənʌmbz).
  (b) segment_agreement vs sentence-mode tăng (ground truth: slices của
      debug_full_vs_sentence + 5 câu ngẫu nhiên seed cố định — tránh overfit).
  (c) learner corpus (5 clip data/audio) không giảm accuracy (> 0.005/clip).
  (d) deletion evidence đáng tin lại: các ca false-del đã biết có max_mass tăng
      mạnh hoặc del biến mất.

Yêu cầu cache từ debug_full_vs_sentence.py (asr_full.json, full.wav,
sent_*_segments.jsonl) — chạy script đó trước nếu thiếu. Chạy:

    python scripts/bench_chunking.py [--skip-learner]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from bench_common import (
    REPO_ROOT,
    SEG_WINDOW_PAD_SEC,
    SLICE_PAD_SEC,
    build_reference_context,
    dtw_stats,
    dump_segments,
    load_jsonl,
    map_sentences_to_spans,
    norm_word,
    run_scoring,
    score_summary,
    segment_comparison,
    slice_wav,
    split_sentences,
    word_block,
)

from src import asr  # noqa: E402
from src.config import load_config  # noqa: E402
from src.phoneme.chunking import compute_chunk_spans  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

TARGET_WORDS = {
    "speaking", "candidate", "identification", "o'clock",
    "about", "close", "friends", "planes",
}
EVIDENCE_WORDS = {"about", "close", "friends", "planes"}

# Cấu hình benchmark (đã chốt trong plan): tên → (strategy, max_chunk_sec).
CONFIGS: list[tuple[str, str | None, float]] = [
    ("off", None, 0.0),
    ("pause_max30", "pause", 30.0),
    ("hybrid_max15", "hybrid", 15.0),
    ("hybrid_max20", "hybrid", 20.0),
    ("hybrid_max30", "hybrid", 30.0),
]
RANDOM_SENTENCE_SEED = 42
RANDOM_WORD_SEED = 7
N_RANDOM_SENTENCES = 5
N_RANDOM_WORDS = 100

LEARNER_CLIPS = [
    "data/audio/sample.wav",
    "data/audio/alley.m4a",
    "data/audio/Recording.m4a",
    "data/audio/Recording (2).m4a",
    "data/audio/Recording (3).m4a",
]


def ipa_validity(diags: list[dict]) -> dict:
    """Metric chính của sprint: đo trên các từ KHÔNG bị skip.

    valid   : matches ≥ 50% số phoneme reference VÀ coverage ≥ 0.5.
    severe  : coverage < 0.5 HOẶC (matches = 0 với từ ≥ 3 phoneme) — "lem nặng".
    """
    scored = [d for d in diags if d["skip_reason"] is None]
    if not scored:
        return {"n": 0, "validity": None, "severe_garbled": 0}
    valid = severe = 0
    for d in scored:
        n_ref = len(d["correspondences"])
        if n_ref and d["matches"] >= 0.5 * n_ref and d["coverage"] >= 0.5:
            valid += 1
        if d["coverage"] < 0.5 or (d["matches"] == 0 and n_ref >= 3):
            severe += 1
    return {
        "n": len(scored),
        "validity": round(valid / len(scored), 4),
        "severe_garbled": severe,
    }


def chunk_length_stats(spans: list[tuple[float, float]] | None) -> dict | None:
    if not spans:
        return None
    lens = [e - s for s, e in spans]
    return {
        "n_chunks": len(spans),
        "min_sec": round(min(lens), 2),
        "median_sec": round(statistics.median(lens), 2),
        "max_sec": round(max(lens), 2),
    }


def predict(predictor, wav: Path, chunk_spans):
    t0 = time.perf_counter()
    segments, warn, posteriors = predictor.predict_with_posteriors(
        str(wav), chunk_spans=chunk_spans)
    elapsed = time.perf_counter() - t0
    if warn:
        raise RuntimeError(f"wav2vec warning: {warn}")
    return segments, posteriors, round(elapsed, 2)


def evidence_cases(diags: list[dict]) -> list[dict]:
    """Các ca deletion-evidence của EVIDENCE_WORDS: từng âm del + max_mass."""
    out = []
    for d in diags:
        if norm_word(d["word"]) not in EVIDENCE_WORDS:
            continue
        wb = word_block(d)
        if wb["deletion_evidence"]:
            for e in wb["deletion_evidence"]:
                out.append({
                    "word": d["word"], "index": d["index"],
                    "ref": e["ref"], "max_mass": e.get("max_mass"),
                    "argmax": e.get("argmax"),
                })
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/bench_chunking")
    ap.add_argument("--skip-learner", action="store_true")
    args = ap.parse_args()

    config = load_config()
    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache = REPO_ROOT / "outputs/debug_full_vs_sentence"
    full_wav = cache / "full.wav"
    asr_cache = cache / "asr_full.json"
    if not (full_wav.exists() and asr_cache.exists()):
        print("Thiếu cache — chạy trước: python scripts/debug_full_vs_sentence.py")
        return 2

    asr_data = json.loads(asr_cache.read_text(encoding="utf-8"))
    ctx = build_reference_context(config, asr_data)
    transcript, spans = ctx["transcript"], ctx["spans"]
    skips, word_windows, word_probs = (
        ctx["skips"], ctx["word_windows"], ctx["word_probs"])
    whisper_words = [(w["text"], float(w["start"]), float(w["end"]))
                     for w in asr_data["words"]]
    duration = float(asr_data.get("duration") or 0.0)
    print(f"[0] Reference: {len(spans)} words | skips={len(skips)} | "
          f"duration={duration:.1f}s")

    predictor = Wav2VecPhonemePredictor(
        model_id=config.phoneme_wav2vec_model, device=config.phoneme_device,
        min_phoneme_duration=config.phoneme_min_duration_sec,
        confidence_threshold=config.phoneme_confidence_threshold,
    )

    # ── Ground truth agreement: slices sẵn có + 5 câu ngẫu nhiên ─────────────
    sentences = split_sentences(transcript)
    sent_spans = map_sentences_to_spans(sentences, spans)
    gt_slices: dict[int, list[dict]] = {}   # sentence idx → slice segments (abs time)
    for p in sorted(cache.glob("sent_*_segments.jsonl")):
        si = int(p.stem.split("_")[1])
        gt_slices[si] = load_jsonl(p)
    rng = random.Random(RANDOM_SENTENCE_SEED)
    candidates = [
        si for si, idxs in enumerate(sent_spans)
        if si not in gt_slices and len(idxs) >= 4
        and sum(1 for k in idxs if k in word_windows) >= 4
    ]
    random_sents = sorted(rng.sample(candidates, min(N_RANDOM_SENTENCES,
                                                     len(candidates))))
    print(f"[0] GT slices có sẵn: {sorted(gt_slices)} | random mới: {random_sents}")
    for si in random_sents:
        idxs = sent_spans[si]
        wins = [word_windows[k] for k in idxs if k in word_windows]
        s_start = max(0.0, min(w[0] for w in wins) - SLICE_PAD_SEC)
        s_end = max(w[1] for w in wins) + SLICE_PAD_SEC
        wav_path = outdir / f"rand_{si:03d}.wav"
        slice_wav(full_wav, wav_path, s_start, s_end)
        segs, _posts, _t = predict(predictor, wav_path, None)
        gt_slices[si] = [
            {"phoneme": s.phoneme, "start": round(s.start + s_start, 3),
             "end": round(s.end + s_start, 3), "conf": s.confidence}
            for s in segs
        ]
    # Từ nào thuộc câu có ground truth → so agreement được.
    span_to_sent = {k: si for si, idxs in enumerate(sent_spans) for k in idxs}
    gt_word_idxs = [
        k for k, si in span_to_sent.items()
        if si in gt_slices and k in word_windows
    ]

    # Mẫu từ ngẫu nhiên (đánh giá word-level, tránh overfit vào lỗi đã biết).
    rng_w = random.Random(RANDOM_WORD_SEED)
    non_skipped = [k for k in range(len(spans)) if k not in skips]
    random_word_sample = set(rng_w.sample(
        non_skipped, min(N_RANDOM_WORDS, len(non_skipped))))

    # 21 occurrence lỗi đã biết (từ baseline off — tính lại bên dưới cho khớp).
    results = []
    baseline_error_occ: list[int] = []
    for name, strategy, max_sec in CONFIGS:
        print(f"\n[{name}] " + "─" * 60)
        chunk_spans = None
        if strategy is not None:
            chunk_spans = compute_chunk_spans(
                whisper_words, duration, strategy,
                max_chunk_sec=max_sec,
                min_pause_sec=config.phoneme_chunk_min_pause_sec,
                pad_sec=config.phoneme_chunk_pad_sec,
            )
        cstats = chunk_length_stats(chunk_spans)
        print(f"[{name}] chunks: {cstats}")
        segments, posteriors, predict_sec = predict(
            predictor, full_wav, chunk_spans)
        print(f"[{name}] predict {predict_sec}s → {len(segments)} segments")
        segs_dump = dump_segments(segments)
        (outdir / f"{name}_segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in segs_dump),
            encoding="utf-8")

        off = run_scoring(config, segments, posteriors, transcript,
                          skips, word_windows, word_probs, gates_on=False)
        on = run_scoring(config, segments, posteriors, transcript,
                         skips, word_windows, word_probs, gates_on=True)
        diags = off["diags"]
        diag_by_idx = {d["index"]: d for d in diags}
        dtw = dtw_stats([s.phoneme for s in segments], ctx["phonemes"])

        # Lỗi đã biết: xác định trên baseline off, đo lại ở mọi config.
        if name == "off":
            baseline_error_occ = [
                d["index"] for d in diags
                if norm_word(d["word"]) in TARGET_WORDS
                and d["skip_reason"] is None
                and d["substitutions"] + d["deletions"] > 0
            ]
            print(f"[off] known error occurrences: {baseline_error_occ}")
        known_still_bad = [
            k for k in baseline_error_occ
            if k in diag_by_idx
            and diag_by_idx[k]["substitutions"] + diag_by_idx[k]["deletions"] > 0
        ]

        # Agreement vs GT slices: target words + mọi từ thuộc câu GT.
        agrees_target, agrees_all = [], []
        for k in gt_word_idxs:
            si = span_to_sent[k]
            w0, w1 = word_windows[k]
            cmpres = segment_comparison(
                segs_dump, gt_slices[si],
                w0 - SEG_WINDOW_PAD_SEC, w1 + SEG_WINDOW_PAD_SEC)
            a = cmpres["segment_agreement"]
            if a is None:
                continue
            agrees_all.append(a)
            d = diag_by_idx.get(k)
            if d and norm_word(d["word"]) in TARGET_WORDS:
                agrees_target.append(a)

        validity_full = ipa_validity(diags)
        validity_sample = ipa_validity(
            [d for d in diags if d["index"] in random_word_sample])

        result = {
            "config": name, "strategy": strategy, "max_chunk_sec": max_sec,
            "chunks": cstats, "predict_runtime_sec": predict_sec,
            "n_segments": len(segments),
            "dtw": dtw,
            "ipa_validity_full": validity_full,
            "ipa_validity_random100": validity_sample,
            "known_errors_remaining": len(known_still_bad),
            "known_errors_total": len(baseline_error_occ),
            "known_errors_still_bad_idx": known_still_bad,
            "segment_agreement_target_median": (
                round(statistics.median(agrees_target), 3)
                if agrees_target else None),
            "segment_agreement_all_median": (
                round(statistics.median(agrees_all), 3) if agrees_all else None),
            "n_agreement_words": len(agrees_all),
            "score_gates_off": score_summary(off["score"], diags),
            "score_gates_on": score_summary(on["score"], on["diags"]),
            "deletion_evidence_cases": evidence_cases(diags),
        }
        results.append(result)
        print(f"[{name}] validity={validity_full['validity']} "
              f"severe={validity_full['severe_garbled']} "
              f"known_err={len(known_still_bad)}/{len(baseline_error_occ)} "
              f"agree_target={result['segment_agreement_target_median']} "
              f"acc_off={result['score_gates_off']['overall_accuracy']}")

    # ── Learner regression ───────────────────────────────────────────────────
    learner = {}
    if not args.skip_learner:
        print("\n[learner] " + "─" * 60)
        for clip in LEARNER_CLIPS:
            clip_path = REPO_ROOT / clip
            if not clip_path.exists():
                print(f"[learner] thiếu {clip} — bỏ qua")
                continue
            cache_p = outdir / f"asr_{clip_path.stem.replace(' ', '_')}.json"
            if cache_p.exists():
                a = json.loads(cache_p.read_text(encoding="utf-8"))
            else:
                run = asr.transcribe_with_backend(
                    str(clip_path), backend=config.asr_engine_practice,
                    model_size=config.asr_model_practice,
                    device=config.whisper_device,
                )
                tr = run.transcription
                a = {"text": tr.text, "duration": tr.duration,
                     "words": [{"text": w.text, "start": w.start, "end": w.end,
                                "probability": w.probability}
                               for w in tr.words]}
                cache_p.write_text(json.dumps(a, ensure_ascii=False),
                                   encoding="utf-8")
            if not a["text"].strip():
                continue
            cctx = build_reference_context(config, a)
            cw = [(w["text"], float(w["start"]), float(w["end"]))
                  for w in a["words"]]
            per_clip = {}
            for name, strategy, max_sec in CONFIGS:
                cs = None
                if strategy is not None:
                    cs = compute_chunk_spans(
                        cw, float(a.get("duration") or 0.0), strategy,
                        max_chunk_sec=max_sec,
                        min_pause_sec=config.phoneme_chunk_min_pause_sec,
                        pad_sec=config.phoneme_chunk_pad_sec,
                    ) or None
                segs, posts, _t = predict(predictor, clip_path, cs)
                sc_on = run_scoring(config, segs, posts, cctx["transcript"],
                                    cctx["skips"], cctx["word_windows"],
                                    cctx["word_probs"], gates_on=True)
                sc_off = run_scoring(config, segs, posts, cctx["transcript"],
                                     cctx["skips"], cctx["word_windows"],
                                     cctx["word_probs"], gates_on=False)
                per_clip[name] = {
                    "n_chunks": len(cs) if cs else 1,
                    "accuracy_gates_on": sc_on["score"].overall_accuracy,
                    "accuracy_gates_off": sc_off["score"].overall_accuracy,
                    "validity": ipa_validity(sc_on["diags"]),
                }
            learner[clip] = per_clip
            base = per_clip["off"]["accuracy_gates_on"]
            deltas = {n: round(v["accuracy_gates_on"] - base, 4)
                      for n, v in per_clip.items() if n != "off"}
            print(f"[learner] {clip}: base={base} deltas={deltas}")

    # ── Report ───────────────────────────────────────────────────────────────
    report = {
        "video": "data/videos/9.0.mp4",
        "configs": results,
        "learner_regression": learner,
        "gt_sentences": sorted(gt_slices),
        "random_sentences_new": random_sents,
        "random_word_sample_size": len(random_word_sample),
    }
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 108)
    print("BENCHMARK CHUNKING — 9.0.mp4 (metric chính: IPA validity; "
          "agreement so với sentence-mode ground truth)")
    print("=" * 108)
    print(f"{'config':14s} {'chunks':>6s} {'predict_s':>9s} "
          f"{'validity':>8s} {'severe':>6s} {'val_rand':>8s} "
          f"{'known_err':>9s} {'agree_tgt':>9s} {'agree_all':>9s} "
          f"{'acc_off':>7s} {'acc_on':>6s}")
    print("-" * 108)
    for r in results:
        print(f"{r['config']:14s} "
              f"{(r['chunks'] or {}).get('n_chunks', 1):>6d} "
              f"{r['predict_runtime_sec']:>9.1f} "
              f"{r['ipa_validity_full']['validity']:>8.4f} "
              f"{r['ipa_validity_full']['severe_garbled']:>6d} "
              f"{r['ipa_validity_random100']['validity']:>8.4f} "
              f"{r['known_errors_remaining']:>4d}/{r['known_errors_total']:<4d} "
              f"{str(r['segment_agreement_target_median']):>9s} "
              f"{str(r['segment_agreement_all_median']):>9s} "
              f"{r['score_gates_off']['overall_accuracy']:>7.4f} "
              f"{r['score_gates_on']['overall_accuracy']:>6.4f}")
    if learner:
        print("\nLEARNER REGRESSION (accuracy gates ON, delta vs off):")
        for clip, per in learner.items():
            base = per["off"]["accuracy_gates_on"]
            deltas = " ".join(
                f"{n}={v['accuracy_gates_on'] - base:+.4f}"
                for n, v in per.items() if n != "off")
            print(f"  {clip}: base={base:.4f}  {deltas}")
    print(f"\nOutputs: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

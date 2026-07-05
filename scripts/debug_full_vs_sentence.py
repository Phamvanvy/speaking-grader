#!/usr/bin/env python3
"""Debug harness: IPA "lem" khi chấm FULL bài — do wav2vec trên audio dài hay DTW toàn cục?

So sánh CÙNG MỘT CÂU giữa 2 chế độ:
  - FULL mode: tái tạo đúng free-speech path của grade_response (reference = toàn
    transcript, wav2vec 1 forward pass trên cả bài, 1 DTW toàn cục).
  - SENTENCE mode: cắt audio đúng câu (theo Whisper word timestamps của run full),
    reference = chính câu đó (cùng text → cùng G2P), wav2vec + DTW chỉ trong câu.

Discriminator 2 tầng:
  1. Word-level (predicted_ipa/coverage/sub/del) khác nhau giữa 2 mode → lỗi thuộc
     pipeline full-exam, không phải scoring.
  2. Raw-segment-level trong CÙNG khoảng thời gian tuyệt đối:
     - segments giống nhau (agreement cao) → wav2vec ổn, lỗi tại DTW/attribution.
     - segments khác rõ → wav2vec suy giảm trên audio dài → cần chunk ở pipeline.

Kết luận 2026-07-04 (comparison.json): wav2vec suy giảm trên audio dài — dẫn tới
sprint chunking (xem scripts/bench_chunking.py). Helper chung: scripts/bench_common.py.

DIAGNOSTIC ONLY — không sửa điểm, không đổi schema production. Chạy:

    python scripts/debug_full_vs_sentence.py [--video data/videos/9.0.mp4] [--fresh]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from bench_common import (
    REPO_ROOT,
    SLICE_PAD_SEC,
    SEG_WINDOW_PAD_SEC,
    build_reference_context,
    dtw_stats,
    dump_segments,
    extract_wav,
    map_sentences_to_spans,
    norm_word,
    run_scoring,
    score_summary,
    segment_comparison,
    slice_wav,
    split_sentences,
    word_block,
)

from src import asr  # noqa: E402  (bench_common đã thêm REPO_ROOT vào sys.path)
from src.config import load_config  # noqa: E402
from src.phoneme.ipa import text_to_ipa_sequence_with_spans  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

# Từ mục tiêu user báo lỗi (so theo dạng normalize: lowercase, giữ apostrophe).
TARGET_WORDS = {
    "speaking", "candidate", "identification", "o'clock",
    "about", "close", "friends", "planes",
}
MAX_CONTROL_SENTENCES = 4


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="data/videos/9.0.mp4")
    ap.add_argument("--outdir", default="outputs/debug_full_vs_sentence")
    ap.add_argument("--fresh", action="store_true",
                    help="bỏ cache (wav/ASR/full segments), chạy lại từ đầu")
    args = ap.parse_args()

    config = load_config()
    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    video = REPO_ROOT / args.video
    full_wav = outdir / "full.wav"

    # ── Stage 0: audio + ASR ─────────────────────────────────────────────────
    print(f"[0] Extract wav từ {video.name} ...")
    extract_wav(video, full_wav, args.fresh)

    asr_cache = outdir / "asr_full.json"
    if asr_cache.exists() and not args.fresh:
        asr_data = json.loads(asr_cache.read_text(encoding="utf-8"))
        print(f"[0] ASR cache: {len(asr_data['words'])} words")
    else:
        print(f"[0] ASR ({config.asr_engine_practice}/{config.asr_model_practice}, "
              f"device={config.whisper_device}) ...")
        run = asr.transcribe_with_backend(
            str(full_wav), backend=config.asr_engine_practice,
            model_size=config.asr_model_practice, device=config.whisper_device,
        )
        tr = run.transcription
        asr_data = {
            "text": tr.text, "duration": tr.duration,
            "backend": run.backend_used, "elapsed_ms": run.elapsed_ms,
            "words": [
                {"text": w.text, "start": w.start, "end": w.end,
                 "probability": w.probability} for w in tr.words
            ],
        }
        asr_cache.write_text(
            json.dumps(asr_data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[0] ASR xong: {len(asr_data['words'])} words, "
              f"{asr_data['elapsed_ms']} ms")

    transcript = asr_data["text"]
    print(f"[0] Transcript: {len(transcript)} chars | mở đầu: {transcript[:80]!r}")

    # ── Reference + windows + skips (mirror grade_response free-speech) ─────
    ctx = build_reference_context(config, asr_data)
    phonemes, spans = ctx["phonemes"], ctx["spans"]
    skips, word_windows, word_probs = (
        ctx["skips"], ctx["word_windows"], ctx["word_probs"])
    print(f"[0] Reference: {len(spans)} words / {len(phonemes)} phonemes | "
          f"skips={len(skips)} | windows={len(word_windows)}")

    # ── Stage 1: FULL mode ───────────────────────────────────────────────────
    predictor = Wav2VecPhonemePredictor(
        model_id=config.phoneme_wav2vec_model, device=config.phoneme_device,
        min_phoneme_duration=config.phoneme_min_duration_sec,
        confidence_threshold=config.phoneme_confidence_threshold,
    )
    print(f"[1] wav2vec FULL ({config.phoneme_wav2vec_model}, "
          f"device={config.phoneme_device}) ...")
    t0 = time.perf_counter()
    full_segments, warn, full_posteriors = predictor.predict_with_posteriors(
        str(full_wav))
    full_predict_sec = time.perf_counter() - t0
    if warn:
        print(f"[1] wav2vec warning: {warn}")
        return 1
    full_scale = {
        "audio_duration_sec": round(
            full_posteriors.probs.shape[0] * full_posteriors.frame_duration, 2),
        "num_frames": int(full_posteriors.probs.shape[0]),
        "frame_duration_sec": round(full_posteriors.frame_duration, 5),
        "num_segments": len(full_segments),
        "predict_runtime_sec": round(full_predict_sec, 2),
        "device": config.phoneme_device,
    }
    print(f"[1] scale: {full_scale}")
    full_segs_dump = dump_segments(full_segments)
    (outdir / "full_segments.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in full_segs_dump),
        encoding="utf-8")

    print("[1] scoring FULL (gates OFF + ON) ...")
    full_off = run_scoring(config, full_segments, full_posteriors, transcript,
                           skips, word_windows, word_probs, gates_on=False)
    full_on = run_scoring(config, full_segments, full_posteriors, transcript,
                          skips, word_windows, word_probs, gates_on=True)
    print("[1] DTW stats FULL ...")
    full_dtw = dtw_stats([s.phoneme for s in full_segments], phonemes)
    print(f"[1] DTW: {full_dtw}")
    (outdir / "full_diags.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in full_off["diags"]),
        encoding="utf-8")

    full_diag_by_idx = {d["index"]: d for d in full_off["diags"]}

    # ── Stage 2: chọn câu lỗi + câu control ─────────────────────────────────
    sentences = split_sentences(transcript)
    sent_spans = map_sentences_to_spans(sentences, spans)
    span_to_sent = {}
    for si, idxs in enumerate(sent_spans):
        for k in idxs:
            span_to_sent[k] = si

    error_occ, perfect_occ = [], []
    for d in full_off["diags"]:
        if norm_word(d["word"]) in TARGET_WORDS and d["skip_reason"] is None:
            if d["substitutions"] + d["deletions"] > 0:
                error_occ.append(d["index"])
            elif d["matches"] == len(d["correspondences"]):
                perfect_occ.append(d["index"])

    error_sents = sorted({span_to_sent[k] for k in error_occ if k in span_to_sent})
    control_sents = [
        si for si in sorted({span_to_sent[k] for k in perfect_occ
                             if k in span_to_sent})
        if si not in error_sents
    ][:MAX_CONTROL_SENTENCES]
    print(f"[2] target occurrences: error={error_occ} perfect={perfect_occ}")
    print(f"[2] câu lỗi: {error_sents} | câu control: {control_sents}")

    # ── Stage 2b: chấm từng câu ──────────────────────────────────────────────
    sentence_reports = []
    for si in error_sents + control_sents:
        kind = "error" if si in error_sents else "control"
        idxs = sent_spans[si]
        wins = [word_windows[k] for k in idxs if k in word_windows]
        if not wins:
            print(f"[2] câu {si}: không có window nào — bỏ qua")
            continue
        s_start = max(0.0, min(w[0] for w in wins) - SLICE_PAD_SEC)
        s_end = max(w[1] for w in wins) + SLICE_PAD_SEC
        sent_text = sentences[si]
        wav_path = outdir / f"sent_{si:03d}.wav"
        slice_wav(full_wav, wav_path, s_start, s_end)

        t0 = time.perf_counter()
        segs, warn, posts = predictor.predict_with_posteriors(str(wav_path))
        predict_sec = time.perf_counter() - t0
        if warn:
            print(f"[2] câu {si}: wav2vec warning: {warn} — bỏ qua")
            continue
        scale = {
            "audio_duration_sec": round(
                posts.probs.shape[0] * posts.frame_duration, 2),
            "num_frames": int(posts.probs.shape[0]),
            "num_segments": len(segs),
            "predict_runtime_sec": round(predict_sec, 2),
        }
        # Segments của slice, quy về thời gian TUYỆT ĐỐI để so với run full.
        segs_abs = [
            {"phoneme": s.phoneme, "start": round(s.start + s_start, 3),
             "end": round(s.end + s_start, 3), "conf": s.confidence}
            for s in segs
        ]
        (outdir / f"sent_{si:03d}_segments.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in segs_abs),
            encoding="utf-8")

        # Reference câu = đúng text câu trong transcript full (cùng G2P).
        s_phonemes, s_spans, _st, _dp = text_to_ipa_sequence_with_spans(sent_text)
        # Map local span k → global span index (cùng thứ tự token).
        local_to_global = dict(zip(range(len(s_spans)), idxs))
        if len(s_spans) != len(idxs):
            print(f"[WARN] câu {si}: local spans {len(s_spans)} != global {len(idxs)}")
        s_windows = {
            lk: (word_windows[gk][0] - s_start, word_windows[gk][1] - s_start)
            for lk, gk in local_to_global.items() if gk in word_windows
        }
        s_probs = {
            lk: word_probs[gk]
            for lk, gk in local_to_global.items() if gk in word_probs
        }
        sc_off = run_scoring(config, segs, posts, sent_text, {},
                             s_windows, s_probs, gates_on=False)
        sc_on = run_scoring(config, segs, posts, sent_text, {},
                            s_windows, s_probs, gates_on=True)
        s_dtw = dtw_stats([s.phoneme for s in segs], s_phonemes)

        diag_by_local = {d["index"]: d for d in sc_off["diags"]}
        words_cmp = []
        for lk, gk in local_to_global.items():
            fd = full_diag_by_idx.get(gk)
            sd = diag_by_local.get(lk)
            if fd is None or sd is None:
                continue
            is_target = norm_word(fd["word"]) in TARGET_WORDS
            entry = {
                "word": fd["word"], "global_index": gk,
                "is_target": is_target,
                "full_skip_reason": fd["skip_reason"],
                "full": word_block(fd),
                "sentence": word_block(sd),
            }
            # Raw-segment comparison quanh window Whisper của từ.
            if gk in word_windows and (is_target or fd["substitutions"]
                                       + fd["deletions"] > 0):
                w0, w1 = word_windows[gk]
                entry["raw_segments"] = segment_comparison(
                    full_segs_dump, segs_abs,
                    w0 - SEG_WINDOW_PAD_SEC, w1 + SEG_WINDOW_PAD_SEC)
            words_cmp.append(entry)

        sentence_reports.append({
            "sentence_index": si, "kind": kind, "text": sent_text,
            "slice": [round(s_start, 3), round(s_end, 3)],
            "wav2vec_scale": scale,
            "dtw": s_dtw,
            "score_gates_off": score_summary(sc_off["score"], sc_off["diags"]),
            "score_gates_on": score_summary(sc_on["score"], sc_on["diags"]),
            "words": words_cmp,
        })
        tgt = [w["word"] for w in words_cmp if w["is_target"]]
        print(f"[2] câu {si} ({kind}) [{s_start:.1f}-{s_end:.1f}s] targets={tgt} "
              f"| slice acc={sc_off['score'].overall_accuracy:.3f}")

    # ── Stage 3: report ──────────────────────────────────────────────────────
    report = {
        "video": str(video), "transcript_words": len(spans),
        "asr": {"backend": asr_data.get("backend"),
                "elapsed_ms": asr_data.get("elapsed_ms"),
                "duration": asr_data.get("duration")},
        "full_mode": {
            "wav2vec_scale": full_scale,
            "dtw": full_dtw,
            "score_gates_off": score_summary(full_off["score"], full_off["diags"]),
            "score_gates_on": score_summary(full_on["score"], full_on["diags"]),
        },
        "sentences": sentence_reports,
    }
    (outdir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # Console summary
    print("\n" + "=" * 100)
    print("SO SÁNH FULL vs SENTENCE — từ mục tiêu")
    print("=" * 100)
    hdr = (f"{'word':16s} {'sent':>4s} {'kind':7s} "
           f"{'FULL pred_ipa':22s} {'SENT pred_ipa':22s} "
           f"{'cov F/S':>9s} {'S+D F/S':>9s} {'seg_agree':>9s}")
    print(hdr)
    print("-" * 100)
    for sr in sentence_reports:
        for w in sr["words"]:
            if not w["is_target"]:
                continue
            f, s = w["full"], w["sentence"]
            agree = (w.get("raw_segments") or {}).get("segment_agreement")
            print(f"{w['word'][:16]:16s} {sr['sentence_index']:>4d} "
                  f"{sr['kind']:7s} "
                  f"{f['predicted_ipa'][:22]:22s} {s['predicted_ipa'][:22]:22s} "
                  f"{f['coverage']:>4.2f}/{s['coverage']:<4.2f} "
                  f"{f['substitutions'] + f['deletions']:>4d}/"
                  f"{s['substitutions'] + s['deletions']:<4d} "
                  f"{agree if agree is not None else '—':>9}")
    print("\nFULL-mode DTW:", full_dtw)
    print("FULL-mode score (gates OFF):",
          score_summary(full_off["score"], full_off["diags"]))
    print("FULL-mode score (gates ON): ",
          score_summary(full_on["score"], full_on["diags"]))
    print(f"\nOutputs: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

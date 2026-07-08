#!/usr/bin/env python3
"""Diagnose per-word PLAYBACK windows + reference IPA cho các clip "answer for test 1".

Case gốc (2026-07-08, báo cáo UI):
  - câu 1  "through"  ▶ phát cả "day through"
  - câu 6  "the"      IPA đúng hiển thị /ðiː/ nhưng TTS đọc /ðə/
  - câu 7  "read"     ▶ phát cả "read book"
  - câu 8  "eyes"     bạn đọc /z z/ (nghi bleed alignment); "comfortable" nghi cắt ngắn
  - câu 9  "will"     ▶ phát cả "will start"; "workshop" phát lẹm "will" (câu có "9 am")
  - câu 10 "lead"     IPA đúng /led/ thay vì /liːd/ (nghi multiref homograph swap)
  - phần 2 (clip 11, 12): rà mọi từ xem cửa sổ cắt có lẹm từ kề không

Mỗi clip: wav (cache) → ASR (cache) → wav2vec chunked hybrid (như ship) →
compute_phoneme_score ĐỦ tham số production (gates ON, locked truyền xuống,
multiref theo env) + một run multiref OFF để so reference IPA. Dump:
  - bảng từ: whisper window vs playback window (start/end WordPronunciation),
    cờ bleed (playback phủ ≥50% cửa sổ Whisper của từ kề) / truncated (<60% duration);
  - log "Homograph swap" bắt từ logger toeic.phoneme.scoring;
  - clip WAV cắt đúng cửa sổ playback (+ bản _ctx ±0.6s) cho từ target/bị cờ.

DIAGNOSTIC ONLY — không sửa điểm. Chạy:
    python scripts/diag_test1_windows.py [--clips 1,6,7,8,9,10,11,12] [--fresh]
Outputs: outputs/diag_test1/{diag.json, clips/*.wav}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from bench_common import (
    REPO_ROOT,
    build_reference_context,
    dump_segments,
    extract_wav,
    run_scoring,
    slice_wav,
)
from trace_word_case import collect_audio_files

from src import asr  # noqa: E402

from src.config import load_config  # noqa: E402
from src.phoneme.chunking import compute_chunk_spans  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402

AUDIO_DIR = REPO_ROOT / "data" / "audio" / "answer for test 1"
OUT_DIR = REPO_ROOT / "outputs" / "diag_test1"

# Từ được báo lỗi trên UI — luôn cắt clip + dump correspondences cho các từ này.
TARGETS: dict[str, set[str]] = {
    "1": {"through"},
    "6": {"the"},
    "7": {"read"},
    "8": {"eyes", "comfortable"},
    "9": {"will", "workshop", "am"},
    "10": {"lead"},
}

CTX_PAD = 0.6          # clip _ctx: nghe thêm ngữ cảnh hai bên
BLEED_COVER_FRAC = 0.5  # playback phủ ≥50% cửa sổ Whisper từ kề → bleed
TRUNC_FRAC = 0.6        # playback < 60% duration cửa sổ Whisper → nghi cắt ngắn


def run_asr_engine_cached(engine: str, model: str, device: str,
                          wav: Path, cache: Path, fresh: bool) -> dict:
    """Như trace_word_case.run_asr_cached nhưng engine/model tường minh — cho phép
    chạy đúng engine mock_test (whisperx/large-v3) mà UI "chấm cả đề" dùng."""
    if cache.exists() and not fresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    run = asr.transcribe_with_backend(
        str(wav), backend=engine, model_size=model, device=device)
    tr = run.transcription
    data = {
        "text": tr.text, "duration": tr.duration,
        "backend": run.backend_used, "elapsed_ms": run.elapsed_ms,
        "words": [
            {"text": w.text, "start": w.start, "end": w.end,
             "probability": w.probability} for w in tr.words
        ],
    }
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return data


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def match_words_to_spans(score_words, spans) -> dict[int, object]:
    """WordPronunciation list → {span_index: wp}. _build_word_details đi theo thứ tự
    span nhưng có thể bỏ span không có point → walk song song khớp text."""
    out: dict[int, object] = {}
    wi = 0
    for k, span in enumerate(spans):
        if wi < len(score_words) and score_words[wi].word == span.word:
            out[k] = score_words[wi]
            wi += 1
    if wi != len(score_words):
        print(f"[WARN] word/span walk lệch: matched {wi}/{len(score_words)}")
    return out


def bleed_report(k: int, play: tuple[float, float],
                 word_windows: dict, spans) -> list[dict]:
    """Từ kề nào bị cửa sổ playback của từ k phủ lên (theo cửa sổ Whisper của kề)."""
    s, e = play
    hits = []
    for j, (ws, we) in sorted(word_windows.items()):
        if j == k or we <= ws:
            continue
        ov = min(e, we) - max(s, ws)
        if ov <= 0:
            continue
        frac = ov / (we - ws)
        if frac >= 0.15:  # <15% = đệm biên bình thường, bỏ qua
            hits.append({
                "neighbor_index": j, "neighbor": spans[j].word,
                "overlap_sec": round(ov, 3), "covered_frac": round(frac, 3),
                "major": frac >= BLEED_COVER_FRAC,
            })
    return hits


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default="1,6,7,8,9,10,11,12")
    ap.add_argument("--engine", choices=["practice", "mock"], default="practice",
                    help="practice=faster_whisper; mock=engine mock_test (whisperx)")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    stems = [s.strip() for s in args.clips.split(",") if s.strip()]

    config = load_config()
    if args.engine == "mock":
        engine, model = config.asr_engine_mock_test, config.asr_model_mock_test
    else:
        engine, model = config.asr_engine_practice, config.asr_model_practice
    clips_dir = OUT_DIR / f"clips_{args.engine}"
    (OUT_DIR / "wav").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "asr").mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # Bắt log homograph swap / subtoken / merge từ mọi logger toeic.*
    handler = _ListHandler()
    root = logging.getLogger("toeic")
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    files = {p.stem: p for p in collect_audio_files(AUDIO_DIR)}
    missing = [s for s in stems if s not in files]
    if missing:
        print(f"[!] Thiếu clip: {missing}")
        return 1

    predictor = Wav2VecPhonemePredictor(
        model_id=config.phoneme_wav2vec_model, device=config.phoneme_device,
        min_phoneme_duration=config.phoneme_min_duration_sec,
        confidence_threshold=config.phoneme_confidence_threshold,
    )

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "asr": f"{engine}/{model}",
            "wav2vec": config.phoneme_wav2vec_model,
            "chunking": config.phoneme_chunking_strategy,
            "multiref_env": config.phoneme_homograph_multiref,
        },
        "clips": {},
    }

    for stem in stems:
        f = files[stem]
        print(f"\n{'=' * 78}\n[{f.name}]\n{'=' * 78}")
        wav = OUT_DIR / "wav" / f"{stem}.wav"
        extract_wav(f, wav, args.fresh)
        asr_data = run_asr_engine_cached(
            engine, model, config.whisper_device, wav,
            OUT_DIR / "asr" / f"{stem}_{engine}.json", args.fresh)
        print(f"[ASR] {asr_data['duration']:.1f}s | {asr_data['text'][:150]}")
        ctx = build_reference_context(config, asr_data)
        spans = ctx["spans"]

        chunk_spans = compute_chunk_spans(
            [(w["text"], float(w["start"]), float(w["end"]))
             for w in asr_data["words"]],
            float(asr_data["duration"] or 0.0),
            strategy="hybrid",
            max_chunk_sec=config.phoneme_chunk_max_sec,
            min_pause_sec=config.phoneme_chunk_min_pause_sec,
            pad_sec=config.phoneme_chunk_pad_sec,
        ) or None
        segs, warn, posts = predictor.predict_with_posteriors(
            str(wav), chunk_spans=chunk_spans)
        if warn:
            print(f"[!] wav2vec warning: {warn} — bỏ clip")
            continue
        segs_dump = dump_segments(segs)

        handler.lines.clear()
        res_on = run_scoring(
            config, segs, posts, asr_data["text"], ctx["skips"],
            ctx["word_windows"], ctx["word_probs"], gates_on=True,
            word_windows_locked=ctx["word_windows_locked"])
        logs_on = list(handler.lines)
        handler.lines.clear()
        res_off = run_scoring(
            config, segs, posts, asr_data["text"], ctx["skips"],
            ctx["word_windows"], ctx["word_probs"], gates_on=True,
            homograph_on=False,
            word_windows_locked=ctx["word_windows_locked"])

        diag_on = {d["index"]: d for d in res_on["diags"]}
        diag_off = {d["index"]: d for d in res_off["diags"]}
        wp_by_k = match_words_to_spans(res_on["score"].words, spans)
        targets = TARGETS.get(stem, set())

        rows = []
        for k, span in enumerate(spans):
            wp = wp_by_k.get(k)
            d = diag_on.get(k) or {}
            d0 = diag_off.get(k) or {}
            whisper = ctx["word_windows"].get(k)
            play = (
                (wp.start, wp.end)
                if wp is not None and wp.start is not None and wp.end is not None
                else None
            )
            row: dict = {
                "k": k, "word": span.word,
                "ref_ipa": d.get("reference_ipa"),
                "ref_ipa_multiref_off": d0.get("reference_ipa"),
                "pred_ipa": d.get("predicted_ipa"),
                "whisper_window": list(whisper) if whisper else None,
                "playback_window": list(play) if play else None,
                "locked": k in ctx["word_windows_locked"],
                "skip": str(ctx["skips"][k].reason) if k in ctx["skips"] else None,
                "penalty": d.get("penalty"),
            }
            if d.get("reference_ipa") != d0.get("reference_ipa"):
                row["homograph_swapped"] = True
            if play and whisper and whisper[1] > whisper[0]:
                row["bleed"] = bleed_report(k, play, ctx["word_windows"], spans)
                dur_frac = (play[1] - play[0]) / (whisper[1] - whisper[0])
                row["play_vs_whisper_dur"] = round(dur_frac, 3)
                if dur_frac < TRUNC_FRAC:
                    row["truncated_suspect"] = True
            is_target = span.word.lower().strip(".,;:!?\"'()[]{}") in targets
            if is_target and d:
                row["correspondences"] = [
                    f"{c['ref_symbol']}→{c.get('pred_symbol') or '∅'}[{c['status']}]"
                    for c in d["correspondences"]
                ]
                if whisper:
                    w0, w1 = whisper[0] - 0.3, whisper[1] + 0.3
                    row["raw_segments_near_window"] = [
                        {"ph": s["phoneme"], "t": [round(s["start"], 3),
                                                   round(s["end"], 3)],
                         "conf": round(s["conf"], 3)}
                        for s in segs_dump if s["end"] >= w0 and s["start"] <= w1
                    ]
            flagged = is_target or any(
                b["major"] for b in row.get("bleed", ())
            ) or row.get("truncated_suspect")
            if flagged and play:
                safe = span.word.lower().strip(".,;:!?\"'()[]{}") or "x"
                base = f"{stem}_{k:02d}_{safe}"
                slice_wav(wav, clips_dir / f"{base}.wav", *play)
                slice_wav(wav, clips_dir / f"{base}_ctx.wav",
                          max(0.0, play[0] - CTX_PAD), play[1] + CTX_PAD)
                row["clip"] = f"{clips_dir.name}/{base}.wav"
            rows.append(row)

        # Console: bảng gọn + đánh dấu vấn đề
        for row in rows:
            marks = []
            if row.get("homograph_swapped"):
                marks.append(f"SWAP({row['ref_ipa_multiref_off']}→{row['ref_ipa']})")
            for b in row.get("bleed", ()):
                if b["major"]:
                    marks.append(f"BLEED→{b['neighbor']}({b['covered_frac']})")
            if row.get("truncated_suspect"):
                marks.append(f"TRUNC({row['play_vs_whisper_dur']})")
            if row.get("locked"):
                marks.append("locked")
            ww = row["whisper_window"]
            pw = row["playback_window"]
            print(f"  {row['k']:3d} {row['word']:<14s}"
                  f" whisper={f'{ww[0]:.2f}-{ww[1]:.2f}' if ww else '—':<12s}"
                  f" play={f'{pw[0]:.2f}-{pw[1]:.2f}' if pw else '—':<12s}"
                  f" /{row['ref_ipa'] or ''}/ {' '.join(marks)}")
        swap_lines = [ln for ln in logs_on if "Homograph swap" in ln]
        for ln in swap_lines:
            print(f"  [homograph] {ln}")

        report["clips"][stem] = {
            "file": str(f), "duration": asr_data["duration"],
            "transcript": asr_data["text"],
            "chunk_spans": chunk_spans,
            "homograph_log": swap_lines,
            "words": rows,
        }

    out = OUT_DIR / f"diag_{args.engine}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nOutputs: {out}\nClips:   {clips_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build a LISTEN-AND-LABEL review sheet for remaining deletion cases (read-only).

Validation step before any scoring change: are the dominant remaining deletions REAL
omissions, or normal connected-speech reduction, or Whisper-inserted function words?
This script selects ~30 representative deletion cases from the cached post-gate run
(scratchpad/fresh_diagnostics.jsonl), cuts a short listen clip for each (word window
± pad), and writes a CSV with blank `label` / `notes` columns for manual judgement.

No production code or scoring is touched. Run:
    .venv/Scripts/python.exe scripts/build_deletion_review.py
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.phoneme.ipa import is_vowel  # noqa: E402

OUT_DIR = Path(
    r"C:\Users\ADMIN\AppData\Local\Temp\claude\e--repos-speaking-grader"
    r"\299e6f41-64f0-4a98-a6ee-35775f794d11\scratchpad"
)
FRESH_JSONL = OUT_DIR / "fresh_diagnostics.jsonl"
AUDIO_DIR = _REPO_ROOT / "data" / "audio"
CLIP_DIR = OUT_DIR / "deletion_clips"
CLIP_PAD = 0.35           # s of context on each side of the word window
COLLAPSE_RATIO = 0.6

FUNCTION_WORDS = {
    "the", "to", "a", "an", "i", "are", "is", "was", "of", "and", "that", "it",
    "in", "on", "at", "for", "his", "her", "our", "he", "she", "they", "we",
    "you", "this", "as", "but", "or", "so", "be", "by", "with", "had", "have",
}

# How many of each bucket to include in the review set.
QUOTA = {"function_word_collapse": 10, "consonant_deletion": 10,
         "vowel_deletion": 5, "cluster_deletion": 5}


def load_deletion_words() -> list[dict]:
    out = []
    with open(FRESH_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (r.get("type") == "word" and r.get("deletions", 0) > 0
                    and r.get("window_start") is not None):
                out.append(r)
    return out


def deleted_phonemes(w: dict) -> list[str]:
    return [c.get("ref_symbol") or "" for c in w.get("correspondences", [])
            if c.get("status") == "del"]


def max_del_run(w: dict) -> int:
    run = best = 0
    for c in w.get("correspondences", []):
        if c.get("status") == "del":
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def correspondence_str(w: dict) -> str:
    parts = []
    for c in w.get("correspondences", []):
        stt = c.get("status")
        if stt == "del":
            parts.append(f'{c.get("ref_symbol")}→∅')
        elif stt == "sub":
            parts.append(f'{c.get("ref_symbol")}→{c.get("pred_symbol")}')
        else:
            parts.append(f'{c.get("ref_symbol")}=ok')
    return " ".join(parts)


def bucket_of(w: dict) -> str:
    word = (w.get("word") or "").lower()
    cov = w.get("coverage", 1.0)
    dels = deleted_phonemes(w)
    if max_del_run(w) >= 2 and cov < COLLAPSE_RATIO:
        if word in FUNCTION_WORDS or len(w.get("correspondences", [])) <= 3:
            return "function_word_collapse"
        return "cluster_deletion"
    if word in FUNCTION_WORDS and cov < COLLAPSE_RATIO:
        return "function_word_collapse"
    if max_del_run(w) >= 2:
        return "cluster_deletion"
    if any(is_vowel(p) for p in dels):
        return "vowel_deletion"
    return "consonant_deletion"


def select(words: list[dict]) -> list[dict]:
    """Balanced, varied selection: fill quotas, prefer distinct words per bucket."""
    by_bucket: dict[str, list[dict]] = collections.defaultdict(list)
    for w in words:
        by_bucket[bucket_of(w)].append(w)
    chosen = []
    for bucket, quota in QUOTA.items():
        cands = by_bucket.get(bucket, [])
        # most-collapsed / most-deletions first, but cap repeats of the same word.
        cands.sort(key=lambda w: (w.get("coverage", 1.0), -w.get("deletions", 0)))
        seen_word = collections.Counter()
        picked = []
        for w in cands:
            wl = (w.get("word") or "").lower()
            if seen_word[wl] >= 2:          # allow at most 2 of same word (consistency check)
                continue
            seen_word[wl] += 1
            picked.append(w)
            if len(picked) >= quota:
                break
        chosen.extend(picked)
    return chosen


def extract_clip(audio_path: Path, start: float, end: float, dest: Path) -> bool:
    try:
        import librosa
        import soundfile as sf
    except Exception:  # noqa: BLE001
        return False
    try:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        a = max(0, int((start - CLIP_PAD) * sr))
        b = min(len(y), int((end + CLIP_PAD) * sr))
        if b <= a:
            return False
        sf.write(str(dest), y[a:b], sr)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  clip fail {dest.name}: {e}", file=sys.stderr)
        return False


def main() -> int:
    if not FRESH_JSONL.exists():
        sys.exit(f"Missing {FRESH_JSONL}; run analyze_hallucinations.py --regen first.")
    words = load_deletion_words()
    chosen = select(words)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    audio_paths = {p.name: p for ext in ("*.m4a", "*.wav", "*.mp3", "*.flac")
                   for p in AUDIO_DIR.glob(ext)}

    rows = []
    for i, w in enumerate(sorted(chosen, key=lambda x: (bucket_of(x), x.get("audio_id", ""))), 1):
        aid = w.get("audio_id", "")
        ws, we = w["window_start"], w["window_end"]
        case_id = f"D{i:02d}"
        clip_name = f"{case_id}_{(w.get('word') or 'x').strip().replace(' ', '_')}.wav"
        ap = audio_paths.get(aid)
        clip_ok = extract_clip(ap, ws, we, CLIP_DIR / clip_name) if ap else False
        rows.append({
            "case_id": case_id,
            "bucket": bucket_of(w),
            "word": w.get("word", ""),
            "reference_ipa": w.get("reference_ipa", ""),
            "deleted_phonemes": " ".join(deleted_phonemes(w)),
            "predicted_ipa": w.get("predicted_ipa", "") or "(nothing)",
            "coverage": round(w.get("coverage", 1.0), 3),
            "correspondences": correspondence_str(w),
            "audio_file": aid,
            "window_start": round(ws, 3),
            "window_end": round(we, 3),
            "clip": f"deletion_clips/{clip_name}" if clip_ok else "(no clip)",
            "label": "",                     # FILL: real_omission | correct_reduction | whisper_inserted | unsure
            "notes": "",
        })

    review_csv = OUT_DIR / "deletion_review.csv"
    with open(review_csv, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    readme = OUT_DIR / "deletion_review_README.md"
    readme.write_text(
        "# Deletion review — listen & label\n\n"
        f"{len(rows)} cases sampled from post-gate fresh diagnostics (5 data/audio clips). "
        "For each row: open `clip` (already trimmed to the word ± 0.35s) OR scrub `audio_file` "
        "to `window_start`–`window_end`, listen, and fill the **label** column:\n\n"
        "- **real_omission** — speaker genuinely did NOT pronounce the deleted phoneme(s) "
        "(a true pronunciation error → SHOULD be penalized).\n"
        "- **correct_reduction** — phoneme is reduced/linked as normal connected speech; a "
        "native-like rendition (should NOT be penalized → current scoring is a false positive).\n"
        "- **whisper_inserted** — the reference word/phoneme was barely or never said; Whisper "
        "over-transcribed it (reference problem, not the speaker's).\n"
        "- **unsure** — can't tell.\n\n"
        "Buckets: function_word_collapse (the/to/a fully dropped), consonant_deletion, "
        "vowel_deletion, cluster_deletion (≥2 consecutive). Tally of labels per bucket tells us "
        "whether to build a deletion-aware gate (if mostly correct_reduction/whisper_inserted) "
        "or leave scoring as-is (if mostly real_omission).\n",
        encoding="utf-8")

    n_clips = sum(1 for r in rows if r["clip"] != "(no clip)")
    by_b = collections.Counter(r["bucket"] for r in rows)
    print(f"Review set: {len(rows)} cases ({dict(by_b)}); clips cut: {n_clips}")
    print(f"  → {review_csv}")
    print(f"  → {CLIP_DIR}  ({n_clips} wav)")
    print(f"  → {readme}")
    print("\nNext: open deletion_review.csv, listen to each clip, fill `label`. "
          "Send it back and I'll tally → decide on the deletion-aware gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

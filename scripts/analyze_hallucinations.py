#!/usr/bin/env python3
"""Root-cause analysis of REMAINING phoneme recognizer hallucinations (read-only).

Two strictly-separated datasets (their numbers are NEVER merged):

  FRESH  (PRIMARY, post-gate): re-run the production pipeline on data/audio (5 clips)
         with the recognizer-noise gate ON → correspondences incl. penalty_reason
         "recognizer_noise" + Whisper word-windows (so sub_outside_window is populated).
         Cached to scratchpad/fresh_diagnostics.jsonl (tel3 schema). All 8 analyses run here.
  TEL3   (comparison ONLY, pre-gate): parse tel3.jsonl (legacy, no recognizer_noise).
         Only a confusion matrix + collapse table, written to tel3_* files, labeled pre-gate.

This script changes NO production code and feeds nothing back into scoring. Categorization
thresholds are analysis-only knobs (see CONSTANTS). Run:

    .venv/Scripts/python.exe scripts/analyze_hallucinations.py [--regen] [--no-tel3]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

# UTF-8 stdout (Windows console is cp1258 → would crash on IPA).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.asr import transcribe  # noqa: E402
from src.phoneme.analyzer import HybridPhonemeAnalyzer  # noqa: E402
from src.phoneme.diagnostics import (  # noqa: E402
    DiagnosticsContext,
    TelemetryWriter,
    map_reference_words_to_windows,
    percentile,
)
from src.phoneme.ipa import (  # noqa: E402
    is_real_error_substitution,
    is_vowel,
    text_to_ipa_sequence_with_spans,
)
from src.phoneme.l1_vietnamese import PenaltyReason  # noqa: E402
from src.phoneme.scoring import (  # noqa: E402
    PHONEME_LOW_CONF_FLOOR,
    PHONEME_RECOGNIZER_NOISE_CONF,
    PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    PHONEME_RECOGNIZER_NOISE_SIM,
)

# ── Constants ────────────────────────────────────────────────────────────────────
BASELINE_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"   # production default (won A/B)
WHISPER_MODEL = "base"
DEVICE = "cpu"
AUDIO_DIR = _REPO_ROOT / "data" / "audio"
AUDIO_EXTS = ("*.m4a", "*.wav", "*.mp3", "*.flac")
TEL3_PATH = _REPO_ROOT / "tel3.jsonl"
NOISE_REASON = PenaltyReason.RECOGNIZER_NOISE.value

# Analysis-only categorization thresholds (NOT used by scoring).
COLLAPSE_RATIO = 0.6        # word: predicted_len/ref_len < this → word_collapse
MASSIVE_DEL_FRAC = 0.5      # word: deletions/ref_len >= this → massive_deletion
BOUNDARY_SHIFT_FRAC = 0.5   # word: sub_outside_window/subs > this → boundary_shift (descriptive)
DELETED = "∅"

OUT_DIR = Path(
    r"C:\Users\ADMIN\AppData\Local\Temp\claude\e--repos-speaking-grader"
    r"\299e6f41-64f0-4a98-a6ee-35775f794d11\scratchpad"
)
FRESH_JSONL = OUT_DIR / "fresh_diagnostics.jsonl"


# ── Fresh generation (post-gate) ─────────────────────────────────────────────────
def generate_fresh(regen: bool) -> dict:
    """Re-run pipeline on data/audio with the gate ON; cache to FRESH_JSONL.

    Returns generation metadata (incl. G2P drop counts → data/reference uncertainty signal).
    """
    meta_path = OUT_DIR / "fresh_gen_meta.json"
    if FRESH_JSONL.exists() and not regen:
        print(f"[fresh] reuse cache {FRESH_JSONL.name} (use --regen to rebuild)")
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {"audios": [], "note": "cache reused; no meta"}

    audios: list[str] = []
    for ext in AUDIO_EXTS:
        audios.extend(sorted(str(p) for p in AUDIO_DIR.glob(ext)))
    if not audios:
        sys.exit(f"No audio in {AUDIO_DIR}")

    if FRESH_JSONL.exists():
        FRESH_JSONL.unlink()   # TelemetryWriter appends → start clean
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = TelemetryWriter(FRESH_JSONL)
    analyzer = HybridPhonemeAnalyzer(wav2vec_model=BASELINE_MODEL, device=DEVICE)
    if not analyzer.wav2vec_available:
        sys.exit(f"wav2vec backend unavailable for {BASELINE_MODEL}")

    gen_meta: dict = {"model": BASELINE_MODEL, "whisper": WHISPER_MODEL, "audios": []}
    for audio in audios:
        name = Path(audio).name
        print(f"[fresh] {name} …", flush=True)
        tr = transcribe(audio, model_size=WHISPER_MODEL, device=DEVICE)
        ref = tr.text.strip()
        if not ref:
            print(f"  (empty transcript — skip {name})")
            continue
        # Reference spans (same g2p the analyzer uses) → reference words for windows.
        _phon, spans, _stress, _disp = text_to_ipa_sequence_with_spans(ref)
        reference_words = [s.word for s in spans]
        transcript_words = [(w.text, w.start, w.end) for w in tr.words]
        windows = map_reference_words_to_windows(reference_words, transcript_words)

        collected: list = []
        analyzer.analyze(
            audio, reference_text=ref, skips=None,
            diagnostics_sink=lambda d: collected.extend(d),
            word_windows=windows,
        )
        ctx = DiagnosticsContext(
            session_id="fresh", audio_id=name, utterance_id="data_audio"
        )
        writer.emit(ctx, collected)
        # G2P drop signal: transcript word tokens vs reference words that resolved to IPA.
        gen_meta["audios"].append({
            "audio": name,
            "transcript_words": len(transcript_words),
            "reference_words_resolved": len(reference_words),
            "g2p_dropped": max(0, len(transcript_words) - len(reference_words)),
            "windows_mapped": len(windows),
        })
    meta_path.write_text(json.dumps(gen_meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[fresh] wrote {FRESH_JSONL.name} ({len(gen_meta['audios'])} audios)")
    return gen_meta


# ── Loading / normalizing ────────────────────────────────────────────────────────
def load_word_records(path: Path) -> list[dict]:
    """Read a tel3-schema JSONL; return only type=='word' records."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("type") == "word":
                out.append(r)
    return out


def _err_corrs(words: list[dict]) -> list[tuple[dict, dict]]:
    """Flatten to (word_record, correspondence) for ERROR correspondences (sub|del)."""
    out = []
    for w in words:
        for c in w.get("correspondences", []):
            if c.get("status") in ("sub", "del"):
                out.append((w, c))
    return out


# ── Categorization (POST-GATE, analysis-only) ─────────────────────────────────────
def low_conf_threshold(ref_symbol: str) -> float:
    return (PHONEME_RECOGNIZER_NOISE_CONF_VOWEL if is_vowel(ref_symbol)
            else PHONEME_RECOGNIZER_NOISE_CONF)


def phoneme_category(c: dict, word_is_data_issue: bool) -> str:
    """Mutually-exclusive structural category for ONE error correspondence.

    Precedence documented top-to-bottom. Recognizer_noise (gate verdict) is reported
    SEPARATELY as an overlay, not as a structural category.
    """
    if word_is_data_issue:
        return "whisper_g2p_issue"
    status = c.get("status")
    ref = c.get("ref_symbol") or ""
    if status == "del":
        return "vowel_deletion" if is_vowel(ref) else "consonant_deletion"
    if status == "sub":
        pred = c.get("pred_symbol")
        if pred is None:
            return "other"
        if not is_real_error_substitution(ref, pred, sim_floor=PHONEME_RECOGNIZER_NOISE_SIM):
            return "implausible_sub"
        conf = c.get("confidence")
        if conf is not None and conf < low_conf_threshold(ref):
            return "low_confidence"
        return "plausible_sub"
    return "other"


def word_category(w: dict) -> str:
    """Word-level overlay category (one per word)."""
    corrs = w.get("correspondences", [])
    ref_len = len(corrs) or 1
    coverage = w.get("coverage", 1.0)          # predicted/ref phoneme-count ratio
    dels = w.get("deletions", 0)
    subs = w.get("substitutions", 0)
    out_w = w.get("sub_outside_window", 0)
    if coverage < COLLAPSE_RATIO:
        return "word_collapse"
    if dels / ref_len >= MASSIVE_DEL_FRAC:
        return "massive_deletion"
    if subs > 0 and (out_w / subs) > BOUNDARY_SHIFT_FRAC:
        return "boundary_shift"          # DESCRIPTIVE only — not a causal claim
    return "normal"


def _word_is_data_issue(w: dict) -> bool:
    return bool(w.get("skip_reason")) or not (w.get("reference_ipa") or "").strip()


# ── Analyses ─────────────────────────────────────────────────────────────────────
def overall_stats(words: list[dict]) -> dict:
    n_words = len(words)
    n_phon = sum(len(w.get("correspondences", [])) for w in words)
    subs = dels = ins = noise = implausible = 0
    correct_conf: list[float] = []
    sub_conf: list[float] = []
    noise_conf: list[float] = []
    for w in words:
        ins += w.get("insertions", 0)
        for c in w.get("correspondences", []):
            stt = c.get("status")
            conf = c.get("confidence")
            if stt == "ok":
                if conf is not None:
                    correct_conf.append(conf)
            elif stt == "sub":
                subs += 1
                if conf is not None:
                    sub_conf.append(conf)
                pred = c.get("pred_symbol")
                if pred is not None and not is_real_error_substitution(
                    c.get("ref_symbol") or "", pred, sim_floor=PHONEME_RECOGNIZER_NOISE_SIM
                ):
                    implausible += 1
            elif stt == "del":
                dels += 1
            if c.get("penalty_reason") == NOISE_REASON:
                noise += 1
                if conf is not None:
                    noise_conf.append(conf)
    all_conf = correct_conf + sub_conf + [c.get("confidence") for w in words
                                          for c in w.get("correspondences", [])
                                          if c.get("status") == "del"
                                          and c.get("confidence") is not None]

    def dist(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0}
        return {"n": len(vals), "mean": round(st.mean(vals), 4),
                "median": round(st.median(vals), 4),
                "p10": round(percentile(vals, 10), 4),
                "p50": round(percentile(vals, 50), 4),
                "p90": round(percentile(vals, 90), 4)}

    return {
        "words": n_words, "phonemes": n_phon,
        "substitutions": subs, "deletions": dels, "insertions": ins,
        "recognizer_noise": noise, "implausible_substitutions": implausible,
        "confidence_overall": dist([v for v in all_conf if v is not None]),
        "confidence_correct": dist(correct_conf),
        "confidence_substitution": dist(sub_conf),
        "confidence_recognizer_noise": dist(noise_conf),
        "_arrays": {"correct": correct_conf, "sub": sub_conf, "noise": noise_conf},
    }


def categorize(words: list[dict]) -> dict:
    phon_cat = collections.Counter()
    phon_cat_noise = collections.Counter()   # overlay: how many of each were gate-flagged
    word_cat = collections.Counter()
    insertions_total = 0
    for w in words:
        word_cat[word_category(w)] += 1
        insertions_total += w.get("insertions", 0)
        data_issue = _word_is_data_issue(w)
        for c in w.get("correspondences", []):
            if c.get("status") not in ("sub", "del"):
                continue
            cat = phoneme_category(c, data_issue)
            phon_cat[cat] += 1
            if c.get("penalty_reason") == NOISE_REASON:
                phon_cat_noise[cat] += 1
    # Insertions counted at word level (no ref attribution in correspondences).
    phon_cat["consonant_insertion"] += insertions_total
    total = sum(phon_cat.values()) or 1
    return {
        "phoneme_level": {k: {"count": v, "pct": round(100 * v / total, 2),
                              "gate_flagged_recognizer_noise": phon_cat_noise.get(k, 0)}
                          for k, v in phon_cat.most_common()},
        "phoneme_total_error_events": total,
        "word_level": dict(word_cat.most_common()),
        "note": "boundary_shift is DESCRIPTIVE (predicted segment outside Whisper window), "
                "not a causal attribution. consonant_insertion counted at word level.",
    }


def confusion_matrix(words: list[dict]) -> list[dict]:
    """ref→pred (incl. →∅) ranked by count, with normalized per-occurrence rate."""
    ref_occ = collections.Counter()         # occurrences of each ref phoneme (all statuses)
    pair = collections.Counter()
    pair_conf: dict = collections.defaultdict(list)
    pair_noise = collections.Counter()
    for w in words:
        for c in w.get("correspondences", []):
            ref = c.get("ref_symbol") or ""
            ref_occ[ref] += 1
            stt = c.get("status")
            if stt == "ok":
                continue
            pred = DELETED if stt == "del" else (c.get("pred_symbol") or "?")
            key = (ref, pred)
            pair[key] += 1
            if c.get("confidence") is not None:
                pair_conf[key].append(c["confidence"])
            if c.get("penalty_reason") == NOISE_REASON:
                pair_noise[key] += 1
    rows = []
    for (ref, pred), cnt in pair.most_common():
        confs = pair_conf.get((ref, pred), [])
        rows.append({
            "ref": ref, "pred": pred, "count": cnt,
            "ref_occurrences": ref_occ[ref],
            "normalized_rate": round(cnt / ref_occ[ref], 4) if ref_occ[ref] else 0.0,
            "avg_confidence": round(st.mean(confs), 4) if confs else None,
            "recognizer_noise_freq": pair_noise.get((ref, pred), 0),
        })
    return rows


def word_collapse(words: list[dict]) -> list[dict]:
    """Per word-occurrence collapse ratio + per-distinct-word aggregate rate."""
    rows = []
    by_word: dict = collections.defaultdict(lambda: {"occ": 0, "collapsed": 0,
                                                      "ratios": []})
    for w in words:
        ref_len = len(w.get("correspondences", [])) or 1
        ratio = round(w.get("coverage", 1.0), 3)   # predicted/ref ratio (already computed)
        word = w.get("word", "")
        agg = by_word[word.lower()]
        agg["occ"] += 1
        agg["ratios"].append(ratio)
        if ratio < COLLAPSE_RATIO:
            agg["collapsed"] += 1
        rows.append({
            "word": word, "reference_ipa": w.get("reference_ipa", ""),
            "predicted_ipa": w.get("predicted_ipa", ""),
            "ref_len": ref_len, "ratio": ratio,
            "severe_collapse": ratio < COLLAPSE_RATIO,
            "audio_id": w.get("audio_id", ""),
        })
    rows.sort(key=lambda r: r["ratio"])
    per_word = [{"word": k, "occurrences": v["occ"], "collapse_count": v["collapsed"],
                 "collapse_rate": round(v["collapsed"] / v["occ"], 3),
                 "mean_ratio": round(st.mean(v["ratios"]), 3)}
                for k, v in by_word.items()]
    per_word.sort(key=lambda r: (r["collapse_rate"], -r["occurrences"]), reverse=True)
    return rows, per_word


def confidence_overlap(arrays: dict) -> dict:
    """Overlap of confidence distributions: correct vs substitution vs recognizer_noise."""
    correct = arrays.get("correct", [])
    noise = arrays.get("noise", [])
    sub = arrays.get("sub", [])

    def overlap_fraction(a: list[float], b: list[float]) -> float | None:
        """Histogram-overlap (sum of per-bin min of normalized densities), 20 bins 0..1."""
        if not a or not b:
            return None
        bins = 20
        ha = [0.0] * bins
        hb = [0.0] * bins
        for v in a:
            ha[min(bins - 1, int(v * bins))] += 1 / len(a)
        for v in b:
            hb[min(bins - 1, int(v * bins))] += 1 / len(b)
        return round(sum(min(x, y) for x, y in zip(ha, hb)), 4)

    def within(a: list[float], lo: float, hi: float) -> float | None:
        if not a:
            return None
        return round(sum(1 for v in a if lo <= v <= hi) / len(a), 4)

    c_p10 = percentile(correct, 10) if correct else None
    c_p90 = percentile(correct, 90) if correct else None
    return {
        "overlap_correct_vs_noise": overlap_fraction(correct, noise),
        "overlap_correct_vs_sub": overlap_fraction(correct, sub),
        "noise_within_correct_p10_p90": (within(noise, c_p10, c_p90)
                                         if c_p10 is not None else None),
        "interpretation_hint": "high overlap / high within → confidence alone cannot "
                               "separate this group from correct phonemes.",
    }


def per_word_stats(words: list[dict]) -> list[dict]:
    agg: dict = collections.defaultdict(lambda: {"occ": 0, "ref_phon": 0, "noise": 0,
                                                 "dels": 0, "subs": 0, "ratios": []})
    for w in words:
        a = agg[w.get("word", "").lower()]
        a["occ"] += 1
        ref_len = len(w.get("correspondences", []))
        a["ref_phon"] += ref_len
        a["dels"] += w.get("deletions", 0)
        a["subs"] += w.get("substitutions", 0)
        a["ratios"].append(w.get("coverage", 1.0))
        for c in w.get("correspondences", []):
            if c.get("penalty_reason") == NOISE_REASON:
                a["noise"] += 1
    rows = []
    for word, a in agg.items():
        rp = a["ref_phon"] or 1
        rows.append({
            "word": word, "occurrences": a["occ"],
            "noise_rate": round(a["noise"] / a["occ"], 3),
            "deletion_rate": round(a["dels"] / rp, 3),
            "sub_rate": round(a["subs"] / rp, 3),
            "mean_collapse_ratio": round(st.mean(a["ratios"]), 3),
        })
    return rows


def worst_examples(words: list[dict], n: int = 30) -> list[dict]:
    """Words with the most error events (sub+del), richest detail first."""
    scored = []
    for w in words:
        errs = [c for c in w.get("correspondences", [])
                if c.get("status") in ("sub", "del")]
        if not errs:
            continue
        data_issue = _word_is_data_issue(w)
        scored.append((len(errs), w, data_issue))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for _k, w, data_issue in scored[:n]:
        corr_repr = []
        cats = collections.Counter()
        for c in w.get("correspondences", []):
            if c.get("status") not in ("sub", "del"):
                continue
            pred = DELETED if c["status"] == "del" else (c.get("pred_symbol") or "?")
            cat = phoneme_category(c, data_issue)
            cats[cat] += 1
            conf = c.get("confidence")
            corr_repr.append(
                f'{c.get("ref_symbol")}→{pred}'
                f'[{cat},conf={round(conf,3) if conf is not None else "NA"},'
                f'{c.get("penalty_reason")}]'
            )
        out.append({
            "word": w.get("word", ""), "word_category": word_category(w),
            "reference_ipa": w.get("reference_ipa", ""),
            "predicted_ipa": w.get("predicted_ipa", ""),
            "coverage": w.get("coverage"),
            "dominant_phoneme_category": cats.most_common(1)[0][0] if cats else "",
            "correspondences": " | ".join(corr_repr),
            "audio_id": w.get("audio_id", ""),
        })
    return out


# ── CSV / report writers ──────────────────────────────────────────────────────────
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def build_summary(stats: dict, cats: dict, conf_overlap: dict, confusion: list[dict],
                  per_word: list[dict], gen_meta: dict) -> str:
    """Strict, evidence-tagged root-cause summary (no causal blame)."""
    L = []
    A = L.append
    A("## Root-cause summary (POST-GATE / fresh — evidence-tagged, no causal blame)\n")
    n_err = cats["phoneme_total_error_events"]
    A(f"Error events analyzed (sub+del+ins): **{n_err}** over {stats['phonemes']} "
      f"phonemes / {stats['words']} words. recognizer_noise (gate-flagged): "
      f"**{stats['recognizer_noise']}**; implausible subs: **{stats['implausible_substitutions']}**.\n")

    A("\n### Category breakdown (phoneme-level, mutually exclusive)\n")
    for k, v in cats["phoneme_level"].items():
        A(f"- {k}: {v['count']} ({v['pct']}%)"
          + (f" — {v['gate_flagged_recognizer_noise']} gate-flagged" if v['gate_flagged_recognizer_noise'] else ""))
    A(f"\nWord-level overlay: {cats['word_level']}\n")

    # Top problematic phonemes (by error count) and words.
    A("\n### Most error-prone phonemes (ref → pred, top 10 by count)\n")
    for r in confusion[:10]:
        A(f"- {r['ref']} → {r['pred']}: {r['count']} "
          f"(norm {r['normalized_rate']} of {r['ref_occurrences']} occ, "
          f"avg_conf {r['avg_confidence']}, noise {r['recognizer_noise_freq']})")
    A("\n### Most problematic words (by noise rate, then collapse)\n")
    pw = sorted(per_word, key=lambda r: (r["noise_rate"], r["deletion_rate"]), reverse=True)
    for r in pw[:10]:
        A(f"- {r['word']}: noise_rate {r['noise_rate']}, del_rate {r['deletion_rate']}, "
          f"sub_rate {r['sub_rate']}, mean_ratio {r['mean_collapse_ratio']} "
          f"({r['occurrences']}×)")

    A("\n### Confidence separability\n")
    A(f"- correct conf: {stats['confidence_correct']}")
    A(f"- substitution conf: {stats['confidence_substitution']}")
    A(f"- recognizer_noise conf: {stats['confidence_recognizer_noise']}")
    A(f"- overlap(correct,noise)={conf_overlap['overlap_correct_vs_noise']}, "
      f"noise_within_correct_p10_p90={conf_overlap['noise_within_correct_p10_p90']}")

    # Evidence-tagged observations, 3 separate buckets, labels: no evidence/possible/likely.
    A("\n### Evidence-tagged observations (separate buckets; labels = no evidence / possible / likely)\n")
    impl = cats["phoneme_level"].get("implausible_sub", {}).get("count", 0)
    vdel = cats["phoneme_level"].get("vowel_deletion", {}).get("count", 0)
    cdel = cats["phoneme_level"].get("consonant_deletion", {}).get("count", 0)
    collapse_words = cats["word_level"].get("word_collapse", 0)
    bshift_words = cats["word_level"].get("boundary_shift", 0)
    ov = conf_overlap.get("overlap_correct_vs_noise")
    g2p_drop = sum(a.get("g2p_dropped", 0) for a in gen_meta.get("audios", []))

    A("**MODEL BEHAVIOR**")
    A(f"- implausible substitutions = {impl}; vowel_deletion = {vdel}, "
      f"consonant_deletion = {cdel}. "
      + ("[likely] deletions/implausible subs dominate remaining errors."
         if (vdel + cdel + impl) >= 0.5 * n_err else
         "[possible] structural model errors are a minority of events."))
    if ov is not None:
        A(f"- confidence overlap(correct,noise) = {ov}. "
          + ("[likely] confidence alone cannot separate recognizer_noise from correct."
             if ov >= 0.3 else
             "[possible] confidence partially separates noise from correct."))
    A("\n**ALIGNMENT ARTIFACTS (descriptive only)**")
    A(f"- boundary_shift words = {bshift_words} (predicted segment outside Whisper window). "
      "[descriptive] this flags positional mismatch; it is NOT evidence that DTW caused it.")
    A("\n**DATA / REFERENCE UNCERTAINTY**")
    A(f"- G2P-dropped words (transcript word lacked IPA) ≈ {g2p_drop}; "
      f"whisper_g2p_issue events = {cats['phoneme_level'].get('whisper_g2p_issue', {}).get('count', 0)}. "
      + ("[possible] reference uncertainty contributes." if g2p_drop else
         "[no evidence] of reference-construction issues in this sample."))

    A("\n### What to optimize next (evidence-tagged; model change NOT recommended unless data strongly supports)\n")
    if impl + (vdel + cdel) >= 0.5 * n_err:
        A("- [likely] decoder/gate tuning on the current model: deletions + implausible "
          "subs dominate, so target those (e.g. deletion handling, per-class gate).")
    if ov is not None and ov >= 0.3:
        A("- [possible] confidence-based gating has limited headroom (high overlap) — "
          "consider signals beyond raw confidence.")
    A(f"- [descriptive] {collapse_words} word_collapse + {bshift_words} boundary_shift "
      "cases warrant manual inspection (see worst_examples_fresh.csv) before any alignment work.")
    A("- Model swap: NOT recommended on this evidence (prior A/B already showed the sibling "
      "is worse; nothing here overturns that).")
    A(f"\n_Caveat: FRESH sample = {stats['words']} words from 5 clips → directional, not "
      "statistically conclusive. Pre-gate tel3 numbers are reported separately and never merged._")
    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────────
def run_fresh(gen_meta: dict) -> dict:
    words = load_word_records(FRESH_JSONL)
    stats = overall_stats(words)
    cats = categorize(words)
    confusion = confusion_matrix(words)
    collapse_rows, collapse_per_word = word_collapse(words)
    conf_overlap = confidence_overlap(stats.pop("_arrays"))
    pw = per_word_stats(words)
    worst = worst_examples(words, 30)

    write_csv(OUT_DIR / "confusion_matrix_fresh.csv", confusion)
    write_csv(OUT_DIR / "word_collapse_fresh.csv", collapse_rows)
    write_csv(OUT_DIR / "worst_examples_fresh.csv", worst)

    summary_md = build_summary(stats, cats, conf_overlap, confusion, pw, gen_meta)
    return {
        "overall_stats": stats, "categorization": cats,
        "confidence_overlap": conf_overlap,
        "confusion_matrix_top": confusion[:40],
        "word_collapse_per_word_top": collapse_per_word[:30],
        "per_word_stats_top": sorted(pw, key=lambda r: r["noise_rate"], reverse=True)[:30],
        "generation_meta": gen_meta,
        "summary_md": summary_md,
    }


def run_tel3() -> dict:
    """COMPARISON ONLY (pre-gate). Confusion + collapse → tel3_* files. Never merged."""
    if not TEL3_PATH.exists():
        return {}
    words = load_word_records(TEL3_PATH)
    confusion = confusion_matrix(words)
    collapse_rows, collapse_per_word = word_collapse(words)
    stats = overall_stats(words)
    stats.pop("_arrays", None)
    write_csv(OUT_DIR / "confusion_matrix_tel3.csv", confusion)
    write_csv(OUT_DIR / "word_collapse_tel3.csv", collapse_rows)
    return {
        "_label": "PRE-GATE (tel3.jsonl) — comparison only, NOT merged with fresh",
        "overall_stats": stats,
        "confusion_matrix_top": confusion[:30],
        "word_collapse_per_word_top": collapse_per_word[:20],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="rebuild fresh_diagnostics.jsonl")
    ap.add_argument("--no-tel3", action="store_true", help="skip pre-gate tel3 comparison")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gen_meta = generate_fresh(args.regen)
    fresh = run_fresh(gen_meta)
    tel3 = {} if args.no_tel3 else run_tel3()

    report = {
        "FRESH_post_gate_PRIMARY": {k: v for k, v in fresh.items() if k != "summary_md"},
        "TEL3_pre_gate_COMPARISON_ONLY": tel3,
    }
    (OUT_DIR / "hallucination_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Hallucination Root-Cause Analysis\n",
          fresh["summary_md"],
          "\n\n---\n## Appendix: pre-gate tel3 (comparison only — never merged)\n"]
    if tel3:
        md.append(f"{tel3['_label']}\n")
        md.append(f"Overall (pre-gate): {json.dumps(tel3['overall_stats'], ensure_ascii=False)}\n")
        md.append("Top pre-gate confusions:")
        for r in tel3["confusion_matrix_top"][:10]:
            md.append(f"- {r['ref']} → {r['pred']}: {r['count']} (norm {r['normalized_rate']})")
    else:
        md.append("_tel3 comparison skipped._")
    (OUT_DIR / "hallucination_analysis.md").write_text("\n".join(md), encoding="utf-8")

    # Console digest.
    print("\n" + fresh["summary_md"])
    print(f"\nOutputs → {OUT_DIR}")
    for fn in ("hallucination_analysis.md", "hallucination_analysis.json",
               "confusion_matrix_fresh.csv", "word_collapse_fresh.csv",
               "worst_examples_fresh.csv", "confusion_matrix_tel3.csv",
               "word_collapse_tel3.csv"):
        p = OUT_DIR / fn
        print(f"  {'✓' if p.exists() else '·'} {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

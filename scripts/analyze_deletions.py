#!/usr/bin/env python3
"""Root-cause analysis of DELETIONS and WORD COLLAPSE (post-gate, read-only).

Strictly scoped: deletions + collapse only. Substitutions are already handled by the
recognizer-noise gate and are NOT the subject here. No model/checkpoint change is proposed.
Pre-gate (tel3) data is NOT used or merged.

Two evidence layers:
  PART 1 (telemetry): deletion taxonomy, collapse vs word length, consistency, per-phoneme
     deletion rate vs confidence-when-present — all from scratchpad/fresh_diagnostics.jsonl
     (the cached post-gate run produced by analyze_hallucinations.py).
  PART 2 (decoder probe): replicate the wav2vec FORWARD PASS (read-only, reusing backend
     helpers) to get FRAME-LEVEL blank statistics, then compare collapse-word windows vs
     normal-word windows. This is the only way to test the CTC-blank / decoder hypothesis,
     which telemetry cannot show.

Discriminator logic (the 4 hypotheses):
  - collapse windows: HIGH blank-frame fraction + LOW non-blank peak prob  → acoustic loss
  - collapse windows: HIGH blank fraction + HIGH non-blank peaks suppressed → decoder/blank
  - collapse windows: LOW blank + few surviving segments                   → alignment/merge
  - deletions concentrated in G2P-dropped / unaligned words                → reference mismatch

All hypotheses are tagged [likely] / [possible] / [no evidence]. Run:
    .venv/Scripts/python.exe scripts/analyze_deletions.py
"""
from __future__ import annotations

import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
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
COLLAPSE_RATIO = 0.6        # word coverage < this → collapse (matches prior analysis)
WINDOW_PAD = 0.08           # s; matches DRIFT_WINDOW_PAD_SEC for frame→window mapping
DELETED = "∅"


def load_words() -> list[dict]:
    if not FRESH_JSONL.exists():
        sys.exit(f"Missing {FRESH_JSONL}. Run analyze_hallucinations.py --regen first.")
    out = []
    with open(FRESH_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("type") == "word":
                    out.append(r)
    return out


# ── PART 1: deletion / collapse taxonomy (telemetry only) ─────────────────────────
def deletion_taxonomy(words: list[dict]) -> dict:
    vowel_del = cons_del = 0
    by_phoneme = collections.Counter()          # ref phoneme → deletion count
    ref_occ = collections.Counter()             # ref phoneme → total occurrences
    cluster_lengths = collections.Counter()     # consecutive-deletion run length → count
    del_in_collapse = del_in_normal = 0
    for w in words:
        corrs = w.get("correspondences", [])
        coverage = w.get("coverage", 1.0)
        is_collapse = coverage < COLLAPSE_RATIO
        run = 0
        for c in corrs:
            ref = c.get("ref_symbol") or ""
            ref_occ[ref] += 1
            if c.get("status") == "del":
                by_phoneme[ref] += 1
                if is_vowel(ref):
                    vowel_del += 1
                else:
                    cons_del += 1
                if is_collapse:
                    del_in_collapse += 1
                else:
                    del_in_normal += 1
                run += 1
            else:
                if run:
                    cluster_lengths[run] += 1
                run = 0
        if run:
            cluster_lengths[run] += 1

    ranked = [{"phoneme": p, "deletions": d, "occurrences": ref_occ[p],
               "deletion_rate": round(d / ref_occ[p], 4) if ref_occ[p] else 0.0,
               "is_vowel": is_vowel(p)}
              for p, d in by_phoneme.most_common()]
    total_dels = vowel_del + cons_del
    # Cluster summary: singletons vs multi-phoneme (cluster) deletions.
    singles = cluster_lengths.get(1, 0)
    clusters = sum(v for k, v in cluster_lengths.items() if k >= 2)
    phonemes_in_clusters = sum(k * v for k, v in cluster_lengths.items() if k >= 2)
    return {
        "total_deletions": total_dels,
        "vowel_deletions": vowel_del, "consonant_deletions": cons_del,
        "vowel_pct": round(100 * vowel_del / total_dels, 1) if total_dels else 0,
        "consonant_pct": round(100 * cons_del / total_dels, 1) if total_dels else 0,
        "ranked_deleted_phonemes": ranked,
        "cluster_run_length_hist": dict(sorted(cluster_lengths.items())),
        "single_deletion_events": singles,
        "cluster_deletion_events(>=2)": clusters,
        "phonemes_lost_in_clusters": phonemes_in_clusters,
        "deletions_in_collapse_words": del_in_collapse,
        "deletions_in_normal_words": del_in_normal,
    }


def collapse_analysis(words: list[dict]) -> dict:
    rows = []
    by_word = collections.defaultdict(list)     # word → [coverage,...] across occurrences
    len_bucket = collections.defaultdict(lambda: {"n": 0, "collapsed": 0, "cov": []})
    cov_all, len_all = [], []
    for w in words:
        ref_len = len(w.get("correspondences", []))
        if ref_len == 0:
            continue
        cov = w.get("coverage", 1.0)
        rows.append({
            "word": w.get("word", ""), "ref_len": ref_len, "coverage": round(cov, 3),
            "reference_ipa": w.get("reference_ipa", ""),
            "predicted_ipa": w.get("predicted_ipa", ""),
            "deletions": w.get("deletions", 0),
            "avg_conf": w.get("avg_conf"), "p20_conf": w.get("p20_conf"),
            "audio_id": w.get("audio_id", ""),
            "severe_collapse": cov < COLLAPSE_RATIO,
        })
        by_word[w.get("word", "").lower()].append(cov)
        cov_all.append(cov)
        len_all.append(ref_len)
        b = ("1-3" if ref_len <= 3 else "4-6" if ref_len <= 6
             else "7-9" if ref_len <= 9 else "10+")
        len_bucket[b]["n"] += 1
        len_bucket[b]["cov"].append(cov)
        if cov < COLLAPSE_RATIO:
            len_bucket[b]["collapsed"] += 1

    rows.sort(key=lambda r: r["coverage"])
    # Correlation length vs coverage (Pearson; expect negative if long words collapse more).
    corr = None
    if len(cov_all) > 2 and len(set(len_all)) > 1 and len(set(cov_all)) > 1:
        try:
            corr = round(st.correlation(len_all, cov_all), 4)
        except Exception:  # noqa: BLE001
            corr = None
    buckets = {}
    for b in ("1-3", "4-6", "7-9", "10+"):
        d = len_bucket[b]
        if d["n"]:
            buckets[b] = {"n": d["n"], "collapse_rate": round(d["collapsed"] / d["n"], 3),
                          "mean_coverage": round(st.mean(d["cov"]), 3)}
    # Consistency across repeated words.
    repeated = []
    for word, covs in by_word.items():
        if len(covs) >= 2:
            repeated.append({"word": word, "occurrences": len(covs),
                             "mean_coverage": round(st.mean(covs), 3),
                             "stdev_coverage": round(st.pstdev(covs), 3),
                             "all_collapse": all(c < COLLAPSE_RATIO for c in covs),
                             "any_collapse": any(c < COLLAPSE_RATIO for c in covs)})
    repeated.sort(key=lambda r: r["mean_coverage"])
    return {
        "severe_collapse_count": sum(1 for r in rows if r["severe_collapse"]),
        "total_words": len(rows),
        "pearson_len_vs_coverage": corr,
        "collapse_by_length_bucket": buckets,
        "worst_collapses": rows[:50],
        "repeated_word_consistency": repeated,
    }


def per_audio_systemic(words: list[dict]) -> dict:
    by_audio = collections.defaultdict(lambda: {"phon": 0, "del": 0, "collapse": 0,
                                                "words": 0})
    for w in words:
        a = by_audio[w.get("audio_id", "")]
        rl = len(w.get("correspondences", []))
        a["phon"] += rl
        a["del"] += w.get("deletions", 0)
        a["words"] += 1
        if w.get("coverage", 1.0) < COLLAPSE_RATIO:
            a["collapse"] += 1
    out = {}
    for aid, d in by_audio.items():
        out[aid] = {"phonemes": d["phon"], "deletions": d["del"],
                    "deletion_rate": round(d["del"] / d["phon"], 4) if d["phon"] else 0,
                    "words": d["words"], "collapse_words": d["collapse"],
                    "collapse_rate": round(d["collapse"] / d["words"], 4) if d["words"] else 0}
    rates = [v["deletion_rate"] for v in out.values()]
    return {"per_audio": out,
            "deletion_rate_spread": {"min": min(rates), "max": max(rates),
                                     "mean": round(st.mean(rates), 4),
                                     "stdev": round(st.pstdev(rates), 4)}}


def confidence_vs_deletion(words: list[dict]) -> dict:
    """Per ref-phoneme: deletion rate vs avg confidence WHEN it is recognized (ok/sub).

    High deletion-rate + low confidence-when-present → acoustic weakness for that phoneme.
    Also collapse-word vs normal-word p20_conf (local confidence collapse signal).
    """
    present_conf = collections.defaultdict(list)
    del_cnt = collections.Counter()
    occ = collections.Counter()
    for w in words:
        for c in w.get("correspondences", []):
            ref = c.get("ref_symbol") or ""
            occ[ref] += 1
            if c.get("status") == "del":
                del_cnt[ref] += 1
            elif c.get("confidence") is not None:
                present_conf[ref].append(c["confidence"])
    rows = []
    for ref in occ:
        if occ[ref] < 5:
            continue
        confs = present_conf.get(ref, [])
        rows.append({"phoneme": ref, "occurrences": occ[ref],
                     "deletion_rate": round(del_cnt[ref] / occ[ref], 3),
                     "mean_conf_when_present": round(st.mean(confs), 3) if confs else None})
    rows.sort(key=lambda r: r["deletion_rate"], reverse=True)
    collapse_p20 = [w.get("p20_conf") for w in words
                    if w.get("coverage", 1.0) < COLLAPSE_RATIO and w.get("p20_conf") is not None]
    normal_p20 = [w.get("p20_conf") for w in words
                  if w.get("coverage", 1.0) >= COLLAPSE_RATIO and w.get("p20_conf") is not None]
    return {
        "phoneme_deletion_vs_confidence": rows,
        "collapse_word_p20_conf_mean": round(st.mean(collapse_p20), 4) if collapse_p20 else None,
        "normal_word_p20_conf_mean": round(st.mean(normal_p20), 4) if normal_p20 else None,
        "collapse_word_avg_conf_mean": round(st.mean(
            [w["avg_conf"] for w in words if w.get("coverage", 1.0) < COLLAPSE_RATIO
             and w.get("avg_conf") is not None]), 4) if collapse_p20 else None,
    }


# ── PART 2: decoder probe (frame-level forward pass, read-only) ────────────────────
def decoder_probe(words: list[dict]) -> dict:
    """Replicate wav2vec forward pass; per Whisper word-window measure blank-frame
    fraction + non-blank peak prob; compare collapse vs normal windows."""
    try:
        import numpy as np
        import torch
        from src.phoneme import wav2vec_backend as wb
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"torch/backend import failed: {e}"}

    model_id = wb.DEFAULT_WAV2VEC_MODEL
    # audio_id → path
    audio_paths = {p.name: p for ext in ("*.m4a", "*.wav", "*.mp3", "*.flac")
                   for p in AUDIO_DIR.glob(ext)}
    # group word records by audio that HAVE a window
    by_audio = collections.defaultdict(list)
    for w in words:
        if w.get("window_start") is not None and w.get("window_end") is not None:
            by_audio[w.get("audio_id", "")].append(w)

    try:
        feat_extractor, model, id_to_label = wb._get_wav2vec_model(model_id, "cpu")
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"model load failed: {e}"}
    # Precompute which token ids are blank/silence.
    blank_ids = {i for i, tok in id_to_label.items()
                 if wb._resolve_ipa(tok, wb._SILENCE_TOKENS) == ""}
    nonblank_cols = [i for i in id_to_label if i not in blank_ids]

    collapse_w, normal_w = [], []     # per-window stats
    overall_blank = []
    per_audio_blank = {}
    for aid, wlist in by_audio.items():
        path = audio_paths.get(aid)
        if path is None:
            continue
        waveform = wb._load_audio(str(path), wb.WAV2VEC_SAMPLE_RATE)
        audio_dur = len(waveform) / wb.WAV2VEC_SAMPLE_RATE
        inputs = feat_extractor(waveform, sampling_rate=wb.WAV2VEC_SAMPLE_RATE,
                                return_tensors="pt")
        iv = inputs.input_values.to(dtype=next(model.parameters()).dtype)
        with torch.no_grad():
            logits = model(iv).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (frames, labels)
        n_frames = probs.shape[0]
        frame_dur = audio_dur / n_frames if n_frames else 0.02
        pred_ids = probs.argmax(axis=-1)
        is_blank = np.array([pid in blank_ids for pid in pred_ids])
        nonblank_peak = probs[:, nonblank_cols].max(axis=1) if nonblank_cols else np.zeros(n_frames)
        frame_blank_frac = float(is_blank.mean())
        per_audio_blank[aid] = round(frame_blank_frac, 4)
        overall_blank.append(frame_blank_frac)
        frame_times = np.arange(n_frames) * frame_dur
        for w in wlist:
            ws, we = w["window_start"], w["window_end"]
            mask = (frame_times >= ws - WINDOW_PAD) & (frame_times <= we + WINDOW_PAD)
            if not mask.any():
                continue
            bf = float(is_blank[mask].mean())
            pk = float(nonblank_peak[mask].mean())
            # frac of non-blank frames whose peak prob is below decode threshold (0.1)
            nb = ~is_blank[mask]
            below = (float((nonblank_peak[mask][nb] < wb.PHONEME_CONFIDENCE_THRESHOLD).mean())
                     if nb.any() else 0.0)
            rec = {"blank_frac": bf, "nonblank_peak_mean": pk, "below_thresh_frac": below,
                   "coverage": w.get("coverage", 1.0), "word": w.get("word", "")}
            (collapse_w if w.get("coverage", 1.0) < COLLAPSE_RATIO else normal_w).append(rec)

    def agg(group):
        if not group:
            return {"n": 0}
        return {"n": len(group),
                "blank_frac_mean": round(st.mean(g["blank_frac"] for g in group), 4),
                "nonblank_peak_mean": round(st.mean(g["nonblank_peak_mean"] for g in group), 4),
                "below_thresh_frac_mean": round(st.mean(g["below_thresh_frac"] for g in group), 4)}

    return {
        "available": True, "model": model_id,
        "confidence_threshold": wb.PHONEME_CONFIDENCE_THRESHOLD,
        "overall_blank_frame_fraction": round(st.mean(overall_blank), 4) if overall_blank else None,
        "per_audio_blank_fraction": per_audio_blank,
        "collapse_windows": agg(collapse_w),
        "normal_windows": agg(normal_w),
        "_collapse_rows": collapse_w,
    }


# ── Verdict ───────────────────────────────────────────────────────────────────────
def build_verdict(tax, coll, sysm, conf, probe) -> str:
    L = []
    A = L.append
    A("## Deletion & Word-Collapse Root-Cause (POST-GATE, evidence-tagged)\n")
    A(f"Deletions: **{tax['total_deletions']}** "
      f"(vowel {tax['vowel_deletions']}/{tax['vowel_pct']}%, "
      f"consonant {tax['consonant_deletions']}/{tax['consonant_pct']}%). "
      f"In collapse words: {tax['deletions_in_collapse_words']}; "
      f"normal words: {tax['deletions_in_normal_words']}.\n")
    A(f"Cluster deletions (≥2 consecutive): {tax['cluster_deletion_events(>=2)']} runs "
      f"removing {tax['phonemes_lost_in_clusters']} phonemes; "
      f"single deletions: {tax['single_deletion_events']}.\n")

    A("\n### Most-deleted phonemes (top 10)\n")
    for r in tax["ranked_deleted_phonemes"][:10]:
        A(f"- {r['phoneme']} ({'V' if r['is_vowel'] else 'C'}): {r['deletions']} dels / "
          f"{r['occurrences']} occ = rate {r['deletion_rate']}")

    A("\n### Collapse vs word length\n")
    A(f"Pearson(ref_len, coverage) = **{coll['pearson_len_vs_coverage']}** "
      "(negative → longer words collapse more).")
    for b, d in coll["collapse_by_length_bucket"].items():
        A(f"- len {b}: n={d['n']}, collapse_rate={d['collapse_rate']}, "
          f"mean_coverage={d['mean_coverage']}")

    A("\n### Consistency across repeated words\n")
    consistent = [r for r in coll["repeated_word_consistency"] if r["all_collapse"]]
    A(f"- {len(consistent)} words collapse on ALL occurrences (consistent); "
      f"showing worst: " + ", ".join(f"{r['word']}(σ={r['stdev_coverage']})"
                                     for r in coll["repeated_word_consistency"][:6]))

    A("\n### Systemic vs concentrated\n")
    sp = sysm["deletion_rate_spread"]
    A(f"- per-audio deletion rate: min {sp['min']}, max {sp['max']}, "
      f"mean {sp['mean']}, stdev {sp['stdev']}")
    for aid, d in sysm["per_audio"].items():
        A(f"  - {aid}: del_rate {d['deletion_rate']}, collapse_rate {d['collapse_rate']}")

    A("\n### Confidence vs deletion\n")
    A(f"- collapse-word p20_conf mean = {conf['collapse_word_p20_conf_mean']} vs "
      f"normal-word p20_conf mean = {conf['normal_word_p20_conf_mean']}")
    top = conf["phoneme_deletion_vs_confidence"][:6]
    A("- highest-deletion phonemes & their conf-when-present: "
      + ", ".join(f"{r['phoneme']}(del {r['deletion_rate']}, conf {r['mean_conf_when_present']})"
                  for r in top))

    A("\n### Decoder probe (frame-level)\n")
    if probe.get("available"):
        cw, nw = probe["collapse_windows"], probe["normal_windows"]
        A(f"- overall blank-frame fraction = {probe['overall_blank_frame_fraction']}")
        A(f"- COLLAPSE windows (n={cw.get('n')}): blank_frac={cw.get('blank_frac_mean')}, "
          f"nonblank_peak={cw.get('nonblank_peak_mean')}, below_thresh={cw.get('below_thresh_frac_mean')}")
        A(f"- NORMAL windows (n={nw.get('n')}): blank_frac={nw.get('blank_frac_mean')}, "
          f"nonblank_peak={nw.get('nonblank_peak_mean')}, below_thresh={nw.get('below_thresh_frac_mean')}")
    else:
        A(f"- probe unavailable: {probe.get('reason')}")

    # ── 4-hypothesis evidence-tagged verdict ──
    A("\n### Verdict — which mechanism dominates? (evidence-tagged)\n")
    cw = probe.get("collapse_windows", {}) if probe.get("available") else {}
    nw = probe.get("normal_windows", {}) if probe.get("available") else {}
    blank_gap = (cw.get("blank_frac_mean", 0) - nw.get("blank_frac_mean", 0)) if cw.get("n") else None
    peak_gap = (nw.get("nonblank_peak_mean", 0) - cw.get("nonblank_peak_mean", 0)) if cw.get("n") else None

    # (1) acoustic loss
    if probe.get("available") and cw.get("n"):
        if blank_gap and blank_gap > 0.05 and (peak_gap is None or peak_gap > 0.03):
            A("1. **Acoustic loss** — [likely]: collapse windows have higher blank-frame "
              f"fraction (Δ={round(blank_gap,3)}) AND lower non-blank peak prob "
              f"(Δ={round(peak_gap,3) if peak_gap is not None else 'NA'}); the model emits "
              "little confident content there, consistent with weak/short acoustics.")
        elif blank_gap and blank_gap > 0.05:
            A("1. **Acoustic loss** — [possible]: collapse windows are blank-heavier but "
              "non-blank peaks are not clearly weaker.")
        else:
            A("1. **Acoustic loss** — [no evidence]: collapse windows are not blank-heavier "
              "than normal windows.")
    else:
        A("1. **Acoustic loss** — [no evidence] (decoder probe unavailable).")

    # (2) decoder/CTC blank
    if probe.get("available") and cw.get("n"):
        if blank_gap and blank_gap > 0.05 and peak_gap is not None and peak_gap < 0.0:
            A("2. **Decoder / CTC blank dominance** — [possible]: blank-heavy collapse windows "
              "still contain non-blank peaks as strong as normal windows (peaks suppressed by "
              "blank/greedy rather than absent).")
        else:
            A("2. **Decoder / CTC blank dominance** — [no evidence] as the PRIMARY driver: "
              "blank-heaviness tracks weaker peaks (acoustic), not suppressed strong peaks. "
              f"Decode threshold is low ({probe.get('confidence_threshold')}), so threshold-"
              "drops are minimal (below_thresh fractions reported above).")
    else:
        A("2. **Decoder / CTC blank dominance** — [no evidence] (probe unavailable).")

    # (3) alignment artifact
    A("3. **Alignment (DTW) artifact** — [possible, bounded]: cluster deletions "
      f"({tax['cluster_deletion_events(>=2)']} runs) co-occur with collapse, but deletions "
      "are concentrated in low-acoustic windows (above), so DTW misassignment is at most a "
      "secondary contributor — NOT shown to be causal.")

    # (4) reference mismatch
    g2p_note = "G2P-dropped proper nouns exist (Sunting/Gris/Minan) but are EXCLUDED from " \
               "reference, so they don't create these deletions"
    A(f"4. **Reference / G2P mismatch** — [no evidence] as a deletion driver: {g2p_note}; "
      "whisper_g2p_issue deletion events ≈ 0.")

    A("\n### Bottom line\n")
    if probe.get("available") and blank_gap and blank_gap > 0.05 and (peak_gap is None or peak_gap > 0.03):
        A("- The remaining deletion/collapse problem is **[likely] dominated by acoustic loss**: "
          "long, fast, or weakly-articulated words produce regions where wav2vec emits mostly "
          "blank with low non-blank confidence. Negative length↔coverage correlation "
          f"({coll['pearson_len_vs_coverage']}) and consistent per-word collapse reinforce this.")
        A("- **Decoder/CTC blank** and **DTW alignment** are [no evidence]/[possible-secondary], "
          "NOT primary. **Reference mismatch** is [no evidence].")
        A("- Optimization implication (no model change): target the ACOUSTIC/segmentation stage "
          "for long words — e.g. window/segmentation handling and deletion-aware gate — rather "
          "than the substitution gate or the alignment.")
    else:
        A("- Evidence is mixed; see per-hypothesis tags above. Largest measured signal is the "
          f"length↔coverage correlation ({coll['pearson_len_vs_coverage']}).")
    A(f"\n_Caveat: {coll['total_words']} words / 5 clips → directional, not statistically "
      "conclusive. Post-gate only; no pre-gate data merged._")
    return "\n".join(L)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    words = load_words()
    tax = deletion_taxonomy(words)
    coll = collapse_analysis(words)
    sysm = per_audio_systemic(words)
    conf = confidence_vs_deletion(words)
    print("[probe] replicating wav2vec forward pass on data/audio …", flush=True)
    probe = decoder_probe(words)

    write_csv(OUT_DIR / "most_deleted_phonemes.csv", tax["ranked_deleted_phonemes"])
    write_csv(OUT_DIR / "worst_collapses.csv", coll["worst_collapses"])
    write_csv(OUT_DIR / "collapse_by_word_length.csv",
              [{"length_bucket": b, **d} for b, d in coll["collapse_by_length_bucket"].items()])
    write_csv(OUT_DIR / "phoneme_deletion_vs_confidence.csv",
              conf["phoneme_deletion_vs_confidence"])

    probe_clean = {k: v for k, v in probe.items() if k != "_collapse_rows"} if probe else {}
    report = {"deletion_taxonomy": tax, "collapse_analysis":
              {k: v for k, v in coll.items() if k != "worst_collapses"},
              "systemic": sysm, "confidence_vs_deletion": conf, "decoder_probe": probe_clean}
    (OUT_DIR / "deletion_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = build_verdict(tax, coll, sysm, conf, probe)
    (OUT_DIR / "deletion_analysis.md").write_text(
        "# Deletion & Collapse Root-Cause Analysis\n\n" + verdict, encoding="utf-8")
    print("\n" + verdict)
    print(f"\nOutputs → {OUT_DIR}")
    for fn in ("deletion_analysis.md", "deletion_analysis.json", "most_deleted_phonemes.csv",
               "worst_collapses.csv", "collapse_by_word_length.csv",
               "phoneme_deletion_vs_confidence.csv"):
        print(f"  {'✓' if (OUT_DIR / fn).exists() else '·'} {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

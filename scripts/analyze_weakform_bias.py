#!/usr/bin/env python3
"""Đo phạm vi lỗi "weak-form reference" trên function words đa-entry CMUdict.

Case gốc (2026-07-05): "are" luôn được chọn reference = entry RÚT GỌN (ER0 → /ɜː/)
vì `_entry_score` ưu tiên 0-nhấn cho FUNCTION_WORDS — bất kể ngữ cảnh (kể cả đầu câu
hỏi "Are you...?" luôn đọc đầy đủ /ɑːr/). Trace xác nhận: cả 2 lần "are" trong
data/audio/answer for test 1/3.weba, wav2vec nghe đúng /ɑː/ nhưng bị chấm sub
ɜː→ɑː (medium) vì reference khoá cứng vào dạng rút gọn.

Script này quét TOÀN BỘ 12 clip cache (outputs/case_project/wav+asr) để đo mức độ
lan rộng: với mỗi function word đa-entry, đếm substitution thật xảy ra và xem có
khớp mẫu "ref=dạng rút gọn, pred=dạng mạnh/citation" không — bằng chứng thực tế
thay vì suy đoán ngôn ngữ học thuần túy.

DIAGNOSTIC ONLY. Chạy:
    python scripts/analyze_weakform_bias.py [--outdir outputs/case_project]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from bench_common import REPO_ROOT, build_reference_context, run_scoring

from src.config import load_config  # noqa: E402
from src.phoneme.ipa.g2p import (  # noqa: E402
    _arpabet_tokens_to_ipa_stress,
    _entry_score,
    _finalize_stress,
    _get_cmudict,
)
from src.phoneme.ipa.phoneme_set import FUNCTION_WORDS  # noqa: E402
from src.phoneme.wav2vec_backend import Wav2VecPhonemePredictor  # noqa: E402


def ipa_of(entry: list[str]) -> str:
    symbols, stresses, syllables = _arpabet_tokens_to_ipa_stress(entry)
    symbols, stresses = _finalize_stress(symbols, stresses, syllables)
    return "".join(symbols)


def multi_entry_function_words() -> dict[str, dict]:
    """{word: {chosen_func_ipa, chosen_content_ipa, differs, entries}} cho từ đa-entry."""
    d = _get_cmudict()
    out = {}
    for w in sorted(FUNCTION_WORDS):
        entries = d.get(w)
        if not entries or len(entries) < 2:
            continue
        scored = [
            (e, _entry_score(e, is_function_word=True), _entry_score(e, is_function_word=False))
            for e in entries
        ]
        chosen_func = min(scored, key=lambda t: t[1])[0]
        chosen_content = min(scored, key=lambda t: t[2])[0]
        ipa_func = ipa_of(chosen_func)
        ipa_content = ipa_of(chosen_content)
        out[w] = {
            "entries": [" ".join(e) for e in entries],
            "chosen_func_ipa": ipa_func,
            "chosen_content_ipa": ipa_content,
            "differs": ipa_func != ipa_content,
        }
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="outputs/case_project")
    args = ap.parse_args()

    config = load_config()
    cache = REPO_ROOT / args.outdir
    wavs = sorted((cache / "wav").glob("*.wav"),
                  key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, 0))
    if not wavs:
        print(f"[!] Không có wav cache tại {cache/'wav'} — chạy trace_word_case trước.")
        return 1

    fw_info = multi_entry_function_words()
    differs_words = {w for w, info in fw_info.items() if info["differs"]}
    print(f"[*] {len(fw_info)} function word đa-entry, {len(differs_words)} có chọn func≠content")

    predictor = Wav2VecPhonemePredictor(
        model_id=config.phoneme_wav2vec_model, device=config.phoneme_device,
        min_phoneme_duration=config.phoneme_min_duration_sec,
        confidence_threshold=config.phoneme_confidence_threshold,
    )

    # {word: Counter[(ref_word_ipa, pred_word_ipa)]} — gộp theo TỪ (không phải phoneme
    # đơn lẻ) vì ta cần biết cả từ được chấm sao, không chỉ 1 phoneme trong từ.
    word_occurrences: dict[str, list[dict]] = collections.defaultdict(list)

    n_clips = 0
    for wav in wavs:
        asr_path = cache / "asr" / f"{wav.stem}.json"
        if not asr_path.exists():
            continue
        asr_data = json.loads(asr_path.read_text(encoding="utf-8"))
        ctx = build_reference_context(config, asr_data)
        segs, warn, posts = predictor.predict_with_posteriors(str(wav))
        if warn:
            print(f"[!] {wav.name}: wav2vec warning {warn} — bỏ qua")
            continue
        results = {
            mode: run_scoring(
                config, segs, posts, asr_data["text"], ctx["skips"],
                ctx["word_windows"], ctx["word_probs"], gates_on=False,
                homograph_on=on,
            )
            for mode, on in (("homograph_off", False), ("homograph_on", True))
        }
        n_clips += 1
        # Gộp diag OFF/ON theo (word, index) — index span không đổi giữa 2 mode.
        for d_off in results["homograph_off"]["diags"]:
            w = (d_off["word"] or "").lower().strip(".,;:!?\"'()[]{}")
            if w not in differs_words:
                continue
            d_on = next(
                (d for d in results["homograph_on"]["diags"]
                 if d["index"] == d_off["index"]), None)
            occ = {"clip": wav.stem, "index": d_off["index"]}
            for mode, d in (("off", d_off), ("on", d_on)):
                if d is None:
                    occ[mode] = None
                    continue
                subs = [c for c in d["correspondences"] if c["status"] == "sub"]
                occ[mode] = {
                    "reference_ipa": d["reference_ipa"],
                    "predicted_ipa": d["predicted_ipa"],
                    "n_sub": len(subs),
                    "subs": [(c["ref_symbol"], c.get("pred_symbol")) for c in subs],
                }
            word_occurrences[w].append(occ)

    print(f"\n[*] Quét {n_clips} clip xong. Occurrences theo từ (chỉ từ đa-entry differs):\n")
    report_rows = []
    for w in sorted(word_occurrences):
        occs = word_occurrences[w]
        info = fw_info[w]
        n_flag_off = sum(1 for o in occs if o["off"] and o["off"]["n_sub"] > 0)
        n_flag_on = sum(1 for o in occs if o["on"] and o["on"]["n_sub"] > 0)
        print(f"{w:8s} chọn=/{info['chosen_func_ipa']}/ (citation=/{info['chosen_content_ipa']}/) "
              f"entries={info['entries']}")
        print(f"         {len(occs)} lần xuất hiện | flagged: OFF={n_flag_off}  ON={n_flag_on}")
        for o in occs:
            off, on = o["off"], o["on"]
            if (off and off["n_sub"]) or (on and on["n_sub"]):
                off_s = (f"ref=/{off['reference_ipa']}/ pred=/{off['predicted_ipa']}/ "
                         f"subs={off['subs']}") if off else "—"
                on_s = (f"ref=/{on['reference_ipa']}/ pred=/{on['predicted_ipa']}/ "
                        f"subs={on['subs']}") if on else "—"
                print(f"           [{o['clip']}] OFF: {off_s}")
                print(f"           {'':>{len(o['clip'])+2}} ON : {on_s}")
        report_rows.append({
            "word": w, "chosen_func_ipa": info["chosen_func_ipa"],
            "chosen_content_ipa": info["chosen_content_ipa"],
            "entries": info["entries"],
            "n_occurrences": len(occs),
            "n_flagged_off": n_flag_off, "n_flagged_on": n_flag_on,
            "occurrences": occs,
        })

    # Từ có differs=True nhưng KHÔNG xuất hiện trong 12 clip — chưa có bằng chứng thực tế.
    unseen = sorted(differs_words - set(word_occurrences))
    if unseen:
        print(f"\n[*] Không xuất hiện trong corpus (chưa có bằng chứng thực tế): {unseen}")

    outdir = REPO_ROOT / "outputs" / "homographs"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "weakform_bias.json"
    out_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_clips": n_clips,
        "all_multi_entry_function_words": fw_info,
        "differs_words": sorted(differs_words),
        "unseen_in_corpus": unseen,
        "rows": report_rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nOutputs: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

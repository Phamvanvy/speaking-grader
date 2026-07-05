#!/usr/bin/env python3
"""Thống kê homograph trong CMUdict — đo blast radius của cơ chế chọn entry hiện tại.

Bối cảnh (case "project" 2026-07-05): `_rank_cmudict_entries` là context-free
(min `_entry_score`) nên MỌI từ đa-entry luôn bị chọn CÙNG một entry bất kể ngữ
cảnh — với "project" nó chọn entry động từ /prədʒekt/, gây 2 false sub khi user
đọc danh từ. Script này trả lời 3 câu hỏi trước khi thiết kế lại:

  1. CMUdict có bao nhiêu từ đa-entry (homograph candidate)?
  2. Bao nhiêu từ trong số đó việc chọn entry KHÔNG ảnh hưởng scoring
     (các entry tương đương sau normalize_ipa / phonemes_match tolerance)?
  3. Bao nhiêu từ chọn sai entry sẽ tạo sub/del thật ở scorer (= cần POS/ngữ
     cảnh để chọn đúng), và trong đó bao nhiêu mang chữ ký noun/verb stress-shift
     (record, project, permit...)?

Phân loại 1 từ đa-entry (so entry ĐƯỢC RANKER CHỌN với từng entry còn lại,
trên chuỗi IPA đã normalize_ipa — đúng dạng scorer so khớp):
  - identical_after_normalize: mọi entry cho cùng chuỗi normalize → vô hại.
  - tolerance_equivalent: khác chuỗi nhưng cùng độ dài và mọi cặp lệch đều
    phonemes_match(reducible=True) → tolerance nuốt được, gần như vô hại.
  - material: có ít nhất 1 entry thay thế lệch thật (sub không match hoặc
    thêm/bớt âm) → chọn sai entry = false error. Đây là tập cần context.
    - material + stress_shift: các entry có vị trí primary stress khác nhau
      trên cùng số âm tiết → chữ ký noun/verb homograph cổ điển, chắc chắn
      cần POS.

DIAGNOSTIC ONLY. Chạy:  python scripts/analyze_homographs.py
Ghi outputs/homographs/homograph_stats.json + summary console.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bench_common import REPO_ROOT, edit_align

from src.phoneme.ipa import normalize_ipa  # noqa: E402
from src.phoneme.ipa.g2p import _entry_score, _get_cmudict  # noqa: E402
from src.phoneme.ipa.phoneme_set import ARPABET_TO_IPA  # noqa: E402
from src.phoneme.ipa.similarity import phonemes_match  # noqa: E402

# Danh sách user nêu + project — kiểm tra chi tiết từng từ.
WATCHLIST = [
    "project", "record", "object", "present", "permit", "contract", "survey",
    "increase", "decrease", "progress", "import", "export", "conflict", "contest",
]

_VOWEL_BASES = {b for b, ipa in ARPABET_TO_IPA.items()
                if normalize_ipa(ipa) in {"ɔ", "æ", "ə", "aʊ", "aɪ", "e", "ɜ",
                                          "eɪ", "ɪ", "i", "əʊ", "ɔɪ", "ʊ", "u"}}


def entry_ipa_norm(tokens: list[str]) -> tuple[str, ...]:
    """ARPAbet entry → chuỗi IPA đã normalize (đúng dạng scorer so khớp)."""
    return tuple(
        normalize_ipa(ARPABET_TO_IPA[t.rstrip("012")])
        for t in tokens if t.rstrip("012") in ARPABET_TO_IPA
    )


def entry_ipa_raw(tokens: list[str]) -> str:
    return "".join(ARPABET_TO_IPA.get(t.rstrip("012"), "?") for t in tokens)


def primary_stress_syllable(tokens: list[str]) -> int | None:
    """Index âm tiết (0-based, đếm theo nguyên âm) mang stress digit 1."""
    syl = 0
    for t in tokens:
        base = t.rstrip("012")
        if base in _VOWEL_BASES:
            if t.endswith("1"):
                return syl
            syl += 1
    return None


def syllable_count(tokens: list[str]) -> int:
    return sum(1 for t in tokens if t.rstrip("012") in _VOWEL_BASES)


def pair_difference(picked: tuple[str, ...], alt: tuple[str, ...]) -> str:
    """So 2 chuỗi normalize: 'identical' | 'tolerance' | 'material'."""
    if picked == alt:
        return "identical"
    if len(picked) != len(alt):
        return "material"  # thêm/bớt âm → false del/ins
    for a, b in zip(picked, alt):
        if a != b and not phonemes_match(a, b, reducible=True):
            return "material"
    return "tolerance"


def classify_word(word: str, entries: list[list[str]]) -> dict:
    scores = [_entry_score(e, is_function_word=False) for e in entries]
    picked_i = min(range(len(entries)), key=lambda i: scores[i])
    picked_norm = entry_ipa_norm(entries[picked_i])

    worst = "identical"
    for i, e in enumerate(entries):
        if i == picked_i:
            continue
        d = pair_difference(picked_norm, entry_ipa_norm(e))
        if d == "material":
            worst = "material"
            break
        if d == "tolerance":
            worst = "tolerance"

    stress_shift = False
    if worst == "material":
        stress_positions = set()
        for e in entries:
            pos = primary_stress_syllable(e)
            if pos is not None and syllable_count(e) == syllable_count(entries[0]):
                stress_positions.add(pos)
        stress_shift = len(stress_positions) > 1

    return {
        "word": word,
        "n_entries": len(entries),
        "picked_index": picked_i,
        "picked_arpabet": " ".join(entries[picked_i]),
        "picked_ipa": entry_ipa_raw(entries[picked_i]),
        "entries": [
            {"arpabet": " ".join(e), "ipa": entry_ipa_raw(e),
             "score": round(scores[i], 4)}
            for i, e in enumerate(entries)
        ],
        "category": ("identical_after_normalize" if worst == "identical"
                     else "tolerance_equivalent" if worst == "tolerance"
                     else "material"),
        "stress_shift": stress_shift,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    outdir = REPO_ROOT / "outputs" / "homographs"
    outdir.mkdir(parents=True, exist_ok=True)

    cmu = _get_cmudict()
    multi = {w: es for w, es in cmu.items() if len(es) >= 2}
    print(f"[1] CMUdict: {len(cmu)} từ | đa-entry (homograph candidate): "
          f"{len(multi)} ({100 * len(multi) / len(cmu):.1f}%)")
    print("[2] Cơ chế hiện tại context-free → 100% từ đa-entry LUÔN bị chọn "
          "cùng 1 entry (min _entry_score), bất kể ngữ cảnh.\n")

    cats: dict[str, list[dict]] = {
        "identical_after_normalize": [], "tolerance_equivalent": [], "material": [],
    }
    for w, es in multi.items():
        r = classify_word(w, es)
        cats[r["category"]].append(r)

    material = cats["material"]
    stress_shift = [r for r in material if r["stress_shift"]]
    # Từ "sạch" (chỉ chữ cái, không dấu nháy/số) — loại tên riêng viết tắt kiểu "aaa".
    alpha_material = [r for r in material if r["word"].isalpha()]
    alpha_stress = [r for r in stress_shift if r["word"].isalpha()]

    print(f"[3] Phân loại {len(multi)} từ đa-entry theo tác động scoring "
          f"(so entry ranker chọn vs mọi entry còn lại, trên IPA normalize):")
    print(f"    - identical_after_normalize : {len(cats['identical_after_normalize']):6d}  (vô hại — chỉ khác stress digit/duplicate)")
    print(f"    - tolerance_equivalent      : {len(cats['tolerance_equivalent']):6d}  (tolerance nuốt được)")
    print(f"    - material                  : {len(material):6d}  (chọn sai = false sub/del → CẦN context)")
    print(f"        trong đó stress-shift noun/verb signature: {len(stress_shift)}")
    print(f"        material chỉ-chữ-cái (loại viết tắt/tên lạ): {len(alpha_material)}")
    print(f"        stress-shift chỉ-chữ-cái: {len(alpha_stress)}")

    print(f"\n[4] Watchlist ({len(WATCHLIST)} từ user nêu):")
    watch_rows = []
    for w in WATCHLIST:
        es = cmu.get(w)
        if not es:
            print(f"    {w:12s} KHÔNG có trong CMUdict")
            continue
        r = classify_word(w, es) if len(es) >= 2 else {
            "word": w, "n_entries": 1, "category": "single_entry",
            "picked_ipa": entry_ipa_raw(es[0]), "stress_shift": False,
            "picked_arpabet": " ".join(es[0]), "picked_index": 0,
            "entries": [{"arpabet": " ".join(es[0]),
                         "ipa": entry_ipa_raw(es[0]), "score": None}],
        }
        watch_rows.append(r)
        alts = " | ".join(f"/{e['ipa']}/ s={e['score']}" for e in r["entries"])
        flag = " ⚠ STRESS-SHIFT" if r.get("stress_shift") else ""
        print(f"    {w:12s} {r['category']:26s} chọn /{r['picked_ipa']}/"
              f"{flag}\n{'':16s}entries: {alts}")

    sample_stress = sorted(alpha_stress, key=lambda r: r["word"])[:40]
    print("\n[5] Mẫu 40 từ material+stress-shift (chỉ chữ cái):")
    print("    " + ", ".join(r["word"] for r in sample_stress))

    stats = {
        "cmudict_total_words": len(cmu),
        "multi_entry_words": len(multi),
        "always_same_entry_pct": 100.0,
        "categories": {k: len(v) for k, v in cats.items()},
        "material_stress_shift": len(stress_shift),
        "material_alpha_only": len(alpha_material),
        "stress_shift_alpha_only": len(alpha_stress),
        "watchlist": watch_rows,
        "material_words_alpha": sorted(r["word"] for r in alpha_material),
        "stress_shift_words_alpha": sorted(r["word"] for r in alpha_stress),
    }
    (outdir / "homograph_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    # Bảng chi tiết material để review khi thiết kế lại.
    (outdir / "material_words.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False)
                  for r in sorted(material, key=lambda r: r["word"])),
        encoding="utf-8")
    print(f"\nOutputs: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

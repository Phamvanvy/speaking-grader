#!/usr/bin/env python3
"""Sinh corpus bench acoustic tiếng Hàn (M2) bằng edge-tts + ffmpeg.

Hai nhóm clip:
  - native: câu chuẩn, TTS đọc ĐÚNG chính tả — đo false-error rate của pipeline
    (reference đúng + audio đúng → lỗi báo ra là lỗi HỆ THỐNG, không phải học viên).
  - error: TTS đọc bản RESPELL mô phỏng lỗi học viên có chủ đích (aspirated→plain,
    ʌ→o, ɯ→u, thiếu 비음화/유음화...), chấm против reference GỐC — đo khả năng
    PHÁT HIỆN đúng lỗi + độ tách điểm so với twin native.

Deterministic về nội dung (câu + respell cố định trong file này); audio phụ thuộc
phiên bản voice edge-tts — corpus sinh ra được commit/lưu lại, KHÔNG regen mỗi lần
bench (so sánh giữa model phải cùng audio).

Usage:
    python scripts/gen_bench_ko_corpus.py            # sinh vào data/bench/ko/
    python scripts/gen_bench_ko_corpus.py --fresh    # ghi đè clip đã có
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "bench" / "ko"

VOICES = ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"]

# Câu native — mỗi câu nhắm ≥1 hiện tượng 표준발음법 / nguyên âm khó.
NATIVE_SENTENCES: list[tuple[str, str]] = [
    ("n01", "국물이 뜨거워요"),            # 비음화 궁물 + liaison 무리
    ("n02", "학교에 갑니다"),              # 경음화 학꾜 + 비음화 감니다
    ("n03", "같이 가요"),                  # 구개음화 가치
    ("n04", "정말 좋아요"),                # ㅎ탈락 조아요
    ("n05", "입학을 축하합니다"),          # 격음화 이팍 / 추카
    ("n06", "신라 시대 역사"),             # 유음화 실라
    ("n07", "옷이 예뻐요"),                # liaison 오시
    ("n08", "책을 읽어 보세요"),           # double-coda liaison 일거
    ("n09", "앉아서 기다려요"),            # 안자서
    ("n10", "있어요 없어요"),              # 이써요 업써요
    ("n11", "감사합니다"),                 # 비음화 함니다
    ("n12", "밥을 먹었어요"),              # 바블 머거써요
    ("n13", "꽃이 피었어요"),              # 꼬치
    ("n14", "부엌에서 요리해요"),          # 부어케서
    ("n15", "어머니와 아버지"),            # nguyên âm ʌ
    ("n16", "오늘은 월요일이에요"),        # o / wʌ
    ("n17", "하늘이 흐려요"),              # ɯ + liaison 하느리
    ("n18", "우유를 마셔요"),              # u + ㅅ trước i (ɕ)
    ("n19", "커피가 뜨거워요"),            # kʰ + ㄸ
    ("n20", "가게에서 과자를 사요"),       # lenis voiced giữa nguyên âm + wa
    ("n21", "저는 베트남 사람이에요"),     # tɕ + liaison 사라미에요
    ("n22", "한국어 공부가 재미있어요"),   # tổng hợp
    ("n23", "천천히 말해 주세요"),         # h giữa nguyên âm (yếu trong khẩu ngữ)
    ("n24", "물을 좀 주세요"),             # coda ㄹ + liaison 무를
]

# Bản respell mô phỏng lỗi: (id, twin native id, text TTS đọc, error_type,
# (ref_phoneme, heard_phoneme HOẶC None nếu deletion), mô tả)
ERROR_SENTENCES: list[tuple[str, str, str, str, tuple[str, str | None], str]] = [
    ("e01", "n03", "가지 가요", "aspirated_to_plain", ("tɕʰ", "tɕ"),
     "같이[가치] đọc thành 가지 — mất bật hơi ㅊ→ㅈ"),
    ("e02", "n02", "하교에 갑니다", "coda_stop_missing", ("k", None),
     "학교 đọc thành 하교 — mất batchim ㄱ"),
    ("e03", "n01", "구물이 뜨거워요", "nasal_coda_missing", ("ŋ", None),
     "국물[궁물] đọc thành 구물 — mất coda mũi ŋ"),
    ("e04", "n15", "오모니와 아버지", "vowel_eo_to_o", ("ʌ", "o"),
     "어머니 đọc thành 오모니 — ㅓ→ㅗ (lỗi VN kinh điển)"),
    ("e05", "n17", "하늘이 후려요", "vowel_eu_to_u", ("ɯ", "u"),
     "흐려요 đọc thành 후려요 — ㅡ→ㅜ"),
    ("e06", "n06", "신나 시대 역사", "lateralization_missing", ("l", "n"),
     "신라[실라] đọc thành 신나 — thiếu 유음화"),
    ("e07", "n19", "거피가 뜨거워요", "aspirated_to_plain", ("kʰ", "k"),
     "커피 đọc thành 거피 — mất bật hơi ㅋ"),
    ("e08", "n12", "밥을 머거어요", "coda_ss_missing", ("s", None),
     "먹었어요[머거써요] đọc thành 머거어요 — mất ㅆ"),
    ("e09", "n21", "저는 베트남 사란이에요", "batchim_m_to_n", ("m", "n"),
     "사람 đọc thành 사란 — batchim ㅁ→ㄴ"),
    ("e10", "n10", "있어요 어서요", "coda_cluster_missing", ("p", None),
     "없어요[업써요] đọc thành 어서요 — mất coda ㅂ"),
]


async def _tts(text: str, voice: str, mp3_path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice).save(str(mp3_path))


def _to_wav(mp3_path: Path, wav_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path),
         "-ac", "1", "-ar", "16000", str(wav_path)],
        check=True,
    )
    mp3_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Ghi đè clip đã có")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    jobs: list[tuple[str, str, str, dict]] = []  # (clip_id, tts_text, voice, meta)
    for sid, text in NATIVE_SENTENCES:
        for vi, voice in enumerate(VOICES):
            clip_id = f"{sid}_v{vi}"
            jobs.append((clip_id, text, voice, {
                "id": clip_id, "kind": "native", "reference": text,
                "voice": voice, "sentence_id": sid,
            }))
    # Lỗi mô phỏng: 1 voice là đủ (đo detection, không đo variance giọng).
    for eid, twin, tts_text, etype, (ref_ph, heard_ph), note in ERROR_SENTENCES:
        ref_text = dict(NATIVE_SENTENCES)[twin]
        clip_id = f"{eid}_v0"
        jobs.append((clip_id, tts_text, VOICES[0], {
            "id": clip_id, "kind": "error", "reference": ref_text,
            "tts_text": tts_text, "voice": VOICES[0], "sentence_id": eid,
            "twin_id": f"{twin}_v0", "error_type": etype,
            "expect_ref_phoneme": ref_ph, "expect_heard_phoneme": heard_ph,
            "note": note,
        }))

    async def run_all() -> None:
        for clip_id, tts_text, voice, meta in jobs:
            wav = OUT_DIR / f"{clip_id}.wav"
            meta["wav"] = wav.name
            manifest.append(meta)
            if wav.exists() and not args.fresh:
                print(f"skip (có sẵn) {clip_id}")
                continue
            mp3 = OUT_DIR / f"{clip_id}.mp3"
            await _tts(tts_text, voice, mp3)
            _to_wav(mp3, wav)
            print(f"gen {clip_id}  [{voice}]  {tts_text}")

    asyncio.run(run_all())

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nManifest: {manifest_path} ({len(manifest)} clips)")


if __name__ == "__main__":
    sys.exit(main())

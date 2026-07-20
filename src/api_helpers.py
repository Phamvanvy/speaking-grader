"""Helper thuần (stateless) cho src/api.py — validate input + biến đổi dữ liệu.

Tách khỏi api.py để file route mỏng hơn. KHÔNG phụ thuộc `app` hay `grade_response`
(các thứ đó ở lại api.py). api.py import lại các tên này nên chúng vẫn là attribute
của `src.api` (test gọi `api._normalize_mode`, `from src.api import _validate_exam`...).
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from fastapi import HTTPException

from .rubrics import EXAM_REGISTRIES, resolve_question_type
from .rubrics.base import QuestionType, exam_language, exam_score_field
from .tts import MAX_TEXT_LEN as _TTS_MAX_TEXT

logger = logging.getLogger("toeic.api")

# Định dạng input chấp nhận (audio + một số video container có track audio).
# faster-whisper đọc qua ffmpeg nên có thể xử lý clip có tiếng (vd .mp4/.mov).
_ALLOWED_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".webm",
    ".weba",  # WebM audio-only (cùng container .webm — vài nguồn upload đặt đuôi này)
    ".aac",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
}

# Video containers cần extract audio trước (ffmpeg → .wav mono 16kHz)
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}

# User-facing modes. Engine/model là implementation detail (xem config.py),
# tách rời khỏi business mode ở đây.
_ALLOWED_MODES = {"practice", "mock_test"}
# Map giá trị cũ (client/bản ghi cũ) → mode mới, không fail request.
_LEGACY_MODE_ALIASES = {
    "auto": "practice",
    "default": "practice",
    "fast": "practice",
    "review": "mock_test",
}

_VALID_ACCENTS: frozenset[str] = frozenset({"default", "gb", "us"})


def _validate_exam(exam: str) -> str:
    """Chuẩn hoá + kiểm tra mã kỳ thi (sai → HTTP 400)."""
    value = (exam or "").strip().lower()
    if value not in EXAM_REGISTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"Kỳ thi không hợp lệ: '{exam}'. Hợp lệ: {sorted(EXAM_REGISTRIES)}",
        )
    return value


def _ensure_exam_lang_enabled(exam: str, config) -> None:
    """Chặn kỳ thi nói tiếng Hàn khi flag TOEIC_LANG_KO_ENABLED tắt (→ 400).

    Gate Ở ĐÂY (request level) chứ không trong core: pipeline lang=ko chưa qua
    bench M2 thì không nhận request — default OFF theo văn hoá flag của repo.
    """
    if exam_language(exam) == "ko" and not getattr(config, "lang_ko_enabled", False):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Kỳ thi '{exam}' (tiếng Hàn) chưa được bật trên server này. "
                "Bật TOEIC_LANG_KO_ENABLED=1 để dùng."
            ),
        )


def _validate_accent(accent: str) -> str:
    """Chuẩn hoá accent giọng tham chiếu. Giá trị lạ → fallback "default" (không 400 —
    accent chỉ điều biến tolerance phát âm, không nên chặn cả request)."""
    value = (accent or "").strip().lower()
    return value if value in _VALID_ACCENTS else "default"


def _has_provided_info(provided_info: str | None) -> bool:
    """True nếu provided_info thực sự có nội dung.

    Frontend có thể gửi '' hoặc 'null' khi để trống → KHÔNG dùng bool() thô (sẽ
    nhận nhầm mọi bài thành Part 2).
    """
    if not provided_info:
        return False
    s = provided_info.strip()
    return bool(s) and s.lower() != "null"


def _resolve(key: str, exam: str) -> QuestionType:
    try:
        return resolve_question_type(key, exam=exam)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _pick_question_type(
    text: str | None,
    has_image: bool,
    provided_info: str | None,
    question_type: str | None,
    exam: str,
) -> QuestionType:
    """Chọn dạng câu theo kỳ thi: ưu tiên override 'question_type', không thì auto-detect."""
    if question_type:
        return _resolve(question_type, exam)

    if exam == "ielts":
        # IELTS: provided_info → Part 2 (cue card). KHÔNG đoán Part 1 vs Part 3
        # (đều Q&A text-only, không phân biệt được) → bắt client nêu rõ.
        if _has_provided_info(provided_info):
            return _resolve("part2_long_turn", exam)
        raise HTTPException(
            status_code=400,
            detail=(
                "IELTS: không tự suy ra được dạng câu. Hãy truyền 'question_type' "
                "rõ ràng (part1_interview / part2_long_turn / part3_discussion), "
                "hoặc 'provided_info' (cue card) cho Part 2."
            ),
        )

    # TOEIC: text → read_aloud, image → describe_picture.
    if text is not None and has_image:
        raise HTTPException(
            status_code=400,
            detail="Truyền 'text' HOẶC 'image', không phải cả hai "
            "(hoặc chỉ định 'question_type' rõ ràng).",
        )
    if text is not None:
        return _resolve("read_aloud", exam)
    if has_image:
        return _resolve("describe_picture", exam)
    raise HTTPException(
        status_code=400,
        detail=(
            "Không xác định được dạng câu. Hãy truyền một trong: "
            "'text' (script → read_aloud), 'image' (ảnh → describe_picture), "
            "hoặc 'question_type' rõ ràng "
            "(read_aloud / describe_picture / respond_questions / "
            "respond_with_info / express_opinion)."
        ),
    )


def _audio_suffix(filename: str | None) -> str:
    """Lấy & kiểm tra phần mở rộng audio (Whisper đọc theo đường dẫn)."""
    suffix = Path(filename or "").suffix.lower() or ".wav"
    if suffix not in _ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng audio không hỗ trợ: '{suffix}'. "
            f"Chấp nhận: {sorted(_ALLOWED_AUDIO_SUFFIXES)}",
        )
    return suffix


def _normalize_mode(mode: str | None) -> str:
    """Chuẩn hoá mode về {practice, mock_test}.

    Nhận chuỗi bẩn (hoa/thường, thừa khoảng trắng), map alias cũ
    (auto/default/fast → practice, review → mock_test). Giá trị rỗng/lạ → practice
    (mặc định an toàn) thay vì 400 để client/bản ghi cũ không bị chặn.
    """
    value = (mode or "").strip().lower()
    value = _LEGACY_MODE_ALIASES.get(value, value)
    if value not in _ALLOWED_MODES:
        return "practice"
    return value


def _overall_score(scores: dict | None, exam: str) -> float | None:
    """Điểm tổng theo kỳ thi, trả về float thống nhất (TOEIC 0-200 / IELTS 0-9).

    Tránh để downstream (scoreBeforeReview/After) phải xử lý union int|float.
    """
    if not scores:
        return None
    value = scores.get(exam_score_field(exam))
    return float(value) if value is not None else None


def _extract_telemetry_signals(output: dict) -> tuple[float, float, float]:
    """Trả về (confidence, silence_ratio, coverage) từ output."""
    features = output.get("features") or {}
    confidence = float(features.get("avg_word_probability") or 0.0)
    audio_dur = float(features.get("audio_duration_sec") or 0.0)
    silence_sec = float(features.get("silence_sec") or 0.0)
    silence_ratio = silence_sec / audio_dur if audio_dur > 0 else 0.0
    acc = features.get("accuracy_metrics") or {}
    # Không có reference script thì không có coverage; coi như đạt để không tự trigger.
    coverage = float(acc.get("coverage") or 1.0)
    return confidence, silence_ratio, coverage


def _extract_audio_from_video(video_bytes: bytes, suffix: str) -> tuple[bytes, str]:
    """Extract audio track from video container → WAV mono 16kHz.

    Returns (audio_bytes, ".wav"). Falls back to raw bytes if ffmpeg unavailable.
    """
    try:
        # Write video to temp file for ffmpeg
        video_tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        video_tmp.write(video_bytes)
        video_tmp.close()

        # Extract audio: mono, 16kHz, WAV format (optimal for ASR + wav2vec)
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_tmp.name,
                "-vn",                # no video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", "16000",       # 16kHz sample rate
                "-ac", "1",           # mono
                "-y",                 # overwrite output
                "-f", "wav",          # force WAV format
                "-",                  # stdout
            ],
            capture_output=True,
            timeout=60,               # 60s timeout for extraction
        )

        audio_output = result.stdout
        Path(video_tmp.name).unlink(missing_ok=True)

        if not audio_output:
            logger.warning("ffmpeg extract returned empty audio, falling back to raw bytes.")
            return video_bytes, suffix

        logger.info(
            "Extracted audio from video (%.1fKB → %.1fKB WAV, 16kHz mono)",
            len(video_bytes) / 1024,
            len(audio_output) / 1024,
        )
        return audio_output, ".wav"

    except FileNotFoundError:
        logger.warning("ffmpeg not found — passing video bytes as-is (ASR may fail).")
        return video_bytes, suffix
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg extraction timed out — passing video bytes as-is.")
        return video_bytes, suffix
    except Exception as e:
        logger.warning("ffmpeg audio extraction failed: %s — passing as-is.", e)
        return video_bytes, suffix


def _validate_tts_text(text: str) -> str:
    """Chuẩn hoá text cho TTS — NỚI LỎNG: chỉ strip + bỏ ký tự điều khiển + chặn rỗng
    + trần độ dài. KHÔNG whitelist hẹp chữ cái: TTS đọc được text tự nhiên, whitelist
    sẽ chặn nhầm từ hợp lệ (it's, co-op) và cụm từ tương lai. Mục tiêu chỉ là chống
    lạm dụng (độ dài), không lọc nội dung."""
    value = "".join(ch for ch in (text or "") if ch >= " " or ch == "\t")
    value = " ".join(value.split())
    if not value:
        raise HTTPException(status_code=400, detail="Thiếu 'text'.")
    if len(value) > _TTS_MAX_TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"'text' quá dài ({len(value)} > {_TTS_MAX_TEXT} ký tự).",
        )
    return value


def _validate_tts_ipa(ipa: str) -> str:
    """Chuẩn hoá chuỗi IPA tuỳ chọn cho TTS (nhánh đọc-IPA).

    IPA là OPTIONAL — rỗng/thiếu → trả "" để endpoint fallback sang `text` (KHÔNG
    400). Chỉ strip ký tự điều khiển, gộp khoảng trắng, và chặn quá dài (chống lạm
    dụng). KHÔNG whitelist ký hiệu IPA ở đây: src/tts.py tự bỏ token ngoài bảng
    phoneme của voice, nên input rác chỉ ra audio ngắn/rỗng chứ không nguy hiểm."""
    value = "".join(ch for ch in (ipa or "") if ch >= " " or ch == "\t")
    value = " ".join(value.split())
    if len(value) > _TTS_MAX_TEXT:
        raise HTTPException(
            status_code=400,
            detail=f"'ipa' quá dài ({len(value)} > {_TTS_MAX_TEXT} ký tự).",
        )
    return value

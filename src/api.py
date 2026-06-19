"""HTTP API chấm TOEIC Speaking (FastAPI).

Endpoint:
- POST /grade        : chấm 1 audio.
- POST /grade-batch  : chấm NHIỀU audio (vd cả lớp 30-40 học sinh) cho CÙNG đề
  bài (text/ảnh/thời lượng dùng chung), trả kết quả theo từng file.
- GET  /health       : kiểm tra sống + cấu hình.

Trường multipart/form-data dùng chung:
- text    (optional): script tham chiếu → chấm Read Aloud (so transcript).
- image   (optional): ảnh đề bài → chấm Describe Picture (gửi LLM dạng vision).
- expected_duration_sec (optional): thời lượng kỳ vọng (giây).
- question_type (optional): ép dạng câu (read_aloud / describe_picture / ...).
- feedback_lang (optional): ngôn ngữ nhận xét (vd 'vi', 'en').
- prompt  (optional): đề bài hiển thị cho thí sinh.
- no_ai   (optional): chỉ ASR + features, bỏ qua LLM.

Quy ước: truyền `text` → Read Aloud; truyền `image` → Describe Picture. Không
được truyền cả hai (trừ khi tự chỉ định question_type). Chạy:

    uvicorn src.api:app --reload --port 8000

Tài liệu Swagger tự sinh tại /docs.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import os
import tempfile
from time import perf_counter
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
import subprocess

from .config import Config, load_config
from .core import grade_response
from .logging_setup import setup_logging
from .rubrics.toeic import QuestionType, get_question_type

logger = logging.getLogger("toeic.api")

app = FastAPI(
    title="TOEIC Speaking Grader API",
    description="Chấm bài nói: so audio với text (đọc to) hoặc ảnh (tả tranh).",
    version="1.1.0",
)

# Nạp config 1 lần lúc khởi động (model Whisper cache trong asr theo process).
_BASE_CONFIG: Config = load_config()
setup_logging()

# Định dạng input chấp nhận (audio + một số video container có track audio).
# faster-whisper đọc qua ffmpeg nên có thể xử lý clip có tiếng (vd .mp4/.mov).
_ALLOWED_AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".flac",
    ".webm",
    ".aac",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
}

# Video containers cần extract audio trước (ffmpeg → .wav mono 16kHz)
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}

# Trần số file trong 1 batch (chặn lạm dụng; 1 lớp thực tế ~40 em).
_MAX_BATCH = 100
_ALLOWED_MODES = {"default", "fast", "review", "auto"}

# Frontend tĩnh: thư mục web/ (index.html + styles.css + app.js) ở repo root
# (src/ là con của root). Mount ở "/" cùng origin với API → không cần CORS, và ô
# "API Base URL" tự điền đúng domain. Xem mount ở CUỐI file (phải đăng ký SAU mọi
# route API để /grade, /health, /docs... được ưu tiên; static chỉ là fallback).
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _resolve_config(feedback_lang: str | None) -> Config:
    """Trả về Config (ghi đè feedback_lang theo request nếu có)."""
    if feedback_lang:
        return dataclasses.replace(_BASE_CONFIG, feedback_lang=feedback_lang)
    return _BASE_CONFIG


def _pick_question_type(
    text: str | None, has_image: bool, question_type: str | None
) -> QuestionType:
    """Chọn dạng câu: ưu tiên override; không thì text→read_aloud, image→describe_picture."""
    if question_type:
        try:
            return get_question_type(question_type)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    if text is not None and has_image:
        raise HTTPException(
            status_code=400,
            detail="Truyền 'text' HOẶC 'image', không phải cả hai "
            "(hoặc chỉ định 'question_type' rõ ràng).",
        )
    if text is not None:
        return get_question_type("read_aloud")
    if has_image:
        return get_question_type("describe_picture")
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
    value = (mode or "auto").strip().lower()
    if value not in _ALLOWED_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"mode không hợp lệ: '{mode}'. Hợp lệ: {sorted(_ALLOWED_MODES)}"
            ),
        )
    return value


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


def _grade_bytes(
    audio_bytes: bytes,
    suffix: str,
    config: Config,
    qt: QuestionType,
    *,
    reference_script: str | None,
    image_b64: str | None,
    image_media_type: str | None,
    expected_duration_sec: float | None,
    prompt: str,
    no_ai: bool,
    mode: str,
    user_requested_review: bool,
    provided_info: str | None = None,
) -> dict:
    """Ghi audio ra file tạm rồi chạy pipeline (HÀM CHẶN — gọi qua threadpool).

    Tự dọn file tạm. Trả dict kết quả (đã bỏ audio_path tạm).
    Video containers (mp4, mov, avi) được extract audio trước bằng ffmpeg.
    """
    # Pre-convert video to audio if needed
    if suffix in _VIDEO_SUFFIXES:
        audio_bytes, suffix = _extract_audio_from_video(audio_bytes, suffix)

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        submission_id = str(uuid4())
        requested_mode = _normalize_mode(mode)
        review_reasons: list[str] = []
        started = perf_counter()

        def _run_once(asr_backend: str, phoneme: bool | None = None) -> dict:
            # phoneme: gắn wav2vec theo mode — review=True (bật), fast=False (ưu
            # tiên tốc độ), default/auto=None (theo config.phoneme_analysis_enabled).
            return grade_response(
                tmp.name,
                config,
                qt,
                prompt_text=prompt,
                reference_script=reference_script,
                expected_duration_sec=expected_duration_sec,
                image_b64=image_b64,
                image_media_type=image_media_type,
                provided_info=provided_info,
                asr_backend=asr_backend,
                no_ai=no_ai,
                phoneme_analysis=phoneme,
                question_id=qt.key,
                save=False,
            )

        score_before_review = None
        score_after_review = None
        used_mode = requested_mode
        review_triggered = False
        fallback_reason: str | None = None

        if requested_mode == "review":
            # review = chấm kỹ → bật wav2vec để có bằng chứng phát âm cấp âm vị.
            output = _run_once(config.asr_backend_review, phoneme=True)
            review_triggered = True
        elif requested_mode == "fast":
            # fast = ưu tiên throughput → tắt wav2vec (nặng) bất kể config.
            if not config.fast_backend_enabled:
                used_mode = "default"
                fallback_reason = "fast_backend_unavailable"
                review_reasons.append("fast backend disabled by config")
                output = _run_once(config.asr_backend_default, phoneme=False)
            else:
                try:
                    output = _run_once(config.asr_backend_fast, phoneme=False)
                except Exception as fast_err:  # noqa: BLE001 - fallback đúng theo quyết định kiến trúc
                    used_mode = "default"
                    fallback_reason = "fast_backend_failed"
                    review_reasons.append(
                        f"fast_fallback_default: {type(fast_err).__name__}: {fast_err}"
                    )
                    output = _run_once(config.asr_backend_default, phoneme=False)
        elif requested_mode == "auto":
            # auto bắt đầu ở default (theo config); chỉ bật wav2vec nếu leo lên review.
            output = _run_once(config.asr_backend_default)
            confidence, silence_ratio, coverage = _extract_telemetry_signals(output)
            if confidence < config.auto_confidence_threshold:
                review_reasons.append(
                    f"low_confidence<{config.auto_confidence_threshold:.2f}"
                )
            if silence_ratio > config.auto_silence_ratio_threshold:
                review_reasons.append(
                    f"high_silence_ratio>{config.auto_silence_ratio_threshold:.2f}"
                )
            if coverage < config.auto_coverage_threshold:
                review_reasons.append(
                    f"low_coverage<{config.auto_coverage_threshold:.2f}"
                )
            if user_requested_review:
                review_reasons.append("user_requested_review")

            if review_reasons:
                score_before_review = (
                    (output.get("scores") or {}).get("estimated_toeic_score")
                )
                try:
                    output = _run_once(config.asr_backend_review, phoneme=True)
                    review_triggered = True
                    used_mode = "review"
                    score_after_review = (
                        (output.get("scores") or {}).get("estimated_toeic_score")
                    )
                except Exception as review_err:  # noqa: BLE001
                    review_reasons.append(
                        f"review_failed_kept_default: {type(review_err).__name__}: {review_err}"
                    )
                    used_mode = "default"
            else:
                used_mode = "default"
        else:
            output = _run_once(config.asr_backend_default)
            used_mode = "default"

        confidence, silence_ratio, _ = _extract_telemetry_signals(output)
        total_ms = int((perf_counter() - started) * 1000)
        telemetry = output.get("telemetry") or {}
        telemetry.update(
            {
                "submissionId": submission_id,
                "modeRequested": requested_mode,
                "modeUsed": used_mode,
                "durationSeconds": float((output.get("features") or {}).get("audio_duration_sec") or 0.0),
                "transcriptionTimeMs": int(telemetry.get("transcription_time_ms") or 0),
                "totalProcessingTimeMs": total_ms,
                "confidence": round(confidence, 4),
                "silenceRatio": round(silence_ratio, 4),
                "wpm": float((output.get("features") or {}).get("speech_rate_wpm") or 0.0),
                "reviewTriggered": review_triggered,
                "reviewReason": ", ".join(review_reasons) if review_reasons else "",
                "fallbackReason": fallback_reason,
                "scoreBeforeReview": score_before_review,
                "scoreAfterReview": score_after_review,
            }
        )
        output["telemetry"] = telemetry
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    output.pop("audio_path", None)
    return output


async def _read_image(image: UploadFile | None) -> tuple[str | None, str | None]:
    """Đọc ảnh (nếu có) → (base64, media_type)."""
    if image is None:
        return None, None
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")
    return base64.b64encode(image_bytes).decode("ascii"), (
        image.content_type or "image/jpeg"
    )


@app.get("/health")
def health() -> dict:
    """Kiểm tra sống + cấu hình backend hiện tại."""
    return {
        "status": "ok",
        "backend": _BASE_CONFIG.backend,
        "model": (
            _BASE_CONFIG.local_model
            if _BASE_CONFIG.is_local
            else _BASE_CONFIG.model
        ),
        "whisper_model": _BASE_CONFIG.whisper_model,
        "max_tokens": _BASE_CONFIG.max_tokens,
    }


@app.post("/grade")
async def grade(
    audio: UploadFile = File(
        ..., description="File audio/clip (.wav/.mp3/.m4a/.webm/.mp4/...)"
    ),
    text: str | None = Form(None, description="Script tham chiếu (Read Aloud)"),
    image: UploadFile | None = File(None, description="Ảnh đề bài (Describe Picture)"),
    expected_duration_sec: float | None = Form(None),
    question_type: str | None = Form(None),
    feedback_lang: str | None = Form(None),
    prompt: str = Form("", description="Đề bài hiển thị cho thí sinh (optional)"),
    provided_info: str | None = Form(
        None, description="Tài liệu cho sẵn dạng text (Respond with info, Q8-10)"
    ),
    no_ai: bool = Form(False),
    mode: str = Form("auto", description="default | fast | review | auto"),
    user_requested_review: bool = Form(
        False, description="Ép review khi mode=auto"
    ),
) -> dict:
    """Chấm 1 bài nói. Trả về JSON {transcript, features, scores}."""
    has_image = image is not None
    qt = _pick_question_type(text, has_image, question_type)
    image_b64, image_media_type = await _read_image(image)

    suffix = _audio_suffix(audio.filename)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File audio rỗng.")

    config = _resolve_config(feedback_lang)
    try:
        return await run_in_threadpool(
            _grade_bytes,
            audio_bytes,
            suffix,
            config,
            qt,
            reference_script=text,
            image_b64=image_b64,
            image_media_type=image_media_type,
            expected_duration_sec=expected_duration_sec,
            prompt=prompt,
            provided_info=provided_info,
            no_ai=no_ai,
            mode=mode,
            user_requested_review=user_requested_review,
        )
    except Exception as e:  # noqa: BLE001 - trả lỗi gọn cho client
        logger.exception("Lỗi khi chấm")
        raise HTTPException(status_code=500, detail=f"Lỗi khi chấm: {e}") from e


@app.post("/grade-batch")
async def grade_batch(
    audios: list[UploadFile] = File(
        ..., description="Nhiều file audio (mỗi file = 1 học sinh)"
    ),
    text: str | None = Form(None, description="Script tham chiếu (Read Aloud)"),
    image: UploadFile | None = File(None, description="Ảnh đề bài (Describe Picture)"),
    expected_duration_sec: float | None = Form(None),
    question_type: str | None = Form(None),
    feedback_lang: str | None = Form(None),
    prompt: str = Form("", description="Đề bài hiển thị cho thí sinh (optional)"),
    provided_info: str | None = Form(
        None, description="Tài liệu cho sẵn dạng text (Respond with info, Q8-10)"
    ),
    no_ai: bool = Form(False),
    mode: str = Form("auto", description="default | fast | review | auto"),
    user_requested_review: bool = Form(
        False, description="Ép review khi mode=auto"
    ),
    max_concurrency: int = Form(
        0, description="Số bài chấm song song; 0 = tự (1 cho local, 4 cho cloud)"
    ),
) -> dict:
    """Chấm nhiều audio cho CÙNG đề bài. Trả kết quả theo từng file.

    Mỗi file chấm độc lập; 1 file lỗi không làm hỏng cả batch (gói vào trường
    `error` của file đó). Đề bài (text/ảnh/thời lượng) dùng chung cho mọi file.

    Lưu ý hiệu năng: backend local (llama.cpp) thường chỉ xử lý 1 request/lúc và
    model Whisper không an toàn khi gọi song song nhiều luồng → mặc định chạy
    TUẦN TỰ (concurrency=1). Chỉ tăng max_concurrency khi backend là cloud
    (Anthropic) và bạn hiểu rủi ro tranh chấp Whisper trên GPU.
    """
    if not audios:
        raise HTTPException(status_code=400, detail="Cần ít nhất 1 file audio.")
    if len(audios) > _MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Quá nhiều file ({len(audios)} > {_MAX_BATCH}). Chia nhỏ batch.",
        )

    has_image = image is not None
    qt = _pick_question_type(text, has_image, question_type)
    requested_mode = _normalize_mode(mode)
    image_b64, image_media_type = await _read_image(image)
    config = _resolve_config(feedback_lang)

    # Đọc sẵn toàn bộ bytes (UploadFile là stream, không dùng được trong thread
    # khác sau khi handler trả về) + validate phần mở rộng từng file.
    items: list[tuple[str, bytes, str | None]] = []  # (filename, bytes, suffix_or_None)
    for up in audios:
        name = up.filename or "audio"
        try:
            suffix = _audio_suffix(up.filename)
        except HTTPException as e:
            items.append((name, b"", None))  # đánh dấu lỗi định dạng
            logger.warning("Bỏ qua %s: %s", name, e.detail)
            continue
        data = await up.read()
        items.append((name, data, suffix))

    # Ưu tiên: tham số form > cấu hình env (TOEIC_BATCH_CONCURRENCY) > tự chọn.
    # ASR (Whisper) đã được serialize bằng lock GPU riêng nên đặt >1 cho local là
    # an toàn: chỉ một ASR chạy/lúc, tầng LLM chồng lấn để tăng throughput batch.
    if max_concurrency and max_concurrency > 0:
        concurrency = max_concurrency
    elif config.batch_concurrency and config.batch_concurrency > 0:
        concurrency = config.batch_concurrency
    else:
        concurrency = 1 if config.is_local else 4
    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int, filename: str, data: bytes, suffix: str | None) -> dict:
        if suffix is None:
            return {"index": idx, "audio_filename": filename,
                    "error": "Định dạng audio không hỗ trợ."}
        if not data:
            return {"index": idx, "audio_filename": filename,
                    "error": "File audio rỗng."}
        async with sem:
            try:
                result = await run_in_threadpool(
                    _grade_bytes,
                    data,
                    suffix,
                    config,
                    qt,
                    reference_script=text,
                    image_b64=image_b64,
                    image_media_type=image_media_type,
                    expected_duration_sec=expected_duration_sec,
                    prompt=prompt,
                    provided_info=provided_info,
                    no_ai=no_ai,
                    mode=requested_mode,
                    user_requested_review=user_requested_review,
                )
                return {"index": idx, "audio_filename": filename, "result": result}
            except Exception as e:  # noqa: BLE001 - lỗi 1 em không làm hỏng cả lớp
                logger.exception("Lỗi khi chấm %s", filename)
                return {"index": idx, "audio_filename": filename, "error": str(e)}

    results = await asyncio.gather(
        *(_one(i, name, data, suffix) for i, (name, data, suffix) in enumerate(items))
    )
    succeeded = sum(1 for r in results if "result" in r)
    return {
        "question_type": qt.key,
        "mode_requested": requested_mode,
        "count": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "concurrency": concurrency,
        "results": results,
    }


# Mount frontend tĩnh ở "/" — PHẢI đặt sau mọi route API ở trên. Starlette so khớp
# route theo thứ tự đăng ký, nên /grade, /health, /docs... (đăng ký trước) được ưu
# tiên; StaticFiles chỉ bắt phần còn lại (GET / → index.html, /styles.css, /app.js).
# html=True → "/" trả index.html. check_dir=False để app vẫn khởi động được nếu
# thiếu web/ (vd chạy chỉ-API trong test); request tới "/" khi đó trả 404 gọn.
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
else:  # pragma: no cover - chỉ xảy ra khi deploy thiếu thư mục web/
    logger.warning("Không thấy thư mục web/ (%s) — frontend tĩnh bị tắt.", _WEB_DIR)

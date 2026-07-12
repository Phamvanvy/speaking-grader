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
import json
import logging
import os
import tempfile
from time import perf_counter
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import history
from .config import Config, load_config
from .core import grade_response
from .exam_import import ExamImportError, extract_exam
from .exam_paper import ExamPaper
from .logging_setup import setup_logging
from .rubrics import EXAM_REGISTRIES
from .rubrics.base import QuestionType
from .scoring import compute_exam_overall
from .suggest import default_target_band, suggest_answer
from .tts import TtsUnavailable, synthesize as _tts_synthesize
from .api_helpers import (
    _VIDEO_SUFFIXES,
    _audio_suffix,
    _extract_audio_from_video,
    _extract_telemetry_signals,
    _has_provided_info,
    _normalize_mode,
    _overall_score,
    _pick_question_type,
    _resolve,
    _validate_accent,
    _validate_exam,
    _validate_tts_text,
)

logger = logging.getLogger("toeic.api")

app = FastAPI(
    title="TOEIC Speaking Grader API",
    description="Chấm bài nói: so audio với text (đọc to) hoặc ảnh (tả tranh).",
    version="1.1.0",
)

# Nạp config 1 lần lúc khởi động (model Whisper cache trong asr theo process).
_BASE_CONFIG: Config = load_config()
setup_logging()

# CORS: cho phép web/app ở origin khác gọi API qua trình duyệt (Swagger ở
# /docs cùng origin nên không cần CORS, nhưng client ngoài thì cần). Origins
# cấu hình qua CORS_ALLOW_ORIGINS (CSV). Khi là "*" thì allow_credentials phải
# tắt theo chuẩn CORS (trình duyệt từ chối "*" + credentials); API này không
# dùng cookie nên không ảnh hưởng.
_CORS_ORIGINS = _BASE_CONFIG.cors_origins_list or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials="*" not in _CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trần số file trong 1 batch (chặn lạm dụng; 1 lớp thực tế ~40 em).
_MAX_BATCH = 100

# Dọn thư mục audio mồ côi (crash giữa ghi audio và insert DB) — best-effort,
# không được làm hỏng startup.
if _BASE_CONFIG.history_enabled:
    try:
        history.sweep_orphans(_BASE_CONFIG)
    except Exception:  # noqa: BLE001
        logger.exception("Lịch sử: sweep_orphans lỗi (bỏ qua).")


def _history_save(fn, /, *args, **kwargs) -> None:
    """Lưu lịch sử best-effort: lỗi chỉ log, KHÔNG BAO GIỜ hỏng response chấm."""
    if not _BASE_CONFIG.history_enabled:
        return
    try:
        fn(_BASE_CONFIG, *args, **kwargs)
    except Exception:  # noqa: BLE001
        logger.exception("Lưu lịch sử thất bại (bỏ qua — không ảnh hưởng kết quả chấm)")


def _require_history_enabled() -> None:
    if not _BASE_CONFIG.history_enabled:
        raise HTTPException(status_code=404, detail="Lịch sử chấm bài đang tắt.")


def _valid_user_id_or_400(user_id: str) -> str:
    try:
        return history.validate_user_id(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

# Frontend tĩnh: thư mục web/ (index.html + CSS + JS) ở repo root
# (src/ là con của root). Mount ở "/" cùng origin với API → không cần CORS, và ô
# "API Base URL" tự điền đúng domain. Xem mount ở CUỐI file (phải đăng ký SAU mọi
# route API để /grade, /health, /docs... được ưu tiên; static chỉ là fallback).
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _resolve_config(feedback_lang: str | None) -> Config:
    """Trả về Config (ghi đè feedback_lang theo request nếu có)."""
    if feedback_lang:
        return dataclasses.replace(_BASE_CONFIG, feedback_lang=feedback_lang)
    return _BASE_CONFIG


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
    accent: str = "default",
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

        def _run_once(
            asr_backend: str, asr_model: str | None = None, phoneme: bool | None = None
        ) -> dict:
            # phoneme: gắn wav2vec theo mode — mock_test=True (bật), practice=None
            # (theo config.phoneme_analysis_enabled), True khi practice leo lên mock_test.
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
                asr_model=asr_model,
                no_ai=no_ai,
                phoneme_analysis=phoneme,
                question_id=qt.key,
                save=False,
                accent=accent,
            )

        score_before_review = None
        score_after_review = None
        used_mode = requested_mode
        review_triggered = False
        fallback_reason: str | None = None

        if requested_mode == "mock_test":
            # mock_test = thí sinh chủ động chọn chấm kỹ → engine tốt nhất + wav2vec.
            # KHÔNG phải auto-escalation nên review_triggered vẫn False.
            output = _run_once(
                config.asr_engine_mock_test, config.asr_model_mock_test, phoneme=True
            )
            used_mode = "mock_test"
        else:
            # practice: chạy lane nhanh trước (phoneme theo config); chỉ bật wav2vec
            # nếu tự leo lên pipeline mock_test do tín hiệu kém.
            output = _run_once(config.asr_engine_practice, config.asr_model_practice)
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
                score_before_review = _overall_score(output.get("scores"), qt.exam)
                try:
                    output = _run_once(
                        config.asr_engine_mock_test,
                        config.asr_model_mock_test,
                        phoneme=True,
                    )
                    review_triggered = True
                    used_mode = "mock_test"
                    score_after_review = _overall_score(output.get("scores"), qt.exam)
                except Exception as review_err:  # noqa: BLE001
                    review_reasons.append(
                        f"mock_test_failed_kept_practice: {type(review_err).__name__}: {review_err}"
                    )
                    used_mode = "practice"
            else:
                used_mode = "practice"

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
                # reserved for future use — luôn None từ khi bỏ fast lane; giữ key
                # để payload ổn định cho frontend/CSV cũ.
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
    exam: str = Form(_BASE_CONFIG.default_exam, description="Kỳ thi: toeic | ielts"),
    question_type: str | None = Form(None),
    feedback_lang: str | None = Form(None),
    prompt: str = Form("", description="Đề bài hiển thị cho thí sinh (optional)"),
    provided_info: str | None = Form(
        None,
        description="Tài liệu cho sẵn dạng text (TOEIC Q8-10 / IELTS Part 2 cue card)",
    ),
    no_ai: bool = Form(False),
    mode: str = Form("practice", description="practice | mock_test"),
    user_requested_review: bool = Form(
        False, description="Ép review khi mode=auto"
    ),
    accent: str = Form(
        "default", description="Giọng tham chiếu phát âm: default | gb | us"
    ),
    user_id: str | None = Form(
        None, description="ID ẩn danh của user (bật lưu lịch sử khi có)"
    ),
    history_session_id: str | None = Form(
        None, description="UUID phiên thi cả đề (SPA chấm từng câu qua /grade)"
    ),
    history_session_title: str | None = Form(None),
    history_seq: int | None = Form(None),
    history_question_id: str | None = Form(None),
) -> dict:
    """Chấm 1 bài nói. Trả về JSON {transcript, features, scores}."""
    has_image = image is not None
    exam = _validate_exam(exam)
    accent = _validate_accent(accent)
    qt = _pick_question_type(text, has_image, provided_info, question_type, exam)
    image_b64, image_media_type = await _read_image(image)

    suffix = _audio_suffix(audio.filename)
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File audio rỗng.")

    config = _resolve_config(feedback_lang)
    try:
        output = await run_in_threadpool(
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
            accent=accent,
        )
    except Exception as e:  # noqa: BLE001 - trả lỗi gọn cho client
        logger.exception("Lỗi khi chấm")
        raise HTTPException(status_code=500, detail=f"Lỗi khi chấm: {e}") from e

    # Lưu lịch sử (best-effort, sau khi payload sẵn sàng). Có history_session_id
    # → đây là 1 câu của phiên thi cả đề (SPA chấm rời từng câu); không thì là
    # bài chấm lẻ. Ghi audio MB-scale → chạy trong threadpool.
    if user_id and history_session_id:
        await run_in_threadpool(
            _history_save,
            history.add_exam_item,
            session_id=history_session_id,
            user_id=user_id,
            exam=exam,
            title=history_session_title,
            mode=_normalize_mode(mode),
            seq=history_seq,
            question_id=history_question_id,
            result=output,
            audio_bytes=audio_bytes,
            suffix=suffix,
        )
    elif user_id:
        await run_in_threadpool(
            _history_save,
            history.save_single,
            user_id=user_id,
            filename=audio.filename,
            mode=_normalize_mode(mode),
            audio_bytes=audio_bytes,
            suffix=suffix,
            result=output,
        )
    return output


@app.post("/grade-batch")
async def grade_batch(
    audios: list[UploadFile] = File(
        ..., description="Nhiều file audio (mỗi file = 1 học sinh)"
    ),
    text: str | None = Form(None, description="Script tham chiếu (Read Aloud)"),
    image: UploadFile | None = File(None, description="Ảnh đề bài (Describe Picture)"),
    expected_duration_sec: float | None = Form(None),
    exam: str = Form(_BASE_CONFIG.default_exam, description="Kỳ thi: toeic | ielts"),
    question_type: str | None = Form(None),
    feedback_lang: str | None = Form(None),
    prompt: str = Form("", description="Đề bài hiển thị cho thí sinh (optional)"),
    provided_info: str | None = Form(
        None,
        description="Tài liệu cho sẵn dạng text (TOEIC Q8-10 / IELTS Part 2 cue card)",
    ),
    no_ai: bool = Form(False),
    mode: str = Form("practice", description="practice | mock_test"),
    user_requested_review: bool = Form(
        False, description="Ép review khi mode=auto"
    ),
    accent: str = Form(
        "default", description="Giọng tham chiếu phát âm: default | gb | us"
    ),
    max_concurrency: int = Form(
        0, description="Số bài chấm song song; 0 = tự (1 cho local, 4 cho cloud)"
    ),
    user_id: str | None = Form(
        None, description="ID ẩn danh của user (bật lưu lịch sử khi có)"
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
    exam = _validate_exam(exam)
    accent = _validate_accent(accent)
    qt = _pick_question_type(text, has_image, provided_info, question_type, exam)
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
                    accent=accent,
                )
                return {"index": idx, "audio_filename": filename, "result": result}
            except Exception as e:  # noqa: BLE001 - lỗi 1 em không làm hỏng cả lớp
                logger.exception("Lỗi khi chấm %s", filename)
                return {"index": idx, "audio_filename": filename, "error": str(e)}

    batch_started = perf_counter()
    results = await asyncio.gather(
        *(_one(i, name, data, suffix) for i, (name, data, suffix) in enumerate(items))
    )
    total_ms = int((perf_counter() - batch_started) * 1000)
    succeeded = sum(1 for r in results if "result" in r)
    response = {
        "exam": exam,
        "question_type": qt.key,
        "mode_requested": requested_mode,
        "count": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "concurrency": concurrency,
        # Wall-clock của cả batch (các file chạy song song theo `concurrency`,
        # nên giá trị này thường nhỏ hơn tổng thời gian từng file).
        "total_processing_time_ms": total_ms,
        "results": results,
    }
    if user_id:
        await run_in_threadpool(
            _history_save,
            history.save_batch,
            user_id=user_id,
            mode=requested_mode,
            batch_response=response,
            files=items,
        )
    return response


@app.get("/tts")
async def tts(text: str = "", accent: str = "default") -> Response:
    """Tổng hợp audio mẫu (Piper TTS) cho 1 từ/cụm ngắn → WAV.

    Dùng cho nút 🔊 "nghe phát âm đúng" ở bảng lỗi phát âm. Param đặt tên trung lập
    để sau thêm `ipa=` mà giữ nguyên route (xem src/tts.py:synthesize). Voice chưa
    cài → 503; text sai → 400.
    """
    clean = _validate_tts_text(text)
    accent = _validate_accent(accent)
    try:
        wav = await run_in_threadpool(
            _tts_synthesize, text=clean, accent=accent, config=_BASE_CONFIG
        )
    except TtsUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - trả lỗi gọn cho client
        logger.exception("Lỗi TTS")
        raise HTTPException(status_code=500, detail=f"Lỗi TTS: {e}") from e
    return Response(
        content=wav,
        media_type="audio/wav",
        # Audio mẫu ổn định theo (text, accent, version) → cho trình duyệt cache.
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ── Gợi ý bài nói MẪU (band cao) cho dạng câu mở ─────────────────────────────
# Sinh bài mẫu để người học tham khảo; KHÔNG chấm điểm, KHÔNG đụng luồng /grade.


@app.post("/suggest")
async def suggest(
    exam: str = Form(_BASE_CONFIG.default_exam, description="Kỳ thi: toeic | ielts"),
    question_type: str = Form(
        ..., description="Dạng câu mở (vd part2_long_turn / describe_picture)"
    ),
    prompt: str = Form("", description="Đề bài / câu hỏi hiển thị cho thí sinh"),
    provided_info: str | None = Form(
        None, description="Cue card / tài liệu cho sẵn (IELTS Part 2 / TOEIC Q8-10)"
    ),
    expected_duration_sec: float | None = Form(None),
    target_band: str = Form(
        "", description="Mức nhắm tới (vd '9.0'); rỗng → cao nhất theo kỳ thi"
    ),
    feedback_lang: str | None = Form(None),
    image: UploadFile | None = File(None, description="Ảnh đề bài (Describe Picture)"),
) -> dict:
    """Sinh MỘT bài nói mẫu chất lượng cao (mặc định band 9.0 / TOEIC max).

    Chỉ cho dạng câu MỞ. `read_aloud` bị chặn (bài mẫu chính là reference script).
    """
    exam = _validate_exam(exam)
    qt = _resolve(question_type, exam)
    if qt.key == "read_aloud":
        raise HTTPException(
            status_code=400,
            detail="Read Aloud đã có sẵn script mẫu — không cần gợi ý bài mẫu.",
        )
    image_b64, image_media_type = await _read_image(image)
    band = target_band.strip() or default_target_band(exam)
    config = _resolve_config(feedback_lang)
    try:
        result = await run_in_threadpool(
            suggest_answer,
            config,
            qt,
            prompt_text=prompt,
            provided_info=provided_info,
            image_b64=image_b64,
            image_media_type=image_media_type,
            target_band=band,
            expected_duration_sec=expected_duration_sec,
        )
    except Exception as e:  # noqa: BLE001 - trả lỗi gọn cho client
        logger.exception("Lỗi khi sinh bài mẫu")
        raise HTTPException(status_code=500, detail=f"Lỗi khi sinh bài mẫu: {e}") from e
    out = result.model_dump()
    out["question_type"] = qt.key
    out["exam"] = exam
    return out


# ── Thi cả đề (cá nhân): upload đề thật → chấm từng câu/phần → gộp ────────────
# THÊM MỚI, không đụng luồng /grade & /grade-batch (chấm lẻ / cả lớp giữ nguyên).


@app.post("/exam/import")
async def exam_import(
    file: UploadFile = File(..., description="Tài liệu đề thi (.pdf/.docx/ảnh)"),
    exam: str = Form(_BASE_CONFIG.default_exam, description="Kỳ thi: toeic | ielts"),
) -> dict:
    """Bóc tách đề từ tài liệu → ExamPaper JSON (kèm warnings) cho UI review/sửa.

    Ảnh Describe Picture trả về dạng base64 trong từng câu (client giữ, gửi lại khi
    chấm) — server KHÔNG lưu file ảnh.
    """
    exam = _validate_exam(exam)
    suffix = Path(file.filename or "").suffix.lower()
    file_bytes = await file.read()
    try:
        paper, warnings = await run_in_threadpool(
            extract_exam, file_bytes, suffix, exam, _BASE_CONFIG
        )
    except ExamImportError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - trả lỗi gọn cho client
        logger.exception("Lỗi bóc tách đề")
        raise HTTPException(status_code=500, detail=f"Lỗi bóc tách đề: {e}") from e
    out = paper.to_dict()
    out["warnings"] = warnings
    return out


@app.post("/exam/grade")
async def exam_grade(
    paper: str = Form(..., description="Định nghĩa đề (JSON: {exam,title,questions[]})"),
    audios: list[UploadFile] = File(..., description="Audio từng câu (đã ghi âm)"),
    audio_question_ids: str = Form(
        ..., description="JSON list[str] question_id song song với 'audios'"
    ),
    feedback_lang: str | None = Form(None),
    mode: str = Form("practice", description="practice | mock_test"),
    accent: str = Form("default"),
    user_id: str | None = Form(
        None, description="ID ẩn danh của user (bật lưu lịch sử khi có)"
    ),
) -> dict:
    """Chấm TRỌN một đề: mỗi câu chấm độc lập qua pipeline hiện có, rồi gộp overall.

    audio_question_ids[i] cho biết file audios[i] thuộc câu nào (map theo
    question_id — KHÔNG theo index, vì UI cho reorder). Câu thiếu audio → bỏ qua.
    """
    try:
        paper_obj = ExamPaper.from_dict(json.loads(paper))
        qids = [str(x) for x in json.loads(audio_question_ids)]
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"'paper'/'audio_question_ids' JSON sai: {e}") from e

    exam = _validate_exam(paper_obj.exam)
    accent = _validate_accent(accent)
    config = _resolve_config(feedback_lang)
    requested_mode = _normalize_mode(mode)
    questions = {q.id: q for q in paper_obj.ordered()}

    if len(qids) != len(audios):
        raise HTTPException(
            status_code=400,
            detail=f"Số audio ({len(audios)}) ≠ số question_id ({len(qids)}).",
        )
    if len(audios) > _MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"Quá nhiều câu (> {_MAX_BATCH}).")

    # Đọc sẵn bytes (UploadFile là stream) + validate dạng câu/định dạng audio.
    items: list[tuple[str, bytes, str | None, QuestionType | None]] = []
    for qid, up in zip(qids, audios):
        q = questions.get(qid)
        if q is None:
            items.append((qid, b"", None, None))
            continue
        try:
            qt = _resolve(q.type, exam)
            suffix = _audio_suffix(up.filename)
        except HTTPException:
            items.append((qid, b"", None, None))
            continue
        data = await up.read()
        items.append((qid, data, suffix, qt))

    concurrency = config.batch_concurrency or (1 if config.is_local else 4)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(qid: str, data: bytes, suffix: str | None, qt: QuestionType | None) -> dict:
        q = questions.get(qid)
        base = {"question_id": qid, "sequence": q.sequence if q else None,
                "type": q.type if q else None}
        if q is None or qt is None or suffix is None:
            return {**base, "error": "Câu không hợp lệ hoặc audio sai định dạng."}
        if not data:
            return {**base, "error": "Thiếu audio cho câu này."}
        async with sem:
            try:
                result = await run_in_threadpool(
                    _grade_bytes,
                    data,
                    suffix,
                    config,
                    qt,
                    reference_script=q.reference_script,
                    image_b64=q.image_b64,
                    image_media_type=q.image_media_type,
                    expected_duration_sec=q.expected_duration_sec,
                    prompt=q.prompt,
                    provided_info=q.provided_info,
                    no_ai=False,
                    mode=requested_mode,
                    user_requested_review=False,
                    accent=accent,
                )
                return {**base, "result": result}
            except Exception as e:  # noqa: BLE001 - 1 câu lỗi không hỏng cả đề
                logger.exception("Lỗi khi chấm câu %s", qid)
                return {**base, "error": str(e)}

    graded = await asyncio.gather(*(_one(qid, d, s, qt) for qid, d, s, qt in items))
    graded.sort(key=lambda r: (r.get("sequence") is None, r.get("sequence") or 0))

    overall = compute_exam_overall(
        exam, [r.get("result", {}).get("scores") for r in graded if "result" in r]
    )
    response = {
        "exam": exam,
        "title": paper_obj.title,
        "overall": overall,
        "overall_max": 9 if exam == "ielts" else 200,
        "overall_estimated": True,  # ƯỚC TÍNH nội bộ — không phải điểm thi official
        "count": len(paper_obj.questions),
        "graded": sum(1 for r in graded if "result" in r),
        "questions": graded,
    }
    if user_id:
        # Đường /exam/grade (API client trực tiếp — SPA đi đường /grade từng câu).
        # items: (qid, bytes, suffix, qt) — chỉ giữ câu có audio hợp lệ.
        audio_by_qid = {
            qid: (data, suffix)
            for qid, data, suffix, _qt in items
            if data and suffix
        }
        await run_in_threadpool(
            _history_save,
            history.save_exam_full,
            user_id=user_id,
            mode=requested_mode,
            exam_response=response,
            audio_by_qid=audio_by_qid,
        )
    return response


_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
}
# Gốc project (để giải image_path tương đối của ngân hàng câu hỏi: "data/image/...").
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_builtin_image(image_path: str | None) -> tuple[str | None, str | None]:
    """Đọc ảnh đề (image_path tương đối từ gốc project) → (base64, media_type).

    Thiếu đường dẫn / file không tồn tại → (None, None) để câu vẫn dùng được (UI
    hiển thị "chưa có ảnh") thay vì làm hỏng cả đề mẫu.
    """
    if not image_path:
        return None, None
    path = (_PROJECT_ROOT / image_path).resolve()
    if not path.is_file():
        logger.warning("Đề mẫu: không thấy ảnh %s", path)
        return None, None
    media = _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    return base64.b64encode(path.read_bytes()).decode("ascii"), media


@app.get("/exam/builtin/{exam}/sets")
def exam_builtin_sets(exam: str) -> dict:
    """Danh sách bộ đề mẫu có sẵn cho 1 kỳ thi (để UI cho user chọn trước khi thi)."""
    exam = _validate_exam(exam)
    from .questions import list_sets  # ngân hàng câu hỏi tĩnh

    return {"exam": exam, "sets": list_sets(exam)}


@app.get("/exam/builtin/{exam}")
def exam_builtin(exam: str, set_id: str = "set1") -> dict:
    """Xuất 1 bộ đề mẫu có sẵn thành đề để thi (test nhanh không cần upload)."""
    exam = _validate_exam(exam)
    from .questions import _load_set  # ngân hàng câu hỏi tĩnh

    try:
        title, bank = _load_set(exam, set_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    questions = []
    for seq, q in enumerate(bank.values(), start=1):
        image_b64, image_media_type = _load_builtin_image(q.image_path)
        questions.append(
            {
                "id": f"q{seq}-{q.type}",
                "sequence": seq,
                "type": q.type,
                "prompt": q.prompt,
                "reference_script": q.reference_script,
                "provided_info": q.provided_info,
                "expected_duration_sec": q.expected_duration_sec,
                "image_b64": image_b64,
                "image_media_type": image_media_type,
            }
        )
    return {"exam": exam, "title": title, "questions": questions, "warnings": []}


@app.post("/exam/overall")
def exam_overall(
    exam: str = Form(..., description="Kỳ thi: toeic | ielts"),
    scores: str = Form(
        ..., description="JSON list các dict `scores` từng câu (null = câu lỗi/bỏ qua)"
    ),
    user_id: str | None = Form(
        None, description="ID ẩn danh của user (điền điểm tổng cho phiên lịch sử)"
    ),
    history_session_id: str | None = Form(
        None, description="UUID phiên thi đã gửi kèm từng câu qua /grade"
    ),
) -> dict:
    """Gộp điểm tổng cả đề từ danh sách điểm từng câu (client chấm từng câu qua /grade).

    Tách khỏi /exam/grade để client chấm RỜI từng câu (request ngắn, tránh timeout
    proxy với đề dài/model local chậm) rồi gọi 1 lần tính overall TẤT ĐỊNH ở đây —
    dùng đúng `compute_exam_overall` để tránh lệch làm tròn so với chấm gộp.
    """
    exam = _validate_exam(exam)
    try:
        per_question = json.loads(scores)
        if not isinstance(per_question, list):
            raise ValueError("'scores' phải là JSON list.")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"'scores' JSON sai: {e}") from e

    overall = compute_exam_overall(exam, per_question)
    field = "estimated_ielts_band" if exam == "ielts" else "estimated_toeic_score"
    graded = sum(1 for s in per_question if s and s.get(field) is not None)
    response = {
        "exam": exam,
        "overall": overall,
        "overall_max": 9 if exam == "ielts" else 200,
        "overall_estimated": True,  # ƯỚC TÍNH nội bộ — không phải điểm thi official
        "count": len(per_question),
        "graded": graded,
    }
    if user_id and history_session_id:
        # Endpoint sync → Starlette đã chạy trong threadpool, gọi thẳng được.
        _history_save(
            history.finalize_exam_session,
            session_id=history_session_id,
            user_id=user_id,
            overall=overall,
            overall_max=response["overall_max"],
            summary=response,
        )
    return response


# ── Lịch sử chấm bài ─────────────────────────────────────────────────────
# Đọc/xoá lịch sử của 1 user (uuid ẩn danh phía client). Ghi lịch sử nằm trong
# chính /grade, /grade-batch, /exam/grade, /exam/overall ở trên.

_HISTORY_AUDIO_MEDIA: dict[str, str] = {
    ".webm": "audio/webm", ".ogg": "audio/ogg", ".mp3": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".flac": "audio/flac", ".mp4": "video/mp4", ".mov": "video/quicktime",
}


# LƯU Ý path: KHÔNG dùng bare GET /history cho API — đó là path "ảo" của tab
# Lịch sử trên SPA (router.js); F5 trên /history phải rơi xuống catch-all để trả
# index.html. Vì vậy list nằm ở /history/list (đăng ký TRƯỚC /history/{record_id}
# để chữ "list" không bị nuốt làm record_id).
@app.get("/history/list")
def history_list(user_id: str, limit: int = 20, offset: int = 0) -> dict:
    """Danh sách bản ghi lịch sử của user (mới nhất trước, phân trang)."""
    _require_history_enabled()
    user_id = _valid_user_id_or_400(user_id)
    try:
        return history.list_records(_BASE_CONFIG, user_id, limit, offset)
    except Exception as e:  # noqa: BLE001
        logger.exception("Lỗi đọc lịch sử")
        raise HTTPException(status_code=500, detail=f"Lỗi đọc lịch sử: {e}") from e


@app.get("/history/{record_id}")
def history_detail(record_id: str, user_id: str) -> dict:
    """Chi tiết 1 bản ghi (kèm items). 404 nếu không tồn tại HOẶC sai user."""
    _require_history_enabled()
    user_id = _valid_user_id_or_400(user_id)
    rec = history.get_record(_BASE_CONFIG, user_id, record_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Không thấy bản ghi lịch sử.")
    return rec


@app.get("/history/{record_id}/audio")
def history_audio(record_id: str, user_id: str, item_id: str | None = None) -> FileResponse:
    """Audio đã lưu của bản ghi (single: không cần item_id; exam/batch: bắt buộc)."""
    _require_history_enabled()
    user_id = _valid_user_id_or_400(user_id)
    path = history.get_audio_path(_BASE_CONFIG, user_id, record_id, item_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Không thấy audio của bản ghi này.")
    media = _HISTORY_AUDIO_MEDIA.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)


@app.delete("/history/{record_id}")
def history_delete(record_id: str, user_id: str) -> dict:
    """Xoá 1 bản ghi (cascade items) + toàn bộ audio của nó trên đĩa."""
    _require_history_enabled()
    user_id = _valid_user_id_or_400(user_id)
    if not history.delete_record(_BASE_CONFIG, user_id, record_id):
        raise HTTPException(status_code=404, detail="Không thấy bản ghi lịch sử.")
    return {"deleted": True}


# Phục vụ frontend tĩnh + fallback SPA ở "/" — PHẢI đăng ký SAU mọi route API ở
# trên. Starlette so khớp route theo thứ tự đăng ký, nên /grade, /health, /docs...
# (đăng ký trước) được ưu tiên; catch-all này chỉ bắt phần còn lại.
#
# Không dùng StaticFiles(html=True) nữa vì nó chỉ fallback "/" → index.html, còn
# path "ảo" của client-side router (vd /exam/toeic/set2/q/3 — xem web/js/router.js)
# sẽ bị 404 khi tải lại trang / mở link trực tiếp. Route này: khớp file tĩnh thật
# (css/js/vendor/...) thì trả đúng file; path lạ thì trả index.html để JS tự dựng
# lại đúng màn hình từ URL.
_INDEX_HTML = _WEB_DIR / "index.html"

if not _WEB_DIR.is_dir():  # pragma: no cover - chỉ xảy ra khi deploy thiếu thư mục web/
    logger.warning("Không thấy thư mục web/ (%s) — frontend tĩnh bị tắt.", _WEB_DIR)


@app.get("/{full_path:path}")
def web_spa(full_path: str) -> FileResponse:
    candidate = (_WEB_DIR / full_path).resolve()
    if full_path and candidate.is_file() and candidate.is_relative_to(_WEB_DIR.resolve()):
        return FileResponse(candidate)
    if not _INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="Frontend tĩnh (web/) không có sẵn.")
    return FileResponse(_INDEX_HTML)

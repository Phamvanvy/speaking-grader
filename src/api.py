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
from fastapi.staticfiles import StaticFiles

from .config import Config, load_config
from .core import grade_response
from .exam_import import ExamImportError, extract_exam
from .exam_paper import ExamPaper
from .logging_setup import setup_logging
from .rubrics import EXAM_REGISTRIES
from .rubrics.base import QuestionType
from .scoring import compute_exam_overall
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
            accent=accent,
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
    return {
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
    return {
        "exam": exam,
        "title": paper_obj.title,
        "overall": overall,
        "overall_max": 9 if exam == "ielts" else 200,
        "overall_estimated": True,  # ƯỚC TÍNH nội bộ — không phải điểm thi official
        "count": len(paper_obj.questions),
        "graded": sum(1 for r in graded if "result" in r),
        "questions": graded,
    }


@app.get("/exam/builtin/{exam}")
def exam_builtin(exam: str) -> dict:
    """Xuất ngân hàng câu hỏi sẵn có thành 1 đề mẫu (test nhanh không cần upload)."""
    exam = _validate_exam(exam)
    from .questions import _load_all  # ngân hàng câu hỏi tĩnh

    bank = _load_all(exam)
    questions = []
    for seq, q in enumerate(bank.values(), start=1):
        questions.append(
            {
                "id": f"q{seq}-{q.type}",
                "sequence": seq,
                "type": q.type,
                "prompt": q.prompt,
                "reference_script": q.reference_script,
                "provided_info": q.provided_info,
                "expected_duration_sec": q.expected_duration_sec,
                "image_b64": None,
                "image_media_type": None,
            }
        )
    return {"exam": exam, "title": f"Đề mẫu {exam.upper()}", "questions": questions, "warnings": []}


# Mount frontend tĩnh ở "/" — PHẢI đặt sau mọi route API ở trên. Starlette so khớp
# route theo thứ tự đăng ký, nên /grade, /health, /docs... (đăng ký trước) được ưu
# tiên; StaticFiles chỉ bắt phần còn lại (GET / → index.html, /styles.css, /app.js).
# html=True → "/" trả index.html. check_dir=False để app vẫn khởi động được nếu
# thiếu web/ (vd chạy chỉ-API trong test); request tới "/" khi đó trả 404 gọn.
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
else:  # pragma: no cover - chỉ xảy ra khi deploy thiếu thư mục web/
    logger.warning("Không thấy thư mục web/ (%s) — frontend tĩnh bị tắt.", _WEB_DIR)

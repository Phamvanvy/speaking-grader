"""HTTP API chấm TOEIC Speaking (FastAPI).

Một endpoint `POST /grade` nhận multipart/form-data:
- audio   (bắt buộc): file ghi âm (.wav/.mp3/...).
- text    (optional): script tham chiếu → chấm dạng Read Aloud (so transcript).
- image   (optional): ảnh đề bài → chấm dạng Describe Picture (gửi LLM dạng vision).
- expected_duration_sec (optional): thời lượng kỳ vọng (giây).
- question_type (optional): ép dạng câu (read_aloud / describe_picture / ...).
- feedback_lang (optional): ngôn ngữ nhận xét (vd 'vi', 'en').
- no_ai   (optional): chỉ ASR + features, bỏ qua LLM.

Quy ước: truyền `text` → Read Aloud; truyền `image` → Describe Picture. Không
được truyền cả hai (trừ khi tự chỉ định question_type). Chạy:

    uvicorn src.api:app --reload --port 8000

Tài liệu Swagger tự sinh tại /docs.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import Config, load_config
from .core import grade_response
from .logging_setup import setup_logging
from .rubrics.toeic import get_question_type

logger = logging.getLogger("toeic.api")

app = FastAPI(
    title="TOEIC Speaking Grader API",
    description="Chấm bài nói: so audio với text (đọc to) hoặc ảnh (tả tranh).",
    version="1.0.0",
)

# Nạp config 1 lần lúc khởi động (model Whisper cache trong asr theo process).
_BASE_CONFIG: Config = load_config()
setup_logging()

# Định dạng audio chấp nhận (theo phần mở rộng, dùng để đặt tên file tạm cho Whisper).
_ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


def _resolve_config(feedback_lang: str | None) -> Config:
    """Trả về Config (ghi đè feedback_lang theo request nếu có)."""
    if feedback_lang:
        return dataclasses.replace(_BASE_CONFIG, feedback_lang=feedback_lang)
    return _BASE_CONFIG


def _pick_question_type(
    text: str | None, has_image: bool, question_type: str | None
):
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
        detail="Cần ít nhất 'text' (đọc to) hoặc 'image' (tả tranh).",
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
    }


@app.post("/grade")
async def grade(
    audio: UploadFile = File(..., description="File audio (.wav/.mp3/...)"),
    text: str | None = Form(None, description="Script tham chiếu (Read Aloud)"),
    image: UploadFile | None = File(None, description="Ảnh đề bài (Describe Picture)"),
    expected_duration_sec: float | None = Form(None),
    question_type: str | None = Form(None),
    feedback_lang: str | None = Form(None),
    prompt: str = Form("", description="Đề bài hiển thị cho thí sinh (optional)"),
    no_ai: bool = Form(False),
) -> dict:
    """Chấm 1 bài nói. Trả về JSON {transcript, features, scores}."""
    has_image = image is not None
    qt = _pick_question_type(text, has_image, question_type)

    # Đọc ảnh (nếu có) → base64.
    image_b64: str | None = None
    image_media_type: str | None = None
    if has_image:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="File ảnh rỗng.")
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        image_media_type = image.content_type or "image/jpeg"

    # Lưu audio ra file tạm (Whisper đọc theo đường dẫn, cần phần mở rộng đúng).
    suffix = Path(audio.filename or "").suffix.lower() or ".wav"
    if suffix not in _ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng audio không hỗ trợ: '{suffix}'. "
            f"Chấp nhận: {sorted(_ALLOWED_AUDIO_SUFFIXES)}",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="File audio rỗng.")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        config = _resolve_config(feedback_lang)
        output = grade_response(
            tmp.name,
            config,
            qt,
            prompt_text=prompt,
            reference_script=text,
            expected_duration_sec=expected_duration_sec,
            image_b64=image_b64,
            image_media_type=image_media_type,
            no_ai=no_ai,
            question_id=qt.key,
            save=False,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - trả lỗi gọn cho client
        logger.exception("Lỗi khi chấm")
        raise HTTPException(status_code=500, detail=f"Lỗi khi chấm: {e}") from e
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    # Không trả về audio_path tạm cho client.
    output.pop("audio_path", None)
    return output

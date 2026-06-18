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
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

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

# Định dạng audio chấp nhận (theo phần mở rộng, dùng để đặt tên file tạm cho Whisper).
_ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}

# Trần số file trong 1 batch (chặn lạm dụng; 1 lớp thực tế ~40 em).
_MAX_BATCH = 100


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
        detail="Cần ít nhất 'text' (đọc to) hoặc 'image' (tả tranh).",
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
) -> dict:
    """Ghi audio ra file tạm rồi chạy pipeline (HÀM CHẶN — gọi qua threadpool).

    Tự dọn file tạm. Trả dict kết quả (đã bỏ audio_path tạm).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        output = grade_response(
            tmp.name,
            config,
            qt,
            prompt_text=prompt,
            reference_script=reference_script,
            expected_duration_sec=expected_duration_sec,
            image_b64=image_b64,
            image_media_type=image_media_type,
            no_ai=no_ai,
            question_id=qt.key,
            save=False,
        )
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
            no_ai=no_ai,
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
    no_ai: bool = Form(False),
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

    if max_concurrency and max_concurrency > 0:
        concurrency = max_concurrency
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
                    no_ai=no_ai,
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
        "count": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "concurrency": concurrency,
        "results": results,
    }

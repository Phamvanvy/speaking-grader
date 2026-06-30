"""Bóc tách "đề thi đầy đủ" từ tài liệu (PDF / ảnh / Word) → `ExamPaper`.

Pipeline tách thành 5 BƯỚC THUẦN, độc lập, dễ test:

    read_document → normalize_to_llm_input → call_llm_extract
                                              → validate_extracted → build_exam_paper

Hàm public `extract_exam()` chỉ là orchestrator nối 5 bước.

NGUYÊN TẮC:
- KHÔNG tin trực tiếp output LLM: `validate_extracted` áp luật TẤT ĐỊNH (type hợp
  lệ theo kỳ thi, field bắt buộc theo dạng câu, gán sequence/id do server quyết).
- Vision-first cho scanned PDF: luôn render trang ra ảnh làm nguồn chính; text-layer
  (nếu có) chỉ là ngữ cảnh hỗ trợ.
- Tài liệu lớn: chặn kích thước + số trang; cảnh báo (không nhồi vô hạn vào LLM).
- Ảnh không lưu ra đĩa: trả base64 cho client giữ (xem exam_paper.py).
- Lib nặng (`fitz`, `docx`) import bên trong hàm + guard → thiếu lib chỉ tắt tính
  năng này, không sập app (mirror pattern TTS).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from .config import Config
from .exam_paper import ExamPaper, ExamQuestion
from .rubrics import EXAM_REGISTRIES, resolve_question_type
from .rubrics.base import QuestionType

logger = logging.getLogger("toeic.exam_import")

# Giới hạn an toàn cho tài liệu lớn.
MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_PAGES = 20  # số trang PDF tối đa render/gửi LLM
_PDF_RENDER_DPI = 130  # đủ nét cho vision, không quá nặng
# Trần token cho bước bóc tách bằng model LOCAL: output là JSON cấu trúc (vài KB),
# không cần trần lớn như chấm điểm. Chặn model local sinh lan man kéo dài import.
_LOCAL_EXTRACT_MAX_TOKENS = 16000

# Thời lượng kỳ vọng mặc định (giây) theo dạng câu — điền khi tài liệu không nêu.
_DEFAULT_DURATION: dict[str, float] = {
    "read_aloud": 45,
    "describe_picture": 45,
    "respond_questions": 15,
    "respond_with_info": 15,
    "express_opinion": 60,
    "part1_interview": 40,
    "part2_long_turn": 120,
    "part3_discussion": 60,
}


class ExamImportError(Exception):
    """Lỗi nghiệp vụ khi bóc tách đề (thiếu lib, file quá lớn, không đọc được...)."""


# ── BƯỚC 1: đọc tài liệu → text-layer + ảnh từng trang ────────────────────────


@dataclass
class DocumentContent:
    """Nội dung thô đã đọc từ tài liệu."""

    text: str  # text-layer (có thể rỗng nếu scanned/ảnh)
    page_images: list[tuple[str, str]]  # [(base64, media_type)] theo thứ tự trang


def read_document(file_bytes: bytes, suffix: str) -> DocumentContent:
    """Đọc tài liệu → (text-layer, danh sách ảnh trang base64).

    - .jpg/.png... → 1 ảnh, không text.
    - .pdf        → text mỗi trang (nếu có) + render mỗi trang ra ảnh PNG (vision-first).
    - .docx       → text các đoạn + ảnh nhúng.
    """
    suffix = (suffix or "").lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        media = "image/jpeg" if suffix in {".jpg", ".jpeg"} else f"image/{suffix.lstrip('.')}"
        return DocumentContent(text="", page_images=[(base64.b64encode(file_bytes).decode("ascii"), media)])
    if suffix == ".pdf":
        return _read_pdf(file_bytes)
    if suffix == ".docx":
        return _read_docx(file_bytes)
    raise ExamImportError(
        f"Định dạng đề không hỗ trợ: '{suffix}'. Hỗ trợ: .pdf, .docx, .jpg, .png."
    )


def _read_pdf(file_bytes: bytes) -> DocumentContent:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover - phụ thuộc tuỳ chọn
        raise ExamImportError(
            "Đọc PDF cần gói 'pymupdf'. Cài: pip install pymupdf (hoặc upload ảnh/.docx)."
        ) from e

    texts: list[str] = []
    images: list[tuple[str, str]] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        n_pages = min(len(doc), MAX_PAGES)
        zoom = _PDF_RENDER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i in range(n_pages):
            page = doc[i]
            texts.append(page.get_text() or "")
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            png = pix.tobytes("png")
            images.append((base64.b64encode(png).decode("ascii"), "image/png"))
    return DocumentContent(text="\n\n".join(texts).strip(), page_images=images)


def _read_docx(file_bytes: bytes) -> DocumentContent:
    try:
        import io

        from docx import Document  # python-docx
    except ImportError as e:  # pragma: no cover - phụ thuộc tuỳ chọn
        raise ExamImportError(
            "Đọc Word cần gói 'python-docx'. Cài: pip install python-docx (hoặc upload PDF/ảnh)."
        ) from e

    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
    images: list[tuple[str, str]] = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                blob = rel.target_part.blob
                ctype = getattr(rel.target_part, "content_type", "image/png") or "image/png"
                images.append((base64.b64encode(blob).decode("ascii"), ctype))
            except Exception:  # noqa: BLE001 - ảnh hỏng thì bỏ qua, không sập import
                continue
    return DocumentContent(text=text.strip(), page_images=images[:MAX_PAGES])


# ── BƯỚC 2: chuẩn hoá → input cho LLM (chặn tài liệu lớn) ─────────────────────


@dataclass
class LlmInput:
    text: str
    page_images: list[tuple[str, str]]
    warnings: list[str]


def normalize_to_llm_input(content: DocumentContent) -> LlmInput:
    """Cắt gọn & cảnh báo cho tài liệu lớn — KHÔNG nhồi vô hạn vào LLM."""
    warnings: list[str] = []
    images = content.page_images
    if len(images) > MAX_PAGES:
        warnings.append(
            f"Tài liệu có nhiều trang/ảnh ({len(images)}) — chỉ xử lý {MAX_PAGES} đầu, "
            "có thể bỏ sót câu cuối. Hãy review."
        )
        images = images[:MAX_PAGES]
    # Text-layer chỉ là hỗ trợ; cắt trần để tránh prompt phình (vision là chính).
    text = content.text or ""
    if len(text) > 20000:
        text = text[:20000]
        warnings.append("Text-layer dài — đã cắt bớt phần hỗ trợ (ảnh trang vẫn đầy đủ).")
    return LlmInput(text=text, page_images=images, warnings=warnings)


# ── BƯỚC 3: gọi LLM structured-output ────────────────────────────────────────


def _extract_system_prompt(exam: str) -> str:
    registry = EXAM_REGISTRIES[exam]
    type_lines = "\n".join(f"  - {key}: {qt.label}" for key, qt in registry.items())
    return _EXTRACT_SYSTEM_TEMPLATE.format(exam=exam.upper(), type_lines=type_lines)


def _extract_user_text(llm_input: LlmInput) -> str:
    text = "Đây là tài liệu đề thi cần bóc tách thành các câu hỏi có cấu trúc."
    if llm_input.text:
        text += "\n\n[Text-layer trích từ tài liệu]:\n" + llm_input.text
    return text


def call_llm_extract(llm_input: LlmInput, exam: str, config: Config) -> "ExtractedExam":
    """Gọi LLM → ExtractedExam. Theo backend cấu hình (anthropic | local).

    - anthropic: Claude vision (messages.parse) — đọc được cả PDF scan/ảnh.
    - local    : model local OpenAI-compatible (ép JSON schema, giống _score_local).
                 Đọc ẢNH chỉ khi model local có VISION (vd Qwen-VL); model text thuần
                 chỉ bóc được tài liệu có text-layer (PDF số / .docx).
    """
    if config.is_local:
        return _extract_local(llm_input, exam, config)
    return _extract_anthropic(llm_input, exam, config)


def _extract_anthropic(llm_input: LlmInput, exam: str, config: Config) -> "ExtractedExam":
    if not config.has_api_key:
        raise ExamImportError(
            "Bóc tách đề (Anthropic) cần ANTHROPIC_API_KEY (model vision). Đặt trong "
            ".env, hoặc dùng backend local (TOEIC_BACKEND=local)."
        )
    import anthropic

    blocks: list[dict] = [
        {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}}
        for b64, media in llm_input.page_images
    ]
    blocks.append({"type": "text", "text": _extract_user_text(llm_input)})

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    response = client.messages.parse(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        system=_extract_system_prompt(exam),
        messages=[{"role": "user", "content": blocks}],
        output_format=ExtractedExam,
    )
    result = response.parsed_output
    if result is None:
        raise ExamImportError(
            f"LLM không trả về cấu trúc đề hợp lệ (stop_reason={response.stop_reason})."
        )
    return result


def _extract_local(llm_input: LlmInput, exam: str, config: Config) -> "ExtractedExam":
    """Bóc tách bằng model local (OpenAI-compatible). Ép JSON schema của ExtractedExam.

    Model text thuần không có text-layer để đọc → báo lỗi rõ (không gửi ảnh vô ích).
    """
    if not llm_input.text and llm_input.page_images:
        # Có ảnh nhưng không có text-layer → cần model local CÓ VISION. Vẫn thử gửi
        # ảnh (data URI); nếu model text thuần sẽ lỗi/cho kết quả rỗng → báo gợi ý.
        logger.info("Local extract: tài liệu chỉ có ảnh — cần model local có vision.")

    from .scoring.backends import _get_local_client

    try:
        client = _get_local_client(config.local_base_url, config.local_api_key)
    except ImportError as e:  # pragma: no cover
        raise ExamImportError("Backend local cần gói 'openai'. Cài: pip install openai") from e

    # Quyết định có gửi ẢNH cho model local không:
    # - local_vision_extract=True (server có --mmproj): gửi ảnh + text → đọc được
    #   tranh/bảng/đề scan kể cả khi đã có text-layer.
    # - Ngược lại: chỉ gửi ảnh khi KHÔNG có text-layer (model text-thuần nhồi base64
    #   ảnh sẽ choke + cực chậm). Có text-layer → TEXT-ONLY cho nhanh.
    user_text = _extract_user_text(llm_input)
    send_images = bool(llm_input.page_images) and (
        config.local_vision_extract or not llm_input.text
    )
    if send_images:
        user_content: object = [
            *({"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}}
              for b64, media in llm_input.page_images),
            {"type": "text", "text": user_text},
        ]
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": _extract_system_prompt(exam)},
        {"role": "user", "content": user_content},
    ]
    try:
        response = client.chat.completions.create(
            model=config.local_model,
            # Trần token có giới hạn: output là JSON cấu trúc (vài KB), KHÔNG cần cả
            # max_tokens chấm điểm (vd 180k). Tránh model sinh lan man.
            max_tokens=min(config.max_tokens, _LOCAL_EXTRACT_MAX_TOKENS),
            temperature=0,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ExtractedExam", "schema": ExtractedExam.model_json_schema()},
            },
            # TẮT "thinking" (Qwen3...) — bật sẽ sinh hàng chục nghìn token reasoning
            # → import mất nhiều phút. Bóc tách không cần reasoning dài.
            extra_body={"chat_template_kwargs": {"enable_thinking": config.local_enable_thinking}},
        )
    except Exception as e:  # noqa: BLE001 - bọc lỗi mạng/model cho rõ
        raise ExamImportError(
            f"Gọi model local thất bại: {e}. Nếu đề là ảnh/PDF scan, cần model local có "
            f"vision (vd Qwen-VL); model text thuần chỉ bóc được tài liệu có text-layer."
        ) from e

    content = response.choices[0].message.content
    if not content:
        raise ExamImportError(
            "Model local không trả về nội dung. Nếu đề là ảnh/scan, model local cần có vision."
        )
    try:
        return ExtractedExam.model_validate_json(content)
    except Exception as e:  # noqa: BLE001
        raise ExamImportError(f"Model local trả JSON không đúng schema: {e}") from e


# ── BƯỚC 4: validate (KHÔNG tin LLM) ─────────────────────────────────────────


def _required_inputs_ok(qt: QuestionType, q: "ExtractedQuestion", has_image: bool) -> list[str]:
    """Trả về danh sách field BẮT BUỘC còn THIẾU cho dạng câu này."""
    missing: list[str] = []
    if qt.uses_reference_script and not (q.reference_script or "").strip():
        missing.append("reference_script")
    if qt.key == "describe_picture" and not has_image:
        missing.append("image")
    if qt.uses_provided_info and not (q.provided_info or "").strip():
        missing.append("provided_info")
    return missing


def validate_extracted(
    raw: "ExtractedExam", exam: str, page_images: list[tuple[str, str]]
) -> tuple[list[ExamQuestion], list[str]]:
    """Áp luật tất định lên output LLM → (câu hợp lệ, warnings).

    - type lạ với kỳ thi → loại câu + warning.
    - gán sequence (server quyết) + id ổn định.
    - field bắt buộc thiếu → câu vẫn giữ nhưng warning (để UI sửa).
    - điền expected_duration_sec default theo dạng câu nếu thiếu.
    """
    warnings: list[str] = []
    questions: list[ExamQuestion] = []
    seq = 0
    for idx, rq in enumerate(raw.questions):
        key = (rq.type or "").strip().lower()
        try:
            qt = resolve_question_type(key, exam=exam)
        except KeyError:
            warnings.append(
                f"Câu #{idx + 1}: dạng '{rq.type}' không thuộc kỳ thi {exam.upper()} — đã bỏ qua."
            )
            continue
        seq += 1
        # Ảnh cho describe_picture: lấy theo image_index LLM chỉ ra (kẹp về dải hợp lệ).
        image_b64 = image_media = None
        if qt.key == "describe_picture" and page_images:
            i = rq.image_index if rq.image_index is not None else (seq - 1)
            if 0 <= i < len(page_images):
                image_b64, image_media = page_images[i]
            else:
                image_b64, image_media = page_images[0]
        dur = rq.expected_duration_sec
        if dur is None or dur <= 0:
            dur = _DEFAULT_DURATION.get(qt.key)
        q = ExamQuestion(
            id=f"q{seq}-{key}",
            sequence=seq,
            type=qt.key,
            prompt=(rq.prompt or "").strip(),
            reference_script=(rq.reference_script or "").strip() or None,
            provided_info=(rq.provided_info or "").strip() or None,
            expected_duration_sec=dur,
            image_b64=image_b64,
            image_media_type=image_media,
        )
        missing = _required_inputs_ok(qt, rq, has_image=bool(image_b64))
        if missing:
            warnings.append(
                f"Câu #{seq} ({qt.label}): thiếu {', '.join(missing)} — hãy bổ sung trước khi thi."
            )
        questions.append(q)

    if not questions:
        warnings.append("Không bóc tách được câu hỏi hợp lệ nào — hãy kiểm tra lại tài liệu.")
    return questions, warnings


# ── BƯỚC 5: dựng ExamPaper ───────────────────────────────────────────────────


def build_exam_paper(exam: str, title: str, questions: list[ExamQuestion]) -> ExamPaper:
    return ExamPaper(exam=exam, title=title or f"Đề {exam.upper()}", questions=questions)


# ── Orchestrator ─────────────────────────────────────────────────────────────


def extract_exam(
    file_bytes: bytes, suffix: str, exam: str, config: Config
) -> tuple[ExamPaper, list[str]]:
    """Bóc tách đề từ tài liệu → (ExamPaper, warnings). Nối 5 bước trên."""
    if not file_bytes:
        raise ExamImportError("File rỗng.")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise ExamImportError(
            f"File quá lớn ({len(file_bytes) // (1024 * 1024)} MB > {MAX_FILE_BYTES // (1024 * 1024)} MB)."
        )
    if exam not in EXAM_REGISTRIES:
        raise ExamImportError(f"Kỳ thi không hợp lệ: '{exam}'.")

    content = read_document(file_bytes, suffix)
    llm_input = normalize_to_llm_input(content)
    raw = call_llm_extract(llm_input, exam, config)
    questions, warnings = validate_extracted(raw, exam, llm_input.page_images)
    paper = build_exam_paper(exam, (raw.title or "").strip(), questions)
    return paper, llm_input.warnings + warnings


# ── Schema structured-output (Pydantic) ──────────────────────────────────────
# Đặt CUỐI file để dùng được type hint string ở trên; import pydantic ở module-level
# an toàn vì anthropic SDK (đã là dependency) kéo theo pydantic.

from pydantic import BaseModel, Field  # noqa: E402


class ExtractedQuestion(BaseModel):
    type: str = Field(description="Một trong các key dạng câu hợp lệ của kỳ thi.")
    prompt: str = Field(default="", description="Đề bài/câu hỏi hiển thị cho thí sinh.")
    reference_script: str | None = Field(
        default=None, description="CHỈ cho read_aloud: nguyên văn đoạn cần đọc to."
    )
    provided_info: str | None = Field(
        default=None,
        description="Tài liệu/cue card cho sẵn (respond_with_info / part2_long_turn).",
    )
    expected_duration_sec: float | None = Field(
        default=None, description="Thời lượng kỳ vọng (giây) nếu tài liệu có nêu."
    )
    image_index: int | None = Field(
        default=None,
        description="CHỈ cho describe_picture: index (0-based) của ảnh trang chứa bức tranh.",
    )


class ExtractedExam(BaseModel):
    title: str = Field(default="", description="Tiêu đề đề thi nếu có.")
    questions: list[ExtractedQuestion] = Field(default_factory=list)


_EXTRACT_SYSTEM_TEMPLATE = (
    "Bạn là trợ lý bóc tách ĐỀ THI NÓI {exam} từ tài liệu (ảnh trang quét/PDF/Word).\n"
    "Đọc kỹ các trang và trích ra DANH SÁCH CÂU HỎI theo đúng thứ tự xuất hiện.\n\n"
    "Mỗi câu hỏi gán 'type' là MỘT trong các key hợp lệ sau (KHÔNG dùng key khác):\n"
    "{type_lines}\n\n"
    "QUY TẮC:\n"
    "- read_aloud: copy NGUYÊN VĂN đoạn cần đọc vào 'reference_script'.\n"
    "- describe_picture: đặt 'image_index' = số thứ tự (0-based) của ảnh trang chứa bức tranh.\n"
    "- respond_with_info / part2_long_turn: đưa tài liệu/cue card cho sẵn vào 'provided_info'.\n"
    "- Câu hỏi/chỉ dẫn hiển thị cho thí sinh → 'prompt'.\n"
    "- Nếu tài liệu nêu thời gian chuẩn bị/trả lời → 'expected_duration_sec' (giây).\n"
    "- KHÔNG bịa câu không có trong tài liệu. Giữ nguyên thứ tự."
)

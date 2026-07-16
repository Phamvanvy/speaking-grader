"""Chấm điểm bằng LLM với structured output.

Gửi đề bài + (script) + transcript + số liệu khách quan + cờ gating cho model,
nhận về SpeakingResult đúng schema (không phải tự parse JSON).

Hai backend (xem Config.backend):
- "anthropic": Claude qua Anthropic SDK (messages.parse + adaptive thinking).
- "local": model local (vd Qwen3 qua llama.cpp server) qua API
  OpenAI-compatible, ép schema bằng response_format json_schema.

Tổ chức (package):
  - compute.py: quy đổi điểm tiêu chí → điểm tổng (TOEIC/IELTS) TẤT ĐỊNH
  - prompts.py: dựng system/user prompt + JSON schema
  - validation.py: validate + clean output LLM
  - api_logging.py: log prompt/response (console + file)
  - backends.py: gọi Anthropic / model local
  - __init__.py (đây): score() orchestrator + re-export công khai
"""

from __future__ import annotations

import logging

from ..asr import Transcription
from ..config import Config
from ..features import Features
from ..gating import GatingResult
from ..phoneme.models import PhonemeResult
from ..rubrics.base import Exam, QuestionType
from ..schema import SpeakingResult

# Submodule cùng package src.scoring → import bằng `.`; module cấp src ở trên dùng `..`.
from .backends import (
    _get_local_client,
    _local_client_cache,
    _score_anthropic,
    _score_local,
    generate,
)
from .compute import (
    _compute_ielts_band,
    _compute_toeic_score,
    _compute_topik_score,
    _interp_crit_points,
    _round_half,
    compute_exam_overall,
)
from .prompts import (
    _build_system_prompt,
    _build_user_prompt,
    _compact_phoneme_data,
    _local_response_schema,
)
from .validation import (
    _drop_invalid_corrections,
    _is_truncated,
    _norm_for_match,
    _validate_result,
)

logger = logging.getLogger("toeic.scoring")


def score(
    config: Config,
    qt: QuestionType,
    prompt_text: str,
    reference_script: str | None,
    transcription: Transcription,
    features: Features,
    gating: GatingResult,
    phoneme_result: PhonemeResult | None = None,
    image_b64: str | None = None,
    image_media_type: str | None = None,
    provided_info: str | None = None,
) -> tuple[SpeakingResult, dict]:
    """Gọi LLM (Claude / OpenRouter / model local) → (SpeakingResult, meta).

    meta (từ backends.generate): {backend_used, model, latency_ms,
    fallback_reason} — backend nào THẬT SỰ chấm, đưa vào telemetry của bài.

    phoneme_result: kết quả phoneme analysis từ wav2vec/MFA (optional).
        Nếu có thì thêm vào payload để AI dùng làm evidence cho pronunciation.
    image_b64/image_media_type: ảnh đề bài (vd Describe Picture) gửi kèm dạng
    vision. Cả hai backend đều hỗ trợ; bỏ trống nếu không có ảnh.
    provided_info: tài liệu cho sẵn (Q8-10) dạng text; chỉ đưa vào payload khi
        dạng câu có uses_provided_info.
    """
    system_prompt = _build_system_prompt(qt, config.feedback_lang)
    user_prompt = _build_user_prompt(
        qt,
        prompt_text,
        reference_script,
        transcription,
        features,
        gating,
        phoneme_result=phoneme_result,
        has_image=bool(image_b64),
        provided_info=provided_info,
    )

    # Gọi backend rồi validate; nếu output rác thì retry 1 lần và raise rõ ràng
    # thay vì âm thầm lưu điểm hỏng. Bắt glitch JSON hiếm (thiếu tiêu chí /
    # suggestions lẫn tên key / text cụt) mà schema Pydantic không chặn được.
    max_attempts = 2
    last_problems: list[str] = []
    for attempt in range(1, max_attempts + 1):
        raw, gen_meta = generate(
            config,
            system_prompt,
            user_prompt,
            SpeakingResult,
            # Schema siết theo qt (ép đúng N tiêu chí + enum key) — chỉ các
            # backend OpenAI-compatible dùng; anthropic đi messages.parse.
            json_schema=_local_response_schema(qt),
            schema_name="SpeakingResult",
            image_b64=image_b64,
            image_media_type=image_media_type,
        )
        assert isinstance(raw, SpeakingResult)
        result = raw
        last_problems = _validate_result(result, qt)
        if not last_problems:
            # Loại các correction mà `said` không thực sự có trong transcript
            # (LLM vẫn có thể paraphrase dù prompt đã cấm). Chạy NGAY sau parse,
            # trước khi result rời score() → JSON/report/UI không bao giờ thấy
            # correction bịa.
            _drop_invalid_corrections(result, transcription.text)
            # Ghi đè điểm tổng bằng giá trị TÍNH TẤT ĐỊNH từ điểm tiêu chí —
            # bỏ qua số (nếu có) mà LLM trả về để đảm bảo nhất quán giữa các lần.
            # Chỉ set field của đúng kỳ thi; field còn lại để None.
            if qt.exam == Exam.IELTS.value:
                result.estimated_ielts_band = _compute_ielts_band(result)
                result.estimated_toeic_score = None
                result.estimated_topik_score = None
            elif qt.exam == Exam.TOPIK.value:
                # question_type trong result chỉ là echo của LLM — ghi đè bằng
                # key authoritative vì compute_exam_overall weight overall theo
                # scores["question_type"] (mức câu sơ/trung/cao cấp).
                result.question_type = qt.key
                result.estimated_topik_score = _compute_topik_score(result, qt)
                result.estimated_toeic_score = None
                result.estimated_ielts_band = None
            else:
                result.estimated_toeic_score = _compute_toeic_score(result)
                result.estimated_ielts_band = None
                result.estimated_topik_score = None
            return result, gen_meta
        logger.warning(
            "Kết quả chấm không hợp lệ (lần %d/%d): %s",
            attempt,
            max_attempts,
            "; ".join(last_problems),
        )
    raise RuntimeError(
        f"LLM trả kết quả hỏng sau {max_attempts} lần (schema hợp lệ nhưng "
        f"nội dung rác): {'; '.join(last_problems)}"
    )


__all__ = [
    "score",
    # compute
    "_compute_toeic_score",
    "_compute_ielts_band",
    "_compute_topik_score",
    "_round_half",
    "_interp_crit_points",
    # prompts
    "_build_system_prompt",
    "_build_user_prompt",
    "_local_response_schema",
    "_compact_phoneme_data",
    # validation
    "_validate_result",
    "_is_truncated",
    "_drop_invalid_corrections",
    "_norm_for_match",
    # backends
    "generate",
    "_score_anthropic",
    "_score_local",
    "_get_local_client",
]

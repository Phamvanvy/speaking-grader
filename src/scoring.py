"""Chấm điểm bằng LLM với structured output.

Gửi đề bài + (script) + transcript + số liệu khách quan + cờ gating cho model,
nhận về SpeakingResult đúng schema (không phải tự parse JSON).

Hai backend (xem Config.backend):
- "anthropic": Claude qua Anthropic SDK (messages.parse + adaptive thinking).
- "local": model local (vd Qwen3 qua llama.cpp server) qua API
  OpenAI-compatible, ép schema bằng response_format json_schema.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from .asr import Transcription
from .config import Config, resolve_language_name
from .features import Features
from .gating import GatingResult
from .phoneme.models import PhonemeResult
from .rubrics.base import Exam, QuestionType
from .schema import CompletionLevel, SpeakingResult

logger = logging.getLogger("toeic.scoring")

# Tái dùng OpenAI client cho backend local: trước đây mỗi bài tạo client mới →
# connection pool mới mỗi lần. Cache theo (base_url, api_key) để batch dùng lại
# một pool. Client của openai-python thread-safe nên dùng chung giữa các luồng.
_local_client_cache: dict[tuple[str, str], object] = {}


def _get_local_client(base_url: str, api_key: str):
    key = (base_url, api_key)
    client = _local_client_cache.get(key)
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        _local_client_cache[key] = client
    return client

# --- Quy đổi điểm tiêu chí (0-3) → điểm TOEIC Speaking (0-200) -----------------
# Vì sao có khối này: trước đây estimated_toeic_score do LLM TỰ CHỌN trong prose
# ("rơi vào khoảng 80-90 → 85"), nên cùng một bộ điểm tiêu chí mỗi lần lại ra số
# khác (85 vs 75) — nhất là với model local nhỏ/nén. Giờ LLM chỉ chấm 0-3 mỗi
# tiêu chí + mức hoàn thành; con số 0-200 được TÍNH bằng công thức dưới đây nên
# CỐ ĐỊNH với cùng input. Hằng số để lộ ở module-level cho dễ tinh chỉnh.
#
# Mốc neo theo thang proficiency ETS: điểm tiêu chí 2/3 ("đạt, vài lỗi") ~ level
# 5 (~110đ), 3/3 ~ level 7-8 (~190đ). Nội suy tuyến tính cho điểm float.
_CRIT_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 60.0),
    (2.0, 110.0),
    (3.0, 190.0),
)

# task_completion / content_relevance dưới mức 'high' nhân phạt vào điểm tổng
# (mắt xích yếu nhất quyết định). Đảm bảo bài làm dở/lạc đề không được điểm cao
# dù phát âm tốt — khớp yêu cầu gating trong system prompt.
_LEVEL_PENALTY: dict[CompletionLevel, float] = {
    CompletionLevel.very_low: 0.35,
    CompletionLevel.low: 0.60,
    CompletionLevel.medium: 0.85,
    CompletionLevel.high: 1.0,
}


def _interp_crit_points(score: float) -> float:
    """Nội suy điểm tiêu chí (0-3) → điểm thành phần (0-190) theo _CRIT_ANCHORS."""
    s = max(0.0, min(3.0, score))
    for (x0, y0), (x1, y1) in zip(_CRIT_ANCHORS, _CRIT_ANCHORS[1:]):
        if s <= x1:
            return y0 + (y1 - y0) * (s - x0) / (x1 - x0)
    return _CRIT_ANCHORS[-1][1]


def _compute_toeic_score(result: SpeakingResult) -> int:
    """Tính estimated_toeic_score (0-200) TẤT ĐỊNH từ điểm tiêu chí + mức hoàn thành.

    Cùng một bộ (điểm tiêu chí, task_completion, content_relevance) luôn cho cùng
    một số → loại bỏ dao động do LLM tự bốc số. Làm tròn về bội số của 10 (thang
    TOEIC Speaking báo theo bước 10).
    """
    if not result.criteria:
        return 0
    base = sum(_interp_crit_points(c.score) for c in result.criteria) / len(
        result.criteria
    )
    penalty = min(
        _LEVEL_PENALTY.get(result.task_completion, 1.0),
        _LEVEL_PENALTY.get(result.content_relevance, 1.0),
    )
    raw = base * penalty
    return max(0, min(200, int(round(raw / 10.0) * 10)))


# --- Quy đổi điểm tiêu chí (band 0-9) → overall band IELTS (0-9) ---------------
# IELTS Speaking: LLM chấm mỗi tiêu chí trên band 0-9; overall = TRUNG BÌNH 4 tiêu
# chí, làm tròn về 0.5 gần nhất (đúng cách giám khảo IELTS tổng hợp). Tính trong
# code (không để LLM bốc) nên cùng bộ band tiêu chí luôn ra cùng một overall.

# Trần overall band khi task_completion / content_relevance thấp — GUARDRAIL NỘI
# BỘ (không phải công thức IELTS official) chống "nói mượt nhưng lạc đề/quá ngắn".
# Đặt nới tay: chỉ thực sự cắn khi completion very_low/low; medium ~6.5 để bài
# tốt nhưng hơi ngắn không bị tụt quá đáng.
_IELTS_LEVEL_CAP: dict[CompletionLevel, float] = {
    CompletionLevel.very_low: 3.0,
    CompletionLevel.low: 4.5,
    CompletionLevel.medium: 6.5,
    CompletionLevel.high: 9.0,
}


def _round_half(x: float) -> float:
    """Làm tròn về bội 0.5 theo quy tắc IELTS (round-half-UP).

    KHÔNG dùng round() built-in (banker's rounding: round(6.25*2)/2 = 6.0 — sai).
    Làm sạch sai số nhị phân (round(x, 4)) TRƯỚC khi floor để tránh 6.75 lưu thành
    13.4999… → floor lệch về 6.5. Cận: 6.124→6.0, 6.25→6.5, 6.74→6.5, 6.75→7.0.
    """
    clean = round(x, 4)
    return math.floor(clean * 2 + 0.5) / 2


def _compute_ielts_band(result: SpeakingResult) -> float:
    """Tính estimated_ielts_band (0-9, bước 0.5) TẤT ĐỊNH từ band tiêu chí.

    overall = trung bình band 4 tiêu chí, áp trần theo completion (guardrail), rồi
    làm tròn 0.5 và kẹp [0, 9].
    """
    if not result.criteria:
        return 0.0
    mean = sum(c.score for c in result.criteria) / len(result.criteria)
    cap = min(
        _IELTS_LEVEL_CAP.get(result.task_completion, 9.0),
        _IELTS_LEVEL_CAP.get(result.content_relevance, 9.0),
    )
    capped = min(mean, cap)
    return max(0.0, min(9.0, _round_half(capped)))

# Directory to store prompt logs for debugging / model comparison.
# Enable by setting env var TOEIC_LOG_PROMPTS=1 or Config.log_prompts=True.
_PROMPT_LOG_DIR = Path("outputs/prompt_logs")

# Chính sách xoay log: mỗi LẦN CHẠY (process mới) sẽ dọn sạch log của lần chạy
# trước ở lần ghi đầu tiên — nên chạy lại app = log mới "ghi đè" log cũ. Nhưng
# trong CÙNG một phiên, các lần chấm tiếp theo chỉ GHI THÊM (append), không xoá
# nhau. Cờ dưới đảm bảo bước dọn chạy đúng một lần cho mỗi process.
_prompt_log_reset_done = False


def _ensure_prompt_log_dir() -> None:
    """Tạo thư mục log; lần ĐẦU trong process thì xoá log của lần chạy trước.

    Hiệu ứng: chạy lại app → log mới ghi đè (dọn) log cũ; nhiều lần chấm trong
    cùng một phiên → tích luỹ thêm (append), không đè lên nhau.
    """
    global _prompt_log_reset_done
    _PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _prompt_log_reset_done:
        for old in _PROMPT_LOG_DIR.glob("*.json"):
            try:
                old.unlink()
            except OSError as e:  # pragma: no cover - file bị khoá/đang mở hiếm gặp
                logger.warning("Không xoá được log cũ %s: %s", old, e)
        _prompt_log_reset_done = True


def _build_system_prompt(qt: QuestionType, feedback_lang: str) -> str:
    criteria_lines = "\n".join(
        f"- {c.key} ({c.label}): {c.description}" for c in qt.criteria
    )
    language_name = resolve_language_name(feedback_lang)

    # Khác biệt theo kỳ thi: văn phong giám khảo + thang điểm + khối "FINAL SCORE".
    # Phần còn lại của prompt (evidence rules, task completion, ngôn ngữ) dùng chung.
    is_ielts = qt.exam == Exam.IELTS.value
    exam_label = "IELTS" if is_ielts else "TOEIC"
    if is_ielts:
        scale_label = "band 0-9 (steps of 0.5)"
        final_score_block = (
            "FINAL SCORE (important):\n"
            "- Do NOT compute or output the overall band (estimated_ielts_band). "
            "It is derived AUTOMATICALLY by the system as the average of your four "
            "per-criterion band scores, rounded to the nearest 0.5 (and capped if "
            "task_completion / content_relevance is low). Your job is ONLY to score "
            "each of the four criteria on the 0-9 band scale accurately and "
            "consistently.\n"
            "- Be calibrated on the 0-9 band scale: anchor every criterion to the "
            "SCORING SCALE above and to the objective evidence, since the overall "
            "band depends entirely on these per-criterion bands."
        )
        rationale_total = "overall band"
    else:
        scale_label = "0-3"
        final_score_block = (
            "FINAL SCORE (important):\n"
            "- Do NOT compute or output the 0-200 estimated_toeic_score. It is "
            "derived AUTOMATICALLY by the system from your per-criterion 0-3 scores "
            "plus task_completion / content_relevance. Your job is ONLY to score "
            "each criterion 0-3 accurately and consistently (TOEIC does NOT use "
            "IELTS bands).\n"
            "- Be calibrated on the 0-3 scale: anchor every criterion to the "
            "SCORING SCALE above and to the objective evidence, since the final "
            "number depends entirely on these per-criterion scores."
        )
        rationale_total = "0-200 number"

    return f"""You are an experienced {exam_label} Speaking examiner. Score one spoken \
response for the task type: {qt.label}.

TASK GUIDANCE:
{qt.guidance}

CRITERIA TO SCORE (only these):
{criteria_lines}

SCORING SCALE:
{qt.scale_description}

EVIDENCE RULES (important):
- Use the OBJECTIVE METRICS provided (speech_rate_wpm, pause_count, \
longest_pause_sec, filler_count, and for read-aloud the accuracy_metrics: wer, \
substitutions, insertions, deletions) as PRIMARY evidence.
- Use analysis of the transcript text as SECONDARY evidence.
- Do NOT rely solely on ASR confidence (avg_word_probability / \
min_word_probability). It is affected by microphone quality, accent, and \
background noise — treat it ONLY as weak supporting evidence, never as the \
pronunciation score itself.
- For read-aloud, accuracy_metrics.word_issues lists places where the ASR \
transcript diverged from the script (substitution / insertion / deletion), e.g. \
expected "morning" but recognized "warning". These are NOT confirmed \
mispronunciations — the ASR may have misheard due to noise, accent, or its own \
limits. Use them ONLY as "words worth reviewing": you may say the ASR may have \
misheard a word and suggest the test-taker double-check it. NEVER state with \
certainty that the test-taker pronounced a specific word wrong based on a \
word_issue alone.
- PHONEME METRICS (if available): phoneme_data provides deep pronunciation \
evidence at the phoneme level (IPA). Use overall_accuracy as a STRONG signal for \
the pronunciation score. High-severity errors indicate clear mispronunciations. \
Pay special attention to substitution errors where similar-sounding phonemes are \
confused (e.g. /θ/ → /s/, /æ/ → /ɛ/) — these are common ESL mistakes. Low \
severity errors may be acceptable regional variants. If phoneme_data is null or \
disabled, rely on word-level evidence only.
- Each phoneme error may include a `word` field — the exact reference word that \
contains the mispronounced phoneme. ONLY mention a word if it appears in the \
phoneme error data. Do NOT infer or guess spoken words from phoneme \
substitutions. When you cite a phoneme error, name that exact `word` (e.g. "âm \
/d/ trong từ 'floods' bị phát âm thành /aɪ/"); if `word` is null, describe the \
phoneme generically without naming any word.

TASK COMPLETION:
- task_completion reflects whether the response actually fulfils the prompt \
(answered fully, long enough, on-topic). A grammatically perfect but far too \
short or off-topic answer must get a LOW task_completion.
- If a completion floor is provided by upstream rule-based checks, do not score \
task_completion higher than that floor.

{final_score_block}
Give concrete, actionable suggestions for each criterion.

VOCABULARY CORRECTIONS (lexical_resource / vocabulary criterion):
- For the lexical_resource (IELTS) / vocabulary (TOEIC) criterion, populate its \
`corrections` list with one entry per wrong, unnatural, or imprecise word choice \
you find: `said` = the candidate's phrase, `suggested` = the correct word/phrase, \
`reason` = a short why, `example` = one natural sentence using the suggested word.
### CRITICAL — the `said` field MUST be an exact substring of the candidate's \
transcript. Do not paraphrase, normalize, translate, or correct the candidate's \
mistake inside `said`. Leave `corrections` empty for every other criterion.

EXPLAIN YOUR REASONING (important):
- Each criterion's `justification` must be a clear, logical chain: cite the \
specific objective metric or transcript evidence, say what it implies, then why \
that lands the criterion at this {scale_label} score and not one higher or lower.
- `score_rationale` must explain which criteria are strong vs weak, how \
task_completion / content_relevance and any gating floor affect the overall \
quality, and what level the response is at. Do NOT state a specific {rationale_total} \
— that total is computed automatically from your per-criterion scores.

OUTPUT LANGUAGE (important):
- Write ALL human-readable text — every `justification`, every entry in \
`suggestions`, `score_rationale`, and `summary_feedback` — in {language_name}.
- Keep machine fields unchanged and in English: the `criterion` field must stay \
the lowercase English key (e.g. "pronunciation", "intonation_stress"), and the \
enum values for task_completion / content_relevance (very_low/low/medium/high) \
stay as-is. Only the explanatory prose is translated."""


def _compact_phoneme_data(phoneme_result: PhonemeResult) -> dict:
    """Bản gọn của phoneme_result để nhúng vào prompt LLM.

    Vì sao: PhonemeResult.to_dict() kèm `segments` thô (164–259 frame, mỗi frame
    {phoneme,start,end,confidence,backend}) + danh sách `reference_phonemes` đầy
    đủ + `audio_path`. Khối này chiếm ~95% kích thước phoneme_data trong prompt
    (~40k ký tự) nhưng VÔ DỤNG với model text: timestamp/khung thời gian không
    giúp chấm phát âm. Chỉ giữ bằng chứng model thực sự dùng — điểm tổng hợp +
    top lỗi (score.to_dict() đã cap errors ở 20) + metadata backend. Cắt phần này
    giảm prompt mạnh → prefill nhanh hơn hẳn, và tăng tỉ trọng system prompt
    (được prefix-cache) trong tổng prompt.

    Lưu ý: chỉ ảnh hưởng prompt chấm điểm. to_dict() đầy đủ (gồm segments) vẫn
    được dùng nguyên vẹn cho report/JSON output ở core.py.
    """
    return {
        "backend_used": phoneme_result.backend_used,
        "backend_available": phoneme_result.backend_available,
        "warning": phoneme_result.warning,
        # score.to_dict() đã gồm overall_accuracy, *_count, reference_count,
        # predicted_count, avg_confidence và errors[:20] (đã sort theo severity).
        "score": phoneme_result.score.to_dict() if phoneme_result.score else None,
    }


def _build_user_prompt(
    qt: QuestionType,
    prompt_text: str,
    reference_script: str | None,
    transcription: Transcription,
    features: Features,
    gating: GatingResult,
    phoneme_result: PhonemeResult | None = None,
    has_image: bool = False,
    provided_info: str | None = None,
) -> str:
    payload: dict = {
        "task_prompt": prompt_text,
        "reference_script": reference_script if qt.uses_reference_script else None,
        "transcript": transcription.text,
        "objective_metrics": features.to_dict(),
        "rule_based_gating": {
            "task_completion_floor": gating.task_completion_floor,
            "reasons": gating.reasons,
            "reference_coverage": gating.reference_coverage,
            "fail_reference_match": gating.fail_reference_match,
        },
    }

    # Tài liệu cho sẵn (Q8-10): chỉ đưa vào khi dạng câu dùng provided_info.
    if qt.uses_provided_info and provided_info:
        payload["provided_info"] = provided_info

    # Include phoneme data if available — bản GỌN (không kèm segments thô) để
    # prompt nhẹ hơn; xem _compact_phoneme_data.
    if phoneme_result is not None:
        payload["phoneme_data"] = _compact_phoneme_data(phoneme_result)
        logger.info(
            "Phoneme data included in scoring payload: "
            "backend=%s | available=%s | segments=%d",
            phoneme_result.backend_used,
            phoneme_result.backend_available,
            len(phoneme_result.segments),
        )

    if not has_image:
        image_note = ""
    elif qt.uses_provided_info:
        # Q8-10: ảnh là TÀI LIỆU NGUỒN (lịch trình/agenda...), không phải tranh để tả.
        image_note = (
            "An IMAGE is attached: it is the SOURCE DOCUMENT (e.g. a schedule, "
            "agenda, itinerary, or information table) that the test-taker had to "
            "answer from. Do NOT treat it as a picture to describe. Judge whether "
            "the spoken transcript answers the question using the information in "
            "this document ACCURATELY — wrong facts (times, dates, names, rooms, "
            "prices) or invented details must lower relevance / content_relevance.\n\n"
        )
    else:
        image_note = (
            "An IMAGE of the picture the test-taker was asked to describe is attached "
            "to this message. Judge whether the spoken transcript accurately and "
            "completely describes what is actually in the picture (objects, people, "
            "actions, setting). A description that does not match the picture must "
            "lower content_relevance / relevance.\n\n"
        )
    exam_label = "IELTS" if qt.exam == Exam.IELTS.value else "TOEIC"
    return (
        f"Score the following {exam_label} Speaking response. All numeric metrics "
        "are pre-computed and objective.\n\n"
        + image_note
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


# Ký tự mở ngoặc ở cuối chuỗi → dấu hiệu text bị cắt giữa chừng (JSON degenerate).
_DANGLING_OPEN = ("(", "[", "{", "（", "［", "｛")


def _is_truncated(text: str) -> bool:
    """True nếu chuỗi rỗng hoặc kết thúc bằng dấu mở ngoặc (bị cắt giữa chừng)."""
    s = (text or "").strip()
    return not s or s.endswith(_DANGLING_OPEN)


def _norm_for_match(s: str) -> str:
    """Chuẩn hoá để so khớp substring khoan dung: lower + gộp khoảng trắng."""
    return " ".join(s.lower().split())


def _drop_invalid_corrections(result: SpeakingResult, transcript: str) -> None:
    """Bỏ các LexicalCorrection mà `said` không có trong transcript (mutate result).

    LLM vẫn có thể paraphrase `said` dù prompt cấm → mọi correction phải truy
    ngược được về điều thí sinh thực sự nói. So khớp khoan dung (case-insensitive,
    gộp khoảng trắng) để tránh loại nhầm vì khác hoa/thường hay spacing.
    """
    haystack = _norm_for_match(transcript)
    dropped = 0
    for c in result.criteria:
        if not c.corrections:
            continue
        kept = [
            corr for corr in c.corrections
            if corr.said and _norm_for_match(corr.said) in haystack
        ]
        dropped += len(c.corrections) - len(kept)
        c.corrections = kept
    if dropped:
        logger.info(
            "Đã loại %d correction có `said` không khớp transcript (LLM paraphrase).",
            dropped,
        )


def _validate_result(result: SpeakingResult, qt: QuestionType) -> list[str]:
    """Bắt output 'hợp lệ schema nhưng rác' mà Pydantic không chặn được.

    Trả về danh sách mô tả lỗi (rỗng nếu OK). Chỉ gắn cờ 3 dạng hỏng đã quan
    sát thực tế: thiếu tiêu chí bắt buộc, suggestions điền nhầm tên key tiêu chí,
    và text bị cắt/rỗng. KHÔNG bắt suggestions rỗng — model trả thiếu suggestions
    vẫn là output hợp lệ.
    """
    problems: list[str] = []
    required = {c.key for c in qt.criteria}

    present = {c.criterion for c in result.criteria}
    missing = required - present
    if missing:
        problems.append(f"thiếu tiêu chí bắt buộc: {sorted(missing)}")

    for c in result.criteria:
        polluted = [s for s in c.suggestions if s in required]
        if polluted:
            problems.append(
                f"suggestions của '{c.criterion}' chứa tên tiêu chí: {polluted}"
            )
        if _is_truncated(c.justification):
            problems.append(f"justification của '{c.criterion}' bị cắt/rỗng")

    if _is_truncated(result.score_rationale):
        problems.append("score_rationale bị cắt/rỗng")
    if _is_truncated(result.summary_feedback):
        problems.append("summary_feedback bị cắt/rỗng")

    return problems


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
) -> SpeakingResult:
    """Gọi LLM (Claude hoặc model local) và trả về SpeakingResult.

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
        if config.is_local:
            result = _score_local(
                config, system_prompt, user_prompt, image_b64, image_media_type
            )
        else:
            result = _score_anthropic(
                config, system_prompt, user_prompt, image_b64, image_media_type
            )
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
            else:
                result.estimated_toeic_score = _compute_toeic_score(result)
                result.estimated_ielts_band = None
            return result
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


def _score_anthropic(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    if not config.has_api_key:
        raise RuntimeError(
            "Thiếu ANTHROPIC_API_KEY. Đặt trong .env, dùng TOEIC_BACKEND=local "
            "để chấm bằng model local, hoặc chạy với --no-ai để chỉ lấy "
            "transcript + features."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # Không có ảnh → giữ nguyên content dạng chuỗi (hành vi cũ). Có ảnh → khối
    # image (base64) đứng trước, rồi khối text để Claude nhìn tranh trước khi đọc.
    if image_b64:
        content: object = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type or "image/jpeg",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": user_prompt},
        ]
    else:
        content = user_prompt

    messages = [{"role": "user", "content": content}]

    # Always log the messages being sent (sanitized). Anthropic giữ system prompt
    # tách riêng nên log kèm để thấy đủ system + user.
    _log_messages(
        logger, "anthropic", config.model, messages, system_prompt=system_prompt
    )

    # Log the full request payload (includes system prompt for Anthropic)
    if config.log_prompts:
        _log_api_request(
            config, "anthropic",
            model=config.model,
            base_url=None,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=0,
            extra_body=None,
            system_prompt=system_prompt,
        )

    t0 = time.monotonic()
    response = client.messages.parse(
        model=config.model,
        max_tokens=config.max_tokens,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=messages,
        output_format=SpeakingResult,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Claude chấm xong | model=%s | latency=%.2fs | "
        "input_tokens=%s output_tokens=%s",
        config.model,
        latency,
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )

    result = response.parsed_output

    # Log AI response
    if config.log_prompts:
        response_json = result.model_dump_json() if result else "null"
        _log_response(config, "anthropic", response_json)
    if result is None:
        # stop_reason refusal / max_tokens → parsed_output có thể None
        hint = (
            f" JSON bị cắt vì chạm trần max_tokens={config.max_tokens} — "
            f"tăng TOEIC_MAX_TOKENS."
            if response.stop_reason == "max_tokens"
            else ""
        )
        raise RuntimeError(
            f"Claude không trả về kết quả đúng schema "
            f"(stop_reason={response.stop_reason}).{hint}"
        )
    return result


def _score_local(
    config: Config,
    system_prompt: str,
    user_prompt: str,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> SpeakingResult:
    """Chấm bằng model local qua API OpenAI-compatible (vd llama.cpp server).

    Ép đúng schema bằng response_format json_schema — llama.cpp chuyển schema
    thành GBNF grammar nên JSON trả về luôn hợp lệ. Không có 'thinking' của
    Claude; nếu model hỗ trợ reasoning (Qwen3) có thể bật qua chat template.
    """
    try:
        client = _get_local_client(config.local_base_url, config.local_api_key)
    except ImportError as e:  # pragma: no cover - phụ thuộc tuỳ chọn
        raise RuntimeError(
            "Backend local cần gói 'openai'. Cài: pip install openai"
        ) from e

    # Định dạng vision OpenAI-compatible: data URI base64. Cần model local có
    # thị giác (vd Qwen2.5-VL); model thuần text sẽ bỏ qua/lỗi khối ảnh.
    if image_b64:
        data_uri = f"data:{image_media_type or 'image/jpeg'};base64,{image_b64}"
        user_content: object = [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": user_prompt},
        ]
    else:
        user_content = user_prompt

    # Tắt reasoning cho model kiểu Qwen3 trừ khi bật rõ ràng. Truyền qua
    # chat_template_kwargs (llama.cpp với --jinja sẽ áp dụng vào chat template;
    # các server khác bỏ qua key lạ). Tắt thinking nhanh ~6.7× (xem Config).
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": config.local_enable_thinking}
    }
    # Prefix caching phía server (llama.cpp): tái dùng KV-cache của system prompt
    # (rubric) — giống nhau giữa mọi bài cùng đề trong batch nên prefill chỉ tính
    # 1 lần. Server không hỗ trợ key này sẽ bỏ qua. (vLLM bật bằng cờ server
    # --enable-prefix-caching, không qua field này.)
    if config.local_prefix_cache:
        extra_body["cache_prompt"] = True

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Quan sát được prefix caching: system prompt (rubric) là phần ổn định, đứng
    # đầu messages → server (llama.cpp) tái dùng KV-cache của nó giữa các bài cùng
    # đề. Log để xác nhận cache đang bật + tỉ trọng phần ổn định so với tổng prompt
    # (càng cao càng tiết kiệm prefill — sau khi cắt segments, phần này tăng mạnh).
    if config.local_prefix_cache:
        user_chars = (
            len(user_content)
            if isinstance(user_content, str)
            else sum(len(p.get("text", "")) for p in user_content if isinstance(p, dict))
        )
        logger.info(
            "Prefix cache ON (cache_prompt=true) | system_chars=%d | user_chars=%d",
            len(system_prompt),
            user_chars,
        )

    # Always log the messages being sent (sanitized: image base64 stripped,
    # long text truncated) so ta thấy đúng prompt model local nhận được.
    _log_messages(logger, "local", config.local_model, messages)

    # Log the full request payload being sent to the local API
    # (system prompt is already embedded in messages[0] for OpenAI-compatible)
    if config.log_prompts:
        _log_api_request(
            config, "local",
            model=config.local_model,
            base_url=config.local_base_url,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=0,
            extra_body=extra_body,
        )

    t0 = time.monotonic()
    response = client.chat.completions.create(
        model=config.local_model,
        max_tokens=config.max_tokens,
        temperature=0,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SpeakingResult",
                "schema": SpeakingResult.model_json_schema(),
                "strict": True,
            },
        },
        extra_body=extra_body,
    )
    latency = time.monotonic() - t0

    usage = response.usage
    logger.info(
        "Model local chấm xong | model=%s | base_url=%s | latency=%.2fs | "
        "prompt_tokens=%s completion_tokens=%s",
        config.local_model,
        config.local_base_url,
        latency,
        getattr(usage, "prompt_tokens", "?"),
        getattr(usage, "completion_tokens", "?"),
    )

    finish = response.choices[0].finish_reason
    content = response.choices[0].message.content
    if finish == "length":
        raise RuntimeError(
            f"Model local bị cắt vì chạm trần max_tokens={config.max_tokens} "
            f"(finish_reason=length) → JSON dở dang. Tăng TOEIC_MAX_TOKENS, "
            f"hoặc giảm độ dài nhận xét."
        )
    if not content:
        raise RuntimeError(
            f"Model local không trả về nội dung (finish_reason={finish})."
        )

    # Log AI response
    if config.log_prompts:
        _log_response(config, "local", content)

    try:
        return SpeakingResult.model_validate_json(content)
    except Exception as e:  # noqa: BLE001 - bọc lỗi parse cho rõ
        raise RuntimeError(
            f"Model local trả JSON không đúng schema SpeakingResult: {e}\n"
            f"Nội dung: {content[:500]}"
        ) from e


# ---- Prompt logging helpers -------------------------------------------------

# Độ dài tối đa của mỗi đoạn text khi log ra console (tránh ngập log).
_LOG_TEXT_PREVIEW = 4000


def _preview_content(content: object) -> object:
    """Rút gọn content của 1 message để log: bỏ base64 ảnh, cắt text dài."""
    if isinstance(content, str):
        if len(content) > _LOG_TEXT_PREVIEW:
            return content[:_LOG_TEXT_PREVIEW] + f"... [+{len(content) - _LOG_TEXT_PREVIEW} chars]"
        return content
    if isinstance(content, list):
        parts: list[object] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype in ("image_url", "image"):
                    parts.append({"type": ptype, "data": "[IMAGE REDACTED]"})
                elif ptype == "text":
                    parts.append({"type": "text", "text": _preview_content(part.get("text", ""))})
                else:
                    parts.append(part)
            else:
                parts.append(part)
        return parts
    return content


def _log_messages(
    log: logging.Logger,
    backend: str,
    model: str,
    messages: list[dict],
    *,
    system_prompt: str | None = None,
) -> None:
    """Log nội dung messages gửi lên LLM (sanitize ảnh + cắt text dài).

    Luôn chạy (không phụ thuộc config.log_prompts) để debug nhanh prompt thực tế
    model nhận. Ảnh base64 bị thay bằng [IMAGE REDACTED]; text > _LOG_TEXT_PREVIEW
    ký tự bị cắt.
    """
    preview = [
        {"role": m.get("role"), "content": _preview_content(m.get("content"))}
        for m in messages
    ]
    if system_prompt is not None:
        # Anthropic truyền system tách khỏi messages → log riêng cho đủ ngữ cảnh.
        preview.insert(0, {"role": "system", "content": _preview_content(system_prompt)})
    log.info(
        "LLM request | backend=%s | model=%s | messages=%s",
        backend,
        model,
        json.dumps(preview, ensure_ascii=False, indent=2),
    )


def _log_api_request(
    config: Config,
    backend: str,
    *,
    model: str,
    base_url: str | None,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    extra_body: dict | None,
    system_prompt: str | None = None,
) -> None:
    """Log the full API request payload (messages, params) to outputs/prompt_logs/."""
    _ensure_prompt_log_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    
    # Sanitize messages: strip image base64 data, keep structure
    sanitized_messages = []
    for msg in messages:
        sanitized_msg = dict(msg)
        content = sanitized_msg.get("content")
        if isinstance(content, list):
            # Vision message: strip image data
            sanitized_content = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type", "")
                    if ptype in ("image_url", "image"):
                        sanitized_content.append({"type": ptype, "[...]": "[IMAGE REDACTED]"})
                    elif ptype == "text":
                        text = part.get("text", "")
                        if len(text) > 5000:
                            text = text[:5000] + "... [truncated]"
                        sanitized_content.append({"type": "text", "text": text})
                    else:
                        sanitized_content.append(part)
                else:
                    sanitized_content.append(part)
            sanitized_msg["content"] = sanitized_content
        elif isinstance(content, str) and len(content) > 5000:
            sanitized_msg["content"] = content[:5000] + "... [truncated]"
        sanitized_messages.append(sanitized_msg)

    # Build a single payload for the request
    # Use a combined hash for the stem to avoid collisions
    content_hash = hash(json.dumps(messages, ensure_ascii=False)[:1000]) % 100000
    stem = f"{ts}_{backend}_req_{content_hash:05d}"
    log_file = _PROMPT_LOG_DIR / f"{stem}.json"
    
    payload = {
        "backend": backend,
        "model": model,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": extra_body,
    }
    # Include system prompt (Anthropic passes it as a separate parameter)
    if system_prompt:
        payload["system_prompt"] = system_prompt[:5000]
    payload["messages"] = sanitized_messages
    
    log_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("API request logged to %s", log_file)


def _log_response(config: Config, backend: str, response_json: str) -> None:
    """Log AI response JSON to outputs/prompt_logs/."""
    _ensure_prompt_log_dir()
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_{backend}_resp_{hash(response_json) % 100000:05d}"
    log_file = _PROMPT_LOG_DIR / f"{stem}.json"

    try:
        pretty = json.loads(response_json)
        content = json.dumps(pretty, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        content = response_json

    log_file.write_text(content, encoding="utf-8")
    logger.info("Response logged to %s", log_file)

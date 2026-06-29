"""Dựng system/user prompt + JSON schema cho LLM scoring.

Tách phần "soạn prompt" khỏi phần gọi backend: dễ test (test_prompt_building) và đọc.
"""

from __future__ import annotations

import json
import logging

from ..asr import Transcription
from ..config import resolve_language_name
from ..features import Features
from ..gating import GatingResult
from ..phoneme.models import PhonemeResult
from ..rubrics.base import Exam, QuestionType
from ..schema import SpeakingResult

logger = logging.getLogger("toeic.scoring")


def _build_system_prompt(qt: QuestionType, feedback_lang: str) -> str:
    criteria_lines = "\n".join(
        f"- {c.key} ({c.label}): {c.description}" for c in qt.criteria
    )
    criterion_keys = ", ".join(c.key for c in qt.criteria)
    # Chỉ thị cứng: model local nhỏ thỉnh thoảng bỏ sót một tiêu chí (vd
    # grammatical_range) → nêu rõ số lượng và yêu cầu xuất đúng-đủ, không lặp.
    criteria_count_rule = (
        f"You MUST output EXACTLY {len(qt.criteria)} criterion objects — one for "
        f"each key listed above ({criterion_keys}). Do NOT omit, merge, rename, or "
        f"duplicate any criterion; every key must appear exactly once."
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
{criteria_count_rule}

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

SUGGESTIONS (required for EVERY criterion):
- The `suggestions` list of every criterion MUST contain 2-4 concrete, \
actionable improvement tips the test-taker can practice. NEVER leave it empty — \
even a strong criterion has something to refine.
- Each suggestion must tie back to the specific weakness or evidence named in \
that criterion's justification. Avoid generic advice like "practice more".

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


#: Số lỗi phoneme tối đa nhúng vào prompt (đã sort theo severity high→low). Khớp cap
#: của PhonemeScore.to_dict() — model chỉ cần các lỗi nặng nhất, không cần toàn bộ.
_PROMPT_MAX_ERRORS: int = 20


def _compact_phoneme_data(phoneme_result: PhonemeResult) -> dict:
    """Bản gọn của phoneme_result để nhúng vào prompt LLM.

    Vì sao: PhonemeResult.to_dict() kèm `segments` thô (mỗi frame {phoneme,start,end,
    confidence,backend}) + `reference_phonemes` + `audio_path`; và score.to_dict() còn
    kèm `words` — phát âm CHI TIẾT từng từ × từng phoneme (tới MAX_WORDS_RETURNED từ),
    dữ liệu phục vụ UI kiểu ELSA. Riêng `words` chiếm ~95% kích thước phoneme_data
    (~136k ký tự cho 1 bài Part 2) và VÔ DỤNG với model text: nó đã có `errors[:20]`
    (kèm `word` của từng lỗi) + điểm tổng hợp để chấm. Nhồi `words` vào prompt ăn hết
    context window của model local → output JSON bị cắt (finish_reason=length).

    Vì thế ở đây dựng score gọn bằng ALLOWLIST (liệt kê tường minh field model dùng)
    thay vì to_dict()+pop(): nếu sau này PhonemeScore thêm field UI/diagnostic lớn,
    nó KHÔNG vô tình lọt vào prompt. Field model thực sự dùng (xem system prompt
    `_build_system_prompt`): overall_accuracy (tín hiệu mạnh) + các *_count (độ lớn lỗi)
    + avg_confidence + errors[:20] (đã sort severity, mỗi lỗi kèm `word`). KHÔNG gồm
    `words` per-word, cũng KHÔNG gồm penalty/L1 metadata nội bộ (raw_penalty,
    l1_adjustment_ratio…) — prompt không tham chiếu tới chúng.

    Lưu ý: chỉ ảnh hưởng prompt chấm điểm. UI/report vẫn dùng đường riêng
    (`_compact_phoneme_output` ở core.py) có đủ `words` (kèm start/end) để hiển thị.
    """
    score = phoneme_result.score
    compact_score = None
    if score is not None:
        compact_score = {
            "overall_accuracy": round(score.overall_accuracy, 4),
            "substitution_count": score.substitution_count,
            "deletion_count": score.deletion_count,
            "insertion_count": score.insertion_count,
            "reference_count": score.reference_count,
            "predicted_count": score.predicted_count,
            "avg_confidence": round(score.avg_confidence, 4),
            "errors": [e.to_dict() for e in score.errors[:_PROMPT_MAX_ERRORS]],
        }
    return {
        "backend_used": phoneme_result.backend_used,
        "backend_available": phoneme_result.backend_available,
        "warning": phoneme_result.warning,
        "score": compact_score,
    }


def _local_response_schema(qt: QuestionType) -> dict:
    """JSON schema gửi backend local, siết `criteria` đúng N tiêu chí của qt.

    Vì sao: schema gốc của SpeakingResult để `criteria` là array ĐỘ DÀI TỰ DO, nên
    grammar GBNF (llama.cpp dịch từ json_schema) chỉ ràng buộc hình dạng từng phần
    tử — KHÔNG ép phải đủ N tiêu chí. Model nhỏ/nén vì thế thỉnh thoảng bỏ sót một
    tiêu chí (vd grammatical_range) mà vẫn hợp lệ schema → hỏng cả bài chấm. Ở đây
    ta:
      - đặt minItems = maxItems = N để grammar ép đúng N phần tử;
      - giới hạn field `criterion` vào enum đúng tập key của qt để mỗi phần tử chỉ
        có thể là một tiêu chí hợp lệ.

    GIỚI HẠN (quan trọng): hai ràng buộc trên chỉ siết SỐ LƯỢNG và TẬP KEY hợp lệ,
    KHÔNG ràng buộc theo VỊ TRÍ → về lý thuyết model vẫn có thể lặp một key và bỏ
    key khác. Đây chỉ là biện pháp GIẢM XÁC SUẤT, không triệt để; `_validate_result`
    vẫn là lưới an toàn thiết yếu bắt trường hợp trùng/thiếu còn sót lại. Nếu build
    llama.cpp đang dùng dịch được `prefixItems` (JSON Schema 2020-12 tuple) sang
    GBNF, có thể nâng cấp: gán mỗi vị trí một `criterion` qua `const` để ép từng
    tiêu chí xuất hiện đúng một lần — ràng buộc theo vị trí, mạnh hơn enum. Chưa bật
    ở đây vì chưa xác minh build hiện tại hỗ trợ.

    Chỉ ảnh hưởng backend local. model_json_schema() trả dict MỚI mỗi lần gọi nên
    mutate ở đây không đụng schema dùng nơi khác.
    """
    schema = SpeakingResult.model_json_schema()
    keys = [c.key for c in qt.criteria]
    n = len(keys)
    crit = schema["properties"]["criteria"]
    crit["minItems"] = n
    crit["maxItems"] = n
    schema["$defs"]["CriterionScore"]["properties"]["criterion"]["enum"] = keys
    return schema


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

"""Lõi pipeline chấm điểm — dùng chung cho CLI ([main.py]) và API ([api.py]).

Tách khỏi main() để cùng một luồng ASR → features → gating → scoring → report
phục vụ được cả dòng lệnh lẫn HTTP, không phụ thuộc ngân hàng câu hỏi: đầu vào
(script tham chiếu / ảnh / thời lượng kỳ vọng) được truyền thẳng vào.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

from . import asr, features as features_mod, gating, report, scoring
from .config import Config
from .phoneme.analyzer import HybridPhonemeAnalyzer
from .phoneme.chunking import CHUNKING_STRATEGIES, compute_chunk_spans
from .phoneme.diagnostics import (
    DiagnosticsContext,
    TelemetryWriter,
    map_reference_words_to_indices,
    subtoken_window,
)
from .phoneme.ipa.profile import get_profile
from .phoneme.l1 import get_l1_profile
from .phoneme.models import PhonemeResult
from .phoneme.reliability import (
    RecognizerEvidence,
    SkipDecision,
    SkipReason,
    assess_asr_confidence,
    assess_reliability,
)
from .rubrics.base import QuestionType, exam_language, exam_score_field

logger = logging.getLogger("toeic.core")

# Định danh phiên cho telemetry (PR2): gom được các audio trong cùng lần chạy.
# Ưu tiên TOEIC_SESSION_ID (vd CI/batch đặt sẵn), không thì sinh 1 id mỗi process.
_RUN_SESSION_ID: str = os.getenv("TOEIC_SESSION_ID") or uuid.uuid4().hex[:12]

# Thông báo khi chỉ chấm phát âm vì thiếu đề bài (không có prompt/script/ảnh).
_PRONUNCIATION_ONLY_NOTICE = (
    "Chưa nhập đề/câu hỏi cho dạng câu này nên không thể chấm điểm tổng — "
    "chỉ chấm phát âm. Nhập đề bài để chấm đầy đủ."
)


def _compact_phoneme_output(phoneme_result: PhonemeResult | None) -> dict | None:
    """Bản gọn của phoneme analysis cho JSON output + UI.

    Chỉ giữ headline + lỗi đã gắn từ (KHÔNG kèm segments thô / reference_phonemes
    đầy đủ). Shape nested theo `score` để khớp model backend và frontend reader
    (`data.phoneme.score.errors`). None nếu không có score.
    """
    if phoneme_result is None or phoneme_result.score is None:
        return None
    score = phoneme_result.score.to_dict()
    return {
        "backend_used": phoneme_result.backend_used,
        "warning": phoneme_result.warning,
        "score": {
            "overall_accuracy": score["overall_accuracy"],
            "errors": score["errors"],
            "words": score["words"],
            "words_truncated": score["words_truncated"],
            "words_total": score["words_total"],
        },
    }


def grade_response(
    audio_path: str,
    config: Config,
    qt: QuestionType,
    *,
    prompt_text: str = "",
    reference_script: str | None = None,
    expected_duration_sec: float | None = None,
    image_b64: str | None = None,
    image_media_type: str | None = None,
    provided_info: str | None = None,
    asr_backend: str = "faster_whisper",
    asr_model: str | None = None,
    no_ai: bool = False,
    phoneme_analysis: bool | None = None,
    question_id: str = "adhoc",
    save: bool = True,
    accent: str = "default",
    lang: str | None = None,
    phoneme_strict: bool = False,
    asr_initial_prompt: str | None = None,
) -> dict[str, Any]:
    """Chạy toàn bộ pipeline cho 1 audio và trả về dict kết quả (build_output).

    - qt: dạng câu (quyết định tiêu chí + có dùng script tham chiếu không).
    - reference_script: text dùng cho Read Aloud (để so transcript ra WER/coverage).
    - image_b64 / image_media_type: ảnh đề bài cho Describe Picture (gửi LLM dạng vision).
    - provided_info: tài liệu cho sẵn (Q8-10) dạng text; vào payload chấm khi dạng câu
      có uses_provided_info.
    - expected_duration_sec: optional, vào features (reading_pace) + gating.
    - no_ai: chỉ chạy ASR + features, bỏ qua LLM.
    - asr_model: model ASR cho lần chấm này (vd "large-v3-turbo" cho practice,
      "large-v3" cho mock_test). None = dùng config.whisper_model chung.
    - phoneme_analysis: ép bật/tắt phoneme analysis (wav2vec) cho lần chấm này,
      bất kể config. None = theo config.phoneme_analysis_enabled. API dùng cờ này
      để gắn wav2vec theo mode: mock_test → True, practice → None (theo config),
      và True khi practice tự leo lên mock_test.
    - save: ghi JSON ra outputs/ (CLI cần; API có thể tắt).
    - accent: giọng tham chiếu phát âm ("default" | "gb" | "us"). CHỈ truyền xuống phoneme
      analyzer; "default" chấp nhận coda /r/ non-rhotic (Anh-Anh), không trừ điểm.
    - lang: ngôn ngữ NÓI đang chấm ("en" | "ko"...). None = suy từ qt.exam
      (exam_language). Quyết định ASR language + LangProfile (G2P/similarity).
      KHÁC config.feedback_lang (ngôn ngữ lời nhận xét).
    - phoneme_strict: chấm phoneme CHẶT (popup luyện 1 từ): tắt các lớp leniency
      thiết kế cho câu dài/nói tự do — L1 (nuốt phụ âm cuối + trung hoà sub
      conf thấp), coverage gate, drift cap, collapse gate. GIỮ các guard thuần
      về recognizer (noise gate, confidence knee, s-cluster, multiref, boundary
      refine) vì đó là nhiễu model, không phải leniency với người học.
      False (mặc định) = mọi đường chấm hiện hành không đổi bit-for-bit.
    - asr_initial_prompt: bias decoder Whisper (popup luyện 1 từ truyền chính từ
      đang luyện — clip 1 từ không ngữ cảnh Whisper rất hay nghe sai → reliability
      skip oan). None (mặc định) = ASR như cũ.
    """
    lang = lang or exam_language(qt.exam)
    # LangProfile: bộ hàm G2P/similarity/tokenizer theo ngôn ngữ đang chấm.
    # get_profile raise với mã lạ — chặn sớm thay vì chấm sai câm.
    lang_profile = get_profile(lang)
    phoneme_enabled = (
        config.phoneme_analysis_enabled
        if phoneme_analysis is None
        else phoneme_analysis
    )
    # Thiếu "đề bài" của dạng câu (vd IELTS Part 2 bỏ trống prompt, Describe
    # Picture không có ảnh) → KHÔNG chấm điểm tổng (LLM tự suy diễn độ liên quan
    # là không đáng tin), chỉ chấm phát âm. Bỏ qua khi no_ai (user chủ động chỉ
    # lấy ASR). qt.has_task_context là nguồn chân lý duy nhất cho quyết định này.
    task_context_missing = not no_ai and not qt.has_task_context(
        prompt=prompt_text,
        reference=reference_script,
        image=bool(image_b64),
        provided_info=provided_info,
    )
    # Khi chỉ chấm phát âm → ép bật phoneme để luôn có báo cáo (không overwrite
    # cờ đã bật; không chạy lại nếu trước đó đã bật theo mode/config).
    phoneme_enabled = phoneme_enabled or task_context_missing
    active_model = config.local_model if config.is_local else config.model
    logger.info(
        "Chấm | audio=%s | question=%s | type=%s | backend=%s | model=%s | no_ai=%s",
        audio_path,
        question_id,
        qt.key,
        config.backend,
        active_model,
        no_ai,
    )
    pipeline_started = time.perf_counter()
    step_timings_ms: dict[str, int] = {}

    # [1] ASR
    step_started = time.perf_counter()
    # language: trước đây không truyền → asr tự default "en" (khoá ngầm tiếng Anh).
    # Truyền tường minh theo lang — "en" cho toeic/ielts = hành vi cũ bit-for-bit.
    asr_run = asr.transcribe_with_backend(
        audio_path,
        backend=asr_backend,
        model_size=asr_model or config.whisper_model,
        device=config.whisper_device,
        language=lang,
        batch_size=config.whisper_batch_size,
        initial_prompt=asr_initial_prompt,
    )
    transcription = asr_run.transcription
    step_timings_ms["asr"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=asr | backend=%s | duration_ms=%d | words=%d | audio_sec=%.2f",
        question_id,
        asr_run.backend_used,
        step_timings_ms["asr"],
        transcription.word_count,
        transcription.duration,
    )

    # [2] Features
    step_started = time.perf_counter()
    feats = features_mod.extract_features(
        transcription,
        reference_script=reference_script,
        expected_duration_sec=expected_duration_sec,
    )
    step_timings_ms["features"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=features | duration_ms=%d | wpm=%.1f | pauses=%d",
        question_id,
        step_timings_ms["features"],
        feats.speech_rate_wpm,
        feats.pause_count,
    )

    # [2b] Phoneme analysis (optional — Phase 1: wav2vec)
    phoneme_result = None
    if phoneme_enabled and transcription.text.strip():
        step_started = time.perf_counter()
        # Bọc try/except: lỗi phoneme (vd wav2vec OOM/CPU nghẽn) KHÔNG được làm
        # hỏng cả request — nhất là khi pronunciation-only thì notice vẫn phải
        # trả về được thay vì 500.
        try:
            # lang=ko: (a) model acoustic riêng (config, mặc định dùng chung
            # xlsr-espeak); (b) ÉP TẮT các rule đặc thù tiếng Anh bất kể config —
            # homograph (CMUdict), s-cluster, connected-speech elision, coda-r
            # accent (accent="ko" → analyzer map accept_accent_variants=False).
            # L1 layer key theo cặp (l1, target): vi→en dùng flag/bảng hiện hành;
            # vi→ko (M5) flag RIÊNG default OFF + bảng src/phoneme/l1/vi_ko.py.
            _is_ko = lang == "ko"
            _l1_ko = _is_ko and config.phoneme_l1_ko_enabled
            # phoneme_strict (popup luyện 1 từ): các gate dưới đây tha lỗi THẬT
            # trên phát âm chủ động 1 từ (tín hiệu âm học sạch, mục đích là bắt
            # lỗi) nên tắt hết; guard nhiễu recognizer (noise gate/knee/s-cluster)
            # giữ nguyên — xem docstring.
            if phoneme_strict:
                _l1_ko = False
            phoneme_analyzer = HybridPhonemeAnalyzer(
                wav2vec_model=(
                    config.phoneme_wav2vec_model_ko
                    if _is_ko
                    else config.phoneme_wav2vec_model
                ),
                device=config.phoneme_device,
                max_words=config.phoneme_max_words,
                confidence_knee=config.phoneme_confidence_knee,
                l1_enabled=(
                    (config.phoneme_l1_enabled and not _is_ko) or _l1_ko
                ) and not phoneme_strict,
                l1_profile=get_l1_profile("vi", "ko") if _l1_ko else None,
                l1_min_confidence=config.phoneme_l1_min_confidence,
                low_conf_floor=config.phoneme_l1_low_conf_floor,
                recognizer_noise_sim=config.phoneme_recognizer_noise_sim,
                recognizer_noise_conf=config.phoneme_recognizer_noise_conf,
                recognizer_noise_conf_vowel=config.phoneme_recognizer_noise_conf_vowel,
                connected_speech_enabled=(
                    config.phoneme_connected_speech_enabled and not _is_ko
                ),
                coverage_gate_enabled=(
                    config.phoneme_coverage_gate_enabled and not phoneme_strict
                ),
                coverage_gate_cap=config.phoneme_coverage_gate_cap,
                coverage_gate_max_len=config.phoneme_coverage_gate_max_len,
                # whisperx word "probability" là alignment score (thang thấp hơn
                # logprob faster-whisper) → ngưỡng riêng, cùng pattern
                # phoneme_asr_conf_min_whisperx bên dưới.
                coverage_gate_min_asr_prob=(
                    config.phoneme_coverage_gate_min_asr_prob_whisperx
                    if asr_run.backend_used == "whisperx"
                    else config.phoneme_coverage_gate_min_asr_prob
                ),
                drift_cap_enabled=(
                    config.phoneme_drift_cap_enabled and not phoneme_strict
                ),
                drift_sub_cap=config.phoneme_drift_sub_cap,
                drift_window_pad=config.phoneme_drift_window_pad,
                deletion_evidence_enabled=config.phoneme_deletion_evidence_enabled,
                homograph_selection_enabled=(
                    config.phoneme_homograph_multiref and not _is_ko
                ),
                accent_dualref_enabled=(
                    config.phoneme_accent_dualref and not _is_ko
                ),
                boundary_refine_enabled=config.phoneme_boundary_refine_enabled,
                s_cluster_enabled=config.phoneme_s_cluster_enabled and not _is_ko,
                collapse_gate_enabled=(
                    config.phoneme_collapse_gate_enabled and not phoneme_strict
                ),
                profile=lang_profile,
                # TOEIC_PHONEME_DEVICES (vd "cuda:0,cuda:1"): chia chunk phoneme
                # song song lên nhiều GPU. Rỗng → None = tuần tự như cũ.
                devices=(
                    [d.strip() for d in config.phoneme_devices.split(",") if d.strip()]
                    or None
                ),
            )
            # Read Aloud có script mẫu → so phát âm với script. Câu nói tự do (IELTS
            # Speaking, Describe Picture, Respond...) không có script → fallback về
            # transcript ASR: đo phát âm của chính những từ thí sinh đã nói (kiểu ELSA).
            #
            # Tầng reliability (TRÊN scorer) — quyết định từ nào KHÔNG đáng tin để chấm,
            # keyed theo CHỈ SỐ TỪ chuẩn (occurrence) nên "the" lặp nhiều lần không bị
            # skip oan. reference_words dựng từ cùng hàm mà analyzer dùng (deterministic)
            # → chỉ số khớp spans của scorer. Hai nhánh:
            #   - Có script: so transcript recognizer với script (cross-source) —
            #     assess_reliability (vd Son Tinh→Andy).
            #   - Free-speech (reference == transcript): không có nguồn chéo → gate bằng
            #     (a) word probability của chính Whisper (assess_asr_confidence) và
            #     (b) từ OOV lấy IPA từ eSpeak (cả transcript lẫn G2P đều là đoán).
            skips: dict = {}
            word_windows = None
            word_windows_locked = None
            # reference_text mà scorer dùng (KHỚP analyzer): có script → script; không
            # → transcript (free-speech, đo phát âm của chính từ thí sinh đã nói).
            phoneme_reference_text = reference_script or transcription.text
            ref_spans: list = []
            if phoneme_reference_text.strip():
                _ph, ref_spans, _st, _ds = lang_profile.text_to_ipa_with_spans(
                    phoneme_reference_text
                )
            reference_words = [s.word for s in ref_spans]
            if reference_script and reference_words:
                evidence = RecognizerEvidence.from_transcript(
                    transcription.text, token_re=lang_profile.transcript_token_re
                )
                skips = dict(assess_reliability(
                    reference_words, evidence, skip_ratio=config.phoneme_skip_ratio
                ))
            elif reference_words:
                min_prob = (
                    config.phoneme_asr_conf_min_whisperx
                    if asr_run.backend_used == "whisperx"
                    else config.phoneme_asr_conf_min
                )
                skips = dict(assess_asr_confidence(
                    reference_words,
                    [(w.text, w.probability) for w in transcription.words],
                    min_probability=min_prob,
                    transcript_text=phoneme_reference_text,
                    token_re=lang_profile.word_token_re,
                ))
                # Từ OOV (IPA từ eSpeak) — setdefault để không ghi đè lý do ASR.
                for k, s in enumerate(ref_spans):
                    if s.source == "espeak":
                        skips.setdefault(
                            k, SkipDecision(k, SkipReason.OOV_ESPEAK)
                        )
            # Map TỪ THAM CHIẾU → Whisper word đã khớp (MỘT alignment difflib, cùng kỹ
            # thuật Recognition Reliability), rồi đọc 2 field: (start,end) → word_windows
            # cho (a) UI phát lại từng từ, (b) telemetry drift (PR3-0), (c) evidence cho
            # coverage/drift gate khi flags bật; probability → word_probs làm guard cho
            # coverage gate (không coi transcript là ground truth tuyệt đối).
            word_probs = None
            if reference_words and transcription.words:
                _widx = map_reference_words_to_indices(
                    reference_words, [w.text for w in transcription.words]
                )
                # Cửa sổ qua subtoken_window: token alphanumeric của Whisper ("9am")
                # bị tokenizer reference rơi phần số → ref "am" map vào NGUYÊN token,
                # cửa sổ thô phát cả "nine"; cắt còn đúng phần ref theo tỉ lệ ký tự.
                # word_windows cũng feed telemetry drift + evidence coverage/drift
                # gate (flags default OFF) — cửa sổ cắt chính xác hơn cho từ bị token
                # gộp, chấp nhận telemetry đổi cho các case này. word_probs GIỮ NGUYÊN
                # probability cả token. Từ bị cắt → word_windows_locked: playback
                # KHÔNG siết theo seg_times (DTW attribution nhiễm vì âm phần số bị
                # rơi không có trong reference — xem _merge_playback_windows).
                word_windows = {}
                word_windows_locked = set()
                for i, j in _widx.items():
                    w = transcription.words[j]
                    raw = (float(w.start), float(w.end))
                    win = subtoken_window(reference_words[i], w.text, *raw)
                    word_windows[i] = win
                    if win != raw:
                        word_windows_locked.add(i)
                word_probs = {
                    i: float(getattr(transcription.words[j], "probability", 0.0) or 0.0)
                    for i, j in _widx.items()
                }
            # Chunk audio TRƯỚC wav2vec (fix IPA "lem" trên audio dài — model suy
            # giảm khi nhận cả bài trong 1 forward pass). Chunk theo Whisper word
            # timestamps (khoảng lặng/câu); CHỈ đổi đầu vào segments — scoring
            # bất biến. "off"/không có words → None = single-pass như cũ. Strategy
            # lạ (env gõ sai) → cảnh báo + fallback single-pass, không chết request.
            chunk_spans = None
            _chunk_strategy = config.phoneme_chunking_strategy
            if _chunk_strategy != "off" and transcription.words:
                if _chunk_strategy in CHUNKING_STRATEGIES:
                    chunk_spans = compute_chunk_spans(
                        [(w.text, float(w.start), float(w.end))
                         for w in transcription.words],
                        float(transcription.duration or 0.0),
                        strategy=_chunk_strategy,
                        max_chunk_sec=config.phoneme_chunk_max_sec,
                        min_pause_sec=config.phoneme_chunk_min_pause_sec,
                        pad_sec=config.phoneme_chunk_pad_sec,
                    ) or None
                    logger.info(
                        "Phoneme | question=%s | chunking=%s → %d chunks",
                        question_id, _chunk_strategy, len(chunk_spans or []),
                    )
                else:
                    logger.warning(
                        "Phoneme | question=%s | TOEIC_PHONEME_CHUNKING=%r không hợp lệ "
                        "(hợp lệ: off, %s) — fallback single-pass.",
                        question_id, _chunk_strategy,
                        ", ".join(sorted(CHUNKING_STRATEGIES)),
                    )
            # Telemetry (PR2): chỉ bật khi config bật — sink ghi JSONL per-word, KHÔNG
            # ảnh hưởng điểm. Tắt → sink None → scorer không tính diagnostics.
            diagnostics_sink = None
            if config.phoneme_telemetry_enabled:
                ctx = DiagnosticsContext(
                    session_id=_RUN_SESSION_ID,
                    audio_id=Path(audio_path).name,
                    utterance_id=question_id,
                )
                diagnostics_sink = partial(
                    TelemetryWriter(config.phoneme_telemetry_path).emit, ctx
                )
            phoneme_result = phoneme_analyzer.analyze(
                audio_path,
                reference_text=reference_script or transcription.text,
                skips=skips,
                diagnostics_sink=diagnostics_sink,
                word_windows=word_windows,
                word_windows_locked=word_windows_locked,
                word_probs=word_probs,
                # "ko" không phải mode accent EN nào → accept_accent_variants=False
                # (coda-r non-rhotic là chuyện tiếng Anh).
                accent="ko" if _is_ko else accent,
                chunk_spans=chunk_spans,
            )
        except Exception:  # noqa: BLE001 - phoneme là phụ trợ, lỗi không fatal
            logger.exception("Phoneme | question=%s | analyzer crashed", question_id)
            phoneme_result = None
        step_timings_ms["phoneme"] = int((time.perf_counter() - step_started) * 1000)
        if phoneme_result is None:
            pass  # đã log exception ở trên
        elif phoneme_result.score:
            logger.info(
                "Phoneme | question=%s | accuracy=%.2f | substitutions=%d | deletions=%d | insertions=%d",
                question_id,
                phoneme_result.score.overall_accuracy,
                phoneme_result.score.substitution_count,
                phoneme_result.score.deletion_count,
                phoneme_result.score.insertion_count,
            )
        else:
            logger.info(
                "Phoneme | question=%s | skipped (%s)",
                question_id,
                phoneme_result.warning or "no reference",
            )
    else:
        step_timings_ms["phoneme"] = 0
        if not phoneme_enabled:
            logger.info("Phoneme | question=%s | disabled (mode/config)", question_id)

    # [3] Gating
    step_started = time.perf_counter()
    gate = gating.evaluate(
        transcription,
        feats,
        expected_duration_sec=expected_duration_sec,
        question_type=qt,
    )
    step_timings_ms["gating"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=gating | duration_ms=%d | skip_ai=%s | floor=%s",
        question_id,
        step_timings_ms["gating"],
        gate.should_skip_ai,
        gate.task_completion_floor,
    )
    for reason in gate.reasons:
        logger.info("Gating: %s", reason)

    # [4] Scoring (trừ khi no_ai hoặc audio rỗng)
    scores_dict = None
    scoring_status = "skipped"
    scoring_meta: dict = {}
    if no_ai:
        logger.info("Bỏ qua chấm điểm (no_ai).")
    elif gate.should_skip_ai:
        logger.warning("Audio rỗng/không nhận ra lời — không gọi LLM.")
    elif task_context_missing:
        scoring_status = "pronunciation_only"
        logger.info(
            "Skip AI scoring | reason=missing_task_context | question=%s | phoneme_only=True",
            qt.key,
        )
    else:
        step_started = time.perf_counter()
        result, scoring_meta = scoring.score(
            config=config,
            qt=qt,
            prompt_text=prompt_text,
            reference_script=reference_script,
            transcription=transcription,
            features=feats,
            gating=gate,
            phoneme_result=phoneme_result,
            image_b64=image_b64,
            image_media_type=image_media_type,
            provided_info=provided_info,
        )
        step_timings_ms["scoring"] = int((time.perf_counter() - step_started) * 1000)
        scores_dict = result.model_dump(mode="json")
        scoring_status = "completed"
        _score_field = exam_score_field(qt.exam)
        logger.info(
            "Timing | question=%s | step=scoring | duration_ms=%d | exam=%s | "
            "score=%s | backend=%s | fallback=%s",
            question_id,
            step_timings_ms["scoring"],
            qt.exam,
            scores_dict.get(_score_field),
            scoring_meta.get("backend_used"),
            scoring_meta.get("fallback_reason"),
        )

    if scoring_status != "completed":
        step_timings_ms["scoring"] = 0
        if no_ai:
            _status_label = "no_ai"
        elif scoring_status == "pronunciation_only":
            _status_label = "pronunciation_only"
        else:
            _status_label = "skipped_by_gating"
        logger.info(
            "Timing | question=%s | step=scoring | duration_ms=0 | status=%s",
            question_id,
            _status_label,
        )

    # [5] Report
    step_started = time.perf_counter()
    output = report.build_output(
        audio_path=audio_path,
        question_id=question_id,
        question_type=qt.key,
        exam=qt.exam,
        transcript=transcription.text,
        features=feats.to_dict(),
        scores=scores_dict,
        phoneme=_compact_phoneme_output(phoneme_result),
        pronunciation_only=task_context_missing,
        reason="missing_task_context" if task_context_missing else None,
        notice=_PRONUNCIATION_ONLY_NOTICE if task_context_missing else None,
        telemetry={
            "asr_backend_used": asr_run.backend_used,
            "transcription_time_ms": asr_run.elapsed_ms,
            "step_timings_ms": step_timings_ms,
            # Backend LLM THẬT SỰ chấm bài này ("local_fallback" = OpenRouter
            # hỏng, local cứu) — theo dõi fallback rate sau khi ship openrouter.
            "scoring_backend_used": scoring_meta.get("backend_used"),
            "scoring_model": scoring_meta.get("model"),
            "scoring_fallback_reason": scoring_meta.get("fallback_reason"),
        },
    )
    step_timings_ms["report_build"] = int((time.perf_counter() - step_started) * 1000)
    logger.info(
        "Timing | question=%s | step=report_build | duration_ms=%d",
        question_id,
        step_timings_ms["report_build"],
    )
    if save:
        save_started = time.perf_counter()
        stem = f"{Path(audio_path).stem}__{question_id}"
        out_path = report.save_json(output, stem=stem)
        step_timings_ms["report_save"] = int((time.perf_counter() - save_started) * 1000)
        logger.info("Đã lưu kết quả: %s", out_path)
        logger.info(
            "Timing | question=%s | step=report_save | duration_ms=%d",
            question_id,
            step_timings_ms["report_save"],
        )
    else:
        step_timings_ms["report_save"] = 0

    total_ms = int((time.perf_counter() - pipeline_started) * 1000)
    output["telemetry"]["step_timings_ms"] = step_timings_ms
    output["telemetry"]["pipeline_total_ms"] = total_ms
    logger.info(
        "Timing | question=%s | total_ms=%d | asr=%d | features=%d | phoneme=%d | gating=%d | scoring=%d | report_build=%d | report_save=%d",
        question_id,
        total_ms,
        step_timings_ms["asr"],
        step_timings_ms["features"],
        step_timings_ms.get("phoneme", 0),
        step_timings_ms["gating"],
        step_timings_ms["scoring"],
        step_timings_ms["report_build"],
        step_timings_ms["report_save"],
    )
    return output

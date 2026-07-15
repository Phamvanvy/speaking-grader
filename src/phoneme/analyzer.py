"""HybridPhonemeAnalyzer — điểm vào chính cho phoneme analysis.

Kết nối wav2vec backend + IPA reference + scoring thành 1 API đơn giản:

    analyzer = HybridPhonemeAnalyzer()
    result = analyzer.analyze(audio_path, reference_text="The quick brown fox")
    print(result.score.overall_accuracy)  # 0.85
    print(result.score.errors)  # [PhonemeError(...), ...]

Architecture (hybrid-ready):
  - Phase 1: wav2vec only (current implementation)
  - Phase 2: thêm MFA alignment backend → kết hợp timestamps (MFA) + confidence (wav2vec)
  - Graceful degradation: nếu wav2vec unavailable → trả về empty result + warning
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

from .diagnostics import DRIFT_WINDOW_PAD_SEC, WordDiagnostic
from .ipa.profile import LangProfile, get_profile
from .models import PhonemeResult
from .reliability import SkipDecision
from .scoring import (
    MAX_WORDS_RETURNED,
    PHONEME_CONFIDENCE_KNEE,
    PHONEME_L1_MIN_CONFIDENCE,
    PHONEME_LOW_CONF_FLOOR,
    PHONEME_RECOGNIZER_NOISE_CONF,
    PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
    PHONEME_RECOGNIZER_NOISE_SIM,
    compute_phoneme_score,
)
from .scoring.constants import (
    PHONEME_COVERAGE_GATE_CAP,
    PHONEME_COVERAGE_GATE_MAX_LEN,
    PHONEME_COVERAGE_GATE_MIN_ASR_PROB,
    PHONEME_DRIFT_SUB_CAP,
)
from .wav2vec_backend import (
    DEFAULT_WAV2VEC_MODEL,
    MIN_PHONEME_DURATION_SEC,
    PHONEME_CONFIDENCE_THRESHOLD,
    Wav2VecPhonemePredictor,
)

logger = logging.getLogger("toeic.phoneme.analyzer")


class HybridPhonemeAnalyzer:
    """Phoneme analyzer — wav2vec backend (Phase 1), MFA-ready (Phase 2).

    Args:
        wav2vec_model: model ID trên HuggingFace (default: facebook/wav2vec2-xlsr-53-espeak-cv-ft)
        device: "cpu" | "cuda" | "auto"
        min_phoneme_duration: segment ngắn nhất được giữ (giây)
        confidence_threshold: probability threshold cho phoneme
        enable_phoneme_analysis: bật/tắt phoneme analysis (mặc định: bật)
        max_words: số từ tối đa trả về trong word-detail (cắt theo ranh giới từ)

    Usage:
        # Basic
        analyzer = HybridPhonemeAnalyzer()
        result = analyzer.analyze("audio.wav", reference_text="Hello world")

        # Custom config
        analyzer = HybridPhonemeAnalyzer(
            device="cuda",
            confidence_threshold=0.15,
        )

        # Without reference (chỉ predict phonemes)
        result = analyzer.analyze("audio.wav")

        # Check backend availability
        if not analyzer.wav2vec_available:
            print("wav2vec not available — phoneme analysis skipped")
    """

    def __init__(
        self,
        wav2vec_model: str = DEFAULT_WAV2VEC_MODEL,
        device: str = "cpu",
        min_phoneme_duration: float = MIN_PHONEME_DURATION_SEC,
        confidence_threshold: float = PHONEME_CONFIDENCE_THRESHOLD,
        enable_phoneme_analysis: bool = True,
        max_words: int = MAX_WORDS_RETURNED,
        confidence_knee: float = PHONEME_CONFIDENCE_KNEE,
        l1_enabled: bool = False,
        l1_min_confidence: float = PHONEME_L1_MIN_CONFIDENCE,
        low_conf_floor: float = PHONEME_LOW_CONF_FLOOR,
        recognizer_noise_sim: float = PHONEME_RECOGNIZER_NOISE_SIM,
        recognizer_noise_conf: float = PHONEME_RECOGNIZER_NOISE_CONF,
        recognizer_noise_conf_vowel: float = PHONEME_RECOGNIZER_NOISE_CONF_VOWEL,
        connected_speech_enabled: bool = True,
        coverage_gate_enabled: bool = False,
        coverage_gate_cap: float = PHONEME_COVERAGE_GATE_CAP,
        coverage_gate_max_len: int = PHONEME_COVERAGE_GATE_MAX_LEN,
        coverage_gate_min_asr_prob: float = PHONEME_COVERAGE_GATE_MIN_ASR_PROB,
        drift_cap_enabled: bool = False,
        drift_sub_cap: float = PHONEME_DRIFT_SUB_CAP,
        drift_window_pad: float = DRIFT_WINDOW_PAD_SEC,
        deletion_evidence_enabled: bool = True,
        homograph_selection_enabled: bool = False,
        boundary_refine_enabled: bool = False,
        s_cluster_enabled: bool = False,
        collapse_gate_enabled: bool = False,
        profile: LangProfile | None = None,
    ):
        self.enable_phoneme_analysis = enable_phoneme_analysis
        # LangProfile — bộ hàm G2P/similarity theo NGÔN NGỮ ĐANG CHẤM (không phải
        # feedback_lang). None = tiếng Anh, wrap đúng các hàm cũ (bit-for-bit).
        self._profile = profile or get_profile("en")
        self._max_words = max_words
        self._confidence_knee = confidence_knee
        self._l1_enabled = l1_enabled
        self._l1_min_confidence = l1_min_confidence
        self._low_conf_floor = low_conf_floor
        self._recognizer_noise_sim = recognizer_noise_sim
        self._recognizer_noise_conf = recognizer_noise_conf
        self._recognizer_noise_conf_vowel = recognizer_noise_conf_vowel
        self._connected_speech_enabled = connected_speech_enabled
        self._coverage_gate_enabled = coverage_gate_enabled
        self._coverage_gate_cap = coverage_gate_cap
        self._coverage_gate_max_len = coverage_gate_max_len
        self._coverage_gate_min_asr_prob = coverage_gate_min_asr_prob
        self._drift_cap_enabled = drift_cap_enabled
        self._drift_sub_cap = drift_sub_cap
        self._drift_window_pad = drift_window_pad
        # Deletion-evidence probe (SHADOW): giữ frame posteriors của wav2vec để đo
        # bằng chứng âm học cho mỗi âm bị thiếu — CHỈ telemetry, không đổi điểm.
        # Tắt (false) để tiết kiệm RAM (~5MB/60s audio trong lúc chấm).
        self._deletion_evidence_enabled = deletion_evidence_enabled
        # Multi-reference homograph: chọn lại entry CMUdict khớp acoustic nhất cho
        # từ đa-entry (xem scoring/homograph.py). Default OFF = bit-for-bit như cũ.
        self._homograph_selection_enabled = homograph_selection_enabled
        # Boundary refinement: sửa segment bị DTW gán nhầm sang từ kề trên path
        # trước khi chấm (xem scoring/alignment.py). Default OFF = bit-for-bit.
        self._boundary_refine_enabled = boundary_refine_enabled
        # S-cluster leniency: /p t k/ sau /s/ đầu từ không bật hơi — recognizer hay
        # gán nhầm voicing/chỗ cấu âm (xem scoring/alignment._is_s_cluster_stop).
        # Default OFF = bit-for-bit như cũ.
        self._s_cluster_enabled = s_cluster_enabled
        # Recognizer-collapse gate: cap del/sub bị wav2vec CTC blank-collapse (âm có
        # mass posterior nhưng argmax=<pad>) — mở rộng coverage gate cho collapse từng
        # phần (xem scoring/word_details._apply_recognizer_collapse_gate). Default OFF.
        self._collapse_gate_enabled = collapse_gate_enabled
        self._wav2vec = Wav2VecPhonemePredictor(
            model_id=wav2vec_model,
            device=device,
            min_phoneme_duration=min_phoneme_duration,
            confidence_threshold=confidence_threshold,
        )
        # MFA backend stub — sẽ implement ở Phase 2
        self._mfa_available: bool = False

    @property
    def wav2vec_available(self) -> bool:
        """Check wav2vec backend có sẵn sàng không."""
        return self._wav2vec.is_available

    @property
    def mfa_available(self) -> bool:
        """Check MFA backend có sẵn sàng không (Phase 2)."""
        return self._mfa_available

    def analyze(
        self,
        audio_path: str,
        reference_text: str | None = None,
        skips: Mapping[int, SkipDecision] | None = None,
        diagnostics_sink: Callable[[list[WordDiagnostic]], None] | None = None,
        word_windows: Mapping[int, tuple[float, float]] | None = None,
        word_windows_locked: Collection[int] | None = None,
        word_probs: Mapping[int, float] | None = None,
        accent: str = "default",
        chunk_spans: list[tuple[float, float]] | None = None,
    ) -> PhonemeResult:
        """Phân tích phonemes trong audio, optional so với reference text.

        Args:
            audio_path: đường dẫn file audio (.wav, .mp3, .m4a, ...)
            reference_text: text tham chiếu (vd script đọc aloud) → chuyển thành
                IPA sequence để so sánh. None = chỉ predict, không scoring.
            skips: quyết định bỏ qua từ Recognition Reliability (tầng TRÊN), keyed
                theo chỉ số từ tham chiếu chuẩn (xem compute_phoneme_score). None =
                chấm hết. Analyzer chỉ truyền xuống — KHÔNG tự tính reliability.
            diagnostics_sink: optional sink nhận WordDiagnostic để ghi telemetry (PR2);
                chỉ truyền xuống scorer, KHÔNG ảnh hưởng điểm.
            word_windows: optional (PR3-0) — cửa sổ thời gian Whisper theo chỉ số từ chuẩn,
                cho telemetry drift-vs-hallucination + evidence cho coverage gate/drift cap
                (khi flags bật — xem compute_phoneme_score); flags mặc định OFF thì KHÔNG
                ảnh hưởng điểm.
            word_windows_locked: optional — chỉ số từ có cửa sổ đã CẮT sub-token (token
                alphanumeric "9am" → ref "am", xem diagnostics.subtoken_window); playback
                bỏ qua siết seg_times cho các từ này. Chỉ truyền xuống scorer.
            word_probs: optional — Whisper word probability theo chỉ số từ chuẩn (cùng
                nguồn word_windows); guard cho coverage gate, chỉ truyền xuống scorer.
            accent: "default" | "gb" | "us" (giọng tham chiếu phát âm từ UI). CHỈ "default" bật
                accept_accent_variants (chấp nhận coda /r/ non-rhotic — xem compute_phoneme_score).
                "gb"/"us" giữ nguyên cách chấm cũ (chuẩn Mỹ). Map mode-name → bool Ở ĐÂY để scorer
                không phụ thuộc tên mode của frontend.
            chunk_spans: optional — danh sách (start, end) giây từ
                chunking.compute_chunk_spans (caller tính từ Whisper word timestamps).
                None (mặc định) = wav2vec single-pass như cũ. Có spans = predict theo
                từng chunk rồi ghép (fix IPA "lem" trên audio dài) — CHỈ đổi đầu vào
                segments/posteriors, KHÔNG đổi scoring. Analyzer chỉ truyền xuống —
                KHÔNG tự tính chunk (không biết Whisper).

        Returns:
            PhonemeResult với segments, reference_phonemes, score, warning.
        """
        # If phoneme analysis disabled, return empty (check before file I/O)
        if not self.enable_phoneme_analysis:
            return PhonemeResult(
                audio_path=audio_path,
                segments=[],
                reference_phonemes=[],
                score=None,
                backend_used="disabled",
                backend_available=False,
                warning="Phoneme analysis is disabled.",
            )

        # Check audio exists
        if not Path(audio_path).exists():
            return PhonemeResult(
                audio_path=audio_path,
                segments=[],
                reference_phonemes=[],
                score=None,
                backend_used="none",
                backend_available=False,
                warning=f"Audio file không tồn tại: {audio_path}",
            )

        # Build reference phoneme sequence + word spans (để map lỗi → từ)
        if reference_text:
            reference_phonemes, reference_spans, reference_stress, reference_display_stress = (
                self._profile.text_to_ipa_with_spans(reference_text)
            )
        else:
            reference_phonemes, reference_spans, reference_stress = [], [], []
            reference_display_stress = []

        # ── Phase 1: wav2vec only ─────────────────────────────────────────
        # Probe bật → giữ thêm frame posteriors (không tốn forward pass nào thêm).
        if self._deletion_evidence_enabled:
            segments, warning, posteriors = self._wav2vec.predict_with_posteriors(
                audio_path, chunk_spans=chunk_spans
            )
        else:
            segments, warning = self._wav2vec.predict(
                audio_path, chunk_spans=chunk_spans
            )
            posteriors = None

        if warning and not segments:
            # wav2vec unavailable — return empty with warning
            return PhonemeResult(
                audio_path=audio_path,
                segments=[],
                reference_phonemes=reference_phonemes,
                score=None,
                backend_used="none",
                backend_available=False,
                warning=warning,
            )

        # ── Phase 2: MFA alignment (stub — sẽ implement sau) ──────────────
        # if self._mfa_available:
        #     mfa_segments = self._mfa.align(audio_path, reference_text)
        #     segments = _merge_wav2vec_mfa(segments, mfa_segments)
        #     backend_used = "hybrid"
        # else:
        backend_used = "wav2vec"

        # ── Scoring ────────────────────────────────────────────────────────
        # Map mode-name accent → bool TẠI BIÊN analyzer; scorer chỉ thấy bool (decouple UI).
        accept_accent_variants = accent == "default"
        score = None
        if reference_phonemes and segments:
            score = compute_phoneme_score(
                segments, reference_phonemes, reference_spans, reference_stress,
                reference_display_stress=reference_display_stress,
                max_words=self._max_words,
                skips=skips,
                confidence_knee=self._confidence_knee,
                diagnostics_sink=diagnostics_sink,
                word_windows=word_windows,
                word_windows_locked=word_windows_locked,
                l1_enabled=self._l1_enabled,
                l1_min_confidence=self._l1_min_confidence,
                low_conf_floor=self._low_conf_floor,
                recognizer_noise_sim=self._recognizer_noise_sim,
                recognizer_noise_conf=self._recognizer_noise_conf,
                recognizer_noise_conf_vowel=self._recognizer_noise_conf_vowel,
                accept_accent_variants=accept_accent_variants,
                connected_speech_enabled=self._connected_speech_enabled,
                word_probs=word_probs,
                coverage_gate_enabled=self._coverage_gate_enabled,
                coverage_gate_cap=self._coverage_gate_cap,
                coverage_gate_max_len=self._coverage_gate_max_len,
                coverage_gate_min_asr_prob=self._coverage_gate_min_asr_prob,
                drift_cap_enabled=self._drift_cap_enabled,
                drift_sub_cap=self._drift_sub_cap,
                drift_window_pad=self._drift_window_pad,
                posteriors=posteriors,
                homograph_selection_enabled=self._homograph_selection_enabled,
                boundary_refine_enabled=self._boundary_refine_enabled,
                s_cluster_enabled=self._s_cluster_enabled,
                collapse_gate_enabled=self._collapse_gate_enabled,
                profile=self._profile,
            )

        logger.info(
            "Phoneme analysis complete: %s | backend=%s | segments=%d | ref=%d | score=%s",
            audio_path,
            backend_used,
            len(segments),
            len(reference_phonemes),
            score.overall_accuracy if score else "N/A",
        )

        return PhonemeResult(
            audio_path=audio_path,
            segments=segments,
            reference_phonemes=reference_phonemes,
            score=score,
            backend_used=backend_used,
            backend_available=True,
            warning=warning,
        )

    def predict_segments(self, audio_path: str) -> PhonemeResult:
        """Predict phoneme segments mà không cần reference (no scoring).

        Useful cho exploratory analysis hoặc khi không có reference script.
        """
        return self.analyze(audio_path, reference_text=None)
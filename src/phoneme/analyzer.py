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
from pathlib import Path

from .ipa import text_to_ipa_sequence
from .models import PhonemeResult
from .scoring import compute_phoneme_score
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
        wav2vec_model: model ID trên HuggingFace (default: facebook/wav2vec2-lg-960h)
        device: "cpu" | "cuda" | "auto"
        min_phoneme_duration: segment ngắn nhất được giữ (giây)
        confidence_threshold: probability threshold cho phoneme
        enable_phoneme_analysis: bật/tắt phoneme analysis (mặc định: bật)

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
    ):
        self.enable_phoneme_analysis = enable_phoneme_analysis
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
    ) -> PhonemeResult:
        """Phân tích phonemes trong audio, optional so với reference text.

        Args:
            audio_path: đường dẫn file audio (.wav, .mp3, .m4a, ...)
            reference_text: text tham chiếu (vd script đọc aloud) → chuyển thành
                IPA sequence để so sánh. None = chỉ predict, không scoring.

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

        # Build reference phoneme sequence
        reference_phonemes = (
            text_to_ipa_sequence(reference_text) if reference_text else []
        )

        # ── Phase 1: wav2vec only ─────────────────────────────────────────
        segments, warning = self._wav2vec.predict(audio_path)

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
        score = None
        if reference_phonemes and segments:
            score = compute_phoneme_score(segments, reference_phonemes)

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
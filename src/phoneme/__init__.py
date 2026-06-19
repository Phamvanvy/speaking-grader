"""Phoneme analysis module — wav2vec 2.0 backend (Phase 1), hybrid-ready for MFA (Phase 2).

Architecture:
  - analyzer.py: public API — HybridPhonemeAnalyzer (single entry point)
  - wav2vec_backend.py: wav2vec 2.0 / HuBERT phoneme predictor
  - ipa.py: IPA phoneme set, mapping utilities, English word→IPA dictionary
  - scoring.py: phoneme-level scoring (substitution, deletion, insertion, confidence)
  - models.py: data classes for phoneme results

Design principles:
  1. Backend-agnostic — wav2vec is Phase 1, MFA can be added as Phase 2
  2. Graceful degradation — if wav2vec unavailable, return empty results, don't crash
  3. Hybrid-ready — architecture supports combining alignment + confidence scores
"""

from .analyzer import HybridPhonemeAnalyzer
from .models import (
    PhonemeSegment,
    PhonemeResult,
    PhonemeError,
    PhonemeScore,
)

__all__ = [
    "HybridPhonemeAnalyzer",
    "PhonemeSegment",
    "PhonemeResult",
    "PhonemeError",
    "PhonemeScore",
]
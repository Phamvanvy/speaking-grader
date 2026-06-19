"""wav2vec 2.0 phoneme prediction backend.

Dùng model wav2vec 2.0 (train trên LibriSpeech với phoneme labels) để dự đoán
phoneme probabilities cho mỗi frame audio, rồi merge các frame liên tiếp
thành phoneme segments có timestamps.

Architecture:
  - Wav2VecPhonemePredictor: class chính, lazy-load model, cache trong process
  - predict_phonemes(): audio path → list[PhonemeSegment]
  - _frames_to_segments(): merge frame-level predictions → phoneme segments

Model: facebook/wav2vec2-lg-960h (960-hour, phoneme-trained)
  - Size: ~680MB
  - Output: frame-level logits over ~41 phoneme classes ( LibriSpeech ARPAbet set)
  - Frame rate: ~50 Hz (20ms/frame)

Graceful degradation:
  - Nếu torch/transformers không cài → trả về empty segments + warning
  - Nếu model download fail → trả về empty segments + warning
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .models import PhonemeSegment

logger = logging.getLogger("toeic.phoneme.wav2vec")

# ──────────────────────────────────────────────────────────────────────────────
# wav2vec 2.0 model config
# ──────────────────────────────────────────────────────────────────────────────

# Model HF được train trên LibriSpeech với phoneme labels (41 classes)
DEFAULT_WAV2VEC_MODEL: str = "facebook/wav2vec2-lg-960h"

# Sample rate của wav2vec
WAV2VEC_SAMPLE_RATE: int = 16000

# Threshold: probability thấp hơn ngưỡng này bị coi là silence/unspoken
PHONEME_CONFIDENCE_THRESHOLD: float = 0.1

# Số frames liên tiếp cùng phoneme để merge thành 1 segment
# wav2vec frame rate ~50Hz → 20ms/frame, min_duration=0.1s = 5 frames
MIN_PHONEME_DURATION_SEC: float = 0.1

# ──────────────────────────────────────────────────────────────────────────────
# LibriSpeech ARPAbet → IPA mapping (41 phoneme classes của wav2vec)
# ──────────────────────────────────────────────────────────────────────────────

# wav2vec2-lg-960h token labels (từ model config.label2id)
# Đây là mapping chuẩn từ LibriSpeech phoneme labels → IPA
WAV2VEC_LABEL_TO_IPA: dict[str, str] = {
    # Silence
    "<unk>": "",
    "<s>": "",
    "</s>": "",
    "#": "",
    "@": "",
    "sil": "",
    "sp": "",
    "pau": "",
    # Vowels
    "AA": "ɑː",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɒ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "e",
    "ER": "ɜː",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "iː",
    "OW": "əʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "uː",
    # Consonants
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

# Reverse: IPA → label name (cho debugging)
_IPA_TO_LABEL: dict[str, str] = {v: k for k, v in WAV2VEC_LABEL_TO_IPA.items() if v}


# ──────────────────────────────────────────────────────────────────────────────
# Audio loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_audio(audio_path: str, target_sr: int = WAV2VEC_SAMPLE_RATE) -> np.ndarray:
    """Load audio file về mono waveform float32, resample nếu cần.

    Priority: librosy → torchaudio → soundfile + manual resample
    """
    # Try librosa (có resample built-in)
    try:
        import librosa
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return waveform  # librosa đã return float32 [-1, 1]
    except ImportError:
        pass
    except Exception as e:
        logger.warning("librosa không đọc được '%s': %s", audio_path, e)

    # Try torchaudio
    try:
        import torch
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.squeeze(0).numpy().astype(np.float32)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("torchaudio không đọc được '%s': %s", audio_path, e)

    # Try soundfile (không có resample, cần matching sample rate)
    try:
        import soundfile as sf
        waveform, sr = sf.read(audio_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != target_sr:
            logger.warning(
                "Audio '%s' sample rate %d ≠ %d — cần librosa/torchaudio để resample",
                audio_path, sr, target_sr,
            )
        return waveform.astype(np.float32)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("soundfile không đọc được '%s': %s", audio_path, e)

    raise RuntimeError(
        f"Không đọc được audio '{audio_path}'. "
        "Cần cài ít nhất 1 trong: librosa, torchaudio, soundfile."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model cache
# ──────────────────────────────────────────────────────────────────────────────

_model_cache: dict[str, tuple[Any, Any]] = {}  # model_id → (processor, model)
_model_lock = threading.Lock()


def _get_wav2vec_model(model_id: str, device: str = "cpu") -> tuple[Any, Any]:
    """Lazy-load wav2vec model + processor, cache trong process."""
    key = f"{model_id}:{device}"
    if key in _model_cache:
        return _model_cache[key]

    with _model_lock:
        if key in _model_cache:
            return _model_cache[key]

        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForCTC
        except ImportError as e:
            raise RuntimeError(
                "wav2vec backend cần torch + transformers. "
                "Cài: pip install torch transformers"
            ) from e

        torch_device = torch.device(device) if device != "cpu" else torch.device("cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(
            "Đang nạp wav2vec model=%s device=%s (có thể mất 1-2 phút lần đầu)...",
            model_id,
            device,
        )

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

        logger.info("wav2vec model đã sẵn sàng.")
        _model_cache[key] = (feature_extractor, model)
        return _model_cache[key]


# ──────────────────────────────────────────────────────────────────────────────
# Frame-level → segment conversion
# ──────────────────────────────────────────────────────────────────────────────

def _frames_to_segments(
    frame_predictions: list[tuple[str, float]],  # (ipa_phoneme, confidence)
    frame_duration: float,  # giây mỗi frame
    audio_duration: float,
    min_duration: float = MIN_PHONEME_DURATION_SEC,
) -> list[PhonemeSegment]:
    """Merge các frame liên tiếp cùng phoneme → phoneme segments.

    Algorithm:
      1. Group consecutive frames with same phoneme
      2. Merge groups with same phoneme if gap < min_duration
      3. Filter: confidence >= threshold AND duration >= min_duration
    """
    if not frame_predictions:
        return []

    # Step 1: group consecutive same-phoneme frames
    groups: list[tuple[str, int, int, float]] = []  # (phoneme, start_frame, end_frame, avg_conf)
    current_phoneme, current_start, conf_sum = frame_predictions[0][0], 0, 0.0
    current_count = 0

    for i, (phoneme, conf) in enumerate(frame_predictions):
        if phoneme == current_phoneme:
            conf_sum += conf
            current_count += 1
        else:
            if current_count > 0:
                groups.append((current_phoneme, current_start, i - 1, conf_sum / current_count))
            current_phoneme = phoneme
            current_start = i
            conf_sum = conf
            current_count = 1

    # Last group
    if current_count > 0:
        groups.append((current_phoneme, current_start, len(frame_predictions) - 1, conf_sum / current_count))

    # Step 2: merge groups with same phoneme if gap < threshold_frames
    threshold_frames = max(1, int(min_duration / max(frame_duration, 1e-6)))
    merged: list[tuple[str, int, int, float]] = [groups[0]] if groups else []

    for group in groups[1:]:
        prev = merged[-1]
        gap = group[1] - prev[2] - 1  # frames between groups
        if group[0] == prev[0] and gap < threshold_frames:
            # Merge: extend end frame, weighted avg confidence
            prev_len = prev[2] - prev[1] + 1
            curr_len = group[2] - group[1] + 1
            new_conf = (prev[3] * prev_len + group[3] * curr_len) / (prev_len + curr_len)
            merged[-1] = (prev[0], prev[1], group[2], new_conf)
        else:
            merged.append(group)

    # Step 3: filter by confidence and duration
    segments: list[PhonemeSegment] = []
    for phoneme, start_frame, end_frame, avg_conf in merged:
        duration = (end_frame - start_frame + 1) * frame_duration
        if avg_conf < PHONEME_CONFIDENCE_THRESHOLD:
            continue
        if duration < min_duration:
            continue
        start_time = start_frame * frame_duration
        end_time = (end_frame + 1) * frame_duration
        # Clamp to audio duration
        end_time = min(end_time, audio_duration)
        segments.append(PhonemeSegment(
            phoneme=phoneme,
            start=round(start_time, 3),
            end=round(end_time, 3),
            confidence=round(avg_conf, 4),
            backend="wav2vec",
        ))

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Main predictor class
# ──────────────────────────────────────────────────────────────────────────────

class Wav2VecPhonemePredictor:
    """wav2vec 2.0 phoneme predictor.

    Usage:
        predictor = Wav2VecPhonemePredictor()
        segments = predictor.predict("audio.wav")
    """

    def __init__(
        self,
        model_id: str = DEFAULT_WAV2VEC_MODEL,
        device: str = "cpu",
        min_phoneme_duration: float = MIN_PHONEME_DURATION_SEC,
        confidence_threshold: float = PHONEME_CONFIDENCE_THRESHOLD,
    ):
        self.model_id = model_id
        self.device = device
        self.min_phoneme_duration = min_phoneme_duration
        self.confidence_threshold = confidence_threshold
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        """Check wav2vec backend có sẵn sàng không."""
        if self._available is not None:
            return self._available
        try:
            _get_wav2vec_model(self.model_id, self.device)
            self._available = True
        except (RuntimeError, ImportError, OSError):
            self._available = False
        return self._available

    def predict(
        self,
        audio_path: str,
    ) -> tuple[list[PhonemeSegment], str | None]:
        """Predict phoneme segments từ audio file.

        Returns:
            (segments, warning) — warning != None nếu backend không sẵn sàng
        """
        if not Path(audio_path).exists():
            return [], f"Audio file không tồn tại: {audio_path}"

        if not self.is_available:
            return [], (
                "wav2vec backend không sẵn sàng. "
                "Cài: pip install torch transformers librosa"
            )

        try:
            # Load audio
            waveform = _load_audio(audio_path, WAV2VEC_SAMPLE_RATE)
            audio_duration = len(waveform) / WAV2VEC_SAMPLE_RATE

            # Get model
            feature_extractor, model = _get_wav2vec_model(self.model_id, self.device)

            # Extract features
            inputs = feature_extractor(waveform, sampling_rate=WAV2VEC_SAMPLE_RATE, return_tensors="pt")
            input_values = inputs.input_values

            # Forward pass (no grad)
            import torch
            with torch.no_grad():
                logits = model(input_values).logits

            # Convert to probabilities per frame
            probs = torch.softmax(logits, dim=-1)
            prob_numpy = probs[0].numpy()  # (num_frames, num_labels)

            # Get label ids from model config
            label_ids = model.config.label2id
            id_to_label = {v: k for k, v in label_ids.items()}

            # For each frame, get the predicted phoneme
            frame_predictions: list[tuple[str, float]] = []
            num_frames = prob_numpy.shape[0]

            for frame_idx in range(num_frames):
                frame_probs = prob_numpy[frame_idx]
                top_label_id = int(np.argmax(frame_probs))
                top_prob = float(frame_probs[top_label_id])
                label_name = id_to_label.get(top_label_id, "")
                ipa = WAV2VEC_LABEL_TO_IPA.get(label_name, "")

                if ipa:  # Only non-silence phonemes
                    frame_predictions.append((ipa, top_prob))
                # Silence frames are skipped — they create natural gaps

            # Calculate frame duration
            # wav2vec output frames ≠ input frames; need to estimate
            # The feature extractor downsamples ~by 320 (for 16kHz → ~50Hz)
            if num_frames > 0 and audio_duration > 0:
                frame_duration = audio_duration / num_frames
            else:
                frame_duration = 0.02  # fallback: 50Hz

            # Convert to segments
            segments = _frames_to_segments(
                frame_predictions,
                frame_duration=frame_duration,
                audio_duration=audio_duration,
                min_duration=self.min_phoneme_duration,
            )

            logger.info(
                "wav2vec predict: %s → %d phoneme segments (%d frames, %.1fs audio)",
                audio_path,
                len(segments),
                num_frames,
                audio_duration,
            )

            return segments, None

        except RuntimeError:
            raise  # Re-raise model errors
        except Exception as e:
            logger.error("wav2vec prediction failed for '%s': %s", audio_path, e, exc_info=True)
            return [], f"wav2vec prediction error: {e}"

    def get_predicted_phoneme_list(self, segments: list[PhonemeSegment]) -> list[str]:
        """Trích danh sách phonemes từ segments (cho comparison với reference).

        Returns list of phonemes in temporal order.
        """
        return [s.phoneme for s in segments]
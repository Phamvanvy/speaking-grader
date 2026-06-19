"""wav2vec 2.0 phoneme prediction backend.

Dùng model wav2vec 2.0 (train trên LibriSpeech với phoneme labels) để dự đoán
phoneme probabilities cho mỗi frame audio, rồi merge các frame liên tiếp
thành phoneme segments có timestamps.

Architecture:
  - Wav2VecPhonemePredictor: class chính, lazy-load model, cache trong process
  - predict_phonemes(): audio path → list[PhonemeSegment]
  - _frames_to_segments(): merge frame-level predictions → phoneme segments

Model: facebook/wav2vec2-xlsr-53-espeak-cv-ft (phoneme-CTC, output IPA eSpeak)
  - Size: ~1.2GB
  - Output: frame-level logits over các token IPA eSpeak (vocab của tokenizer)
  - Frame rate: ~50 Hz (20ms/frame)

LƯU Ý: phải dùng model phoneme-CTC (output IPA), KHÔNG dùng wav2vec2-*-960h —
các model 960h là CTC ký tự (A-Z), không phải phoneme. Token của model này đã là
IPA nên decode bằng tokenizer trực tiếp, không cần map ARPAbet thủ công.

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

# Model HF phoneme-CTC: output token IPA eSpeak trực tiếp (không phải ký tự).
DEFAULT_WAV2VEC_MODEL: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

# Sample rate của wav2vec
WAV2VEC_SAMPLE_RATE: int = 16000

# Threshold: probability thấp hơn ngưỡng này bị coi là silence/unspoken
PHONEME_CONFIDENCE_THRESHOLD: float = 0.1

# Số frames liên tiếp cùng phoneme để merge thành 1 segment
# wav2vec frame rate ~50Hz → 20ms/frame, min_duration=0.1s = 5 frames
MIN_PHONEME_DURATION_SEC: float = 0.1

# ──────────────────────────────────────────────────────────────────────────────
# ARPAbet → IPA mapping (CHỈ là fallback cho model phoneme dạng ARPAbet)
# ──────────────────────────────────────────────────────────────────────────────

# Model mặc định (espeak-cs-ft) đã output token IPA → không cần bảng này. Giữ lại
# để tương thích nếu ai đó cấu hình một model phoneme dùng nhãn ARPAbet.
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

# Token coi như "không phải phoneme" (silence/blank/special/word-boundary).
# CTC blank = pad token; ngoài ra còn các special token và dấu phân từ "|".
_SILENCE_TOKENS: frozenset[str] = frozenset(
    {"", " ", "|", "sil", "sp", "spn", "pau", "<pad>", "<s>", "</s>", "<unk>"}
)


def _resolve_ipa(token: str, silence_tokens: frozenset[str]) -> str:
    """Quy 1 token của model về ký hiệu IPA, '' nếu là silence/blank.

    - Token nằm trong silence set → '' (bị bỏ qua, tạo khoảng lặng tự nhiên).
    - Token là nhãn ARPAbet (vd 'AA', 'TH') → map qua WAV2VEC_LABEL_TO_IPA.
    - Còn lại: coi token đã là IPA (model espeak) → trả nguyên token.
    """
    if token in silence_tokens:
        return ""
    if token in WAV2VEC_LABEL_TO_IPA:
        return WAV2VEC_LABEL_TO_IPA[token]
    return token.strip()


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

# model_id → (feature_extractor, model, id_to_token)
_model_cache: dict[str, tuple[Any, Any, dict[int, str]]] = {}
_model_lock = threading.Lock()


def _load_id_to_token(model_id: str, model: Any) -> dict[int, str]:
    """Lấy map id → token (IPA) của model, KHÔNG cần tokenizer/phonemizer.

    Model phoneme espeak dùng Wav2Vec2PhonemeCTCTokenizer vốn yêu cầu thư viện
    `phonemizer` (kéo theo espeak-ng) chỉ để phonemize text — không cần cho việc
    decode id → token. Nên ta đọc thẳng vocab.json từ repo (token → id) rồi đảo
    lại. Fallback: model.config.label2id.
    """
    try:
        import json as _json

        from huggingface_hub import hf_hub_download

        vocab_path = hf_hub_download(model_id, "vocab.json")
        with open(vocab_path, encoding="utf-8") as f:
            vocab: dict[str, int] = _json.load(f)
        return {int(idx): tok for tok, idx in vocab.items()}
    except Exception as e:  # noqa: BLE001 - fallback an toàn về config
        logger.warning(
            "Không đọc được vocab.json của %s (%s) — fallback model.config.label2id.",
            model_id,
            e,
        )
        return {int(idx): tok for tok, idx in model.config.label2id.items()}


def _get_wav2vec_model(
    model_id: str, device: str = "cpu"
) -> tuple[Any, Any, dict[int, str]]:
    """Lazy-load feature_extractor + model + id→token map, cache trong process."""
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

        # Chỉ nạp feature_extractor (nhẹ, không cần phonemizer). Việc decode
        # id → token IPA dùng vocab.json đọc riêng (xem _load_id_to_token).
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

        id_to_token = _load_id_to_token(model_id, model)

        logger.info(
            "wav2vec model đã sẵn sàng (vocab=%d tokens).", len(id_to_token)
        )
        _model_cache[key] = (feature_extractor, model, id_to_token)
        return _model_cache[key]


# ──────────────────────────────────────────────────────────────────────────────
# Frame-level → segment conversion
# ──────────────────────────────────────────────────────────────────────────────

def _ctc_decode_segments(
    pred_ids: np.ndarray,       # (num_frames,) argmax token id mỗi frame
    pred_probs: np.ndarray,     # (num_frames,) prob của token đó
    id_to_label: dict[int, str],
    frame_duration: float,      # giây mỗi frame
    audio_duration: float,
    confidence_threshold: float = PHONEME_CONFIDENCE_THRESHOLD,
) -> list[PhonemeSegment]:
    """CTC greedy decode: frame-level argmax → phoneme segments.

    Output của wav2vec CTC rất "spiky": phần lớn frame là blank (<pad>), mỗi
    phoneme chỉ chiếm 1-vài frame. Quy tắc CTC:
      1. Gộp các frame liên tiếp cùng token id thành 1 "run".
      2. Bỏ run là blank/silence (chính các blank này phân tách phoneme lặp).
      3. Mỗi run phoneme còn lại = 1 segment, timestamp theo vị trí frame thật.
    KHÔNG lọc theo min_duration (sẽ giết hết các spike hợp lệ).
    """
    n = len(pred_ids)
    if n == 0:
        return []

    segments: list[PhonemeSegment] = []
    run_start = 0
    for i in range(1, n + 1):
        # Kết thúc 1 run khi đổi id hoặc hết frame.
        if i == n or pred_ids[i] != pred_ids[run_start]:
            token = id_to_label.get(int(pred_ids[run_start]), "")
            ipa = _resolve_ipa(token, _SILENCE_TOKENS)
            if ipa:
                avg_conf = float(pred_probs[run_start:i].mean())
                if avg_conf >= confidence_threshold:
                    start_time = run_start * frame_duration
                    end_time = min(i * frame_duration, audio_duration)
                    segments.append(PhonemeSegment(
                        phoneme=ipa,
                        start=round(start_time, 3),
                        end=round(end_time, 3),
                        confidence=round(avg_conf, 4),
                        backend="wav2vec",
                    ))
            run_start = i

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
        except (RuntimeError, ImportError, OSError) as e:
            self._available = False
            # Log the REAL reason (not generic "install torch" message)
            err_type = type(e).__name__
            err_msg = str(e)
            logger.warning(
                "wav2vec backend KHÔNG khả dụng (%s): %s",
                err_type,
                err_msg,
            )
            # Detect common causes
            if "CUDA" in err_type or "out of memory" in err_msg.lower() or "cuda" in err_msg.lower():
                logger.warning(
                    "Nguyên nhân: GPU không đủ memory (Whisper %s + wav2vec %s cùng lúc). "
                    "Khắc phục: (a) dùng GPU lớn hơn, (b) chạy wav2vec trên CPU bằng "
                    "TOEIC_PHONEME_DEVICE=cpu, hoặc (c) tắt phoneme analysis "
                    "TOEIC_PHONEME_ANALYSIS_ENABLED=false.",
                    self.model_id,
                    self.model_id,
                )
            elif "ImportError" in err_type:
                logger.warning(
                    "Nguyên nhân: thiếu package. Cài: pip install torch transformers librosa"
                )
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
            return [], "wav2vec backend không khả dụng (xem log chi tiết)."

        try:
            import torch

            # Free CUDA memory before loading model (helps avoid OOM with Whisper)
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                logger.debug("CUDA free memory before wav2vec: %.2f GB", free_mem)

            # Load audio
            waveform = _load_audio(audio_path, WAV2VEC_SAMPLE_RATE)
            audio_duration = len(waveform) / WAV2VEC_SAMPLE_RATE

            # Get feature_extractor + model + id→token map
            feature_extractor, model, id_to_label = _get_wav2vec_model(
                self.model_id, self.device
            )

            # Extract features
            inputs = feature_extractor(waveform, sampling_rate=WAV2VEC_SAMPLE_RATE, return_tensors="pt")
            input_values = inputs.input_values

            # Move input to device
            if self.device == "cuda" and torch.cuda.is_available():
                input_values = input_values.to("cuda")

            # Forward pass (no grad)
            with torch.no_grad():
                logits = model(input_values).logits

            # Move results back to CPU before freeing GPU memory
            probs = torch.softmax(logits, dim=-1)
            prob_numpy = probs[0].cpu().numpy()  # (num_frames, num_labels)
            num_frames = prob_numpy.shape[0]

            # Per-frame argmax token id + prob (CTC decode dùng cả blank frames)
            pred_ids = np.argmax(prob_numpy, axis=-1)
            pred_probs = prob_numpy[np.arange(num_frames), pred_ids]

            # Calculate frame duration
            # wav2vec output frames ≠ input frames; need to estimate
            # The feature extractor downsamples ~by 320 (for 16kHz → ~50Hz)
            if num_frames > 0 and audio_duration > 0:
                frame_duration = audio_duration / num_frames
            else:
                frame_duration = 0.02  # fallback: 50Hz

            # CTC greedy decode → segments. id_to_label đến từ vocab.json
            # (model espeak: token = IPA). _resolve_ipa bỏ silence/blank.
            segments = _ctc_decode_segments(
                pred_ids,
                pred_probs,
                id_to_label,
                frame_duration=frame_duration,
                audio_duration=audio_duration,
                confidence_threshold=self.confidence_threshold,
            )

            # Free CUDA memory after prediction
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                logger.debug("CUDA free memory after wav2vec: %.2f GB", free_mem)

            logger.info(
                "wav2vec predict: %s → %d phoneme segments (%d frames, %.1fs audio)",
                audio_path,
                len(segments),
                num_frames,
                audio_duration,
            )

            return segments, None

        except RuntimeError as e:
            # CUDA OOM: suggest CPU fallback
            if self.device == "cuda" and "cuda" in str(e).lower():
                logger.error(
                    "wav2vec CUDA OOM for '%s': %s\n"
                    "Khắc phục: đặt TOEIC_PHONEME_DEVICE=cpu để chạy trên CPU, "
                    "hoặc TOEIC_PHONEME_ANALYSIS_ENABLED=false để tắt phoneme analysis.",
                    audio_path, e,
                )
            raise  # Re-raise for proper upstream handling
        except Exception as e:
            logger.error("wav2vec prediction failed for '%s': %s", audio_path, e, exc_info=True)
            return [], f"wav2vec prediction error: {e}"

    def get_predicted_phoneme_list(self, segments: list[PhonemeSegment]) -> list[str]:
        """Trích danh sách phonemes từ segments (cho comparison với reference).

        Returns list of phonemes in temporal order.
        """
        return [s.phoneme for s in segments]
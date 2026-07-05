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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import EvidenceStats, PhonemeSegment

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

# Đệm cửa sổ khi probe deletion-evidence: nới khoảng frame mỗi bên chừng này
# (~40ms ở 50Hz) để không cắt mất onset/coda do sai số biên segment/window.
EVIDENCE_WINDOW_MARGIN_FRAMES: int = 2

# Số frame mass cao nhất lấy trung bình cho EvidenceStats.top_k_mean.
EVIDENCE_TOP_K_FRAMES: int = 3

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
    "OW": "əʊ",  # cố ý GIỮ əʊ (≠ reference oʊ): bảng predicted-side, CHỈ cho model ARPAbet
                 # không-mặc-định; model mặc định (espeak) output IPA trực tiếp. normalize_ipa
                 # gộp oʊ↔əʊ nên scoring không đổi — không cần đồng bộ với ARPABET_TO_IPA["OW"].
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
# Deletion-evidence probe (SHADOW): giữ lại frame posteriors + tra mass theo âm
# ──────────────────────────────────────────────────────────────────────────────

# model_id → {normalized_ipa: frozenset[token_id]} — vocab tĩnh theo model nên cache
# process-wide (không phụ thuộc audio). Nhóm theo normalize_ipa để /l/ khớp mọi biến
# thể espeak ('l', 'ɫ', ...) và oʊ↔əʊ tự gộp như phía scoring.
_ipa_group_cache: dict[str, dict[str, frozenset[int]]] = {}
_ipa_group_lock = threading.Lock()


def _ipa_token_groups(
    model_id: str, id_to_token: dict[int, str]
) -> dict[str, frozenset[int]]:
    """Nhóm token id của vocab theo IPA đã normalize; bỏ silence/blank/special."""
    cached = _ipa_group_cache.get(model_id)
    if cached is not None:
        return cached
    # Lazy import: giữ wav2vec_backend importable độc lập không kéo chuỗi ipa/g2p.
    from .ipa import normalize_ipa

    groups: dict[str, set[int]] = {}
    for idx, token in id_to_token.items():
        ipa = _resolve_ipa(token, _SILENCE_TOKENS)
        if not ipa:
            continue
        key = normalize_ipa(ipa)
        if not key:
            continue
        groups.setdefault(key, set()).add(idx)
    frozen = {k: frozenset(v) for k, v in groups.items()}
    with _ipa_group_lock:
        _ipa_group_cache[model_id] = frozen
    return frozen


@dataclass(frozen=True)
class FramePosteriors:
    """Frame posteriors của 1 lần predict — sống trong request, KHÔNG serialize.

    `probs` (num_frames × vocab, float32) chính là ma trận wav2vec đã tính sẵn cho
    CTC decode; giữ lại để probe deletion-evidence (SHADOW — chỉ telemetry). ~5MB
    cho 60s audio, giải phóng khi request kết thúc.
    """

    probs: np.ndarray
    frame_duration: float
    id_to_token: dict[int, str]
    model_id: str

    def evidence_stats(
        self, ref_ipa: str, t0: float, t1: float
    ) -> EvidenceStats | None:
        """Thống kê mass của nhóm token khớp `ref_ipa` trong cửa sổ [t0, t1].

        Deterministic thuần (chỉ sum/max/percentile trên ma trận đã có). Trả None
        nếu vocab không có token nào normalize trùng `ref_ipa` (không đo được —
        khác với "đo được và bằng 0"). Cửa sổ rỗng/ngoài biên → stats toàn 0,
        n_frames=0. Margin EVIDENCE_WINDOW_MARGIN_FRAMES nới mỗi bên.
        """
        from .ipa import normalize_ipa

        groups = _ipa_token_groups(self.model_id, self.id_to_token)
        token_ids = groups.get(normalize_ipa(ref_ipa))
        if not token_ids:
            return None
        n_frames = self.probs.shape[0]
        if n_frames == 0 or self.frame_duration <= 0 or t1 <= t0:
            return EvidenceStats(0.0, 0.0, 0.0, 0)
        lo = max(0, int(t0 / self.frame_duration) - EVIDENCE_WINDOW_MARGIN_FRAMES)
        hi = min(
            n_frames,
            int(np.ceil(t1 / self.frame_duration)) + EVIDENCE_WINDOW_MARGIN_FRAMES,
        )
        if lo >= hi:
            return EvidenceStats(0.0, 0.0, 0.0, 0)
        window = self.probs[lo:hi]
        ids = np.fromiter(sorted(token_ids), dtype=np.int64)
        mass = window[:, ids].sum(axis=1)  # (hi-lo,) mass nhóm âm mỗi frame
        best = int(np.argmax(mass))
        top_k = np.sort(mass)[-EVIDENCE_TOP_K_FRAMES:]
        argmax_id = int(np.argmax(window[best]))
        return EvidenceStats(
            max_mass=float(mass[best]),
            top_k_mean=float(top_k.mean()),
            p90=float(np.percentile(mass.astype(np.float64), 90)),
            n_frames=int(hi - lo),
            argmax_token=self.id_to_token.get(argmax_id, ""),
            argmax_prob=float(window[best, argmax_id]),
        )


@dataclass(frozen=True)
class ChunkedFramePosteriors:
    """Posteriors của predict CHUNKED — cùng interface `evidence_stats` với
    FramePosteriors (consumer duy nhất: _attach_deletion_evidence).

    `chunks`: list (chunk_start_sec, FramePosteriors) theo thứ tự thời gian; mỗi
    FramePosteriors sống trong toạ độ THỜI GIAN LOCAL của chunk đó. Query bằng
    thời gian tuyệt đối: trừ offset rồi hỏi từng chunk overlap, trả stats có
    max_mass LỚN HƠN (đơn giản, deterministic — cửa sổ vắt 2 chunk lấy evidence
    mạnh hơn). Cửa sổ rơi hoàn toàn vào gap im lặng giữa các chunk →
    EvidenceStats toàn 0, n_frames=0 (cùng ngữ nghĩa out-of-range của bản đơn).
    """

    chunks: tuple[tuple[float, "FramePosteriors"], ...]

    def evidence_stats(
        self, ref_ipa: str, t0: float, t1: float
    ) -> EvidenceStats | None:
        best: EvidenceStats | None = None
        for chunk_start, post in self.chunks:
            chunk_end = chunk_start + post.probs.shape[0] * post.frame_duration
            if t1 < chunk_start or t0 > chunk_end:
                continue
            stats = post.evidence_stats(ref_ipa, t0 - chunk_start, t1 - chunk_start)
            if stats is None:
                continue  # vocab thiếu token — thử chunk khác (vocab chung, hiếm)
            if best is None or stats.max_mass > best.max_mass:
                best = stats
        if best is not None:
            return best
        # Không chunk nào đo được: hoặc cửa sổ rơi vào gap im lặng ("đo được và
        # bằng 0"), hoặc vocab không có token khớp (None như bản đơn) — phân biệt
        # bằng probe cửa sổ rỗng trên chunk đầu (vocab dùng chung 1 model).
        if self.chunks:
            probe = self.chunks[0][1].evidence_stats(ref_ipa, 0.0, 0.0)
            if probe is None:
                return None
        return EvidenceStats(0.0, 0.0, 0.0, 0)


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


def _is_cuda_device(device: str) -> bool:
    """True nếu device là CUDA ('cuda' hoặc 'cuda:N').

    Cần vì code cũ so sánh cứng `device == "cuda"` → đặt TOEIC_PHONEME_DEVICE=cuda:1
    (chạy wav2vec trên GPU thứ 2) sẽ âm thầm rớt về CPU. torch nhận thẳng chuỗi
    'cuda:1' cho .to()/.device nên chỉ cần nhận diện đúng prefix là đủ.
    """
    return (device or "").strip().lower().startswith("cuda")


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

        is_cuda = _is_cuda_device(device)
        dtype = torch.float16 if is_cuda else torch.float32

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
        if is_cuda and torch.cuda.is_available():
            model = model.to(device)  # 'cuda' hoặc 'cuda:N' — chọn đúng GPU
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
        chunk_spans: list[tuple[float, float]] | None = None,
    ) -> tuple[list[PhonemeSegment], str | None]:
        """Predict phoneme segments từ audio file.

        Returns:
            (segments, warning) — warning != None nếu backend không sẵn sàng
        """
        segments, warning, _posteriors = self.predict_with_posteriors(
            audio_path, chunk_spans=chunk_spans
        )
        return segments, warning

    def _forward_decode(
        self,
        waveform: np.ndarray,
        feature_extractor: Any,
        model: Any,
        id_to_label: dict[int, str],
        torch: Any,
    ) -> tuple[list[PhonemeSegment], FramePosteriors]:
        """Một forward pass wav2vec trên waveform → (segments, posteriors).

        Lõi dùng chung cho cả predict single-pass lẫn per-chunk — đúng trình tự
        ops của bản single-pass cũ (feature extract → cast dtype → forward →
        softmax → CTC greedy decode) để hành vi không đổi.
        """
        audio_duration = len(waveform) / WAV2VEC_SAMPLE_RATE

        inputs = feature_extractor(
            waveform, sampling_rate=WAV2VEC_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values

        if _is_cuda_device(self.device) and torch.cuda.is_available():
            input_values = input_values.to(self.device)

        # Cast input theo dtype thật của model (float16 trên CUDA, float32 CPU).
        model_dtype = next(model.parameters()).dtype
        input_values = input_values.to(dtype=model_dtype)

        with torch.no_grad():
            logits = model(input_values).logits

        probs = torch.softmax(logits, dim=-1)
        prob_numpy = probs[0].cpu().numpy()  # (num_frames, num_labels)
        num_frames = prob_numpy.shape[0]

        pred_ids = np.argmax(prob_numpy, axis=-1)
        pred_probs = prob_numpy[np.arange(num_frames), pred_ids]

        if num_frames > 0 and audio_duration > 0:
            frame_duration = audio_duration / num_frames
        else:
            frame_duration = 0.02  # fallback: 50Hz

        segments = _ctc_decode_segments(
            pred_ids,
            pred_probs,
            id_to_label,
            frame_duration=frame_duration,
            audio_duration=audio_duration,
            confidence_threshold=self.confidence_threshold,
        )
        posteriors = FramePosteriors(
            probs=prob_numpy,
            frame_duration=frame_duration,
            id_to_token=id_to_label,
            model_id=self.model_id,
        )
        return segments, posteriors

    # Chunk ngắn hơn ngưỡng này bị bỏ qua: conv feature encoder của wav2vec cần
    # tối thiểu ~25ms sample; chunk gần rỗng không mang thông tin phoneme.
    _MIN_CHUNK_SEC: float = 0.05

    def _predict_chunked(
        self,
        waveform: np.ndarray,
        chunk_spans: list[tuple[float, float]],
        feature_extractor: Any,
        model: Any,
        id_to_label: dict[int, str],
        torch: Any,
    ) -> tuple[list[PhonemeSegment], ChunkedFramePosteriors]:
        """Predict theo từng chunk span (giây, thời gian tuyệt đối) rồi ghép:
        segment times CỘNG offset chunk_start; posteriors gói ChunkedFramePosteriors
        (cùng interface evidence_stats). Spans không hợp lệ/quá ngắn bị bỏ qua.
        """
        n_samples = len(waveform)
        segments: list[PhonemeSegment] = []
        chunk_posts: list[tuple[float, FramePosteriors]] = []
        for span_start, span_end in chunk_spans:
            s0 = max(0, int(span_start * WAV2VEC_SAMPLE_RATE))
            s1 = min(n_samples, int(span_end * WAV2VEC_SAMPLE_RATE))
            if (s1 - s0) < self._MIN_CHUNK_SEC * WAV2VEC_SAMPLE_RATE:
                continue
            t0 = s0 / WAV2VEC_SAMPLE_RATE  # offset thật sau clamp
            chunk_started = time.perf_counter()
            chunk_segments, chunk_post = self._forward_decode(
                waveform[s0:s1], feature_extractor, model, id_to_label, torch
            )
            logger.debug(
                "wav2vec chunk [%.2f-%.2f]s: %d segments, %d frames, %.2fs",
                t0, s1 / WAV2VEC_SAMPLE_RATE, len(chunk_segments),
                chunk_post.probs.shape[0], time.perf_counter() - chunk_started,
            )
            for seg in chunk_segments:
                segments.append(PhonemeSegment(
                    phoneme=seg.phoneme,
                    start=round(seg.start + t0, 3),
                    end=round(seg.end + t0, 3),
                    confidence=seg.confidence,
                    backend=seg.backend,
                ))
            chunk_posts.append((t0, chunk_post))
        return segments, ChunkedFramePosteriors(chunks=tuple(chunk_posts))

    def predict_with_posteriors(
        self,
        audio_path: str,
        chunk_spans: list[tuple[float, float]] | None = None,
    ) -> tuple[
        list[PhonemeSegment], str | None, FramePosteriors | ChunkedFramePosteriors | None
    ]:
        """Như `predict` nhưng kèm posteriors (ma trận đã tính sẵn cho CTC decode —
        không tốn thêm forward pass) để probe deletion-evidence.

        Args:
            chunk_spans: optional — danh sách (start, end) giây (từ
                chunking.compute_chunk_spans). None/rỗng = single-pass như cũ
                (bit-for-bit). Có spans = forward từng chunk rồi ghép (fix suy
                giảm wav2vec trên audio dài); segment times là thời gian tuyệt
                đối; posteriors là ChunkedFramePosteriors (cùng interface).

        Returns:
            (segments, warning, posteriors) — posteriors None khi backend lỗi.
        """
        if not Path(audio_path).exists():
            return [], f"Audio file không tồn tại: {audio_path}", None

        if not self.is_available:
            return [], "wav2vec backend không khả dụng (xem log chi tiết).", None

        try:
            import torch

            # Free CUDA memory before loading model (helps avoid OOM with Whisper)
            if _is_cuda_device(self.device) and torch.cuda.is_available():
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

            posteriors: FramePosteriors | ChunkedFramePosteriors
            if chunk_spans:
                # Chunked: forward từng span rồi ghép (fix suy giảm trên audio
                # dài). Load audio MỘT lần ở trên; times cộng offset trong helper.
                segments, posteriors = self._predict_chunked(
                    waveform, chunk_spans, feature_extractor, model,
                    id_to_label, torch,
                )
                num_frames = sum(
                    p.probs.shape[0] for _t, p in posteriors.chunks
                )
                mode = f"chunked×{len(posteriors.chunks)}"
            else:
                # Single-pass (đường cũ) — giữ lại posteriors cho deletion-
                # evidence probe (SHADOW): ma trận đã ở CPU sẵn, không vứt đi.
                segments, posteriors = self._forward_decode(
                    waveform, feature_extractor, model, id_to_label, torch
                )
                num_frames = posteriors.probs.shape[0]
                mode = "single"

            # Free CUDA memory after prediction
            if _is_cuda_device(self.device) and torch.cuda.is_available():
                torch.cuda.empty_cache()
                free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
                logger.debug("CUDA free memory after wav2vec: %.2f GB", free_mem)

            logger.info(
                "wav2vec predict [%s]: %s → %d phoneme segments (%d frames, %.1fs audio)",
                mode,
                audio_path,
                len(segments),
                num_frames,
                audio_duration,
            )

            return segments, None, posteriors

        except RuntimeError as e:
            # CUDA OOM: suggest CPU fallback
            if _is_cuda_device(self.device) and "cuda" in str(e).lower():
                logger.error(
                    "wav2vec CUDA OOM for '%s': %s\n"
                    "Khắc phục: đặt TOEIC_PHONEME_DEVICE=cpu để chạy trên CPU, "
                    "hoặc TOEIC_PHONEME_ANALYSIS_ENABLED=false để tắt phoneme analysis.",
                    audio_path, e,
                )
            raise  # Re-raise for proper upstream handling
        except Exception as e:
            logger.error("wav2vec prediction failed for '%s': %s", audio_path, e, exc_info=True)
            return [], f"wav2vec prediction error: {e}", None

    def get_predicted_phoneme_list(self, segments: list[PhonemeSegment]) -> list[str]:
        """Trích danh sách phonemes từ segments (cho comparison với reference).

        Returns list of phonemes in temporal order.
        """
        return [s.phoneme for s in segments]
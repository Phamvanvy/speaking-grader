"""Speech-to-Text local bằng faster-whisper.

Trả về transcript đầy đủ + danh sách từ kèm mốc thời gian và độ tự tin
(logprob → probability). Mốc thời gian từng từ là dữ liệu gốc để
features.py tính tốc độ nói, quãng ngắt...
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger("toeic.asr")

ASRBackend = Literal["faster_whisper", "insanely_fast_whisper", "whisperx"]
_WHISPER_SAMPLE_RATE = 16000


def _register_cuda_dll_dirs() -> None:
    """Cho Windows tìm thấy cuBLAS/cuDNN/nvRTC từ các wheel ``nvidia-*-cu12``.

    ctranslate2 nạp ``cublas64_12.dll`` / ``cudnn64_9.dll`` qua LoadLibrary lúc
    encode. Trên Windows, Python không tự tìm trong site-packages\\nvidia\\*\\bin
    (chỉ ``torch`` mới thêm), nên thiếu các DLL này dù đã ``pip install``.
    Bộ nạp DLL lười của ctranslate2 không tôn trọng ``os.add_dll_directory``,
    nên ta phải prepend trực tiếp các thư mục bin đó vào ``PATH``.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia  # các wheel nvidia-*-cu12 cùng nằm trong namespace này
    except ImportError:
        return
    bin_dirs: list[str] = []
    for pkg_dir in nvidia.__path__:
        for sub in ("cublas", "cudnn", "cuda_nvrtc"):
            bin_dir = os.path.join(pkg_dir, sub, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(bin_dir)
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass
class Transcription:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str = ""
    duration: float = 0.0  # thời lượng audio (giây)

    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class ASRRun:
    transcription: Transcription
    backend_used: str
    elapsed_ms: int


# Cache model trong process để khỏi nạp lại mỗi lần gọi.
_model_cache: dict[tuple[str, str], object] = {}
_ifw_pipe_cache: dict[tuple[str, str], object] = {}
# WhisperX: cache cả ASR model lẫn align model. Trước đây nạp lại mỗi lần gọi
# (load_model + load_align_model) — rất tốn khi chấm cả lớp (batch). Key gồm
# device/compute_type để không lẫn CPU/GPU; align model còn key theo ngôn ngữ.
_whisperx_model_cache: dict[tuple[str, str, str], object] = {}
_whisperx_align_cache: dict[tuple[str, str], tuple[object, object]] = {}
# Khóa nạp model: khi /grade-batch chấm song song, nhiều luồng cùng miss cache
# và cùng nạp large-v3 (vài GB) một lúc → tranh GPU/đĩa. Lock + double-check để
# chỉ 1 luồng nạp, các luồng còn lại chờ rồi dùng lại model trong cache.
_whisperx_load_lock = threading.Lock()

# Khóa inference ASR: cho phép /grade-batch chạy nhiều luồng (concurrency>1) để
# tầng LLM của bài N chồng lấn với ASR của bài N+1, nhưng Whisper KHÔNG an toàn
# khi chạy song song trên cùng GPU → serialize phần inference để chỉ một ASR
# chạy tại một thời điểm. Đây là chokepoint chung cho mọi backend.
_asr_inference_lock = threading.Lock()


def _split_cuda_device(device: str) -> tuple[str, int]:
    """Tách device CUDA kèm chỉ số GPU: 'cuda:1' → ('cuda', 1); 'cuda' → ('cuda', 0).

    Cần vì faster-whisper/ctranslate2 và whisperx.load_model nhận device='cuda'
    KÈM device_index riêng — chúng KHÔNG hiểu chuỗi 'cuda:1'. Các device khác
    ('cpu', 'auto') trả index 0 và giữ nguyên tên. Dùng để chạy Whisper trên GPU
    này còn wav2vec trên GPU kia (vd WHISPER_DEVICE=cuda:0, TOEIC_PHONEME_DEVICE=cuda:1).
    """
    d = (device or "").strip().lower()
    if d.startswith("cuda:"):
        try:
            return "cuda", int(d.split(":", 1)[1])
        except ValueError:
            return "cuda", 0
    return d or "cpu", 0


def _resolve_torch_device(requested_device: str, backend_name: str) -> str:
    """Torch-based backends should degrade to CPU if CUDA is unavailable.

    Faster-Whisper uses ctranslate2 and can still work with CUDA even when the
    installed PyTorch build is CPU-only. WhisperX / IFW cannot, so they need a
    separate device resolution step.
    """
    requested = (requested_device or "").strip().lower() or "auto"

    # Chỉ phân giải các yêu cầu CUDA (auto / cuda / cuda:N). Còn lại (vd 'cpu')
    # giữ nguyên. 'cuda:N' được giữ lại nguyên chỉ số khi CUDA khả dụng.
    if not (requested == "auto" or requested.startswith("cuda")):
        return requested

    try:
        import torch
    except ImportError:
        logger.warning("%s: chưa cài torch; dùng cpu.", backend_name)
        return "cpu"

    torch_cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    has_cuda_build = bool(torch_cuda_build)
    cuda_available = bool(torch.cuda.is_available())

    if cuda_available:
        if requested == "auto":
            logger.info(
                "%s: auto chọn cuda (torch=%s, cuda_build=%s, gpu_count=%d).",
                backend_name,
                getattr(torch, "__version__", "unknown"),
                torch_cuda_build,
                int(torch.cuda.device_count()),
            )
            return "cuda"
        return requested  # giữ nguyên 'cuda' hoặc 'cuda:N' (chọn đúng GPU)

    # Không khả dụng CUDA: log rõ để tránh hiểu nhầm "máy có GPU nhưng torch vẫn CPU-only".
    details = (
        "torch build CPU-only"
        if not has_cuda_build
        else "CUDA runtime không khả dụng trong torch"
    )
    if requested.startswith("cuda"):
        logger.warning(
            "%s: yêu cầu cuda nhưng phải hạ xuống cpu (%s, torch=%s, cuda_build=%s).",
            backend_name,
            details,
            getattr(torch, "__version__", "unknown"),
            torch_cuda_build,
        )
    else:
        logger.info(
            "%s: auto chọn cpu (%s, torch=%s, cuda_build=%s).",
            backend_name,
            details,
            getattr(torch, "__version__", "unknown"),
            torch_cuda_build,
        )
    return "cpu"


def _get_model(model_size: str, device: str):
    key = (model_size, device)
    if key not in _model_cache:
        # 'cuda:1' → base='cuda', index=1: ctranslate2 cần device + device_index
        # tách rời (không hiểu chuỗi 'cuda:1').
        base, device_index = _split_cuda_device(device)
        if base == "cuda":
            _register_cuda_dll_dirs()
        # Import trong hàm để --no-ai và unit test không bắt buộc cài faster-whisper
        from faster_whisper import WhisperModel

        compute_type = "float16" if base == "cuda" else "int8"
        logger.info(
            "Đang nạp Whisper model=%s device=%s index=%d compute_type=%s",
            model_size,
            base,
            device_index,
            compute_type,
        )
        _model_cache[key] = WhisperModel(
            model_size,
            device=base,
            device_index=device_index,
            compute_type=compute_type,
        )
    return _model_cache[key]


def transcribe(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    language: str = "en",
) -> Transcription:
    """Chuyển 1 file audio thành Transcription (có word timestamps)."""
    model = _get_model(model_size, device)

    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,  # lọc khoảng lặng dài để timestamps sạch hơn
    )

    words: list[Word] = []
    text_parts: list[str] = []
    for seg in segments:
        text_parts.append(seg.text.strip())
        for w in seg.words or []:
            words.append(
                Word(
                    text=w.word.strip(),
                    start=float(w.start),
                    end=float(w.end),
                    probability=float(w.probability),
                )
            )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    if duration == 0.0 and words:
        duration = words[-1].end

    transcription = Transcription(
        text=" ".join(p for p in text_parts if p).strip(),
        words=words,
        language=getattr(info, "language", language) or language,
        duration=duration,
    )
    logger.info(
        "ASR xong: %d từ, %.2fs, lang=%s",
        transcription.word_count,
        transcription.duration,
        transcription.language,
    )
    return transcription


def transcribe_with_backend(
    audio_path: str,
    backend: str,
    model_size: str = "base",
    device: str = "cpu",
    language: str = "en",
    batch_size: int = 16,
) -> ASRRun:
    """Chạy ASR theo backend được chỉ định và trả metadata runtime.

    - faster_whisper: backend mặc định, ổn định.
    - whisperx: cần cài thêm whisperx + dependencies alignment.
    - insanely_fast_whisper: để dành fast lane; hiện chưa có adapter Python ổn
      định trong project này nên sẽ raise RuntimeError để caller fallback.
    """
    started = time.perf_counter()
    key = (backend or "faster_whisper").lower()

    # Serialize inference (xem _asr_inference_lock): với batch concurrency>1, các
    # luồng vào đây tuần tự nên chỉ một ASR chạy/lúc, trong khi tầng LLM của bài
    # đã xong ASR vẫn chồng lấn ở luồng khác.
    with _asr_inference_lock:
        if key == "faster_whisper":
            out = transcribe(audio_path, model_size=model_size, device=device, language=language)
        elif key == "whisperx":
            out = _transcribe_whisperx(
                audio_path,
                model_size=model_size,
                device=device,
                language=language,
                batch_size=batch_size,
            )
        elif key == "insanely_fast_whisper":
            out = _transcribe_insanely_fast_whisper(
                audio_path,
                model_size=model_size,
                device=device,
                language=language,
            )
        else:
            raise RuntimeError(f"ASR backend không hợp lệ: {backend}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return ASRRun(transcription=out, backend_used=key, elapsed_ms=elapsed_ms)


def _transcribe_whisperx(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    language: str = "en",
    batch_size: int = 16,
) -> Transcription:
    """ASR bằng WhisperX (kèm alignment từ).

    Ưu tiên loader mặc định của WhisperX (ffmpeg CLI). Nếu máy không có ffmpeg
    trên PATH, fallback sang loader Python bằng torchaudio để review mode vẫn
    chạy được trên Windows/local dev.
    """
    try:
        import whisperx
    except ImportError as e:
        raise RuntimeError(
            "Chưa cài whisperx. Cài trước khi dùng ASR backend 'whisperx'."
        ) from e

    device = _resolve_torch_device(device, "whisperx")
    # 'cuda:1' → base='cuda', index=1 cho load_model (ctranslate2); chuỗi 'cuda:1'
    # đầy đủ vẫn dùng cho align (torch hiểu trực tiếp).
    base, device_index = _split_cuda_device(device)
    if base == "cuda":
        # WhisperX dùng faster-whisper/ctranslate2 bên dưới → cần cuBLAS/cuDNN
        # DLL trên PATH như nhánh faster_whisper, nếu không sẽ lỗi lúc encode.
        _register_cuda_dll_dirs()

    # WhisperX cần audio waveform thay vì path string thuần.
    audio = _load_audio_for_whisperx(audio_path, whisperx)
    compute_type = "float16" if base == "cuda" else "int8"

    # Nạp ASR model 1 lần rồi cache: biết trước language thì truyền vào để bỏ
    # bước tự nhận diện ngôn ngữ mỗi file (tốn thêm thời gian, log "language
    # will be detected").
    model_key = (model_size, device, compute_type)
    model = _whisperx_model_cache.get(model_key)
    if model is None:
        with _whisperx_load_lock:
            # Double-check: luồng khác có thể đã nạp xong khi ta chờ lock.
            model = _whisperx_model_cache.get(model_key)
            if model is None:
                logger.info(
                    "Đang nạp WhisperX model=%s device=%s index=%d compute_type=%s",
                    model_size,
                    base,
                    device_index,
                    compute_type,
                )
                model = whisperx.load_model(
                    model_size,
                    device=base,
                    device_index=device_index,
                    compute_type=compute_type,
                    language=language or None,
                    vad_method="silero",
                )
                _whisperx_model_cache[model_key] = model
    try:
        result = model.transcribe(audio, batch_size=batch_size, language=language)
    except IndexError:
        # Silero VAD không tìm thấy đoạn nói nào (clip quá ngắn/nhỏ tiếng — hay
        # gặp ở popup luyện 1 từ) → whisperx đưa list VAD-segment RỖNG vào
        # transformers pipeline và nổ IndexError (inputs[0]). Coi là "không nghe
        # thấy gì": trả transcript rỗng để tầng trên xử lý (reliability skip hết
        # → UI báo "chưa nghe rõ"), thay vì 500 cả request.
        logger.warning(
            "WhisperX: VAD không tìm thấy đoạn nói nào trong %s — trả transcript rỗng.",
            audio_path,
        )
        return Transcription(
            text="",
            words=[],
            language=language or "",
            duration=float(len(audio)) / 16000.0,
        )

    detected_lang = result.get("language") or language
    align_key = (detected_lang, device)
    cached_align = _whisperx_align_cache.get(align_key)
    if cached_align is None:
        with _whisperx_load_lock:
            cached_align = _whisperx_align_cache.get(align_key)
            if cached_align is None:
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=device,
                )
                _whisperx_align_cache[align_key] = (align_model, metadata)
            else:
                align_model, metadata = cached_align
    else:
        align_model, metadata = cached_align
    aligned = whisperx.align(
        result.get("segments", []),
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    words: list[Word] = []
    text_parts: list[str] = []
    for seg in aligned.get("segments", []):
        seg_text = (seg.get("text") or "").strip()
        if seg_text:
            text_parts.append(seg_text)
        for w in seg.get("words", []) or []:
            token = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            # Một số token alignment có thể không có mốc đầy đủ.
            if not token or start is None or end is None:
                continue
            words.append(
                Word(
                    text=token,
                    start=float(start),
                    end=float(end),
                    probability=float(w.get("score", 0.0) or 0.0),
                )
            )

    duration = 0.0
    if words:
        duration = words[-1].end
    elif aligned.get("segments"):
        duration = float(aligned["segments"][-1].get("end", 0.0) or 0.0)

    return Transcription(
        text=" ".join(p for p in text_parts if p).strip(),
        words=words,
        language=detected_lang,
        duration=float(duration),
    )


_FFMPEG_FALLBACK_DIRS = [r"C:\tools\ffmpeg\bin"] if sys.platform == "win32" else []


def _ensure_ffmpeg_on_path() -> bool:
    """Return True if ffmpeg is (or becomes) findable on PATH."""
    if shutil.which("ffmpeg"):
        return True
    for d in _FFMPEG_FALLBACK_DIRS:
        if shutil.which("ffmpeg", path=d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            logger.info("ffmpeg found at fallback dir %s; added to PATH.", d)
            return True
    return False


def _load_audio_for_whisperx(audio_path: str, whisperx_module) -> object:
    """Load audio for WhisperX, preferring ffmpeg CLI and falling back to torchaudio.

    WhisperX upstream hard-depends on `ffmpeg` executable. On Windows/local dev
    that binary is often missing even though Python packages are installed.
    """
    if _ensure_ffmpeg_on_path():
        return whisperx_module.load_audio(audio_path)

    logger.warning("Không tìm thấy ffmpeg; dùng torchaudio fallback cho WhisperX.")
    try:
        return _load_audio_with_torchaudio(audio_path)
    except Exception as torchaudio_err:  # noqa: BLE001 - thử đường decode khác cho container audio/video
        logger.warning(
            "Torchaudio không đọc được '%s' (%s). Thử PyAV fallback.",
            audio_path,
            torchaudio_err,
        )
        return _load_audio_with_pyav(audio_path)


def _load_audio_with_torchaudio(audio_path: str) -> object:
    """Load audio bằng torchaudio, resample về mono 16k float32."""
    try:
        import numpy as np
        import torch
        import torchaudio
    except ImportError as e:
        raise RuntimeError(
            "Thiếu torchaudio/numpy cho WhisperX fallback loader."
        ) from e

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != _WHISPER_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, _WHISPER_SAMPLE_RATE
        )
    # WhisperX expects mono float32 numpy array.
    mono = waveform.squeeze(0).to(torch.float32).contiguous().cpu().numpy()
    return mono.astype(np.float32, copy=False)


def _load_audio_with_pyav(audio_path: str) -> object:
    """Load audio bằng PyAV khi torchaudio không mở được container như m4a/mp4."""
    try:
        import av
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "Thiếu PyAV/numpy cho WhisperX fallback loader."
        ) from e

    try:
        container = av.open(audio_path)
    except FileNotFoundError:
        raise
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"PyAV không mở được audio: {e}") from e

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=_WHISPER_SAMPLE_RATE,
    )
    chunks: list[np.ndarray] = []
    with container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("File không có audio stream.")
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                chunks.append(arr.reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm16 = np.concatenate(chunks).astype(np.float32, copy=False)
    return pcm16 / 32768.0


def _resolve_ifw_model_id(model_size: str) -> str:
    """Ánh xạ model_size quen thuộc -> model id dùng cho transformers."""
    mapping = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v3",
        "large-v3": "openai/whisper-large-v3",
    }
    return mapping.get((model_size or "").lower(), "openai/whisper-small")


def _transcribe_insanely_fast_whisper(
    audio_path: str,
    model_size: str = "base",
    device: str = "cpu",
    language: str = "en",
) -> Transcription:
    """ASR fast lane theo phong cách Insanely Fast Whisper (transformers).

    Dùng pipeline ASR với attention tối ưu nếu môi trường hỗ trợ.
    """
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise RuntimeError(
            "Thiếu dependency cho insanely_fast_whisper (cần transformers + torch)."
        ) from e

    device = _resolve_torch_device(device, "insanely_fast_whisper")
    base, _device_index = _split_cuda_device(device)

    model_id = _resolve_ifw_model_id(model_size)
    key = (model_id, device)
    if key not in _ifw_pipe_cache:
        torch_dtype = torch.float16 if base == "cuda" else torch.float32
        # transformers pipeline nhận chuỗi torch device ('cuda:1') hoặc 'cpu'.
        device_arg: int | str = device if base == "cuda" else "cpu"
        logger.info(
            "Đang nạp IFW pipeline model=%s device=%s dtype=%s",
            model_id,
            device,
            torch_dtype,
        )
        _ifw_pipe_cache[key] = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device_arg,
        )

    asr_pipe = _ifw_pipe_cache[key]
    result = asr_pipe(
        audio_path,
        return_timestamps="word",
        generate_kwargs={"task": "transcribe", "language": language},
    )

    text = (result.get("text") or "").strip() if isinstance(result, dict) else ""
    chunks = result.get("chunks") if isinstance(result, dict) else None
    words: list[Word] = []
    for chunk in chunks or []:
        token = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp")
        if not token or not ts or len(ts) != 2:
            continue
        start, end = ts
        if start is None or end is None:
            continue
        words.append(
            Word(
                text=token,
                start=float(start),
                end=float(end),
                probability=float(chunk.get("score", 0.0) or 0.0),
            )
        )

    duration = float(words[-1].end) if words else 0.0
    return Transcription(
        text=text,
        words=words,
        language=language,
        duration=duration,
    )

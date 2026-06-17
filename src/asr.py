"""Speech-to-Text local bằng faster-whisper.

Trả về transcript đầy đủ + danh sách từ kèm mốc thời gian và độ tự tin
(logprob → probability). Mốc thời gian từng từ là dữ liệu gốc để
features.py tính tốc độ nói, quãng ngắt...
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field

logger = logging.getLogger("toeic.asr")


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


# Cache model trong process để khỏi nạp lại mỗi lần gọi.
_model_cache: dict[tuple[str, str], object] = {}


def _get_model(model_size: str, device: str):
    key = (model_size, device)
    if key not in _model_cache:
        if device == "cuda":
            _register_cuda_dll_dirs()
        # Import trong hàm để --no-ai và unit test không bắt buộc cài faster-whisper
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(
            "Đang nạp Whisper model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )
        _model_cache[key] = WhisperModel(
            model_size, device=device, compute_type=compute_type
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

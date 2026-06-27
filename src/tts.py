"""Tổng hợp giọng đọc mẫu (Piper TTS) cho tính năng "nghe phát âm đúng".

Sinh audio WAV cho một TỪ (hoặc cụm ngắn) bằng Piper — model neural TTS chạy
OFFLINE. Dùng cho nút 🔊 ở bảng lỗi phát âm (web/app.js): người học nghe giọng
chuẩn để so với chính mình. KHÁC với playWordSegment (phát lại Blob của chính
người học) — đây là audio THAM CHIẾU do máy đọc.

Thiết kế:
- API contract MỞ RỘNG ĐƯỢC: synthesize(text=..., ipa=..., accent=...). Hiện chỉ
  hiện thực nhánh `text`; nhánh `ipa` để sau (Piper nhận được phoneme nhưng chưa
  làm) → thêm phát-âm-theo-IPA về sau KHÔNG phải đổi chữ ký/route.
- Cache đĩa có VERSION trong key → đổi voice/normalization/format chỉ cần bump
  CACHE_VERSION, cache cũ tự "miss" mà không phải xoá tay.
- Voice load LAZY + cache theo process (giống Whisper trong asr): lần đầu mỗi voice
  mới nạp model, các lần sau dùng lại.
"""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import wave
from pathlib import Path

from .config import Config

logger = logging.getLogger("toeic.tts")

# Bump khi đổi voice mặc định, cách chuẩn hoá input, hoặc format output → cache cũ
# tự bị bỏ qua (key đổi) mà không cần xoá thủ công.
CACHE_VERSION = "v1"

# Trần độ dài text TTS (ký tự). Đây là mức từ/cụm ngắn, không phải câu — đủ rộng cho
# cụm nhiều từ nhưng vẫn chặn lạm dụng. (Endpoint cũng validate; đây là lớp thứ hai.)
MAX_TEXT_LEN = 100


class TtsUnavailable(RuntimeError):
    """Voice model thiếu / Piper chưa cài → không tổng hợp được (endpoint trả 503)."""


# Cache voice theo đường dẫn model (mỗi process). Lock để 2 request không nạp đôi.
_voices: dict[str, object] = {}
_voices_lock = threading.Lock()


def _resolve_accent(accent: str, config: Config) -> str:
    """Chuẩn hoá accent → 'us' | 'gb' (giọng CỤ THỂ để phát).

    Khác /grade: ở đó 'default' nghĩa "chấp nhận cả GB lẫn US" (điều biến tolerance),
    nhưng TTS bắt buộc phát MỘT giọng. 'default'/'auto' → theo TTS_DEFAULT_ACCENT
    (mặc định US, vì IPA tham chiếu CMUdict là giọng Mỹ → đồng bộ với IPA hiển thị).
    Giá trị lạ → US.
    """
    a = (accent or "").strip().lower()
    if a in ("", "default", "auto"):
        a = (config.tts_default_accent or "us").strip().lower()
    return "gb" if a == "gb" else "us"


def _voice_path(canonical: str, config: Config) -> str:
    """'us'|'gb' → đường dẫn file voice .onnx tương ứng (có thể rỗng = chưa cấu hình)."""
    return config.tts_voice_gb if canonical == "gb" else config.tts_voice_us


def voice_for_accent(accent: str, config: Config) -> str:
    """Tiện ích public: accent (default|us|gb) → đường dẫn voice model sẽ dùng."""
    return _voice_path(_resolve_accent(accent, config), config)


def _load_voice(model_path: str):
    """Nạp (lazy, cache theo process) một PiperVoice từ đường dẫn .onnx."""
    if not model_path:
        raise TtsUnavailable(
            "Chưa cấu hình voice TTS (đặt TTS_VOICE_US / TTS_VOICE_GB)."
        )
    cached = _voices.get(model_path)
    if cached is not None:
        return cached
    with _voices_lock:
        cached = _voices.get(model_path)  # double-check sau khi giành lock
        if cached is not None:
            return cached
        path = Path(model_path)
        if not path.is_file():
            raise TtsUnavailable(f"Không thấy voice model: {model_path}")
        try:
            from piper import PiperVoice  # import trễ: chỉ cần khi thực sự tổng hợp
        except ImportError as e:  # piper-tts chưa cài
            raise TtsUnavailable(
                "Chưa cài piper-tts (pip install piper-tts)."
            ) from e
        logger.info("Nạp Piper voice: %s", model_path)
        voice = PiperVoice.load(str(path))  # config .onnx.json tự nhận cạnh file
        _voices[model_path] = voice
        return voice


def _synthesize_wav_bytes(voice, text: str) -> bytes:
    """Tổng hợp `text` → WAV bytes. Bao quát thay đổi API giữa các phiên bản Piper."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        # Piper đổi API giữa các bản:
        # - <=1.2.x: voice.synthesize(text, wave.Wave_write)
        # - >=1.3.x: voice.synthesize_wav(text, wave.Wave_write)
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wav_file)
        else:
            voice.synthesize(text, wav_file)
    return buf.getvalue()


def synthesize(
    *,
    text: str | None = None,
    ipa: str | None = None,
    accent: str = "default",
    config: Config,
) -> bytes:
    """Trả WAV bytes của audio mẫu. Chỉ ĐƯỢC set MỘT trong `text` / `ipa`.

    Hiện chỉ hiện thực nhánh `text`; `ipa` để sau (chữ ký giữ ổn định để thêm
    phát-âm-theo-IPA mà không phải đổi route/contract). Cache đĩa có version.
    """
    if (text is None) == (ipa is None):
        raise ValueError("Cần đúng MỘT trong 'text' hoặc 'ipa'.")
    if ipa is not None:
        raise NotImplementedError("Tổng hợp theo IPA chưa hỗ trợ (để sau).")

    norm = " ".join(text.split())  # chuẩn hoá khoảng trắng
    if not norm:
        raise ValueError("'text' rỗng.")

    canonical = _resolve_accent(accent, config)
    model_path = _voice_path(canonical, config)

    # Key cache: version + giọng + tên voice + loại input + text (case-sensitive để
    # không gộp nhầm các biến thể hoa/thường). Đổi bất kỳ phần nào → key khác → miss.
    voice_name = Path(model_path).name or canonical
    key_src = f"{CACHE_VERSION}:{canonical}:{voice_name}:text:{norm}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
    cache_dir = Path(config.tts_cache_dir) / "tts"
    cache_file = cache_dir / f"{key}.wav"
    if cache_file.is_file():
        return cache_file.read_bytes()

    voice = _load_voice(model_path)
    wav = _synthesize_wav_bytes(voice, norm)

    # Ghi atomic (tmp rồi replace) để request đua nhau không đọc phải file rách.
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_name(cache_file.name + ".tmp")
    tmp.write_bytes(wav)
    tmp.replace(cache_file)
    return wav

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
import unicodedata
import wave
from pathlib import Path

from .config import Config

logger = logging.getLogger("toeic.tts")

# Bump khi đổi voice mặc định, cách chuẩn hoá input, hoặc format output → cache cũ
# tự bị bỏ qua (key đổi) mà không cần xoá thủ công.
# v3: thêm noise_scale=0.0 (tất định hoàn toàn) → bỏ các WAV cache-v2 đã trúng draw
# xấu làm phụ âm đầu (vd "store" → gần như mất /s/) bị đóng băng vĩnh viễn.
# v4: đổi voice US lessac→amy (lessac nuốt hẳn /s/ đầu cụm s+stop; amy phát rõ) →
# bỏ cache lessac cũ. Phải giữ khớp TTS_AUDIO_VERSION ở
# frontend/src/features/grading/playback.ts.
# v5: thêm dấu "." cuối khi text không có dấu câu kết thúc. Không có dấu câu, VITS
# thiếu "mỏ neo" cuối phát ngôn → từ đơn bị đọc lem: "at" thành /hɛ/ ("he"), "cat"
# mất /k/, "student" mất /s,t/ (đo bằng Whisper + wav2vec trên 16 từ; thêm "."
# sửa gần như tất cả). Cách khác đã thử và BỎ: chèn ˈ vào phoneme rồi
# phoneme_ids_to_audio (ra /aɪ/, tệ hơn), "!" (sửa "at" nhưng làm "stop"/"for" lem).
# v6: thêm nhánh `ipa` (đọc audio mẫu TỪ IPA hiển thị, không để Piper G2P từ chữ).
# Chỉ ẢNH HƯỞNG key khi kind="ipa"; kind="text" giữ nguyên payload v5 nên WAV text
# cũ vẫn TRÚNG cache (không sinh lại). Xem _ipa_to_phoneme_tokens: chuyển IPA hiển
# thị → token phoneme espeak (r→ɹ, e→ɛ trừ 'eɪ', g→ɡ) rồi tái dùng phonemes_to_ids
# + phoneme_ids_to_audio của Piper — KHÁC hẳn cách v5 đã bỏ (đó là chèn ˈ vào rồi
# tổng hợp không chuẩn hoá ký hiệu). Nhánh ipa nằm sau flag TTS_IPA_SYNTH.
# v7: map ɝ→ɜ (NURSE r-hóa giọng Mỹ, Cambridge us_ipa) trong _ipa_to_phoneme_tokens
# — trước đây ɝ bị bỏ, "church" /tʃɝːtʃ/ mất nguyên âm đọc thành "ch-ch". Bump để bust
# audio ɝ hỏng đang cache ở client (URL /tts?...&v= gắn version). Phải khớp
# TTS_AUDIO_VERSION ở frontend/src/features/grading/playback.ts.
CACHE_VERSION = "v7"

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
    """Tổng hợp `text` → WAV bytes. Bao quát thay đổi API giữa các phiên bản Piper.

    noise_w_scale=0.0: tắt nhiễu ngẫu nhiên ở bộ dự đoán trường độ (duration
    predictor) của VITS. Mặc định có nhiễu → thỉnh thoảng gán trường độ ~0 cho
    phụ âm đầu từ (đo được: "starting"/"studying" im lặng 40-80ms đầu, "s" gần
    như biến mất), và vì synthesize() cache WAV xuống đĩa vĩnh viễn nên 1 lần
    "trúng" draw xấu sẽ làm từ đó bị lỗi mãi cho mọi người dùng.

    noise_scale=0.0: tắt NỐT nhiễu âm học của bộ sinh (flow). Chỉ zero noise_w
    chưa đủ tất định — noise_scale mặc định 0.667 vẫn randomize biên độ, nên phụ
    âm xát đầu (/s/ trong cụm s+phụ âm: store/start/stop) vẫn dao động draw-to-draw
    từ ~0.15 đến ~0.8 lần biên độ nguyên âm; bản yếu bị cache đóng băng → nghe mất
    /s/ ("store" thành "tɔːr"). Zero cả hai → mỗi từ ra ĐÚNG một bản, /s/ luôn đủ
    to (đo: store 0.74, start 0.97, stop 0.91 s/peak, ổn định mọi lần). Đánh đổi:
    giọng đơn điệu hơn — chấp nhận được cho audio mẫu 1 từ (ưu tiên phụ âm rõ +
    nhất quán). Verify bằng lặp lại nhiều lần trên store/start/stop/student.
    """
    from piper.config import SynthesisConfig

    syn_config = SynthesisConfig(noise_w_scale=0.0, noise_scale=0.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        # Piper đổi API giữa các bản:
        # - <=1.2.x: voice.synthesize(text, wave.Wave_write)
        # - >=1.3.x: voice.synthesize_wav(text, wave.Wave_write)
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        else:
            # API cũ (<=1.2.x) không nhận SynthesisConfig — giữ nguyên hành vi cũ.
            voice.synthesize(text, wav_file)
    return buf.getvalue()


# Thay ký hiệu IPA hiển thị (CMUdict/Cambridge, giọng Mỹ) → ký hiệu phoneme mà bảng
# espeak-ng của Piper mong đợi. CHỈ những thay đổi làm SAI hẳn âm nếu để nguyên:
# - 'r' (U+0072): ARPABET R → hiển thị dùng 'r', nhưng trong espeak 'r' là rung lưỡi
#   (rr Tây Ban Nha); phụ âm gần đúng của tiếng Anh là 'ɹ'.
# - 'g' (U+0067): script-g IPA là 'ɡ' (U+0261); 'g' Latin không có trong bảng.
# - ':' ASCII: dấu kéo dài phải là 'ː' (U+02D0).
# - 'ɝ' (U+025D): nguyên âm NURSE r-hóa có nhấn (giọng Mỹ, ví dụ church /tʃɝːtʃ/).
#   Bảng espeak của cả amy(US) lẫn alan(GB) KHÔNG có 'ɝ' (chỉ có 'ɜ' + 'ɚ') → nếu để
#   nguyên thì bị bỏ hẳn, nguyên âm biến mất ("church" đọc thành "ch-ch"). Map về 'ɜ'
#   (base NURSE espeak thật có). 'ɚ' (schwa r-hóa không nhấn) đã có sẵn trong bảng.
# KHÔNG "sửa" các khác biệt thuần phong cách (ɚ↔əɹ, ɜː độ rhotic…) vì mục tiêu là
# đọc ĐÚNG IPA đang hiển thị cho người học, không phải bản espeak-native.
_IPA_CHAR_SUBS = {"r": "ɹ", "g": "ɡ", ":": "ː", "ɝ": "ɜ"}

# Ký tự bao/ngăn cách KHÔNG phải phoneme — bỏ trước khi map (dấu /…/, ngoặc, dấu
# chấm ngắt âm tiết, nửa-dài ˑ, tie-bar). Dấu nhấn ˈ/ˌ GIỮ (Piper có trong bảng).
_IPA_DROP_CHARS = set("/[]()|‿͡ˑ.,;! \t\r\n")
_IPA_STRESS = ("ˈ", "ˌ")
# Nguyên âm IPA (để dời dấu nhấn về đúng trước nhân âm tiết — xem bên dưới).
_IPA_VOWELS = set("aeiouæɐɑɒɔəɘɛɜɞɤɨɪɯɵøœʉʊʌʏy")


def _ipa_to_phoneme_tokens(
    ipa: str, phoneme_id_map: dict
) -> tuple[list[str], list[str]]:
    """IPA hiển thị → (tokens phoneme espeak giữ được, tokens bị bỏ vì ngoài bảng).

    Char-by-char (mọi ký hiệu espeak của Piper đều là 1 codepoint, kể cả 'ː' tách
    riêng), có 1 ngoại lệ ngữ cảnh: 'e' đứng trước 'ɪ' là nguyên âm đôi FACE 'eɪ' →
    giữ 'e'; 'e' còn lại là DRESS /ɛ/ (ARPABET EH hiển thị bằng 'e') → 'ɛ', nếu để
    'e' Piper đọc thành nguyên âm căng hơn (jerry/prepared/measured sai rõ).

    Hai chuẩn hoá prosody bắt buộc (bench scripts/bench_tts_ipa.py: agreement với bộ
    nhận diện phoneme 0.61 → 0.87, vượt cả nhánh text 0.72):
    - Dời dấu nhấn ˈ/ˌ về NGAY TRƯỚC nguyên âm kế tiếp. IPA hiển thị của app đặt nhấn
      ở ĐẦU âm tiết (onset cụm phụ âm) qua place_stress_at_onset; espeak — thứ Piper
      được huấn luyện — đặt nhấn ngay trước nhân âm tiết. Sai vị trí → VITS méo nguyên
      âm (advantage 1.0→0.56). KHÔNG bỏ hẳn nhấn (mất prosody tự nhiên).
    - Thêm '.' cuối (mỏ neo phát ngôn): thiếu → nguyên âm/phụ âm biên sụp như nhánh
      text không dấu câu (xem CACHE_VERSION v5). '.' là token pause trong bảng.
    """
    s = unicodedata.normalize("NFC", (ipa or "").strip())
    chars = list(s)
    n = len(chars)
    tokens: list[str] = []
    i = 0
    while i < n:
        ch = chars[i]
        if ch in _IPA_DROP_CHARS:
            i += 1
            continue
        if ch == "e":
            nxt = chars[i + 1] if i + 1 < n else ""
            tokens.append("e" if nxt == "ɪ" else "ɛ")  # eɪ = FACE, còn lại = ɛ
            i += 1
            continue
        tokens.append(_IPA_CHAR_SUBS.get(ch, ch))
        i += 1

    # Dời dấu nhấn về trước nguyên âm kế tiếp (đệm dấu nhấn tới khi gặp nhân âm tiết).
    repositioned: list[str] = []
    pending_stress: str | None = None
    for t in tokens:
        if t in _IPA_STRESS:
            pending_stress = t  # giữ dấu nhấn mới nhất (bỏ dấu thừa liên tiếp)
            continue
        if pending_stress and t[:1] in _IPA_VOWELS:
            repositioned.append(pending_stress)
            pending_stress = None
        repositioned.append(t)
    # pending_stress còn lại = nhấn không có nguyên âm theo sau (bất thường) → bỏ.

    kept = [t for t in repositioned if t in phoneme_id_map]
    dropped = [t for t in repositioned if t not in phoneme_id_map]
    # Mỏ neo phát ngôn cuối — CHỈ khi có phoneme thật (giữ guard "IPA không map được
    # phoneme nào" ở synthesize: '.' đơn độc không tính là đọc được gì).
    if any(t not in _IPA_STRESS for t in kept):
        kept.append(".")
    return kept, dropped


def _synthesize_ipa_wav_bytes(voice, phonemes: list[str]) -> bytes:
    """Tổng hợp WAV từ danh sách token phoneme (đã map sang bảng espeak của voice).

    Tái dùng ĐÚNG phonemes_to_ids + phoneme_ids_to_audio mà nhánh text đi qua (Piper
    tự chèn BOS/EOS/PAD) nên khác biệt duy nhất so với text là NGUỒN phoneme. Hậu xử
    lý (normalize biên độ → volume → clip → int16) sao chép nguyên si vòng lặp trong
    PiperVoice.synthesize để biên độ khớp nhánh text. noise=0 cho tất định (xem
    _synthesize_wav_bytes).
    """
    import numpy as np
    from piper.config import SynthesisConfig

    syn_config = SynthesisConfig(noise_w_scale=0.0, noise_scale=0.0)
    ids = voice.phonemes_to_ids(phonemes)
    audio = voice.phoneme_ids_to_audio(ids, syn_config)
    if isinstance(audio, tuple):  # (audio, alignments) khi include_alignments
        audio = audio[0]
    audio = np.asarray(audio, dtype=np.float32)
    if syn_config.normalize_audio:
        max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
        audio = audio / max_val if max_val >= 1e-8 else np.zeros_like(audio)
    if syn_config.volume != 1.0:
        audio = audio * syn_config.volume
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(voice.config.sample_rate))
        wav_file.writeframes(pcm.tobytes())
    return buf.getvalue()


def synthesize(
    *,
    text: str | None = None,
    ipa: str | None = None,
    accent: str = "default",
    config: Config,
) -> bytes:
    """Trả WAV bytes của audio mẫu. Chỉ ĐƯỢC set MỘT trong `text` / `ipa`.

    Nhánh `text`: Piper G2P từ chữ viết (như cũ). Nhánh `ipa`: đọc ĐÚNG chuỗi IPA
    tham chiếu đang hiển thị (khớp homograph / dạng trích dẫn / viết tắt). Cache đĩa
    có version; kind ('text'|'ipa') nằm trong key nên hai nhánh không đụng nhau.

    Nhánh `ipa` raise ValueError khi chuỗi IPA rỗng/không map được token nào — caller
    (endpoint /tts) bắt để fallback sang `text`.
    """
    if (text is None) == (ipa is None):
        raise ValueError("Cần đúng MỘT trong 'text' hoặc 'ipa'.")

    canonical = _resolve_accent(accent, config)
    model_path = _voice_path(canonical, config)
    voice_name = Path(model_path).name or canonical
    cache_dir = Path(config.tts_cache_dir) / "tts"

    if ipa is not None:
        # Nạp voice TRƯỚC (cần bảng phoneme để chuẩn hoá + để tính key ổn định theo
        # token đã map, không theo chuỗi thô). Voice thiếu → TtsUnavailable như text.
        voice = _load_voice(model_path)
        phoneme_id_map = voice.config.phoneme_id_map
        tokens, dropped = _ipa_to_phoneme_tokens(ipa, phoneme_id_map)
        if not tokens:
            raise ValueError(f"IPA không map được phoneme nào: {ipa!r}")
        if dropped:
            logger.warning("TTS IPA bỏ token ngoài bảng %s (từ %r)", dropped, ipa)
        payload = " ".join(tokens)
        key_src = f"{CACHE_VERSION}:{canonical}:{voice_name}:ipa:{payload}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
        cache_file = cache_dir / f"{key}.wav"
        if cache_file.is_file():
            return cache_file.read_bytes()
        wav = _synthesize_ipa_wav_bytes(voice, tokens)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".tmp")
        tmp.write_bytes(wav)
        tmp.replace(cache_file)
        return wav

    norm = " ".join(text.split())  # chuẩn hoá khoảng trắng
    if not norm:
        raise ValueError("'text' rỗng.")

    # Tiếng Hàn: chưa cấu hình voice Piper tiếng Hàn → 503 (TtsUnavailable) thay vì
    # để voice EN "đánh vần" Hangul thành rác rồi cache vĩnh viễn. Frontend
    # (playback.js) tự phát hiện Hangul và fallback Web Speech API ko-KR nên nhánh
    # này chỉ chặn client cũ/gọi tay.
    if any("가" <= ch <= "힣" for ch in norm):
        raise TtsUnavailable(
            "Chưa có voice TTS tiếng Hàn trên server — dùng giọng đọc của trình duyệt."
        )

    # Bảo đảm có dấu câu KẾT THÚC: từ/cụm trần (không ".!?") bị VITS đọc dạng
    # "giữa câu" — nguyên âm sụp + artifact /h/ đầu ("at" → nghe thành "he").
    # Thêm "." cho ngữ điệu hạ tự nhiên. Đặt TRƯỚC khi tính key cache để
    # "at" và "at." gộp chung một key (cùng một audio).
    if norm[-1] not in ".!?":
        norm += "."

    # Key cache: version + giọng + tên voice + loại input + text (case-sensitive để
    # không gộp nhầm các biến thể hoa/thường). Đổi bất kỳ phần nào → key khác → miss.
    key_src = f"{CACHE_VERSION}:{canonical}:{voice_name}:text:{norm}"
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
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

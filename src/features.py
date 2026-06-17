"""Trích đặc trưng khách quan từ kết quả ASR (KHÔNG dùng AI).

Gồm: tốc độ nói, quãng ngắt, từ đệm, và (cho Read Aloud) so sánh transcript
với script tham chiếu để ra WER + chi tiết substitutions/insertions/deletions.

LƯU Ý QUAN TRỌNG: avg_word_probability (logprob của Whisper) CHỈ là tín hiệu
phụ, KHÔNG phải thước đo phát âm — nó bị nhiễu bởi mic/giọng/tạp âm. Việc
chấm phát âm do LLM quyết định, confidence chỉ là evidence bổ trợ.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .asr import Transcription
from .config import FEATURES_VERSION

# Ngưỡng coi là một quãng ngắt (giây) giữa hai từ liên tiếp.
PAUSE_THRESHOLD_SEC = 0.3

# Danh sách từ đệm/ấp úng thường gặp.
FILLER_WORDS = {
    "um", "uh", "er", "ah", "erm", "hmm", "mhm",
    "like", "yeah", "you know", "i mean", "sort of", "kind of",
}


@dataclass
class AccuracyMetrics:
    """Kết quả so sánh transcript với script tham chiếu (Read Aloud)."""
    wer: float
    substitutions: int
    insertions: int
    deletions: int
    hits: int
    reference_word_count: int


@dataclass
class Features:
    speech_rate_wpm: float          # số từ / phút (trên thời lượng nói)
    word_count: int
    speaking_duration_sec: float    # thời lượng có nói (từ đầu tới cuối)
    audio_duration_sec: float       # thời lượng audio
    silence_sec: float              # phần im lặng ước tính
    pause_count: int
    total_pause_sec: float
    longest_pause_sec: float
    filler_count: int
    # Tín hiệu phụ — KHÔNG dùng làm điểm phát âm trực tiếp
    avg_word_probability: float
    min_word_probability: float
    # Chỉ có với Read Aloud
    accuracy_metrics: AccuracyMetrics | None = None
    features_version: str = field(default=FEATURES_VERSION)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _normalize_tokens(text: str) -> list[str]:
    """Token hóa đơn giản: lowercase, bỏ dấu câu, tách theo khoảng trắng."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return [t for t in text.split() if t]


def _count_fillers(text: str) -> int:
    lower = text.lower()
    count = 0
    for filler in FILLER_WORDS:
        if " " in filler:
            count += len(re.findall(rf"\b{re.escape(filler)}\b", lower))
        else:
            count += len(re.findall(rf"\b{re.escape(filler)}\b", lower))
    return count


def _compute_accuracy(reference_script: str, hypothesis: str) -> AccuracyMetrics | None:
    ref_tokens = _normalize_tokens(reference_script)
    hyp_tokens = _normalize_tokens(hypothesis)
    if not ref_tokens:
        return None

    try:
        import jiwer

        out = jiwer.process_words(
            [" ".join(ref_tokens)], [" ".join(hyp_tokens)]
        )
        return AccuracyMetrics(
            wer=round(float(out.wer), 4),
            substitutions=int(out.substitutions),
            insertions=int(out.insertions),
            deletions=int(out.deletions),
            hits=int(out.hits),
            reference_word_count=len(ref_tokens),
        )
    except ImportError:
        # jiwer không bắt buộc — fallback: chỉ tính WER thô bằng Levenshtein từ.
        s, i, d, h = _levenshtein_words(ref_tokens, hyp_tokens)
        wer = (s + i + d) / max(1, len(ref_tokens))
        return AccuracyMetrics(
            wer=round(wer, 4),
            substitutions=s,
            insertions=i,
            deletions=d,
            hits=h,
            reference_word_count=len(ref_tokens),
        )


def _levenshtein_words(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Levenshtein cấp độ từ với backtrace → (sub, ins, del, hits)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution / match
            )
    # backtrace
    i, j = n, m
    sub = ins = dele = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dele += 1
            i -= 1
    return sub, ins, dele, hits


def extract_features(
    transcription: Transcription,
    reference_script: str | None = None,
) -> Features:
    words = transcription.words

    if not words:
        return Features(
            speech_rate_wpm=0.0,
            word_count=0,
            speaking_duration_sec=0.0,
            audio_duration_sec=round(transcription.duration, 3),
            silence_sec=round(transcription.duration, 3),
            pause_count=0,
            total_pause_sec=0.0,
            longest_pause_sec=0.0,
            filler_count=0,
            avg_word_probability=0.0,
            min_word_probability=0.0,
            accuracy_metrics=_compute_accuracy(reference_script, "")
            if reference_script
            else None,
        )

    speaking_start = words[0].start
    speaking_end = words[-1].end
    speaking_duration = max(1e-6, speaking_end - speaking_start)

    # Quãng ngắt: khoảng trống giữa các từ liên tiếp vượt ngưỡng.
    pauses: list[float] = []
    for prev, cur in zip(words, words[1:]):
        gap = cur.start - prev.end
        if gap > PAUSE_THRESHOLD_SEC:
            pauses.append(gap)

    probs = [w.probability for w in words]
    speech_rate = len(words) / speaking_duration * 60.0

    return Features(
        speech_rate_wpm=round(speech_rate, 1),
        word_count=len(words),
        speaking_duration_sec=round(speaking_duration, 3),
        audio_duration_sec=round(transcription.duration, 3),
        silence_sec=round(max(0.0, transcription.duration - speaking_duration), 3),
        pause_count=len(pauses),
        total_pause_sec=round(sum(pauses), 3),
        longest_pause_sec=round(max(pauses), 3) if pauses else 0.0,
        filler_count=_count_fillers(transcription.text),
        avg_word_probability=round(sum(probs) / len(probs), 4),
        min_word_probability=round(min(probs), 4),
        accuracy_metrics=_compute_accuracy(reference_script, transcription.text)
        if reference_script
        else None,
    )

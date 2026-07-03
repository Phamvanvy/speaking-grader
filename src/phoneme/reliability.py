"""Recognition Reliability — tầng ĐỘC LẬP đứng TRÊN phoneme scorer.

Trả lời câu hỏi A ("có nên tin recognizer cho từ này không?") TÁCH KHỎI câu hỏi B
("nếu tin thì learner phát âm thế nào?" — việc của scorer). Tầng này:

  - THUẦN: chỉ biết danh sách từ tham chiếu (theo thứ tự chuẩn) + bằng chứng nhận
    dạng (RecognizerEvidence). KHÔNG import WordPronunciation / PhonemePoint / DTW /
    features.AccuracyMetrics.
  - Quyết định KHÔNG bao giờ suy từ DTW match-ratio / phoneme_similarity / penalty
    (sẽ là circular: scorer tự chấm chính nó).
  - Bằng chứng dùng để skip là CROSS-SOURCE (independently-derived): so transcript
    của recognizer với SCRIPT đã biết. (Gọi "cross-source" chứ không "fully
    independent" vì một assessment cũng có thể dùng transcript.)
  - Kết quả keyed theo CHỈ SỐ TỪ THAM CHIẾU (occurrence) — KHÔNG theo chuỗi từ:
    "the" xuất hiện nhiều lần, key theo chuỗi sẽ skip mọi "the".

PR1 chỉ phát 1 reason: WHISPER_MISMATCH. Các tín hiệu recognizer-internal (coverage,
confidence, repetition, duration) là TELEMETRY (PR2), KHÔNG phải đầu vào quyết định.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("toeic.phoneme.reliability")

# Tokenizer DÙNG CHUNG với text_to_ipa_sequence_with_spans (ipa/__init__.py) — phải
# giống hệt để mapping token ↔ span word là 1-1 theo cấu trúc, không cần fuzzy.
_WORD_TOKEN_RE = re.compile(r"[a-zA-Z'-]+")

# Ngưỡng SequenceMatcher.ratio để coi 1 từ substituted là "lệch lớn" → skip.
# Ratio cao (mountains→mountain ≈0.94) giữ lại chấm; thấp (Son Tinh→Andy ≈0) skip.
DEFAULT_SKIP_RATIO: float = 0.6


class SkipReason(str, Enum):
    """Lý do 1 từ bị coi là không đáng tin để chấm phoneme. Mở rộng dần (PR2+)."""

    WHISPER_MISMATCH = "whisper_mismatch"
    # Free-speech (reference == transcript): chính Whisper không chắc đã nghe đúng
    # từ này → reference không đáng tin → không chấm phoneme trên nó.
    ASR_LOW_CONFIDENCE = "asr_low_confidence"
    # Free-speech: IPA của từ lấy từ eSpeak G2P (OOV/tên riêng) — cả transcript lẫn
    # IPA chuẩn đều là đoán → không chấm.
    OOV_ESPEAK = "oov_espeak"


@dataclass(frozen=True)
class SkipDecision:
    """Quyết định bỏ qua 1 TỪ tham chiếu cụ thể (theo occurrence).

    KHÔNG mang trường `word`: suy được từ reference_words[word_index]; giữ cả hai dễ
    lệch (word_index=17 nhưng word="traditional" trong khi reference_words[17]="story").
    """

    word_index: int       # vị trí chuẩn trong reference word sequence (occurrence-specific)
    reason: SkipReason


@dataclass(frozen=True)
class RecognizerEvidence:
    """Bằng chứng nhận dạng độc lập (raw output của recognizer), KHÔNG gắn Azure/jiwer.

    AccuracyMetrics chỉ là MỘT adapter điền vào đây; đổi WhisperX/Google chỉ đổi
    adapter, không đổi tầng reliability. `recognized_words` là transcript đã tách từ.
    """

    recognized_words: tuple[str, ...]

    @classmethod
    def from_transcript(cls, transcript: str) -> RecognizerEvidence:
        """Tách transcript thành danh sách từ (lowercase, bỏ dấu câu)."""
        return cls(tuple(re.findall(r"[a-z0-9']+", (transcript or "").lower())))


def assess_reliability(
    reference_words: list[str],
    evidence: RecognizerEvidence,
    *,
    skip_ratio: float = DEFAULT_SKIP_RATIO,
) -> Mapping[int, SkipDecision]:
    """So reference_words ↔ transcript recognizer → các từ KHÔNG đáng tin (skip).

    Dùng difflib.SequenceMatcher trên 2 DANH SÁCH TỪ → opcodes cho ĐÚNG chỉ số từ
    tham chiếu (occurrence-correct, khác hẳn key theo chuỗi). Quy tắc:
      - 'delete' (từ script không có trong transcript) → recognizer không nghe ra → skip.
      - 'replace' (script ↔ từ khác): với mỗi từ script, lấy ratio tốt nhất so với
        các từ transcript trong block; ratio < skip_ratio → lệch lớn → skip
        (Son Tinh→Andy skip; mountains→mountain giữ lại).
      - 'insert' (transcript thừa từ) → không có từ script tương ứng → bỏ qua.

    Trả về Mapping bất biến {word_index: SkipDecision} (lookup O(1)).
    """
    ref = [w.lower() for w in reference_words]
    hyp = list(evidence.recognized_words)
    skips: dict[int, SkipDecision] = {}
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "insert"):
            continue
        if tag == "delete":
            for i in range(i1, i2):
                skips[i] = SkipDecision(i, SkipReason.WHISPER_MISMATCH)
        elif tag == "replace":
            for i in range(i1, i2):
                best = max(
                    (
                        difflib.SequenceMatcher(None, ref[i], hyp[j]).ratio()
                        for j in range(j1, j2)
                    ),
                    default=0.0,
                )
                if best < skip_ratio:
                    skips[i] = SkipDecision(i, SkipReason.WHISPER_MISMATCH)
    return skips


def assess_asr_confidence(
    reference_words: list[str],
    transcript_words: list[tuple[str, float]],
    *,
    min_probability: float,
    transcript_text: str | None = None,
) -> Mapping[int, SkipDecision]:
    """Free-speech gate: từ mà chính Whisper không chắc → reference không đáng tin → skip.

    CHỈ dùng khi reference == transcript (free-speech). Khi đó reference_words được
    dựng từ CHÍNH transcript_text, nên mapping là THUẦN cấu trúc, KHÔNG fuzzy
    (deterministic — đây là đường ảnh hưởng trực tiếp điểm):

      1. Gán probability theo KÝ TỰ trên transcript_text: định vị từng Whisper word
         trong text bằng str.find với con trỏ ĐƠN ĐIỆU (exact substring, theo thứ tự
         phát sinh — không fuzzy). Xử lý được cả 2 chiều lệch tokenization:
         1 Whisper word sinh nhiều token ("aehelp.com's" → "aehelp" + "com's") LẪN
         nhiều Whisper word gộp thành 1 token ("o'" + "clock" → "o'clock").
         Word không tìm thấy trong text → ký tự của nó giữ prob 0 ("unknown") —
         conservative, không bao giờ skip vì thiếu số liệu.
      2. Tokenize transcript_text bằng đúng regex của text_to_ipa_sequence_with_spans
         (`[a-zA-Z'-]+`) → prob của token = MIN prob các ký tự được phủ (>0); token
         không có ký tự nào được phủ → unknown, không skip.
      3. reference_words (từ spans) là SUBSEQUENCE của chuỗi token trên (từ không tra
         được IPA bị drop khỏi spans) và cùng sinh từ MỘT chuỗi text → khớp bằng con
         trỏ tiến tuần tự, so sánh chuỗi CHÍNH XÁC. Lệch (không thể xảy ra theo cấu
         trúc) → log warning và trả RỖNG: không gate, không rơi về fuzzy.

    Quy tắc skip: 0 < probability < min_probability. probability <= 0 nghĩa là
    "không có số liệu" (whisperx thiếu score, insanely_fast_whisper luôn 0.0) →
    KHÔNG BAO GIỜ skip. min_probability <= 0 → tắt gate (trả rỗng).

    transcript_text=None (tiện cho test): dựng lại bằng " ".join các Whisper word.
    Production LUÔN truyền transcription.text (chuỗi đã dựng reference spans).

    Trả về Mapping {span_index: SkipDecision} — cùng keying với assess_reliability.
    """
    if min_probability <= 0 or not reference_words or not transcript_words:
        return {}
    if transcript_text is None:
        transcript_text = " ".join(t for t, _p in transcript_words)
    if not transcript_text.strip():
        return {}

    # [1] Prob theo ký tự: định vị từng Whisper word bằng con trỏ đơn điệu.
    char_prob = [0.0] * len(transcript_text)
    cursor = 0
    for wtext, prob in transcript_words:
        wt = wtext.strip()
        if not wt:
            continue
        pos = transcript_text.find(wt, cursor)
        if pos < 0:
            continue  # word không thấy trong text → để unknown (không skip oan)
        for i in range(pos, pos + len(wt)):
            char_prob[i] = prob
        cursor = pos + len(wt)

    # [2] Prob theo token: MIN các ký tự được phủ; không phủ → unknown (0.0).
    tokens: list[tuple[str, float]] = []
    for m in _WORD_TOKEN_RE.finditer(transcript_text):
        covered = [p for p in char_prob[m.start():m.end()] if p > 0]
        tokens.append((m.group(), min(covered) if covered else 0.0))

    # [3] reference_words là subsequence của tokens → con trỏ tiến tuần tự.
    skips: dict[int, SkipDecision] = {}
    t = 0
    for k, word in enumerate(reference_words):
        while t < len(tokens) and tokens[t][0] != word:
            t += 1
        if t >= len(tokens):
            logger.warning(
                "assess_asr_confidence: reference word %r (index %d) không khớp "
                "token nào của transcript — cấu trúc lệch, bỏ gate (không skip từ nào)",
                word, k,
            )
            return {}
        prob = tokens[t][1]
        t += 1
        if 0 < prob < min_probability:
            skips[k] = SkipDecision(k, SkipReason.ASR_LOW_CONFIDENCE)
    return skips

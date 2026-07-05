"""Chunk audio theo Whisper word timestamps — chạy TRƯỚC wav2vec, KHÔNG đụng scoring.

Vì sao tồn tại: wav2vec xlsr-53 train trên utterance ngắn (~15s crop); đưa cả bài
thi 11 phút vào MỘT forward pass làm chính posterior/CTC output suy giảm cục bộ
(IPA "lem" — xem outputs/debug_full_vs_sentence/comparison.json, 2026-07-04:
raw segments cùng khoảng thời gian tuyệt đối khác hẳn giữa full-run và slice-run,
median agreement 0.56). Fix: chia audio thành chunk theo khoảng lặng/câu từ Whisper
word timestamps, predict từng chunk rồi cộng offset — scoring phía sau bất biến.

Module này THUẦN + deterministic: chỉ tính toán trên (text, start, end) đã có,
không I/O, không phụ thuộc torch/whisper. Predictor (wav2vec_backend) nhận spans
và tự cắt waveform.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("toeic.phoneme.chunking")

# Các strategy hợp lệ; "off" do caller xử lý (không gọi vào đây).
CHUNKING_STRATEGIES: frozenset[str] = frozenset({"pause", "hybrid"})

# Mặc định đồng bộ với Config (config.py đọc env rồi truyền vào đây).
DEFAULT_MAX_CHUNK_SEC: float = 30.0
DEFAULT_MIN_PAUSE_SEC: float = 0.5
DEFAULT_PAD_SEC: float = 0.25

# Ký tự đóng câu; strip các dấu bao (quote/ngoặc) trước khi xét — Whisper word
# text giữ nguyên dấu câu ("videos.", 'said."').
_SENTENCE_FINAL = (".", "?", "!")
_CLOSING_WRAPPERS = "\"'”’)]}»"


def _ends_sentence(text: str) -> bool:
    """True nếu word text kết thúc câu (sau khi bỏ dấu bao đóng)."""
    return (text or "").rstrip(_CLOSING_WRAPPERS).endswith(_SENTENCE_FINAL)


def _group_by_pause(
    words: list[tuple[str, float, float]], min_pause_sec: float
) -> list[list[tuple[str, float, float]]]:
    """Chia words thành nhóm liên tiếp, cắt tại gap ≥ min_pause_sec."""
    groups: list[list[tuple[str, float, float]]] = [[words[0]]]
    for prev, cur in zip(words, words[1:]):
        if cur[1] - prev[2] >= min_pause_sec:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    return groups


def _split_long_group(
    group: list[tuple[str, float, float]],
    max_chunk_sec: float,
    use_punctuation: bool,
) -> list[list[tuple[str, float, float]]]:
    """Cắt 1 nhóm dài hơn max_chunk_sec thành các nhóm con ≤ max (trừ khi 1 từ
    đơn lẻ đã dài hơn max — giữ nguyên, không thể cắt nhỏ hơn ranh giới từ).

    Greedy trái→phải: tìm từ xa nhất còn giữ nhóm con ≤ max; nếu use_punctuation,
    ưu tiên lùi về ranh giới CÂU gần nhất trong phạm vi đó (tầng 2 của hybrid);
    không có thì hard-cut tại chính từ đó (tầng 3).
    """
    out: list[list[tuple[str, float, float]]] = []
    start = 0
    n = len(group)
    while start < n:
        t0 = group[start][1]
        # limit = từ XA NHẤT sao cho [start..limit] vẫn ≤ max_chunk_sec.
        limit = start
        for i in range(start, n):
            if group[i][2] - t0 <= max_chunk_sec:
                limit = i
            else:
                break
        if limit == n - 1:
            out.append(group[start:])
            break
        cut = limit
        if use_punctuation:
            for i in range(limit, start - 1, -1):
                if _ends_sentence(group[i][0]):
                    cut = i
                    break
        out.append(group[start:cut + 1])
        start = cut + 1
    return out


def compute_chunk_spans(
    words: list[tuple[str, float, float]],
    duration: float,
    strategy: str,
    max_chunk_sec: float = DEFAULT_MAX_CHUNK_SEC,
    min_pause_sec: float = DEFAULT_MIN_PAUSE_SEC,
    pad_sec: float = DEFAULT_PAD_SEC,
) -> list[tuple[float, float]]:
    """Tính danh sách (start, end) chunk từ Whisper word timestamps.

    Args:
        words: (text, start_s, end_s) theo THỨ TỰ thời gian — text CÒN dấu câu
            (faster-whisper word text: "videos.", "Putna,").
        duration: thời lượng audio (giây); dùng để clamp pad biên phải. ≤ 0 thì
            chỉ clamp theo từ cuối + pad.
        strategy: "pause" — cắt tại khoảng lặng ≥ min_pause_sec, chunk quá dài
            thì hard-cut tại biên từ. "hybrid" — như pause nhưng chunk quá dài
            ưu tiên cắt tại ranh giới câu trước, hết mới hard-cut.
        max_chunk_sec: trần độ dài chunk (mọi strategy). Một TỪ đơn lẻ dài hơn
            trần được giữ nguyên (không cắt trong lòng từ).
        min_pause_sec: ngưỡng gap giữa 2 từ để coi là khoảng lặng.
        pad_sec: đệm mỗi biên chunk (không cắt cụt onset/coda); clamp về TRUNG
            ĐIỂM gap với chunk kề để các span KHÔNG BAO GIỜ overlap.

    Returns:
        Spans (start, end) tăng dần, không overlap, phủ mọi từ. [] nếu words rỗng
        (caller fallback single-pass).

    Raises:
        ValueError: strategy không thuộc CHUNKING_STRATEGIES.
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Chunking strategy không hợp lệ: {strategy!r}. "
            f"Hợp lệ: {sorted(CHUNKING_STRATEGIES)} (hoặc 'off' — không gọi hàm này)."
        )
    if not words:
        return []

    groups = _group_by_pause(words, min_pause_sec)
    use_punct = strategy == "hybrid"
    split_groups: list[list[tuple[str, float, float]]] = []
    for g in groups:
        if g[-1][2] - g[0][1] > max_chunk_sec:
            split_groups.extend(_split_long_group(g, max_chunk_sec, use_punct))
        else:
            split_groups.append(g)

    raw = [(g[0][1], g[-1][2]) for g in split_groups]

    # Pad mỗi biên, clamp về trung điểm gap với chunk kề (không overlap) và
    # về [0, duration] (duration ≤ 0 → không clamp phải).
    spans: list[tuple[float, float]] = []
    for i, (s, e) in enumerate(raw):
        lo = s - pad_sec
        hi = e + pad_sec
        if i > 0:
            lo = max(lo, (raw[i - 1][1] + s) / 2.0)
        if i < len(raw) - 1:
            hi = min(hi, (e + raw[i + 1][0]) / 2.0)
        lo = max(0.0, lo)
        if duration > 0:
            hi = min(hi, duration)
        if hi > lo:
            spans.append((round(lo, 3), round(hi, 3)))

    logger.debug(
        "compute_chunk_spans: strategy=%s words=%d → %d chunks "
        "(max=%.1fs pause=%.2fs pad=%.2fs)",
        strategy, len(words), len(spans), max_chunk_sec, min_pause_sec, pad_sec,
    )
    return spans

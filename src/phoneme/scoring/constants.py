"""Hằng số tinh chỉnh cho phoneme scoring (tách riêng → tránh circular import).

Các ngưỡng này được dùng làm DEFAULT trong chữ ký hàm (alignment/__init__) và được
analyzer import qua facade. Override qua Config (env TOEIC_PHONEME_*).
"""

from __future__ import annotations

from typing import Final

# Số lỗi tối đa trả về trong results (tránh payload quá lớn)
MAX_ERRORS_RETURNED: Final[int] = 30

# Số từ tối đa trả về trong word details (cắt theo ranh giới từ, không giữa từ)
MAX_WORDS_RETURNED: Final[int] = 80

# Knee của confidence weighting: predicted phoneme có confidence < knee thì penalty
# của lỗi sub bị hạ tỉ lệ (recognizer không chắc → ít khả năng là lỗi người đọc).
# Mặc định 0.5; override qua Config.phoneme_confidence_knee (env TOEIC_PHONEME_CONFIDENCE_KNEE).
PHONEME_CONFIDENCE_KNEE: Final[float] = 0.5

# L1-aware scoring layer (default OFF — bật qua Config.phoneme_l1_*; xem l1_vietnamese.py).
# l1_min_confidence: ngưỡng confidence để áp L1 *substitution* tolerance (chưa dùng ở v1).
# low_conf_floor: sub có confidence < ngưỡng này → penalty bị TRUNG HOÀ về 0 (soften,
# KHÔNG skip). Chỉ áp cho sub (âm được nhận diện); deletion KHÔNG đi qua confidence.
PHONEME_L1_MIN_CONFIDENCE: Final[float] = 0.70
PHONEME_LOW_CONF_FLOOR: Final[float] = 0.40

# Recognizer-noise gate (ĐỘC LẬP với L1): 1 sub bị coi là wav2vec hallucinate (KHÔNG
# phải lỗi học viên) khi cặp (ref→pred) BẤT KHẢ THI về âm học (sim < SIM, và không nằm
# trong _REAL_ERROR_SUBS) VÀ recognizer KHÔNG chắc (conf < CONF). Khi đó penalty về 0 +
# severity "low" → rơi vào nhóm "Hidden recognizer noise" (không tô đỏ), giống cơ chế
# LOW_CONFIDENCE_NEUTRALIZED nhưng có điều kiện bất-khả-thi nên KHÔNG giấu lỗi near-pair.
#
# Ngưỡng conf THEO LOẠI ÂM (hiệu chỉnh từ telemetry tel3.jsonl): nguyên âm wav2vec/espeak
# vốn confidence thấp hơn nhiều dù ĐÚNG (median ~0.67 vs phụ âm ~0.91) → 1 ngưỡng chung
# sẽ gate oan nguyên âm. CONF=0 → tắt gate (conf < 0 không bao giờ đúng → bit-for-bit như cũ).
#
# GIỚI HẠN ĐÃ BIẾT (sprint này, có chủ đích): gate này CHỈ bắt sub bất khả thi + CONFIDENCE
# THẤP. Nó KHÔNG xử lý "whole-word hallucination" CONFIDENCE CAO — khi wav2vec tự tin nhả
# sai cả từ (vd famous /feɪməs/→/leɪmz/ f→l @0.98) hoặc Whisper chép nhầm từ (blood→"floods"
# nên reference IPA sai). Confidence KHÔNG bắt được các ca này (đang cao), và word-accuracy
# KHÔNG tách được chúng khỏi lỗi phát âm THẬT (vd Vietnam v→b, nuốt cụm cuối first/most) → ẩn
# theo accuracy sẽ giấu lỗi thật. Hướng tương lai: "Word Reliability Gate" thiết kế TỪ DỮ
# LIỆU telemetry per-word (diagnostics.py đã ghi đủ: ref/pred IPA + alignment + per-phone
# confidence), KHÔNG bake heuristic conf/sim ở production.
PHONEME_RECOGNIZER_NOISE_SIM: Final[float] = 0.2
PHONEME_RECOGNIZER_NOISE_CONF: Final[float] = 0.6        # phụ âm
PHONEME_RECOGNIZER_NOISE_CONF_VOWEL: Final[float] = 0.45  # nguyên âm (confidence nền thấp hơn)

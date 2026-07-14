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

# Từ có IPA chuẩn lấy từ eSpeak G2P (OOV/tên riêng — WordSpan.source == "espeak"):
# bản thân reference đã là ĐOÁN nên sub/del trên từ đó không đáng tin là lỗi người đọc.
# Cap penalty dưới ngưỡng "medium" (0.3, xem _severity_from_penalty) → severity "low"
# → UI xếp vào "Hidden recognizer noise", không tô đỏ. (Free-speech thì các từ này bị
# skip hẳn ở tầng reliability; cap này chủ yếu cho chế độ CÓ script.)
PHONEME_G2P_UNCERTAIN_CAP: Final[float] = 0.2

# Coverage gate (Track A — "Word Reliability Gate" đã hứa ở ghi chú trên): từ có 100%
# âm KHÔNG-skip là "del" (coverage=0) trong khi Whisper (LM-biased, nguồn ĐỘC LẬP) đã
# match từ đó trong transcript → khả năng cao wav2vec collapse trên từ ngắn/đọc lướt,
# không phải học viên bỏ từ. ~4.4% tổng số từ (outputs/phoneme_telemetry.jsonl, 5905 từ).
# Guard 3 lớp (xem _apply_coverage_gate): (a) chỉ từ ≤ MAX_LEN âm (phủ hầu hết function
# words, tránh cap nhầm từ NỘI DUNG dài bị bỏ hẳn); (b) KHÔNG có wav2vec segment nào
# overlap Whisper window của từ (có âm trong vùng đó = drift/lỗi thật, không phải im
# lặng); (c) Whisper word prob ≥ MIN_ASR_PROB (transcript KHÔNG phải ground truth tuyệt
# đối — thiếu prob/window thì không cap). Cap dưới "medium" (0.3) → severity "low" →
# Hidden recognizer noise, không tô đỏ oan.
PHONEME_COVERAGE_GATE_CAP: Final[float] = 0.2
PHONEME_COVERAGE_GATE_MAX_LEN: Final[int] = 4
PHONEME_COVERAGE_GATE_MIN_ASR_PROB: Final[float] = 0.60

# Recognizer-collapse gate (mở rộng coverage gate cho collapse TỪNG PHẦN qua posterior
# evidence): coverage gate chỉ bắt từ "del" 100% + wav2vec IM LẶNG. Nhưng wav2vec còn
# collapse KIỂU KHÁC — nhả token BLANK (<pad>) đè lên âm VẪN CÓ trong audio (CTC
# blank-collapse) — làm 1-2 âm giữa từ thành del/sub dù từ được đọc rõ (case "line to"
# 2026-07-14: /aɪ n/ mass posterior 0.28-0.33 nhưng argmax=<pad>; Whisper prob 0.997;
# đối chứng âm VẮNG THẬT mass ~0.001). Gate này cap del/sub về "low" khi âm THAM CHIẾU
# có max_mass ≥ FLOOR (được nói) VÀ argmax là token silence (bị nhả blank) — hai điều
# kiện tách CTC-collapse khỏi (a) nghe ra âm KHÁC = lỗi thật (argmax là IPam) và (b) âm
# vắng thật (mass thấp). Cùng cap/min_asr_prob với coverage gate (COVERAGE_COLLAPSE,
# không tô đỏ). Default OFF. FLOOR 0.10 nằm giữa del-thật (~0.001-0.03, xem
# deletion-evidence telemetry) và blank-collapse (~0.28-0.33).
PHONEME_COLLAPSE_GATE_MASS_FLOOR: Final[float] = 0.10

# Drift cap (Track B): sub có predicted segment NGOÀI cửa sổ Whisper của chính từ đó
# (±DRIFT_WINDOW_PAD_SEC, dùng chung is_within_word_window với telemetry) → khả năng
# DTW "mượn" âm của từ kế bên, không phải lỗi phát âm thật. Đo lại trên 5905 từ:
# drift_fraction = 711/3324 = 21.4% (vẫn dưới ngưỡng kill 40% của PR3-0 → KHÔNG build
# lại alignment, chỉ cap severity dựa trên evidence diagnostics đã tính sẵn). Cùng cap
# 0.2 với G2P_UNCERTAIN để nhất quán UI (Hidden recognizer noise).
PHONEME_DRIFT_SUB_CAP: Final[float] = 0.2

# S-cluster leniency: /p t k/ sau /s/ trong onset cùng từ (speak, stay, school) phát
# âm ĐÚNG là stop KHÔNG bật hơi — về âm học gần voiced counterpart hơn dạng bật hơi,
# wav2vec (train trên espeak citation form) hay gán nhầm: (a) voicing p→b/t→d/k→ɡ,
# (b) CHỖ cấu âm sp→st (case "speak" → /stɪk/, t→p penalty 0.6 = "cao" oan).
# (a) → "ok" (S_CLUSTER_VARIANT); (b) → cap 0.1 dưới cả G2P_UNCERTAIN_CAP: giữ visible
# severity "low" (không loại trừ 100% lỗi steak-for-speak thật, Whisper word-match đã
# guard intelligibility ở tầng trên). Default OFF qua Config.phoneme_s_cluster_enabled.
PHONEME_S_CLUSTER_SUB_CAP: Final[float] = 0.1

"""Biến thể phát âm theo GIỌNG (UK) làm reference thay thế cho dual-reference scoring.

Bối cảnh: reference chấm điểm dựng từ CMUdict = giọng MỸ. Tầng tolerance
(similarity.phonemes_match + normalize_ipa) đã chấp nhận HẦU HẾT khác biệt UK/US khi
accent="default": coda /r/ non-rhotic, LOT ɑː↔ɒ, GOAT oʊ↔əʊ, NURSE/letter rhotacised,
nguyên âm rút gọn (weak/strong form). Đo thực nghiệm (2026-07-21) còn ĐÚNG MỘT gap
phonemic không được tha: **BATH-split** — nhóm từ RP đọc /ɑː/ nơi giọng Mỹ đọc /æ/
(dance /dæns/ ↔ /dɑːns/, path, class, ask, example…). Reference Mỹ /æ/ → học viên đọc
giọng Anh /ɑː/ bị tính SAI.

Module này sinh một reference UK từ reference Mỹ để đưa vào cơ chế fitting đa-reference
(scoring/homograph.py) — người nói khớp BẤT KỲ reference (US HOẶC UK) nào là đúng. Chỉ
áp các phép biến đổi UK/US ĐÃ ĐỊNH NGHĨA RÕ và CHƯA được tolerance bao (hiện: BATH).
KHÔNG đụng các khác biệt tolerance đã lo (tránh candidate thừa trùng lặp).

An toàn: `uk_variant` trả None khi không có luật nào áp được (từ ngoài BATH set) → 0
candidate thừa cho đại đa số từ → bề mặt regression giới hạn đúng bộ từ BATH. Fitting
chỉ swap sang UK khi khớp acoustic HẲN hơn (strict) nên lỗi thật không bị nuốt.
"""
from __future__ import annotations

# Bộ từ BATH (RP /ɑː/ ↔ GenAm /æ/). Nguồn: lexical set BATH (Wells) — chỉ giữ các từ
# BATH ỔN ĐỊNH trong RP (bỏ các từ dao động vùng miền như "plastic/gymnastic"). Key đã
# lower + không dấu; biến thể số nhiều/chia đuôi liệt kê tường minh để khớp reference
# theo từ (homograph key strip giống g2p). "æ" nhấn của các từ này → "ɑː".
_BATH_WORDS: frozenset[str] = frozenset({
    "after", "afternoon", "afternoons", "afters",
    "answer", "answers", "answered", "answering",
    "ask", "asks", "asked", "asking",
    "aunt", "aunts", "aunty", "auntie",
    "bath", "baths", "bathing",  # danh từ /bɑːθ/ (động từ bathe khác)
    "basket", "baskets",
    "blast", "blasts", "blasted",
    "branch", "branches", "branched",
    "brass",
    "calf", "calves",
    "cast", "casts", "casting", "castaway",
    "castle", "castles",
    "chance", "chances", "chanced",
    "chant", "chants", "chanted",
    "class", "classes", "classed", "classroom", "classrooms", "classmate", "classmates",
    "command", "commands", "commanded", "commander", "commanding",
    "dance", "dances", "danced", "dancer", "dancers", "dancing",
    "demand", "demands", "demanded", "demanding",
    "disaster", "disasters", "disastrous",
    "draft", "drafts", "drafted",  # = draught RP
    "enhance", "enhances", "enhanced", "enhancing",
    "example", "examples",
    "fast", "faster", "fastest", "fasten",
    "france",
    "glance", "glances", "glanced", "glancing",
    "glass", "glasses",
    "graph", "graphs",
    "grasp", "grasps", "grasped",
    "grass",
    "half", "halves", "halved",
    "last", "lasts", "lasted", "lasting",
    "laugh", "laughs", "laughed", "laughing", "laughter",
    "mask", "masks", "masked",
    "mast", "masts",
    "master", "masters", "mastered", "mastery",
    "nasty", "nastier",
    "pass", "passes", "passed", "passing", "passenger", "passengers",
    "past", "pasta",
    "path", "paths", "pathway", "pathways",
    "plant", "plants", "planted", "planting",
    "plaster", "plasters", "plastered",
    "raft", "rafts",
    "rather",
    "sample", "samples", "sampled", "sampling",
    "shaft", "shafts",
    "slander",
    "staff", "staffs", "staffed",
    "task", "tasks", "tasked",
    "vast", "vaster",
})

_WORD_STRIP_CHARS = ".,;:!?\"'()[]{}"

# Token IPA (dạng đã dựng reference, ARPABET_TO_IPA): TRAP /æ/, PALM/START /ɑː/.
_TRAP = "æ"
_BATH_TARGET = "ɑː"


def uk_variant(
    symbols: list[str], stresses: list[str | None], word: str | None
) -> tuple[list[str], list[str | None]] | None:
    """Reference UK từ reference Mỹ (symbols/stresses của MỘT từ), hoặc None nếu không
    có luật nào đổi được (→ caller bỏ qua, không thêm candidate).

    Hiện chỉ luật BATH: từ thuộc _BATH_WORDS thì mọi token TRAP /æ/ → /ɑː/. Stresses
    giữ nguyên (đổi chất nguyên âm, không đổi trọng âm). Trả None nếu từ ngoài set hoặc
    không có /æ/ nào (không đổi gì).
    """
    if not word:
        return None
    key = word.lower().strip(_WORD_STRIP_CHARS)
    if key not in _BATH_WORDS or _TRAP not in symbols:
        return None
    new_symbols = [_BATH_TARGET if s == _TRAP else s for s in symbols]
    if new_symbols == symbols:
        return None
    return new_symbols, list(stresses)

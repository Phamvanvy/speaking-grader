"""Hồ sơ "âm yếu" per-user — tổng hợp CHỈ-ĐỌC từ output chấm điểm đã lưu.

Nguồn evidence (không đụng scoring path):
- history result_json: `result.phoneme.score.words[*].phonemes[*]` (status
  ok/sub/del/skipped, heard, severity) — quét TĂNG DẦN bằng con trỏ composite
  (created_at, id) lưu ở words.db (phoneme_profile_state), tally cộng dồn vào
  phoneme_stats. Không hook vào save path: tự backfill history cũ, xoá record
  chỉ ngừng thêm evidence (stats là advisory).
- saved_words snapshot phonemes: merge LIVE lúc đọc (≤500 rows) với trọng số
  cao hơn — là evidence per-word mới nhất, kéo điểm về đúng khi user tiến bộ
  (stats cộng dồn không decay hồi tố).

Consumer: src/word_suggest.py (gợi ý từ luyện tập trong tab Từ đã lưu).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from . import history, words
from .config import Config
from .phoneme.ipa.ko.phoneme_set_ko import normalize_ipa_ko
from .phoneme.ipa.phoneme_set import normalize_ipa
from .rubrics.base import exam_language

logger = logging.getLogger("toeic.phoneme_profile")


def _normalize(symbol: str, lang: str) -> str:
    """Chuẩn hoá symbol theo ngôn ngữ (ko dùng normalize_ipa_ko: fold ㅐ/ㅔ, ɔ→ʌ,
    nhãn espeak…; en dùng normalize_ipa)."""
    return normalize_ipa_ko(symbol) if lang == "ko" else normalize_ipa(symbol)

# Trọng số lỗi theo severity — low = pass, khớp practicePct/practiceIsBad của
# frontend (web/js/practice.js) để backend và UI thống nhất "thế nào là lỗi".
_SEVERITY_ERR = {"high": 1.0, "medium": 0.6, "low": 0.0}

# Trọng số evidence theo tuổi record lúc quét (stats cộng dồn không decay hồi
# tố nên evidence cũ vào sổ với trọng số thấp hơn ngay từ đầu).
_RECENCY_STEPS = ((30, 1.0), (90, 0.5))
_RECENCY_OLD = 0.25

# Snapshot saved_words = evidence mới nhất per-word → trọng số cao hơn history.
_SAVED_SNAPSHOT_WEIGHT = 2.0

# Ngưỡng lọc âm yếu: đủ mẫu + tỉ lệ lỗi (Laplace smoothing) đủ cao.
_MIN_ATTEMPTS = 5.0
_MIN_WEAKNESS = 0.18

# Fallback khi chưa đủ dữ liệu chấm: các âm người Việt hay gặp khó — fricative/
# affricate (nhất là coda, xem _FINAL_DELETION_CATEGORIES trong
# src/phoneme/l1_vietnamese.py) + nguyên âm hay nhầm (æ↔e, ɪ↔iː).
FALLBACK_WEAK_PHONEMES = ["θ", "ð", "s", "z", "ʃ", "tʃ", "dʒ", "v", "æ", "ɪ"]

# Fallback tiếng Hàn (người Việt học tiếng Hàn): âm CĂNG (không có trong tiếng
# Việt) + âm BẬT HƠI (contrast bật hơi/thường) + coda ㄹ [l]. KHÔNG dùng nhóm
# nguyên âm ㅓ/ㅗ vì model phone-mfa fold ɔ→ʌ (contrast không chấm được — xem
# phoneme_set_ko._IPA_EQUIV_KO). Xem src/phoneme/l1/vi_ko.py.
KOREAN_FALLBACK_WEAK_PHONEMES = [
    "p͈", "t͈", "k͈", "s͈", "t͈ɕ",   # tense
    "pʰ", "tʰ", "kʰ", "tɕʰ",        # aspirated
    "l",                             # coda ㄹ
]


def _fallback_for(lang: str) -> list[str]:
    return KOREAN_FALLBACK_WEAK_PHONEMES if lang == "ko" else FALLBACK_WEAK_PHONEMES


def _recency_weight(created_at: str | None, now: datetime) -> float:
    """Trọng số theo tuổi record (ISO 'YYYY-MM-DDTHH:MM:SSZ'); parse lỗi → 1.0."""
    try:
        ts = datetime.strptime(created_at or "", "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return 1.0
    age_days = (now - ts).days
    for max_days, weight in _RECENCY_STEPS:
        if age_days <= max_days:
            return weight
    return _RECENCY_OLD


def _tally_points(
    points: list, tallies: dict[str, dict], weight: float, lang: str = "en"
) -> None:
    """Cộng 1 danh sách PhonemePoint dict vào tallies (key = symbol đã chuẩn hoá
    theo `lang`). tallies vẫn keyed theo SYMBOL (không kèm lang) — refresh_profile
    tự gắn lang khi apply, nên caller cũ (test) truyền dict symbol-keyed vẫn đúng."""
    for p in points:
        if not isinstance(p, dict):
            continue
        symbol = p.get("symbol")
        status = p.get("status")
        if not symbol or status not in ("ok", "sub", "del"):
            continue  # skipped / dữ liệu lạ → không tính
        key = _normalize(symbol, lang)
        t = tallies.setdefault(
            key,
            {"attempts": 0.0, "ok": 0.0, "sub": 0.0, "del": 0.0,
             "err_weighted": 0.0, "heard": {}},
        )
        t["attempts"] += weight
        if status == "ok":
            t["ok"] += weight
            continue
        t[status] += weight
        err = _SEVERITY_ERR.get(p.get("severity"), 1.0)
        t["err_weighted"] += weight * err
        # Cặp nhầm lẫn expected→heard (chỉ sub có lỗi thật — low là near-match).
        if status == "sub" and err > 0 and p.get("heard"):
            heard_key = _normalize(str(p["heard"]), lang)
            t["heard"][heard_key] = t["heard"].get(heard_key, 0) + weight


def _tally_result(
    result: dict, tallies: dict[str, dict], weight: float = 1.0, lang: str = "en"
) -> None:
    """Tally 1 grading result. Duyệt score.words[*].phonemes (KHÔNG dùng
    score.errors — bị cắt top-20). Guard .get mọi tầng: record cha của exam chỉ
    là summary, không có phoneme detail. `lang` chỉ đổi CÁCH chuẩn hoá symbol
    (en/ko) — tallies vẫn symbol-keyed."""
    score = (result.get("phoneme") or {}).get("score") or {}
    for w in score.get("words") or []:
        if not isinstance(w, dict) or w.get("skip_reason"):
            continue
        _tally_points(w.get("phonemes") or [], tallies, weight, lang)


def refresh_profile(cfg: Config, user_id: str) -> None:
    """Quét phần history mới hơn con trỏ, cộng dồn vào phoneme_stats.

    Mỗi lượt tối đa 1 batch (300 blob) — backfill account lớn hội tụ qua vài
    request; steady-state là no-op O(records mới).
    """
    since_at, since_id = words.get_profile_cursor(cfg, user_id)
    results, cursor = history.list_results_since(cfg, user_id, since_at, since_id)
    if cursor == (since_at, since_id):
        return  # không có gì mới
    now = datetime.now(timezone.utc)
    # Tally TÁCH theo ngôn ngữ nói của kỳ thi (exam_language): TOEIC/IELTS→en,
    # TOPIK→ko. Mỗi lang một dict symbol-keyed (dùng normalizer riêng), rồi gộp
    # thành key (symbol, lang) cho apply → hồ sơ âm en/ko không trộn.
    by_lang: dict[str, dict[str, dict]] = {}
    for r in results:
        lang = exam_language((r["result"] or {}).get("exam"))
        _tally_result(
            r["result"], by_lang.setdefault(lang, {}),
            weight=_recency_weight(r["created_at"], now), lang=lang,
        )
    combined = {
        (sym, lang): t for lang, d in by_lang.items() for sym, t in d.items()
    }
    # Apply cả khi rỗng (blob không có phoneme) để con trỏ vẫn tiến.
    words.apply_phoneme_tallies(cfg, user_id, combined, cursor)


def get_weak_phonemes(
    cfg: Config, user_id: str, top_k: int = 5, lang: str = "en"
) -> tuple[list[dict], str]:
    """Top âm yếu của user cho 1 NGÔN NGỮ: (weak-items, source "history"/"fallback").

    weak-item = {symbol, attempts, errors, error_rate, top_heard, fallback}.
    "Phát âm ok rồi thì giảm gợi ý" nằm ở đây: âm ok-rate cao → weakness thấp
    → không vào top. Default lang='en' → hành vi cũ (TOEIC/IELTS + tab Từ đã lưu)
    bit-for-bit; 'ko' cho khóa học TOPIK.
    """
    refresh_profile(cfg, user_id)
    stats = words.get_phoneme_stats(cfg, user_id, lang)
    tallies: dict[str, dict] = {
        sym: {"attempts": s["attempts"], "ok": s["ok"], "sub": s["sub"],
              "del": s["del"], "err_weighted": s["err_weighted"],
              "heard": dict(s["heard"])}
        for sym, s in stats.items()
    }
    # Snapshot saved_words là evidence per-word tiếng ANH (bookmark từ bảng lỗi
    # phát âm en) → chỉ merge cho lang='en'; khóa học ko không dùng.
    if lang == "en":
        for entry in words.list_words(cfg, user_id)["words"]:
            _tally_points(entry.get("phonemes") or [], tallies, _SAVED_SNAPSHOT_WEIGHT, "en")

    ranked = []
    for sym, t in tallies.items():
        if t["attempts"] < _MIN_ATTEMPTS:
            continue
        weakness = (t["err_weighted"] + 1) / (t["attempts"] + 6)  # Laplace
        if weakness < _MIN_WEAKNESS:
            continue
        ranked.append((weakness, t["attempts"], sym, t))
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))

    weak = []
    for _weakness, attempts, sym, t in ranked[:top_k]:
        top_heard = sorted(t["heard"], key=t["heard"].get, reverse=True)[:2]
        weak.append({
            "symbol": sym,
            "attempts": round(attempts, 1),
            "errors": round(t["sub"] + t["del"], 1),
            "error_rate": round(t["err_weighted"] / attempts, 3),
            "top_heard": top_heard,
            "fallback": False,
        })

    source = "history"
    if len(weak) < 2:
        source = "fallback"
        have = {w["symbol"] for w in weak}
        for sym in _fallback_for(lang):
            if len(weak) >= top_k:
                break
            norm = _normalize(sym, lang)
            if norm in have:
                continue
            weak.append({
                "symbol": norm, "attempts": 0, "errors": 0,
                "error_rate": None, "top_heard": [], "fallback": True,
            })
    return weak, source

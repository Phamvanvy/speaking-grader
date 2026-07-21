"""Lịch sử chấm bài per-user: SQLite (metadata + result JSON) + audio gốc trên đĩa.

Thiết kế:
- "User" = uuid ẩn danh do frontend sinh (localStorage), KHÔNG có auth. Mọi truy
  vấn đều lọc theo user_id nên user này không thấy bản ghi của user khác (chỉ là
  cách ly mềm, không phải bảo mật).
- Schema cha/con: `history_records` (single/batch/exam) + `history_items` (từng
  câu của exam / từng file của batch). Exam session của SPA được ghép DẦN qua
  nhiều request /grade độc lập (có thể rơi vào worker khác nhau — app chạy 2
  uvicorn worker) → cha insert kiểu INSERT OR IGNORE (idempotent), con append
  bằng INSERT ngắn. KHÔNG read-modify-write JSON blob.
- An toàn 2 worker: mỗi hàm mở connection MỚI, WAL + busy_timeout, transaction
  ngắn qua `with conn:`.
- Crash-safe: audio ghi atomic (tmp → fsync → os.replace) TRƯỚC, row DB insert
  SAU trong 1 transaction → row không bao giờ trỏ file thiếu/ghi dở. Crash giữa
  2 bước chỉ để lại thư mục audio mồ côi (vô hại) — dọn bằng sweep_orphans()
  lúc startup.
- Schema versioning: PRAGMA user_version (hiện = 1). Migration sau này = chuỗi
  `if version < N: ALTER ...; user_version = N` trong _init_schema.
- Backlog (chưa làm): nén result_json bằng zlib khi blob > ~100KB (thêm cột
  result_compressed + bump user_version); quota theo dung lượng GB.

Dọn tay khi cần: DELETE FROM history_records WHERE ... (cascade sang items) rồi
xoá thư mục data/history_audio/{record_id}/ tương ứng.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import uuid
from pathlib import Path

from .config import Config
from .api_helpers import _overall_score

logger = logging.getLogger("toeic.history")

_SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS history_records (
  id                 TEXT PRIMARY KEY,
  user_id            TEXT NOT NULL,
  kind               TEXT NOT NULL CHECK (kind IN ('single','batch','exam')),
  exam               TEXT,
  question_type      TEXT,
  mode               TEXT,
  title              TEXT,
  overall_score      REAL,
  overall_max        INTEGER,
  pronunciation_only INTEGER NOT NULL DEFAULT 0,
  item_count         INTEGER NOT NULL DEFAULT 0,
  audio_path         TEXT,
  result_json        TEXT,
  created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_history_user_created
  ON history_records(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS history_items (
  id            TEXT PRIMARY KEY,
  record_id     TEXT NOT NULL REFERENCES history_records(id) ON DELETE CASCADE,
  seq           INTEGER,
  question_id   TEXT,
  question_type TEXT,
  label         TEXT,
  score         REAL,
  audio_path    TEXT,
  result_json   TEXT,
  error         TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_history_items_record ON history_items(record_id, seq);
"""

_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Cột cha trả về cho list/detail (không lôi result_json vào list cho nhẹ).
_RECORD_COLS = (
    "id, kind, exam, question_type, mode, title, overall_score, overall_max, "
    "pronunciation_only, item_count, audio_path, created_at, updated_at"
)


# ── Kết nối / schema ─────────────────────────────────────────────────────


def _connect(cfg: Config) -> sqlite3.Connection:
    path = Path(cfg.history_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= _SCHEMA_VERSION:
        return
    with conn:
        # version 0 → schema đầy đủ. Migration tương lai: if version < 2: ALTER...
        conn.executescript(_DDL)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")


# ── Validate input từ client ─────────────────────────────────────────────


def validate_user_id(user_id: str) -> str:
    user_id = (user_id or "").strip()
    if not _USER_ID_RE.match(user_id):
        raise ValueError("user_id không hợp lệ (chỉ [A-Za-z0-9_-], tối đa 64 ký tự).")
    return user_id


def validate_uuid(value: str) -> str:
    """Chuẩn hoá uuid từ client (dùng làm id bản ghi = tên thư mục audio)."""
    return str(uuid.UUID((value or "").strip()))


# ── Audio trên đĩa ───────────────────────────────────────────────────────


def _audio_root(cfg: Config) -> Path:
    return Path(cfg.history_audio_dir).resolve()


def _write_audio(
    cfg: Config, record_id: str, filename_stem: str, audio_bytes: bytes, suffix: str
) -> str:
    """Ghi audio atomic (tmp → fsync → replace). Trả path tương đối với audio root."""
    rel = f"{record_id}/{filename_stem}{suffix}"
    dest = _audio_root(cfg) / record_id / f"{filename_stem}{suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(audio_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dest)
    return rel


def _rm_audio_dir(cfg: Config, record_id: str) -> None:
    shutil.rmtree(_audio_root(cfg) / record_id, ignore_errors=True)


def sweep_orphans(cfg: Config) -> int:
    """Xoá thư mục audio không có row cha tương ứng (rác do crash giữa 2 bước ghi).

    Best-effort, gọi 1 lần lúc startup. Trả số thư mục đã xoá.
    """
    root = _audio_root(cfg)
    if not root.is_dir():
        return 0
    dirs = [d for d in root.iterdir() if d.is_dir()]
    if not dirs:
        return 0
    conn = _connect(cfg)
    try:
        known = {
            row[0]
            for row in conn.execute("SELECT id FROM history_records").fetchall()
        }
    finally:
        conn.close()
    removed = 0
    for d in dirs:
        if d.name not in known:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        logger.info("Lịch sử: dọn %d thư mục audio mồ côi.", removed)
    return removed


# ── Retention ────────────────────────────────────────────────────────────


def _enforce_retention(cfg: Config, conn: sqlite3.Connection, user_id: str) -> None:
    """Giữ tối đa N bản ghi mới nhất của user; bản cũ hơn xoá cả row lẫn audio."""
    max_records = cfg.history_max_records_per_user
    if max_records <= 0:
        return
    stale = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM history_records WHERE user_id = ? "
            "ORDER BY created_at DESC, id LIMIT -1 OFFSET ?",
            (user_id, max_records),
        ).fetchall()
    ]
    if not stale:
        return
    with conn:
        conn.executemany(
            "DELETE FROM history_records WHERE id = ?", [(rid,) for rid in stale]
        )
    for rid in stale:
        _rm_audio_dir(cfg, rid)


# ── Ghi: single / batch / exam ───────────────────────────────────────────


def save_single(
    cfg: Config,
    *,
    user_id: str,
    filename: str | None,
    mode: str | None,
    audio_bytes: bytes,
    suffix: str,
    result: dict,
) -> str:
    """Lưu 1 lần chấm lẻ (/grade). Trả record_id."""
    user_id = validate_user_id(user_id)
    record_id = str(uuid.uuid4())
    audio_rel = _write_audio(cfg, record_id, "audio", audio_bytes, suffix)
    exam = result.get("exam")
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT INTO history_records (id, user_id, kind, exam, question_type,"
                " mode, title, overall_score, overall_max, pronunciation_only,"
                " item_count, audio_path, result_json)"
                " VALUES (?, ?, 'single', ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                (
                    record_id,
                    user_id,
                    exam,
                    result.get("question_type"),
                    mode,
                    filename or "audio",
                    _overall_score(result.get("scores"), exam or ""),
                    _overall_max(exam),
                    1 if result.get("pronunciation_only") else 0,
                    audio_rel,
                    json.dumps(result, ensure_ascii=False),
                ),
            )
        _enforce_retention(cfg, conn, user_id)
    finally:
        conn.close()
    return record_id


def save_batch(
    cfg: Config,
    *,
    user_id: str,
    mode: str | None,
    batch_response: dict,
    files: list[tuple[str, bytes, str | None]],
) -> str:
    """Lưu 1 lần chấm cả lớp (/grade-batch): 1 cha + N con.

    `files[i]` = (filename, bytes, suffix|None) khớp index với
    batch_response["results"][i]. Cha giữ summary (KHÔNG kèm results[*].result —
    body từng bài nằm ở con). Con lỗi vẫn giữ audio nếu có bytes.
    """
    user_id = validate_user_id(user_id)
    record_id = str(uuid.uuid4())
    exam = batch_response.get("exam")
    results = batch_response.get("results") or []

    # Ghi audio TRƯỚC (crash-safe: DB row luôn trỏ file đã hoàn chỉnh).
    audio_rels: list[str | None] = []
    for i, (_name, data, suffix) in enumerate(files):
        if data and suffix:
            audio_rels.append(_write_audio(cfg, record_id, f"item{i:03d}", data, suffix))
        else:
            audio_rels.append(None)

    summary = {k: v for k, v in batch_response.items() if k != "results"}
    succeeded = batch_response.get("succeeded", 0)
    count = batch_response.get("count", len(results))
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT INTO history_records (id, user_id, kind, exam, question_type,"
                " mode, title, overall_score, overall_max, pronunciation_only,"
                " item_count, audio_path, result_json)"
                " VALUES (?, ?, 'batch', ?, ?, ?, ?, NULL, ?, 0, ?, NULL, ?)",
                (
                    record_id,
                    user_id,
                    exam,
                    batch_response.get("question_type"),
                    mode,
                    f"Chấm cả lớp ({succeeded}/{count})",
                    _overall_max(exam),
                    len(results),
                    json.dumps(summary, ensure_ascii=False),
                ),
            )
            for i, entry in enumerate(results):
                result = entry.get("result")
                conn.execute(
                    "INSERT INTO history_items (id, record_id, seq, question_type,"
                    " label, score, audio_path, result_json, error)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        record_id,
                        i,
                        (result or {}).get("question_type"),
                        entry.get("audio_filename") or f"file {i + 1}",
                        _overall_score((result or {}).get("scores"), exam or ""),
                        audio_rels[i] if i < len(audio_rels) else None,
                        json.dumps(result, ensure_ascii=False) if result else None,
                        entry.get("error"),
                    ),
                )
        _enforce_retention(cfg, conn, user_id)
    finally:
        conn.close()
    return record_id


def ensure_exam_session(
    cfg: Config,
    *,
    session_id: str,
    user_id: str,
    exam: str | None,
    title: str | None,
    mode: str | None,
) -> str:
    """Tạo cha kind='exam' nếu chưa có (idempotent — an toàn khi các câu của cùng
    phiên rơi vào worker khác nhau). Trả session_id đã chuẩn hoá."""
    user_id = validate_user_id(user_id)
    session_id = validate_uuid(session_id)
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO history_records (id, user_id, kind, exam,"
                " mode, title, overall_max) VALUES (?, ?, 'exam', ?, ?, ?, ?)",
                (session_id, user_id, exam, mode, title or "Thi cả đề",
                 _overall_max(exam)),
            )
    finally:
        conn.close()
    return session_id


def add_exam_item(
    cfg: Config,
    *,
    session_id: str,
    user_id: str,
    exam: str | None,
    title: str | None,
    mode: str | None,
    seq: int | None,
    question_id: str | None,
    result: dict,
    audio_bytes: bytes,
    suffix: str,
) -> None:
    """Append 1 câu vào phiên thi (tự ensure cha — request đầu tiên tới trước)."""
    session_id = ensure_exam_session(
        cfg, session_id=session_id, user_id=user_id, exam=exam, title=title, mode=mode
    )
    user_id = validate_user_id(user_id)
    item_id = str(uuid.uuid4())
    audio_rel = _write_audio(cfg, session_id, item_id, audio_bytes, suffix)
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                "INSERT INTO history_items (id, record_id, seq, question_id,"
                " question_type, label, score, audio_path, result_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item_id,
                    session_id,
                    seq,
                    question_id,
                    result.get("question_type"),
                    f"Câu {seq}" if seq is not None else (question_id or "Câu ?"),
                    _overall_score(result.get("scores"), result.get("exam") or ""),
                    audio_rel,
                    json.dumps(result, ensure_ascii=False),
                ),
            )
            conn.execute(
                "UPDATE history_records SET item_count = item_count + 1,"
                " updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')"
                " WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
    finally:
        conn.close()


def finalize_exam_session(
    cfg: Config,
    *,
    session_id: str,
    user_id: str,
    overall: float | None,
    overall_max: int | None,
    summary: dict | None,
) -> None:
    """Điền điểm tổng cho phiên thi (SPA gọi /exam/overall sau khi chấm hết câu)."""
    user_id = validate_user_id(user_id)
    session_id = validate_uuid(session_id)
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute(
                "UPDATE history_records SET overall_score = ?, overall_max = ?,"
                " result_json = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')"
                " WHERE id = ? AND user_id = ? AND kind = 'exam'",
                (
                    overall,
                    overall_max,
                    json.dumps(summary, ensure_ascii=False) if summary else None,
                    session_id,
                    user_id,
                ),
            )
        _enforce_retention(cfg, conn, user_id)
    finally:
        conn.close()


def save_exam_full(
    cfg: Config,
    *,
    user_id: str,
    mode: str | None,
    exam_response: dict,
    audio_by_qid: dict[str, tuple[bytes, str]],
) -> str:
    """Lưu trọn phiên /exam/grade (API client trực tiếp — SPA đi đường add_exam_item).

    `audio_by_qid[question_id]` = (bytes, suffix) của câu đó.
    """
    user_id = validate_user_id(user_id)
    session_id = str(uuid.uuid4())
    exam = exam_response.get("exam")
    ensure_exam_session(
        cfg,
        session_id=session_id,
        user_id=user_id,
        exam=exam,
        title=exam_response.get("title"),
        mode=mode,
    )
    for q in exam_response.get("questions") or []:
        result = q.get("result")
        qid = q.get("question_id")
        audio = audio_by_qid.get(qid or "")
        if result and audio:
            add_exam_item(
                cfg,
                session_id=session_id,
                user_id=user_id,
                exam=exam,
                title=exam_response.get("title"),
                mode=mode,
                seq=q.get("sequence"),
                question_id=qid,
                result=result,
                audio_bytes=audio[0],
                suffix=audio[1],
            )
    summary = {k: v for k, v in exam_response.items() if k != "questions"}
    finalize_exam_session(
        cfg,
        session_id=session_id,
        user_id=user_id,
        overall=exam_response.get("overall"),
        overall_max=exam_response.get("overall_max"),
        summary=summary,
    )
    return session_id


# ── Đọc / xoá ────────────────────────────────────────────────────────────


def list_records(cfg: Config, user_id: str, limit: int, offset: int) -> dict:
    user_id = validate_user_id(user_id)
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    conn = _connect(cfg)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM history_records WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        # has_audio của list = "có audio ở BẤT KỲ đâu" (record hoặc item) — exam/
        # batch chỉ lưu audio ở item nên audio_path cấp record là NULL; nút ⬇ tải
        # zip trên hàng lịch sử cần biết tổng thể. (has_audio của get_record vẫn
        # là cấp record vì detail dùng nó cho <audio src> record-level.)
        rows = conn.execute(
            f"SELECT {_RECORD_COLS}, EXISTS(SELECT 1 FROM history_items i"
            " WHERE i.record_id = history_records.id AND i.audio_path IS NOT NULL)"
            " AS item_audio FROM history_records WHERE user_id = ?"
            " ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
    finally:
        conn.close()
    records = []
    for row in rows:
        rec = dict(row)
        rec["has_audio"] = bool(rec.pop("audio_path", None)) or bool(rec.pop("item_audio", 0))
        records.append(rec)
    return {
        "user_id": user_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": records,
    }


def get_record(cfg: Config, user_id: str, record_id: str) -> dict | None:
    user_id = validate_user_id(user_id)
    conn = _connect(cfg)
    try:
        row = conn.execute(
            f"SELECT {_RECORD_COLS}, result_json FROM history_records"
            " WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        ).fetchone()
        if row is None:
            return None
        items = conn.execute(
            "SELECT id, seq, question_id, question_type, label, score, audio_path,"
            " result_json, error, created_at FROM history_items"
            " WHERE record_id = ? ORDER BY seq, created_at",
            (record_id,),
        ).fetchall()
    finally:
        conn.close()
    rec = dict(row)
    rec["has_audio"] = bool(rec.pop("audio_path", None))
    raw_result = rec.pop("result_json", None)
    rec["result"] = json.loads(raw_result) if raw_result else None
    out_items = []
    for it in items:
        item = dict(it)
        item["has_audio"] = bool(item.pop("audio_path", None))
        raw = item.pop("result_json", None)
        item["result"] = json.loads(raw) if raw else None
        out_items.append(item)
    rec["items"] = out_items
    return rec


def list_results_since(
    cfg: Config, user_id: str, since_at: str, since_id: str, limit: int = 300
) -> tuple[list[dict], tuple[str, str]]:
    """Result dicts (đã parse) của records + items mới hơn con trỏ, cho quét
    tăng dần của hồ sơ phoneme (src/phoneme_profile.py). CHỈ ĐỌC.

    Con trỏ composite (created_at, id): tie-break theo id nên record cùng giây
    không bị bỏ sót (id là uuid — cả 2 bảng đều so sánh lexicographic ổn định).
    Records và items được UNION vào 1 dòng thời gian chung; `limit` chặn 1 lượt
    quét để backfill lần đầu account lớn không phình RAM (stats hội tụ qua các
    request sau). Trả (results, cursor_mới); cursor không đổi nếu hết dữ liệu.
    """
    user_id = validate_user_id(user_id)
    limit = max(1, min(int(limit), 1000))
    after = "(created_at > ? OR (created_at = ? AND id > ?))"
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            f"""
            SELECT created_at, id, result_json FROM history_records
              WHERE user_id = ? AND result_json IS NOT NULL AND {after}
            UNION ALL
            SELECT i.created_at, i.id, i.result_json FROM history_items i
              JOIN history_records r ON r.id = i.record_id
              WHERE r.user_id = ? AND i.result_json IS NOT NULL
                AND (i.created_at > ? OR (i.created_at = ? AND i.id > ?))
            ORDER BY created_at ASC, id ASC LIMIT ?
            """,
            (user_id, since_at, since_at, since_id,
             user_id, since_at, since_at, since_id, limit),
        ).fetchall()
    finally:
        conn.close()
    results: list[dict] = []
    cursor = (since_at, since_id)
    for row in rows:
        cursor = (row["created_at"], row["id"])
        try:
            parsed = json.loads(row["result_json"])
        except (TypeError, ValueError):
            continue  # blob hỏng — bỏ qua, không chặn quét
        if isinstance(parsed, dict):
            results.append({"created_at": row["created_at"], "result": parsed})
    return results, cursor


def list_recent_results(
    cfg: Config, user_id: str, limit: int = 50
) -> list[dict]:
    """Result dicts (đã parse) MỚI NHẤT trước — records + items của user. CHỈ ĐỌC.

    Dùng cho khóa học (src/course/content.py) gộp suggestions/corrections gần đây
    của 1 tiêu chí. Khác list_results_since (quét tăng dần ASC theo con trỏ), hàm
    này lấy N blob mới nhất (DESC) không trạng thái. `limit` chặn payload.
    """
    user_id = validate_user_id(user_id)
    limit = max(1, min(int(limit), 300))
    conn = _connect(cfg)
    try:
        rows = conn.execute(
            """
            SELECT created_at, id, result_json FROM history_records
              WHERE user_id = ? AND result_json IS NOT NULL
            UNION ALL
            SELECT i.created_at, i.id, i.result_json FROM history_items i
              JOIN history_records r ON r.id = i.record_id
              WHERE r.user_id = ? AND i.result_json IS NOT NULL
            ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (user_id, user_id, limit),
        ).fetchall()
    finally:
        conn.close()
    out: list[dict] = []
    for row in rows:
        try:
            parsed = json.loads(row["result_json"])
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def get_audio_path(
    cfg: Config, user_id: str, record_id: str, item_id: str | None = None
) -> Path | None:
    """Path tuyệt đối tới file audio đã lưu (đã verify nằm dưới audio root)."""
    user_id = validate_user_id(user_id)
    conn = _connect(cfg)
    try:
        if item_id:
            row = conn.execute(
                "SELECT i.audio_path FROM history_items i"
                " JOIN history_records r ON r.id = i.record_id"
                " WHERE i.id = ? AND i.record_id = ? AND r.user_id = ?",
                (item_id, record_id, user_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT audio_path FROM history_records WHERE id = ? AND user_id = ?",
                (record_id, user_id),
            ).fetchone()
    finally:
        conn.close()
    if row is None or not row[0]:
        return None
    root = _audio_root(cfg)
    path = (root / row[0]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path


def list_audio_paths(
    cfg: Config, user_id: str, record_id: str
) -> list[tuple[str, Path]] | None:
    """Mọi audio đã lưu của 1 bản ghi: [(tên file trong zip, path tuyệt đối)].

    None nếu bản ghi không tồn tại/sai user; [] nếu bản ghi không có audio.
    Tên trong zip: "audio.<ext>" cho audio cấp record (single), item thì
    "<seq+1:02d>-<label>.<ext>" để giữ thứ tự bài và không đụng tên nhau.
    """
    user_id = validate_user_id(user_id)
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT audio_path FROM history_records WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        ).fetchone()
        if row is None:
            return None
        items = conn.execute(
            "SELECT seq, label, audio_path FROM history_items"
            " WHERE record_id = ? AND audio_path IS NOT NULL"
            " ORDER BY seq, created_at",
            (record_id,),
        ).fetchall()
    finally:
        conn.close()
    root = _audio_root(cfg)
    out: list[tuple[str, Path]] = []

    def _add(stem: str, rel: str | None) -> None:
        if not rel:
            return
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return
        out.append((f"{stem}{path.suffix}", path))

    _add("audio", row["audio_path"])
    for i, it in enumerate(items):
        seq = it["seq"] if it["seq"] is not None else i
        label = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", it["label"] or "")
        label = re.sub(r"\s+", " ", label).strip()
        _add(f"{int(seq) + 1:02d}-{label or 'bai'}", it["audio_path"])
    return out


def delete_record(cfg: Config, user_id: str, record_id: str) -> bool:
    user_id = validate_user_id(user_id)
    conn = _connect(cfg)
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM history_records WHERE id = ? AND user_id = ?",
                (record_id, user_id),
            )
        deleted = cur.rowcount > 0
    finally:
        conn.close()
    if deleted:
        _rm_audio_dir(cfg, record_id)
    return deleted


def reassign_user(cfg: Config, from_user_id: str, to_user_id: str) -> int:
    """Chuyển toàn bộ bản ghi lịch sử từ user ẩn danh sang user tài khoản (dùng khi
    /auth/claim gộp lịch sử lúc đăng nhập lần đầu). Items đi theo record_id nên chỉ
    cần đổi user_id ở bảng cha. Trả số bản ghi đã chuyển. KHÔNG enforce retention
    ở đây (caller quyết định) — giữ hàm thuần chuyển khoá."""
    from_user_id = validate_user_id(from_user_id)
    to_user_id = validate_user_id(to_user_id)
    if from_user_id == to_user_id:
        return 0
    conn = _connect(cfg)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE history_records SET user_id = ? WHERE user_id = ?",
                (to_user_id, from_user_id),
            )
        return cur.rowcount
    finally:
        conn.close()


def _overall_max(exam: str | None) -> int:
    return 9 if exam == "ielts" else 200

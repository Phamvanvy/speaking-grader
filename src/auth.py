"""Đăng nhập bằng tài khoản (username + mật khẩu) — tuỳ chọn, để đồng bộ lịch sử
chấm bài qua nhiều thiết bị thay vì phụ thuộc UUID trong localStorage của 1 trình
duyệt.

Thiết kế (đồng bộ triết lý src/history.py, src/words.py):
- "Account" = 1 row `users` có `user_id` cố định (uuid) — CHÍNH là khoá dữ liệu mà
  history.py / words.py đã dùng. Đăng nhập chỉ để lấy đúng `user_id` này, nên
  KHÔNG phải sửa logic chấm/lưu; mọi endpoint per-user chạy nguyên vẹn.
- Bảo mật bằng SESSION TOKEN (không dùng user_id làm credential): login trả token
  ngẫu nhiên; các request kèm `Authorization: Bearer <token>` để server suy ra
  user_id. user_id của tài khoản là "khoá" (is_account_user_id) → truy cập trực
  tiếp bằng user_id mà KHÔNG có token bị từ chối (api.py). UUID ẩn danh cũ vẫn mở
  (cách ly mềm như trước) để tương thích ngược.
- Mật khẩu: hash bằng hashlib.scrypt (stdlib, không thêm dependency). Lưu ĐẦY ĐỦ
  metadata theo dạng PHC-like tự mô tả: `scrypt$ln=<log2 n>,r=..,p=..$<salt_b64>$
  <dk_b64>` → đổi tham số/thuật toán sau này (vd Argon2id cho môi trường nhiều
  người dùng) vẫn verify được hash cũ. So sánh bằng hmac.compare_digest.
- DB RIÊNG (data/auth.db), WAL + busy_timeout, mỗi hàm mở connection mới, an toàn
  nhiều worker. Schema versioning bằng PRAGMA user_version.

Merge "claim": khi user đăng nhập lần đầu trên máy đang có lịch sử ẩn danh, api.py
gọi history.reassign_user + words.merge_user để chuyển dữ liệu của uuid ẩn danh
sang user_id tài khoản (xem /auth/claim).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
import sqlite3
import time
import uuid
from pathlib import Path

import requests

from .config import Config

logger = logging.getLogger("toeic.auth")

_SCHEMA_VERSION = 2

# Thời hạn session (giây). 30 ngày — cân bằng tiện lợi & rủi ro token bị lộ.
SESSION_TTL_SEC = 30 * 24 * 3600

# Tham số scrypt (đủ mạnh cho web tự host; ~16MB RAM/lần hash). ln = log2(N).
_SCRYPT_LN = 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
# 128 * N * r * p < maxmem; N=2**14, r=8, p=1 → ~16MB. Cho biên rộng để chắc chắn.
_SCRYPT_MAXMEM = 64 * 1024 * 1024

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
# Email "đủ tốt" cho định danh đăng nhập (không cố cover full RFC 5322): có đúng
# 1 @, domain có dấu chấm, không khoảng trắng. KHÔNG gửi mail xác thực (app
# self-hosted, không có SMTP) — email ở đây chỉ là username dễ nhớ.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_EMAIL_LEN = 254
_MIN_PASSWORD_LEN = 8

_DDL = """
CREATE TABLE IF NOT EXISTS users (
  username      TEXT PRIMARY KEY,
  user_id       TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  google_sub    TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  updated_at    TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub
  ON users(google_sub) WHERE google_sub IS NOT NULL;

CREATE TABLE IF NOT EXISTS sessions (
  token        TEXT PRIMARY KEY,
  user_id      TEXT NOT NULL,
  created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  last_used_at TEXT,
  expires_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


# ── Kết nối / schema ─────────────────────────────────────────────────────


def _connect(cfg: Config) -> sqlite3.Connection:
    path = Path(cfg.auth_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < _SCHEMA_VERSION:
        with conn:
            # v1 → v2: thêm cột google_sub TRƯỚC executescript (index trong _DDL
            # tham chiếu cột này — chạy sau ALTER mới hợp lệ). DB mới (v0) đã có
            # cột trong CREATE TABLE nên bỏ qua ALTER.
            if version == 1:
                conn.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
            conn.executescript(_DDL)
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    return conn


# ── Băm / kiểm mật khẩu (self-describing, đổi thuật toán tương lai vẫn verify) ──


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def hash_password(password: str) -> str:
    """Trả chuỗi hash tự mô tả: scrypt$ln=..,r=..,p=..$<salt_b64>$<dk_b64>."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=1 << _SCRYPT_LN,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    params = f"ln={_SCRYPT_LN},r={_SCRYPT_R},p={_SCRYPT_P}"
    return f"scrypt${params}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """So mật khẩu với hash đã lưu. Đọc tham số TỪ chuỗi (không hardcode) → hash cũ
    tạo bằng tham số khác vẫn verify đúng."""
    try:
        algo, params, salt_b64, dk_b64 = stored.split("$")
        if algo != "scrypt":
            logger.warning("Thuật toán hash không hỗ trợ: %s", algo)
            return False
        kv = dict(p.split("=", 1) for p in params.split(","))
        n = 1 << int(kv["ln"])
        r = int(kv["r"])
        p = int(kv["p"])
        salt = _b64d(salt_b64)
        expected = _b64d(dk_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt, n=n, r=r, p=p, dklen=len(expected), maxmem=_SCRYPT_MAXMEM,
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, KeyError, TypeError):
        logger.warning("Hash mật khẩu sai định dạng — từ chối.")
        return False


# ── Validate input ───────────────────────────────────────────────────────


def validate_username(username: str) -> str:
    """Định danh đăng nhập: username HOẶC email (cùng 1 cột `username`).

    Email được lowercase (case-insensitive theo quy ước) để "Vy@X.com" và
    "vy@x.com" không thành 2 tài khoản; username giữ nguyên hoa/thường.
    """
    u = (username or "").strip()
    if "@" in u:
        u = u.lower()
        if len(u) > _MAX_EMAIL_LEN or not _EMAIL_RE.match(u):
            raise ValueError("Email không hợp lệ.")
        return u
    if not _USERNAME_RE.match(u):
        raise ValueError(
            "Tên đăng nhập không hợp lệ (3–32 ký tự, chỉ chữ/số/._-) — hoặc dùng email."
        )
    return u


def _check_password_strength(password: str) -> None:
    if len(password or "") < _MIN_PASSWORD_LEN:
        raise ValueError(f"Mật khẩu phải ít nhất {_MIN_PASSWORD_LEN} ký tự.")


# ── Tài khoản ────────────────────────────────────────────────────────────


class AuthError(Exception):
    """Lỗi nghiệp vụ auth (username trùng, sai mật khẩu…) — api.py map sang HTTP."""


def register(cfg: Config, username: str, password: str) -> dict:
    """Tạo tài khoản mới. Trả {token, user_id, username}. Raise AuthError nếu trùng."""
    username = validate_username(username)
    _check_password_strength(password)
    user_id = str(uuid.uuid4())
    pw_hash = hash_password(password)
    conn = _connect(cfg)
    try:
        try:
            with conn:
                conn.execute(
                    "INSERT INTO users (username, user_id, password_hash)"
                    " VALUES (?, ?, ?)",
                    (username, user_id, pw_hash),
                )
        except sqlite3.IntegrityError as e:
            raise AuthError("Tên đăng nhập đã tồn tại.") from e
        token = _new_session(conn, user_id)
    finally:
        conn.close()
    return {"token": token, "user_id": user_id, "username": username}


def login(cfg: Config, username: str, password: str) -> dict:
    """Xác thực → tạo session. Trả {token, user_id, username}. Raise AuthError."""
    username = validate_username(username)
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT user_id, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        # So hash kể cả khi user không tồn tại (giảm rò rỉ thời gian đoán username).
        stored = row["password_hash"] if row else (
            "scrypt$ln=14,r=8,p=1$AAAAAAAAAAAAAAAAAAAAAA==$"
            + _b64e(b"\x00" * _SCRYPT_DKLEN)
        )
        if not verify_password(password, stored) or row is None:
            raise AuthError("Sai tên đăng nhập hoặc mật khẩu.")
        user_id = row["user_id"]
        token = _new_session(conn, user_id)
    finally:
        conn.close()
    return {"token": token, "user_id": user_id, "username": username}


def change_password(
    cfg: Config, user_id: str, old_password: str, new_password: str
) -> None:
    """Đổi mật khẩu (yêu cầu mật khẩu cũ). Giữ nguyên các session hiện có."""
    _check_password_strength(new_password)
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None or not verify_password(old_password, row["password_hash"]):
            raise AuthError("Mật khẩu hiện tại không đúng.")
        with conn:
            conn.execute(
                "UPDATE users SET password_hash = ?,"
                " updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')"
                " WHERE user_id = ?",
                (hash_password(new_password), user_id),
            )
    finally:
        conn.close()


# ── Đăng nhập với Google (id_token từ Google Identity Services) ──────────
# "Sync user" với app khác cùng project GCP: dùng CHUNG OAuth Client ID → cùng
# tài khoản Google = cùng người, danh tính khớp theo email. KHÔNG cần client
# secret (flow id_token phía web) và KHÔNG đọc Firestore của app kia.


_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
_GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}


def verify_google_credential(client_id: str, credential: str) -> dict:
    """Xác thực id_token (JWT `credential` từ GIS) qua endpoint tokeninfo của
    Google — Google tự kiểm chữ ký + hạn; mình kiểm aud/iss/email_verified.

    Chọn tokeninfo thay vì verify chữ ký cục bộ để KHÔNG thêm dependency
    (google-auth/PyJWT); đổi lấy 1 call HTTPS mỗi lần đăng nhập — chấp nhận được
    với app self-hosted. Trả {sub, email}. Raise AuthError nếu token không hợp lệ.
    """
    try:
        resp = requests.get(
            _GOOGLE_TOKENINFO_URL, params={"id_token": credential}, timeout=10
        )
    except requests.RequestException as e:
        raise AuthError("Không kết nối được Google để xác thực token.") from e
    if resp.status_code != 200:
        raise AuthError("Token Google không hợp lệ hoặc đã hết hạn.")
    data = resp.json()
    if data.get("aud") != client_id:
        raise AuthError("Token không phát hành cho app này (sai client_id).")
    if data.get("iss") not in _GOOGLE_ISSUERS:
        raise AuthError("Token không do Google phát hành.")
    # tokeninfo trả chuỗi "true"/"false" (JSON string, không phải bool).
    if str(data.get("email_verified")).lower() != "true":
        raise AuthError("Email của tài khoản Google chưa được xác minh.")
    sub = data.get("sub")
    email = (data.get("email") or "").strip().lower()
    if not sub or not email:
        raise AuthError("Token Google thiếu thông tin định danh.")
    return {"sub": sub, "email": email}


def google_login(cfg: Config, *, sub: str, email: str) -> dict:
    """Đăng nhập/đăng ký bằng danh tính Google ĐÃ verify (sub + email).

    Thứ tự khớp:
    1. google_sub đã liên kết → đăng nhập (email đổi bên Google không ảnh hưởng).
    2. username == email (tài khoản email+mật khẩu tạo trước đó) → LIÊN KẾT
       google_sub vào tài khoản đó rồi đăng nhập — Google đã xác minh người này
       sở hữu email, nên họ là chủ hợp lệ; đường mật khẩu cũ vẫn dùng được.
    3. Chưa có → tạo tài khoản mới: username = email, password_hash = ''
       (chuỗi rỗng KHÔNG verify được với bất kỳ mật khẩu nào → tài khoản
       Google-only, không có đường đăng nhập mật khẩu).
    """
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT user_id, username FROM users WHERE google_sub = ?", (sub,)
        ).fetchone()
        if row is None:
            linked = conn.execute(
                "SELECT user_id, username FROM users WHERE username = ?", (email,)
            ).fetchone()
            if linked is not None:
                with conn:
                    conn.execute(
                        "UPDATE users SET google_sub = ?,"
                        " updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')"
                        " WHERE user_id = ?",
                        (sub, linked["user_id"]),
                    )
                row = linked
            else:
                user_id = str(uuid.uuid4())
                try:
                    with conn:
                        conn.execute(
                            "INSERT INTO users (username, user_id, password_hash,"
                            " google_sub) VALUES (?, ?, '', ?)",
                            (email, user_id, sub),
                        )
                except sqlite3.IntegrityError:
                    # Race 2 worker cùng tạo — đọc lại bản thắng cuộc.
                    row = conn.execute(
                        "SELECT user_id, username FROM users"
                        " WHERE google_sub = ? OR username = ?",
                        (sub, email),
                    ).fetchone()
                    if row is None:
                        raise AuthError("Không tạo được tài khoản Google.") from None
                else:
                    row = {"user_id": user_id, "username": email}
        token = _new_session(conn, row["user_id"])
    finally:
        conn.close()
    return {"token": token, "user_id": row["user_id"], "username": row["username"]}


# ── Session ──────────────────────────────────────────────────────────────


def _new_session(conn: sqlite3.Connection, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = int(time.time()) + SESSION_TTL_SEC
    with conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at),
        )
    return token


def resolve_session(cfg: Config, token: str | None) -> str | None:
    """token hợp lệ & chưa hết hạn → user_id (kèm cập nhật last_used_at). Ngược lại
    None. Token hết hạn bị xoá luôn (dọn rác nhẹ)."""
    if not token:
        return None
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        if int(row["expires_at"]) < int(time.time()):
            with conn:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
        with conn:
            conn.execute(
                "UPDATE sessions SET last_used_at ="
                " strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE token = ?",
                (token,),
            )
        return row["user_id"]
    finally:
        conn.close()


def logout(cfg: Config, token: str | None) -> None:
    if not token:
        return
    conn = _connect(cfg)
    try:
        with conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    finally:
        conn.close()


def get_account(cfg: Config, user_id: str) -> dict | None:
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT username, user_id, created_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def is_account_user_id(cfg: Config, user_id: str) -> bool:
    """True nếu user_id thuộc 1 tài khoản (→ chỉ truy cập được khi có session token).

    Dùng cho api.py để KHOÁ dữ liệu tài khoản: không cho ai truyền thẳng user_id
    của tài khoản mà không đăng nhập. UUID ẩn danh (không có trong users) trả False
    → vẫn mở như cơ chế cách ly mềm cũ.
    """
    conn = _connect(cfg)
    try:
        row = conn.execute(
            "SELECT 1 FROM users WHERE user_id = ? LIMIT 1", (user_id,)
        ).fetchone()
    finally:
        conn.close()
    return row is not None

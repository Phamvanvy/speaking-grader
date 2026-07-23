"""Quyền admin: nhận diện admin theo ENV + cấp phép xem dữ liệu user khác.

Kiểm ở tầng logic (auth/history + api._authz_user_id) chứ không dựng HTTP server,
vì mọi endpoint per-user đều đi qua _authz_user_id nên chặn ở đó là đủ.
"""

from __future__ import annotations

import dataclasses

import pytest

from src import auth, history
from src.config import load_config


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        anthropic_api_key=None,
        admin_users="boss@x.com, Chief",
        auth_db_path=str(tmp_path / "auth.db"),
        history_db_path=str(tmp_path / "history.db"),
        history_audio_dir=str(tmp_path / "history_audio"),
    )


def _save(cfg, user_id):
    return history.save_single(
        cfg,
        user_id=user_id,
        filename="t.webm",
        mode="practice",
        audio_bytes=b"xx",
        suffix=".webm",
        result={"exam": "toeic", "scores": {}, "transcript": "hi"},
    )


def test_admin_user_set_parsing(cfg):
    assert cfg.admin_user_set == {"boss@x.com", "chief"}


def test_is_admin_matches_email_case_insensitive(cfg):
    admin = auth.register(cfg, "Boss@X.com", "password1")  # email → lowercase
    assert auth.is_admin_user_id(cfg, admin["user_id"]) is True


def test_is_admin_matches_username(cfg):
    # Username giữ nguyên hoa/thường khi lưu; so khớp không phân biệt hoa/thường.
    admin = auth.register(cfg, "chief", "password1")
    assert auth.is_admin_user_id(cfg, admin["user_id"]) is True


def test_non_admin_account_is_not_admin(cfg):
    normal = auth.register(cfg, "learner@x.com", "password1")
    assert auth.is_admin_user_id(cfg, normal["user_id"]) is False


def test_anonymous_uuid_never_admin(cfg):
    assert auth.is_admin_user_id(cfg, "anon-browser-uuid-123") is False


def test_no_admins_configured_disables(cfg, tmp_path):
    cfg2 = dataclasses.replace(cfg, admin_users="")
    admin = auth.register(cfg2, "boss@x.com", "password1")
    assert auth.is_admin_user_id(cfg2, admin["user_id"]) is False


def test_authz_admin_can_read_other_account(cfg):
    """Admin (kèm Bearer token của mình) đọc được dữ liệu tài khoản người khác."""
    from src import api

    admin = auth.register(cfg, "boss@x.com", "password1")
    victim = auth.register(cfg, "learner@x.com", "password1")

    with _patched_config(api, cfg):
        # Không token → tài khoản victim bị khoá.
        with pytest.raises(Exception):
            api._authz_user_id(None, victim["user_id"])
        # Token admin → được phép truyền user_id của victim.
        got = api._authz_user_id(f"Bearer {admin['token']}", victim["user_id"])
        assert got == victim["user_id"]


def test_authz_non_admin_cannot_read_other_account(cfg):
    from src import api

    intruder = auth.register(cfg, "sneaky@x.com", "password1")
    victim = auth.register(cfg, "learner@x.com", "password1")
    with _patched_config(api, cfg):
        with pytest.raises(Exception):
            api._authz_user_id(f"Bearer {intruder['token']}", victim["user_id"])


def test_list_all_users_aggregates(cfg):
    _save(cfg, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    _save(cfg, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    _save(cfg, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    data = history.list_all_users(cfg, limit=50, offset=0)
    assert data["total"] == 2
    counts = {u["user_id"]: u["record_count"] for u in data["users"]}
    assert counts["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"] == 2
    assert counts["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"] == 1


class _patched_config:
    """Tạm trỏ api._BASE_CONFIG sang cfg test (module đọc config lúc import)."""

    def __init__(self, api_mod, cfg):
        self.api = api_mod
        self.cfg = cfg
        self.old = None

    def __enter__(self):
        self.old = self.api._BASE_CONFIG
        self.api._BASE_CONFIG = self.cfg
        return self

    def __exit__(self, *exc):
        self.api._BASE_CONFIG = self.old

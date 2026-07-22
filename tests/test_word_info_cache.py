"""Guard hợp đồng dữ liệu `word_info_cache` mà vòng chơi khóa học phụ thuộc.

CourseGameSession nạp qua /word-info rồi dùng `meaning` (Word Match) và
`example_en` (Sentence Builder). Backend KHÔNG đổi cho tính năng này, nhưng
trước đây không có test nào chốt rằng cả `meaning` lẫn `example_en` sống sót qua
kho cache — một regression làm rớt một trong hai sẽ âm thầm bỏ game tương ứng.
Test thuần (không LLM/không server): round-trip put→get + upsert giữ đủ 2 trường.
"""

from __future__ import annotations

import dataclasses

import pytest

from src import words
from src.config import load_config


@pytest.fixture()
def cfg(tmp_path):
    return dataclasses.replace(
        load_config(),
        anthropic_api_key=None,  # LLM thật không bao giờ được gọi trong test
        words_db_path=str(tmp_path / "words.db"),
    )


def test_word_info_roundtrip_keeps_meaning_and_example(cfg):
    words.put_word_info(
        cfg, "advantage", "vi",
        definition_en="a favorable condition",
        example_en="Speed is a key advantage in this sport.",
        meaning="lợi thế",
    )
    got = words.get_word_info(cfg, "advantage", "vi")
    assert got is not None
    # Hai trường vòng chơi tiêu thụ phải còn nguyên vẹn.
    assert got["meaning"] == "lợi thế"
    assert got["example_en"] == "Speed is a key advantage in this sport."
    assert got["definition_en"] == "a favorable condition"


def test_word_info_upsert_overwrites_both_fields(cfg):
    words.put_word_info(cfg, "market", "vi", "old def", "old example one two", "cũ")
    words.put_word_info(cfg, "market", "vi", "new def", "new example three four", "mới")
    got = words.get_word_info(cfg, "market", "vi")
    assert got is not None
    assert got["meaning"] == "mới"
    assert got["example_en"] == "new example three four"


def test_word_info_missing_returns_none(cfg):
    assert words.get_word_info(cfg, "nonexistent", "vi") is None

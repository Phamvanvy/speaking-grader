"""Test offline cho chọn dạng câu + validate kỳ thi ở tầng API.

Gọi thẳng các helper thuần (không cần TestClient/audio): _pick_question_type,
_validate_exam, _has_provided_info, _overall_score. Phủ: auto-detect IELTS,
chống cross-exam, provided_info rỗng/"null" không bị nhận nhầm Part 2, và tương
thích ngược TOEIC (không truyền exam → vẫn đọc estimated_toeic_score).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api import (
    _has_provided_info,
    _overall_score,
    _pick_question_type,
    _validate_exam,
)

_CUE_CARD = "Describe a memorable trip.\nYou should say: where you went..."


# --- _validate_exam ----------------------------------------------------------

def test_validate_exam_accepts_known():
    assert _validate_exam("toeic") == "toeic"
    assert _validate_exam("IELTS") == "ielts"  # chuẩn hoá hoa/thường


def test_validate_exam_rejects_typo():
    with pytest.raises(HTTPException) as e:
        _validate_exam("ielt")
    assert e.value.status_code == 400


# --- _has_provided_info ------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "   ", "null", "NULL", " null "])
def test_has_provided_info_false_for_blank_or_null(value):
    assert _has_provided_info(value) is False


def test_has_provided_info_true_for_real_text():
    assert _has_provided_info(_CUE_CARD) is True


# --- _pick_question_type : IELTS --------------------------------------------

def test_ielts_provided_info_picks_part2():
    qt = _pick_question_type(None, False, _CUE_CARD, None, "ielts")
    assert qt.key == "part2_long_turn"


def test_ielts_explicit_question_type_honored():
    qt = _pick_question_type(None, False, None, "part3_discussion", "ielts")
    assert qt.key == "part3_discussion"


def test_ielts_no_hint_requires_explicit_type():
    # Không provided_info + không question_type → KHÔNG đoán Part 1 vs Part 3 → 400.
    with pytest.raises(HTTPException) as e:
        _pick_question_type(None, False, None, None, "ielts")
    assert e.value.status_code == 400


def test_ielts_blank_provided_info_not_treated_as_part2():
    # provided_info='' / 'null' không được nhận nhầm thành Part 2.
    for blank in ("", "null"):
        with pytest.raises(HTTPException) as e:
            _pick_question_type(None, False, blank, None, "ielts")
        assert e.value.status_code == 400


# --- Cross-exam guard --------------------------------------------------------

def test_cross_exam_toeic_type_under_ielts_is_400():
    with pytest.raises(HTTPException) as e:
        _pick_question_type(None, False, None, "read_aloud", "ielts")
    assert e.value.status_code == 400


def test_cross_exam_ielts_type_under_toeic_is_400():
    with pytest.raises(HTTPException) as e:
        _pick_question_type(None, False, None, "part3_discussion", "toeic")
    assert e.value.status_code == 400


# --- Tương thích ngược TOEIC -------------------------------------------------

def test_toeic_text_autodetects_read_aloud():
    qt = _pick_question_type("reference script here", False, None, None, "toeic")
    assert qt.key == "read_aloud"
    assert qt.exam == "toeic"


def test_toeic_image_autodetects_describe_picture():
    qt = _pick_question_type(None, True, None, None, "toeic")
    assert qt.key == "describe_picture"


# --- _overall_score (float thống nhất) --------------------------------------

def test_overall_score_toeic_returns_float():
    val = _overall_score({"estimated_toeic_score": 120}, "toeic")
    assert val == 120.0 and isinstance(val, float)


def test_overall_score_ielts_reads_band():
    assert _overall_score({"estimated_ielts_band": 6.5}, "ielts") == 6.5


def test_overall_score_none_when_missing():
    assert _overall_score({}, "ielts") is None
    assert _overall_score(None, "toeic") is None

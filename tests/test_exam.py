"""Test luồng "Thi cả đề" — hàm thuần (validate/aggregate) + endpoint (mock LLM)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src import api
from src.exam_import import ExtractedExam, ExtractedQuestion, validate_extracted
from src.scoring import compute_exam_overall


# ── validate_extracted: KHÔNG tin LLM ────────────────────────────────────────


def test_validate_drops_unknown_type_and_resequences():
    raw = ExtractedExam(
        title="T",
        questions=[
            ExtractedQuestion(type="read_aloud", reference_script="Hello."),
            ExtractedQuestion(type="not_a_real_type", prompt="x"),
            ExtractedQuestion(type="express_opinion", prompt="Why?"),
        ],
    )
    qs, warnings = validate_extracted(raw, "toeic", [])
    assert [q.type for q in qs] == ["read_aloud", "express_opinion"]
    # sequence do server gán liên tục, không có khoảng trống dù 1 câu bị loại.
    assert [q.sequence for q in qs] == [1, 2]
    assert any("không thuộc kỳ thi" in w for w in warnings)


def test_validate_attaches_image_and_fills_default_duration():
    raw = ExtractedExam(questions=[ExtractedQuestion(type="describe_picture", image_index=0)])
    qs, _ = validate_extracted(raw, "toeic", [("B64DATA", "image/png")])
    assert qs[0].image_b64 == "B64DATA"
    assert qs[0].expected_duration_sec == 45  # default describe_picture


def test_validate_warns_missing_required_field():
    raw = ExtractedExam(questions=[ExtractedQuestion(type="read_aloud")])  # thiếu script
    qs, warnings = validate_extracted(raw, "toeic", [])
    assert len(qs) == 1
    assert any("reference_script" in w for w in warnings)


def test_validate_out_of_range_image_index_falls_back():
    raw = ExtractedExam(questions=[ExtractedQuestion(type="describe_picture", image_index=9)])
    qs, _ = validate_extracted(raw, "toeic", [("FIRST", "image/png")])
    assert qs[0].image_b64 == "FIRST"  # kẹp về ảnh đầu thay vì rỗng


# ── compute_exam_overall ─────────────────────────────────────────────────────


def test_overall_toeic_rounds_to_10():
    out = compute_exam_overall(
        "toeic",
        [{"estimated_toeic_score": 120}, {"estimated_toeic_score": 150}, None, {"estimated_toeic_score": 80}],
    )
    assert out == 120  # mean 116.67 → 120


def test_overall_ielts_rounds_to_half():
    out = compute_exam_overall(
        "ielts",
        [{"estimated_ielts_band": 6.0}, {"estimated_ielts_band": 6.5}, {"estimated_ielts_band": 7.0}],
    )
    assert out == 6.5


def test_overall_none_when_no_scores():
    assert compute_exam_overall("toeic", [None, {}]) is None


# ── Endpoint /exam/builtin ───────────────────────────────────────────────────


def test_exam_builtin_returns_ordered_questions():
    client = TestClient(api.app)
    res = client.get("/exam/builtin/toeic")
    assert res.status_code == 200
    data = res.json()
    assert data["exam"] == "toeic"
    assert len(data["questions"]) >= 1
    seqs = [q["sequence"] for q in data["questions"]]
    assert seqs == sorted(seqs)


def test_exam_import_rejects_unsupported_format():
    client = TestClient(api.app)
    res = client.post(
        "/exam/import",
        data={"exam": "toeic"},
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400


# ── Endpoint /exam/grade (mock _grade_bytes — không gọi LLM/ASR thật) ─────────


def test_exam_grade_maps_by_question_id_and_aggregates(monkeypatch):
    def fake_grade_bytes(audio_bytes, suffix, config, qt, **kwargs):
        # Điểm phụ thuộc prompt để chắc map đúng câu (không lẫn theo index).
        score = 100 if "first" in kwargs.get("prompt", "") else 160
        return {
            "exam": qt.exam,
            "transcript": "ok",
            "features": {},
            "scores": {"estimated_toeic_score": score},
            "phoneme": None,
            "telemetry": {},
        }

    monkeypatch.setattr(api, "_grade_bytes", fake_grade_bytes)
    client = TestClient(api.app)

    paper = {
        "exam": "toeic",
        "title": "Đề test",
        "questions": [
            {"id": "qa", "sequence": 1, "type": "respond_questions", "prompt": "first question"},
            {"id": "qb", "sequence": 2, "type": "express_opinion", "prompt": "second question"},
        ],
    }
    # Cố tình gửi audio THEO THỨ TỰ NGƯỢC để kiểm tra map theo question_id, không theo index.
    res = client.post(
        "/exam/grade",
        data={
            "paper": json.dumps(paper),
            "audio_question_ids": json.dumps(["qb", "qa"]),
            "mode": "practice",
        },
        files=[
            ("audios", ("b.wav", b"BBBB", "audio/wav")),
            ("audios", ("a.wav", b"AAAA", "audio/wav")),
        ],
    )
    assert res.status_code == 200
    data = res.json()
    assert data["graded"] == 2
    assert data["overall_estimated"] is True
    # overall = mean(100, 160) = 130
    assert data["overall"] == 130
    # kết quả sort theo sequence; câu qa (first→100) đứng trước
    by_seq = data["questions"]
    assert by_seq[0]["question_id"] == "qa"
    assert by_seq[0]["result"]["scores"]["estimated_toeic_score"] == 100
    assert by_seq[1]["question_id"] == "qb"


def test_exam_grade_skips_question_without_audio(monkeypatch):
    monkeypatch.setattr(
        api, "_grade_bytes",
        lambda *a, **k: {"exam": "toeic", "scores": {"estimated_toeic_score": 140}, "features": {}},
    )
    client = TestClient(api.app)
    paper = {
        "exam": "toeic",
        "questions": [
            {"id": "qa", "sequence": 1, "type": "respond_questions", "prompt": "p"},
            {"id": "qb", "sequence": 2, "type": "express_opinion", "prompt": "p"},
        ],
    }
    res = client.post(
        "/exam/grade",
        data={"paper": json.dumps(paper), "audio_question_ids": json.dumps(["qa"])},
        files=[("audios", ("a.wav", b"AAAA", "audio/wav"))],
    )
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 2
    assert data["graded"] == 1
    assert data["overall"] == 140

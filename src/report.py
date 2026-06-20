"""Định dạng kết quả: in ra console (rich nếu có) + lưu JSON đầy đủ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import FEATURES_VERSION

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def build_output(
    audio_path: str,
    question_id: str,
    question_type: str,
    transcript: str,
    features: dict,
    scores: dict | None,
    telemetry: dict | None = None,
    exam: str = "toeic",
    phoneme: dict | None = None,
) -> dict[str, Any]:
    """Gom toàn bộ kết quả thành 1 dict để lưu JSON (đầy đủ để debug sau).

    phoneme: bản gọn của phoneme analysis cho UI/JSON ({backend_used, warning,
        score:{overall_accuracy, errors[...]}}); None nếu không có.
    """
    return {
        "audio_path": audio_path,
        "question_id": question_id,
        "question_type": question_type,
        "exam": exam,
        "features_version": FEATURES_VERSION,
        "transcript": transcript,
        "features": features,
        "scores": scores,
        "phoneme": phoneme,
        "telemetry": telemetry or {},
    }


def _score_display(output: dict[str, Any], scores: dict[str, Any]) -> tuple[str, str]:
    """Trả về (criterion_suffix, overall_line) theo kỳ thi của output.

    TOEIC: ('/3', 'ĐIỂM TOEIC ƯỚC TÍNH: 120/200'); IELTS: ('/9', 'IELTS BAND
    ƯỚC TÍNH: 6.5/9'). Dùng chung cho cả bản plain lẫn rich.
    """
    if output.get("exam") == "ielts":
        band = scores.get("estimated_ielts_band")
        return "/9", f"IELTS BAND ƯỚC TÍNH: {band if band is not None else '--'}/9"
    toeic = scores.get("estimated_toeic_score")
    return "/3", f"ĐIỂM TOEIC ƯỚC TÍNH: {toeic if toeic is not None else '--'}/200"


def save_json(output: dict[str, Any], stem: str) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUTPUT_DIR / f"{stem}.json"
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _word_issue_lines(features: dict[str, Any]) -> list[str]:
    """Tóm tắt word_issues thành các dòng người đọc được (rỗng nếu không có).

    Diễn đạt theo hướng "ASR nghe khác script — nên review", KHÔNG khẳng định
    sai phát âm. Xem features.WordIssue để biết vì sao.
    """
    acc = features.get("accuracy_metrics")
    if not acc:
        return []
    issues = acc.get("word_issues") or []
    lines: list[str] = []
    for it in issues:
        t = it.get("issue_type")
        exp, rec = it.get("expected"), it.get("recognized")
        if t == "substitution":
            lines.append(f"'{exp}' → ASR nghe thành '{rec}' (nên review)")
        elif t == "deletion":
            lines.append(f"'{exp}' → không nghe thấy trong bản đọc (thiếu/bỏ?)")
        elif t == "insertion":
            lines.append(f"thừa từ '{rec}' không có trong script")
    return lines


def print_report(output: dict[str, Any]) -> None:
    try:
        _print_rich(output)
    except ImportError:
        _print_plain(output)


def _print_plain(output: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"Câu hỏi : {output['question_id']} ({output['question_type']})")
    print(f"Audio   : {output['audio_path']}")
    print("-" * 60)
    print("TRANSCRIPT:")
    print(output["transcript"] or "(rỗng)")
    print("-" * 60)
    print("ĐẶC TRƯNG (features):")
    for k, v in output["features"].items():
        print(f"  {k}: {v}")

    issue_lines = _word_issue_lines(output["features"])
    if issue_lines:
        print("-" * 60)
        print("TỪ CẦN REVIEW (ASR nghe khác script — KHÔNG chắc chắn sai phát âm):")
        for line in issue_lines:
            print(f"  • {line}")

    scores = output.get("scores")
    if not scores:
        print("-" * 60)
        print("(Chưa chấm điểm — chạy không có --no-ai để Claude chấm.)")
        print("=" * 60 + "\n")
        return

    crit_suffix, overall_line = _score_display(output, scores)
    print("-" * 60)
    print(f"TASK COMPLETION : {scores['task_completion']}")
    print(f"CONTENT RELEVANCE: {scores['content_relevance']}")
    print("\nĐIỂM TỪNG TIÊU CHÍ:")
    for c in scores["criteria"]:
        print(f"  [{c['score']}{crit_suffix}] {c['criterion']}: {c['justification']}")
        for s in c.get("suggestions", []):
            print(f"        → {s}")
    print(f"\n{overall_line}")
    rationale = scores.get("score_rationale")
    if rationale:
        print("\nLÝ DO RA ĐIỂM:")
        print(rationale)
    print("\nNHẬN XÉT CHUNG:")
    print(scores["summary_feedback"])
    print("=" * 60 + "\n")


def _print_rich(output: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    exam_title = "IELTS Speaking" if output.get("exam") == "ielts" else "TOEIC Speaking"
    console.print(
        Panel.fit(
            f"[bold]{output['question_id']}[/bold] "
            f"([cyan]{output['question_type']}[/cyan])\n{output['audio_path']}",
            title=exam_title,
        )
    )

    console.print(Panel(output["transcript"] or "(rỗng)", title="Transcript"))

    feat_table = Table(title="Đặc trưng khách quan", show_header=True)
    feat_table.add_column("Chỉ số")
    feat_table.add_column("Giá trị", justify="right")
    for k, v in output["features"].items():
        feat_table.add_row(k, str(v))
    console.print(feat_table)

    issue_lines = _word_issue_lines(output["features"])
    if issue_lines:
        console.print(
            Panel(
                "\n".join(f"• {line}" for line in issue_lines),
                title="Từ cần review (ASR nghe khác script — chưa chắc sai phát âm)",
            )
        )

    scores = output.get("scores")
    if not scores:
        console.print("[yellow](Chưa chấm điểm — bỏ --no-ai để Claude chấm.)[/yellow]")
        return

    console.print(
        f"[bold]Task completion:[/bold] {scores['task_completion']}   "
        f"[bold]Relevance:[/bold] {scores['content_relevance']}"
    )

    crit_suffix, overall_line = _score_display(output, scores)
    crit_table = Table(title="Điểm từng tiêu chí", show_header=True)
    crit_table.add_column("Tiêu chí")
    crit_table.add_column("Điểm", justify="center")
    crit_table.add_column("Nhận xét")
    for c in scores["criteria"]:
        suggestions = "\n".join(f"• {s}" for s in c.get("suggestions", []))
        body = c["justification"]
        if suggestions:
            body += f"\n[dim]{suggestions}[/dim]"
        crit_table.add_row(c["criterion"], f"{c['score']}{crit_suffix}", body)
    console.print(crit_table)

    rationale = scores.get("score_rationale")
    if rationale:
        console.print(Panel(rationale, title="Lý do ra điểm"))

    console.print(
        Panel(
            scores["summary_feedback"],
            title=overall_line,
        )
    )

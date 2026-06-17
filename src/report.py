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
) -> dict[str, Any]:
    """Gom toàn bộ kết quả thành 1 dict để lưu JSON (đầy đủ để debug sau)."""
    return {
        "audio_path": audio_path,
        "question_id": question_id,
        "question_type": question_type,
        "features_version": FEATURES_VERSION,
        "transcript": transcript,
        "features": features,
        "scores": scores,
    }


def save_json(output: dict[str, Any], stem: str) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUTPUT_DIR / f"{stem}.json"
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


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

    scores = output.get("scores")
    if not scores:
        print("-" * 60)
        print("(Chưa chấm điểm — chạy không có --no-ai để Claude chấm.)")
        print("=" * 60 + "\n")
        return

    print("-" * 60)
    print(f"TASK COMPLETION : {scores['task_completion']}")
    print(f"CONTENT RELEVANCE: {scores['content_relevance']}")
    print("\nĐIỂM TỪNG TIÊU CHÍ:")
    for c in scores["criteria"]:
        print(f"  [{c['score']}/3] {c['criterion']}: {c['justification']}")
        for s in c.get("suggestions", []):
            print(f"        → {s}")
    print(f"\nĐIỂM TOEIC ƯỚC TÍNH: {scores['estimated_toeic_score']}/200")
    print("\nNHẬN XÉT CHUNG:")
    print(scores["summary_feedback"])
    print("=" * 60 + "\n")


def _print_rich(output: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(
        Panel.fit(
            f"[bold]{output['question_id']}[/bold] "
            f"([cyan]{output['question_type']}[/cyan])\n{output['audio_path']}",
            title="TOEIC Speaking",
        )
    )

    console.print(Panel(output["transcript"] or "(rỗng)", title="Transcript"))

    feat_table = Table(title="Đặc trưng khách quan", show_header=True)
    feat_table.add_column("Chỉ số")
    feat_table.add_column("Giá trị", justify="right")
    for k, v in output["features"].items():
        feat_table.add_row(k, str(v))
    console.print(feat_table)

    scores = output.get("scores")
    if not scores:
        console.print("[yellow](Chưa chấm điểm — bỏ --no-ai để Claude chấm.)[/yellow]")
        return

    console.print(
        f"[bold]Task completion:[/bold] {scores['task_completion']}   "
        f"[bold]Relevance:[/bold] {scores['content_relevance']}"
    )

    crit_table = Table(title="Điểm từng tiêu chí", show_header=True)
    crit_table.add_column("Tiêu chí")
    crit_table.add_column("Điểm", justify="center")
    crit_table.add_column("Nhận xét")
    for c in scores["criteria"]:
        suggestions = "\n".join(f"• {s}" for s in c.get("suggestions", []))
        body = c["justification"]
        if suggestions:
            body += f"\n[dim]{suggestions}[/dim]"
        crit_table.add_row(c["criterion"], f"{c['score']}/3", body)
    console.print(crit_table)

    console.print(
        Panel(
            scores["summary_feedback"],
            title=f"Điểm TOEIC ước tính: {scores['estimated_toeic_score']}/200",
        )
    )

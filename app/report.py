"""
Rich console reporting for the triage pipeline.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from app.evaluate import EvalReport
from app.schema import PipelineRun, Priority

console = Console()

# Priority color map
PRIORITY_COLORS = {
    "P0": "bold red",
    "P1": "red",
    "P2": "yellow",
    "P3": "green",
}

CATEGORY_ICONS = {
    "billing": "💳",
    "technical": "🔧",
    "account": "🔑",
    "feature_request": "💡",
    "complaint": "😤",
    "out_of_scope": "🚫",
    "general_inquiry": "❓",
}


def print_run_summary(run: PipelineRun) -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Messages processed:[/bold] {run.total}\n"
            f"[bold green]Successful:[/bold green]  {run.successful}\n"
            f"[bold red]Failed:[/bold red]       {run.failed}\n"
            f"[bold]Avg latency:[/bold]  {run.avg_latency_ms:.0f} ms\n"
            f"[bold]Total runtime:[/bold] {run.total_runtime_seconds:.1f} sec\n"
            f"[bold]Prompt tokens:[/bold]  {run.total_prompt_tokens:,}\n"
            f"[bold]Completion tokens:[/bold] {run.total_completion_tokens:,}",
            title="[bold cyan]Pipeline Run Summary[/bold cyan]",
            border_style="cyan",
        )
    )


def print_results_table(run: PipelineRun) -> None:
    table = Table(
        title="Triage Results",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
    )

    table.add_column("ID", style="dim", width=9)
    table.add_column("Priority", width=6, justify="center")
    table.add_column("Category", width=16)
    table.add_column("Human?", width=7, justify="center")
    table.add_column("Conf.", width=6, justify="right")
    table.add_column("Summary", ratio=2)
    table.add_column("Action", ratio=2)

    for r in run.results:
        pri_color = PRIORITY_COLORS.get(r.priority.value, "white")
        icon = CATEGORY_ICONS.get(r.category.value, "")
        human_mark = "[bold red]YES[/bold red]" if r.needs_human else "[green]no[/green]"

        table.add_row(
            r.message_id,
            Text(r.priority.value, style=pri_color),
            f"{icon} {r.category.value}",
            human_mark,
            f"{r.confidence:.2f}",
            r.summary[:120],
            r.suggested_action[:120],
        )

    console.print(table)

    if run.failures:
        console.print()
        fail_table = Table(
            title="[bold red]Failed Messages[/bold red]",
            box=box.ROUNDED,
        )
        fail_table.add_column("ID")
        fail_table.add_column("Error")
        for f in run.failures:
            fail_table.add_row(f.message_id, f.error[:200])
        console.print(fail_table)


def print_eval_report(report: EvalReport, run: PipelineRun) -> None:
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Ground-truth cases:[/bold] {report.ground_truth_total}\n\n"
            f"[bold cyan]Category accuracy:[/bold cyan]   {report.category_accuracy:.0%}  "
            f"({report.category_correct}/{report.ground_truth_total})\n"
            f"[bold cyan]Priority accuracy:[/bold cyan]   {report.priority_accuracy:.0%}  "
            f"({report.priority_correct}/{report.ground_truth_total})\n"
            f"[bold cyan]Human-esc accuracy:[/bold cyan] {report.needs_human_accuracy:.0%}  "
            f"({report.needs_human_correct}/{report.ground_truth_total})\n"
            f"[bold cyan]Exact-match rate:[/bold cyan]   {report.exact_match_rate:.0%}  "
            f"({report.exact_match}/{report.ground_truth_total})",
            title="[bold cyan]Evaluation Results[/bold cyan]",
            border_style="cyan",
        )
    )

    if report.disagreements:
        console.print()
        console.print("[bold red]Disagreements with ground truth:[/bold red]")
        for d in report.disagreements:
            console.print(
                Panel(
                    f"[bold]Message:[/bold] {d['message_text']}\n\n"
                    f"[bold green]Expected:[/bold green] {d['expected']}\n"
                    f"[bold red]Actual:  [/bold red] {d.get('actual', 'FAILED')}\n\n"
                    f"[bold yellow]Reason:[/bold yellow] {d['reason']}",
                    title=f"[dim]{d['message_id']}[/dim]",
                    border_style="red",
                )
            )
    else:
        console.print("[bold green]All ground-truth cases matched![/bold green]")

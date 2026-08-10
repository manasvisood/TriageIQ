"""
FRONTLINE — Customer Message Triage System
CLI entry point.

Usage:
    python main.py                        # Run on default dataset
    python main.py --dataset data/messages.json
    python main.py --output results/run.json
    python main.py --dataset data/messages.json --evaluate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

load_dotenv()

from app.pipeline import run_pipeline, save_results
from app.evaluate import evaluate
from app.report import print_run_summary, print_results_table, print_eval_report

console = Console()

DEFAULT_DATASET = "data/messages.json"
DEFAULT_OUTPUT = "results/triage_results.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FRONTLINE — AI Customer Message Triage System"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to input dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to save results JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Run evaluation against ground truth (default: True)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_false",
        dest="evaluate",
        help="Skip evaluation",
    )
    args = parser.parse_args()

    console.print()
    console.rule("[bold cyan]FRONTLINE — Customer Message Triage[/bold cyan]")
    console.print()

    # Quick API key validation before processing any messages
    if not os.environ.get("GROQ_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] GROQ_API_KEY is not set.\n"
            "1. Copy [bold].env.example[/bold] to [bold].env[/bold]\n"
            "2. Add your Groq API key (free at https://console.groq.com)\n"
            "3. Re-run: [bold]python main.py[/bold]"
        )
        return 1

    # Progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Triaging messages...", total=None)

        total_msgs = [0]

        def on_progress(current: int, total: int, msg_id: str) -> None:
            if total_msgs[0] == 0:
                progress.update(task, total=total)
                total_msgs[0] = total
            progress.update(
                task,
                completed=current,
                description=f"Triaging [dim]{msg_id}[/dim]",
            )

        try:
            run = run_pipeline(args.dataset, progress_callback=on_progress)
        except FileNotFoundError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            return 1
        except Exception as exc:
            console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
            return 1

    # Display results
    print_run_summary(run)
    print_results_table(run)

    # Save results
    save_results(run, args.output)
    console.print(f"\n[dim]Results saved to:[/dim] [bold]{args.output}[/bold]")

    # Evaluation
    if args.evaluate:
        console.print()
        console.rule("[bold cyan]Evaluation[/bold cyan]")
        try:
            report = evaluate(run, args.dataset)
            print_eval_report(report, run)
        except Exception as exc:
            console.print(f"[bold red]Evaluation error:[/bold red] {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

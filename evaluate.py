"""
Standalone evaluation script.
Loads a previously saved results file and re-runs evaluation against ground truth.

Usage:
    python evaluate.py
    python evaluate.py --results results/triage_results.json --dataset data/messages.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

from app.schema import PipelineRun, TriageResult, FailedResult
from app.evaluate import evaluate
from app.report import print_eval_report, print_run_summary

console = Console()

DEFAULT_RESULTS = "results/triage_results.json"
DEFAULT_DATASET = "data/messages.json"


def load_pipeline_run(results_path: str | Path) -> PipelineRun:
    """Re-hydrate a PipelineRun from a saved JSON file."""
    path = Path(results_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}\n"
            "Run 'python main.py' first to generate results."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = [TriageResult(**r) for r in data.get("results", [])]
    failures = [FailedResult(**f) for f in data.get("failures", [])]
    summary = data.get("summary", {})

    return PipelineRun(
        total=summary.get("total", len(results) + len(failures)),
        successful=summary.get("successful", len(results)),
        failed=summary.get("failed", len(failures)),
        results=results,
        failures=failures,
        total_runtime_seconds=summary.get("total_runtime_seconds", 0.0),
        avg_latency_ms=summary.get("avg_latency_ms", 0.0),
        total_prompt_tokens=summary.get("total_prompt_tokens", 0),
        total_completion_tokens=summary.get("total_completion_tokens", 0),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FRONTLINE — Evaluate triage results against ground truth"
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help=f"Path to saved results JSON (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to input dataset with ground truth (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()

    console.print()
    console.rule("[bold cyan]FRONTLINE — Evaluation Report[/bold cyan]")
    console.print()

    try:
        run = load_pipeline_run(args.results)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    print_run_summary(run)

    try:
        report = evaluate(run, args.dataset)
        print_eval_report(report, run)
    except Exception as exc:
        console.print(f"[bold red]Evaluation error:[/bold red] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

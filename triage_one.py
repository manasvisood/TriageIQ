"""
Triage a single customer message interactively.
Great for live demos.

Usage:
    python triage_one.py
    python triage_one.py "I was charged twice this month"
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

load_dotenv()

from app.llm_client import LLMClient
from app.report import PRIORITY_COLORS, CATEGORY_ICONS

console = Console()

PRIORITY_LABELS = {
    "P0": "CRITICAL",
    "P1": "HIGH",
    "P2": "MEDIUM",
    "P3": "LOW",
}


def triage_single(message: str) -> None:
    console.print()
    console.rule("[bold cyan]FRONTLINE — Single Message Triage[/bold cyan]")
    console.print()
    console.print(
        Panel(message, title="[bold]Customer Message[/bold]", border_style="dim")
    )
    console.print()

    with console.status("[bold cyan]Analyzing message...[/bold cyan]"):
        client = LLMClient()
        try:
            result = client.triage("demo-msg", message)
        except Exception as exc:
            console.print(f"[bold red]Triage failed:[/bold red] {exc}")
            return

    pri = result.priority.value
    pri_color = PRIORITY_COLORS.get(pri, "white")
    icon = CATEGORY_ICONS.get(result.category.value, "")
    human_mark = "[bold red]YES — needs human review[/bold red]" if result.needs_human else "[green]No — can be handled automatically[/green]"

    console.print(
        Panel.fit(
            f"[bold]Category:[/bold]          {icon} {result.category.value}\n"
            f"[bold]Priority:[/bold]          [{pri_color}]{pri} — {PRIORITY_LABELS.get(pri, '')}[/{pri_color}]\n"
            f"[bold]Needs Human:[/bold]       {human_mark}\n"
            f"[bold]Confidence:[/bold]        {result.confidence:.0%}\n\n"
            f"[bold]Summary:[/bold]\n  {result.summary}\n\n"
            f"[bold]Suggested Action:[/bold]\n  {result.suggested_action}",
            title="[bold cyan]Triage Decision[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    else:
        console.print("[bold cyan]FRONTLINE[/bold cyan] — Enter a customer message to triage.")
        console.print("[dim]Press Enter twice when done, or Ctrl+C to exit.[/dim]")
        console.print()
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
        msg = "\n".join(lines).strip()
        if not msg:
            console.print("[yellow]No message entered. Exiting.[/yellow]")
            sys.exit(0)

    triage_single(msg)

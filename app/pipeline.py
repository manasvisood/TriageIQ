"""
Triage pipeline — orchestrates end-to-end processing of the dataset.

Responsibilities:
- Load and validate input dataset
- Normalize messages
- Call LLM for each message
- Record latency per message
- Handle individual failures gracefully (pipeline always continues)
- Return a PipelineRun summary
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from app.llm_client import LLMClient
from app.schema import FailedResult, InputMessage, PipelineRun, TriageResult


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path) -> list[InputMessage]:
    """
    Load and validate the messages dataset.
    Raises FileNotFoundError or ValueError on structural problems.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "messages" not in raw or not isinstance(raw["messages"], list):
        raise ValueError("Dataset must be a JSON object with a 'messages' list.")

    messages: list[InputMessage] = []
    for i, item in enumerate(raw["messages"]):
        try:
            messages.append(InputMessage(**item))
        except (ValidationError, TypeError) as exc:
            # Log but skip malformed entries
            print(f"[WARNING] Skipping malformed entry at index {i}: {exc}")

    return messages


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def normalize_message(text: str) -> str:
    """
    Light normalization:
    - Strip leading/trailing whitespace
    - Normalize unicode (NFC)
    - Collapse multiple blank lines to one
    - Truncate at a safe maximum length
    """
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    # Collapse runs of 3+ newlines to 2
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Truncate to 4000 characters (well within token limits)
    return text[:4000]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    dataset_path: str | Path,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> PipelineRun:
    """
    Run the full triage pipeline on a dataset file.

    Args:
        dataset_path: Path to messages.json
        progress_callback: Optional fn(current, total, message_id) for live progress

    Returns:
        PipelineRun with results and metrics
    """
    messages = load_dataset(dataset_path)

    client = LLMClient()
    results: list[TriageResult] = []
    failures: list[FailedResult] = []
    latencies_ms: list[float] = []

    pipeline_start = time.perf_counter()

    for idx, msg in enumerate(messages):
        if progress_callback:
            progress_callback(idx + 1, len(messages), msg.id)

        normalized_text = normalize_message(msg.text)

        msg_start = time.perf_counter()
        try:
            result = client.triage(
                message_id=msg.id,
                message_text=normalized_text,
            )
            results.append(result)
            latencies_ms.append((time.perf_counter() - msg_start) * 1000)

        except Exception as exc:
            # One failure MUST NOT stop the run
            failures.append(
                FailedResult(
                    message_id=msg.id,
                    error=str(exc),
                    raw_response=None,
                )
            )
            latencies_ms.append((time.perf_counter() - msg_start) * 1000)

    total_runtime = time.perf_counter() - pipeline_start
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    return PipelineRun(
        total=len(messages),
        successful=len(results),
        failed=len(failures),
        results=results,
        failures=failures,
        total_runtime_seconds=round(total_runtime, 2),
        avg_latency_ms=round(avg_latency, 1),
        total_prompt_tokens=client.total_prompt_tokens,
        total_completion_tokens=client.total_completion_tokens,
    )


def save_results(run: PipelineRun, output_path: str | Path) -> None:
    """Persist the full pipeline results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": {
            "total": run.total,
            "successful": run.successful,
            "failed": run.failed,
            "total_runtime_seconds": run.total_runtime_seconds,
            "avg_latency_ms": run.avg_latency_ms,
            "total_prompt_tokens": run.total_prompt_tokens,
            "total_completion_tokens": run.total_completion_tokens,
        },
        "results": [r.model_dump() for r in run.results],
        "failures": [f.model_dump() for f in run.failures],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

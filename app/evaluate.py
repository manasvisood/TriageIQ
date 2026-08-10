"""
Evaluation module — measures system accuracy against ground truth.

Only messages with ground_truth defined are evaluated.
Reports per-field accuracy and shows all disagreements honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.pipeline import load_dataset
from app.schema import PipelineRun, TriageResult


@dataclass
class GroundTruthCase:
    message_id: str
    message_text: str
    expected_category: str
    expected_priority: str
    expected_needs_human: bool
    actual: Optional[TriageResult] = None


@dataclass
class EvalReport:
    total_messages: int
    successful: int
    failed: int
    ground_truth_total: int
    category_correct: int
    priority_correct: int
    needs_human_correct: int
    exact_match: int  # all three fields match
    disagreements: list[dict] = field(default_factory=list)

    @property
    def category_accuracy(self) -> float:
        if self.ground_truth_total == 0:
            return 0.0
        return self.category_correct / self.ground_truth_total

    @property
    def priority_accuracy(self) -> float:
        if self.ground_truth_total == 0:
            return 0.0
        return self.priority_correct / self.ground_truth_total

    @property
    def needs_human_accuracy(self) -> float:
        if self.ground_truth_total == 0:
            return 0.0
        return self.needs_human_correct / self.ground_truth_total

    @property
    def exact_match_rate(self) -> float:
        if self.ground_truth_total == 0:
            return 0.0
        return self.exact_match / self.ground_truth_total


def evaluate(run: PipelineRun, dataset_path: str | Path) -> EvalReport:
    """
    Compare pipeline results against ground-truth labels in the dataset.
    """
    messages = load_dataset(dataset_path)

    # Build ground truth cases
    gt_cases: list[GroundTruthCase] = []
    for msg in messages:
        if msg.ground_truth is None:
            continue
        gt = msg.ground_truth
        gt_cases.append(
            GroundTruthCase(
                message_id=msg.id,
                message_text=msg.text,
                expected_category=gt.get("category", ""),
                expected_priority=gt.get("priority", ""),
                expected_needs_human=bool(gt.get("needs_human", False)),
            )
        )

    # Index results by message_id
    result_index: dict[str, TriageResult] = {r.message_id: r for r in run.results}

    category_correct = 0
    priority_correct = 0
    needs_human_correct = 0
    exact_match = 0
    disagreements: list[dict] = []

    for case in gt_cases:
        actual = result_index.get(case.message_id)
        case.actual = actual

        if actual is None:
            disagreements.append(
                {
                    "message_id": case.message_id,
                    "message_text": case.message_text[:120] + "..."
                    if len(case.message_text) > 120
                    else case.message_text,
                    "expected": {
                        "category": case.expected_category,
                        "priority": case.expected_priority,
                        "needs_human": case.expected_needs_human,
                    },
                    "actual": None,
                    "reason": "Message failed during pipeline execution",
                }
            )
            continue

        cat_ok = actual.category.value == case.expected_category
        pri_ok = actual.priority.value == case.expected_priority
        nh_ok = actual.needs_human == case.expected_needs_human

        if cat_ok:
            category_correct += 1
        if pri_ok:
            priority_correct += 1
        if nh_ok:
            needs_human_correct += 1
        if cat_ok and pri_ok and nh_ok:
            exact_match += 1

        if not (cat_ok and pri_ok and nh_ok):
            reasons = []
            if not cat_ok:
                reasons.append(
                    f"category: expected '{case.expected_category}', got '{actual.category.value}'"
                )
            if not pri_ok:
                reasons.append(
                    f"priority: expected '{case.expected_priority}', got '{actual.priority.value}'"
                )
            if not nh_ok:
                reasons.append(
                    f"needs_human: expected {case.expected_needs_human}, got {actual.needs_human}"
                )

            disagreements.append(
                {
                    "message_id": case.message_id,
                    "message_text": case.message_text[:120] + "..."
                    if len(case.message_text) > 120
                    else case.message_text,
                    "expected": {
                        "category": case.expected_category,
                        "priority": case.expected_priority,
                        "needs_human": case.expected_needs_human,
                    },
                    "actual": {
                        "category": actual.category.value,
                        "priority": actual.priority.value,
                        "needs_human": actual.needs_human,
                        "confidence": actual.confidence,
                        "summary": actual.summary,
                    },
                    "reason": "; ".join(reasons),
                }
            )

    return EvalReport(
        total_messages=run.total,
        successful=run.successful,
        failed=run.failed,
        ground_truth_total=len(gt_cases),
        category_correct=category_correct,
        priority_correct=priority_correct,
        needs_human_correct=needs_human_correct,
        exact_match=exact_match,
        disagreements=disagreements,
    )

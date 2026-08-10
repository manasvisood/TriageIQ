"""
Output schema and validation for the triage system.

All fields are strictly typed. Pydantic validates every LLM response
before it reaches the rest of the pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Category(str, Enum):
    BILLING = "billing"               # Payment, invoice, subscription, refund
    TECHNICAL = "technical"           # Bugs, errors, crashes, API issues
    ACCOUNT = "account"               # Login, access, password, suspension
    FEATURE_REQUEST = "feature_request"  # New feature asks
    COMPLAINT = "complaint"           # General dissatisfaction, feedback
    OUT_OF_SCOPE = "out_of_scope"    # Illegal requests, off-topic, injection
    GENERAL_INQUIRY = "general_inquiry"  # Questions, info requests, pricing


class Priority(str, Enum):
    P0 = "P0"  # Critical: security breach, full outage, data loss, mass impact
    P1 = "P1"  # High: service unusable, payment failure, account locked
    P2 = "P2"  # Medium: degraded function, billing question, non-critical bug
    P3 = "P3"  # Low: inquiry, feature request, feedback, minor issue


class TriageResult(BaseModel):
    """Structured triage decision output."""

    message_id: str = Field(..., description="Identifier of the source message")
    category: Category = Field(..., description="Message category")
    priority: Priority = Field(..., description="Urgency priority P0-P3")
    summary: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Concise factual summary of the message content",
    )
    suggested_action: str = Field(
        ...,
        min_length=5,
        max_length=300,
        description="Recommended next action for the support team",
    )
    needs_human: bool = Field(
        ..., description="True when human review is required"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence 0.0-1.0"
    )

    @field_validator("summary", "suggested_action", mode="before")
    @classmethod
    def strip_strings(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @model_validator(mode="after")
    def low_confidence_escalates(self) -> "TriageResult":
        """If confidence is below 0.5, force needs_human=True."""
        if self.confidence < 0.5:
            self.needs_human = True
        return self


class FailedResult(BaseModel):
    """Represents a message that could not be triaged after all retries."""

    message_id: str
    error: str
    raw_response: Optional[str] = None


class InputMessage(BaseModel):
    """A single customer message from the dataset."""

    id: str
    text: str
    ground_truth: Optional[dict] = None

    @field_validator("text", mode="before")
    @classmethod
    def text_must_not_be_empty(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            raise ValueError("Message text cannot be empty")
        return v


class PipelineRun(BaseModel):
    """Container for a full pipeline execution."""

    total: int
    successful: int
    failed: int
    results: list[TriageResult]
    failures: list[FailedResult]
    total_runtime_seconds: float
    avg_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int

"""Domain schemas, configuration and typed LangGraph state.

All data crossing a trust boundary (HTTP input, LLM output) is validated by a
Pydantic model. Internal state is a ``TypedDict`` so LangGraph reducers can
append immutable audit events without bespoke merge logic.
"""

from __future__ import annotations

import operator
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Verdict(StrEnum):
    """Decision-support outcome. Never executed automatically by this service."""

    BLOCK = "BLOCK"
    REVIEW = "REVIEW"
    ALLOW = "ALLOW"


class Channel(StrEnum):
    """Transaction origination channel."""

    CARD_PRESENT = "CARD_PRESENT"
    CARD_NOT_PRESENT = "CARD_NOT_PRESENT"
    WIRE = "WIRE"
    ACH = "ACH"
    P2P = "P2P"


class AppSettings(BaseSettings):
    """Environment-driven runtime configuration (see README)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    service_name: str = "agentic-soc-analyzer"
    log_level: str = "INFO"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_api_key: str = "ollama-local"
    llm_retries: int = Field(default=2, ge=0, le=5)
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)


class TransactionIn(BaseModel):
    """Flagged transaction submitted by an upstream fraud-monitoring system."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    customer_id: str = Field(min_length=4, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    amount: Decimal = Field(gt=0, le=Decimal("100000000"), max_digits=18, decimal_places=2)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    occurred_at: datetime
    channel: Channel
    merchant_country: str = Field(pattern=r"^[A-Z]{2}$")
    merchant_category: str = Field(min_length=2, max_length=64, pattern=r"^[A-Za-z0-9 &._-]+$")
    narrative: str = Field(min_length=1, max_length=2000)

    @field_validator("occurred_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone offset")
        return value

    @field_validator("narrative")
    @classmethod
    def _reject_control_characters(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in {"\n", "\t"} for char in value):
            raise ValueError("narrative contains disallowed control characters")
        return value


class CustomerHistory(BaseModel):
    """Minimal customer risk profile returned by the internal data source."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str
    account_age_days: int = Field(ge=0)
    avg_transaction_amount: Decimal = Field(ge=0)
    transactions_last_24h: int = Field(ge=0)
    chargebacks_last_12m: int = Field(ge=0)
    prior_sar_filings: int = Field(ge=0)
    home_country: str = Field(pattern=r"^[A-Z]{2}$")


class RiskAssessment(BaseModel):
    """Strict output contract enforced on the LLM by PydanticAI."""

    model_config = ConfigDict(extra="forbid")

    risk_score: int = Field(ge=0, le=100, description="Fraud risk from 0 (benign) to 100 (severe).")
    verdict: Verdict = Field(description="Decision-support verdict.")
    audit_reason: str = Field(
        min_length=10,
        max_length=500,
        description="Concise, PII-free justification citing the decisive factors.",
    )

    @classmethod
    def human_review(cls, reason: str) -> RiskAssessment:
        """Deterministic fallback used when the model output cannot be trusted."""

        return cls(risk_score=50, verdict=Verdict.REVIEW, audit_reason=reason[:500])


class EvaluationOutcome(BaseModel):
    """Assessment plus provenance for audit and response building."""

    model_config = ConfigDict(extra="forbid")

    assessment: RiskAssessment
    degraded: bool
    model_name: str


class AnalysisResponse(BaseModel):
    """Public API response. Contains no PII and no raw model output."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    transaction_id: str
    risk_score: int
    verdict: Verdict
    audit_reason: str
    degraded: bool
    audit_events: int


AuditValue = str | int | float | bool | None


class AuditEvent(BaseModel):
    """Immutable record of a single state transition. Values must be PII-free."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    request_id: str
    node: str
    action: str
    outcome: str
    occurred_at: datetime
    detail: dict[str, AuditValue] = Field(default_factory=dict)


class AnalysisState(TypedDict, total=False):
    """LangGraph state. ``audit_trail`` uses an append reducer."""

    request_id: str
    transaction: TransactionIn
    masked_transaction: TransactionIn
    history: CustomerHistory
    outcome: EvaluationOutcome
    audit_trail: Annotated[list[AuditEvent], operator.add]


class ErrorDetail(BaseModel):
    """Stable machine-readable error payload."""

    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    """Uniform error envelope returned by every handler."""

    error: ErrorDetail


class SecurityViolation(Exception):
    """Raised by the security gate when untrusted input is rejected.

    The message never contains the offending payload, preventing log injection
    and accidental exfiltration of attacker-controlled content.
    """

    def __init__(self, rule: str) -> None:
        super().__init__("Input rejected by the security gate.")
        self.rule = rule

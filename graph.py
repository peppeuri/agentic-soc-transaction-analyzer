"""LangGraph workflow: security gate -> enrichment -> agentic evaluation.

Each node emits an immutable audit event and returns only a partial state
update. A failure in the security gate aborts the graph before any customer
data is enriched or any LLM call is made.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Final

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent import risk_evaluator
from models import (
    AnalysisState,
    AuditEvent,
    AuditValue,
    CustomerHistory,
    SecurityViolation,
    TransactionIn,
)

logger = logging.getLogger("soc.graph")

_PII_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("PAN", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
)
_PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,16}\d)(?!\w)")

_INJECTION_RULES: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|system|developer)\b[^.\n]{0,20}"
            r"\b(instructions?|rules?|prompts?|messages?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"\b(you are now|act as|pretend to be|new role|developer mode|jailbreak|DAN mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,30}"
            r"\b(system prompt|hidden instructions|developer message|initial prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_breakout",
        re.compile(
            r"</?\s*(system|untrusted_data|transaction|customer_history|instructions)\s*>",
            re.IGNORECASE,
        ),
    ),
    (
        "verdict_manipulation",
        re.compile(
            r"\b(set|change|force|make)\b[^.\n]{0,30}"
            r"\b(risk score|verdict|decision)\b[^.\n]{0,20}"
            r"\b(to|=|allow|block|review)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "override_decision",
        re.compile(r"\b(do not|don't|never)\s+(flag|block|report|decline)\b", re.IGNORECASE),
    ),
    (
        "encoded_payload",
        re.compile(r"\b(base64|rot13|hex)\s*[:=]\s*[A-Za-z0-9+/=]{16,}", re.IGNORECASE),
    ),
)


def _mask_phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_PHONE]" if 9 <= len(digits) <= 15 else match.group(0)


def mask_pii(text: str) -> str:
    """Redact common banking PII. Over-masking is acceptable; leakage is not."""

    masked = text
    for label, pattern in _PII_RULES:
        masked = pattern.sub(f"[REDACTED_{label}]", masked)
    return _PHONE_PATTERN.sub(_mask_phone, masked)


def detect_injection(text: str) -> str | None:
    """Return the first matching heuristic rule id, or ``None``.

    Heuristics are defense-in-depth, not the sole control: the system prompt
    also enforces instruction/data separation and the agent has no tools.
    """

    for rule_id, pattern in _INJECTION_RULES:
        if pattern.search(text):
            return rule_id
    return None


def _audit_event(
    request_id: str,
    node: str,
    action: str,
    outcome: str,
    detail: dict[str, AuditValue] | None = None,
) -> AuditEvent:
    event = AuditEvent(
        event_id=uuid.uuid4().hex,
        request_id=request_id,
        node=node,
        action=action,
        outcome=outcome,
        occurred_at=datetime.now(timezone.utc),
        detail=detail or {},
    )
    logger.info("audit_event", extra={"audit": event.model_dump(mode="json")})
    return event


async def security_gate(state: AnalysisState) -> dict[str, Any]:
    """Reject injection attempts, then mask PII before any downstream processing."""

    request_id = state["request_id"]
    transaction = state["transaction"]

    injection_rule = detect_injection(transaction.narrative) or detect_injection(
        transaction.merchant_category
    )
    if injection_rule is not None:
        _audit_event(
            request_id,
            "security_gate",
            "prompt_injection_detected",
            "BLOCKED",
            {"rule": injection_rule},
        )
        raise SecurityViolation(injection_rule)

    masked = transaction.model_copy(
        update={
            "narrative": mask_pii(transaction.narrative),
            "merchant_category": mask_pii(transaction.merchant_category),
        }
    )
    pii_masked = (
        masked.narrative != transaction.narrative
        or masked.merchant_category != transaction.merchant_category
    )
    event = _audit_event(
        request_id,
        "security_gate",
        "input_sanitized",
        "OK",
        {"pii_masked": pii_masked, "rules_evaluated": len(_PII_RULES) + 1},
    )
    return {"masked_transaction": masked, "audit_trail": [event]}


_CUSTOMER_DB: Final[dict[str, dict[str, Any]]] = {
    "CUST-1001": {
        "account_age_days": 1280,
        "avg_transaction_amount": Decimal("85.40"),
        "transactions_last_24h": 2,
        "chargebacks_last_12m": 0,
        "prior_sar_filings": 0,
        "home_country": "US",
    },
    "CUST-RISK-9001": {
        "account_age_days": 21,
        "avg_transaction_amount": Decimal("1200.00"),
        "transactions_last_24h": 9,
        "chargebacks_last_12m": 4,
        "prior_sar_filings": 1,
        "home_country": "US",
    },
}

_NEW_CUSTOMER_PROFILE: Final[dict[str, Any]] = {
    "account_age_days": 0,
    "avg_transaction_amount": Decimal("0.00"),
    "transactions_last_24h": 1,
    "chargebacks_last_12m": 0,
    "prior_sar_filings": 0,
    "home_country": "US",
}


async def enrichment_node(state: AnalysisState) -> dict[str, Any]:
    """Retrieve the customer profile from the internal data source (mocked)."""

    request_id = state["request_id"]
    transaction = state["masked_transaction"]

    await asyncio.sleep(0)  # simulated non-blocking DB round-trip
    row = _CUSTOMER_DB.get(transaction.customer_id)
    new_customer = row is None
    profile = _NEW_CUSTOMER_PROFILE if new_customer else row
    history = CustomerHistory(customer_id=transaction.customer_id, **profile)

    event = _audit_event(
        request_id,
        "enrichment",
        "customer_history_loaded",
        "OK",
        {
            "new_customer": new_customer,
            "home_country": history.home_country,
            "country_mismatch": transaction.merchant_country != history.home_country,
            "chargebacks_last_12m": history.chargebacks_last_12m,
            "prior_sar_filings": history.prior_sar_filings,
        },
    )
    return {"history": history, "audit_trail": [event]}


async def evaluation_node(state: AnalysisState) -> dict[str, Any]:
    """Run the PydanticAI agent and record whether validation succeeded."""

    request_id = state["request_id"]
    outcome = await risk_evaluator.evaluate(state["masked_transaction"], state["history"])

    event = _audit_event(
        request_id,
        "agentic_evaluation",
        "assessment_degraded" if outcome.degraded else "assessment_completed",
        "DEGRADED" if outcome.degraded else "OK",
        {
            "risk_score": outcome.assessment.risk_score,
            "verdict": outcome.assessment.verdict.value,
            "model": outcome.model_name,
        },
    )
    return {"outcome": outcome, "audit_trail": [event]}


def build_analysis_graph() -> CompiledStateGraph:
    """Compile the linear SOC analysis workflow."""

    builder = StateGraph(AnalysisState)
    builder.add_node("security_gate", security_gate)
    builder.add_node("enrichment", enrichment_node)
    builder.add_node("agentic_evaluation", evaluation_node)
    builder.add_edge(START, "security_gate")
    builder.add_edge("security_gate", "enrichment")
    builder.add_edge("enrichment", "agentic_evaluation")
    builder.add_edge("agentic_evaluation", END)
    return builder.compile()


analysis_graph: CompiledStateGraph = build_analysis_graph()

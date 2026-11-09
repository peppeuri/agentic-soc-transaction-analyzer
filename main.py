"""FastAPI entrypoint, structured audit logging and centralized error handling."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from agent import settings
from graph import analysis_graph
from models import (
    AnalysisResponse,
    AnalysisState,
    ErrorDetail,
    ErrorEnvelope,
    EvaluationOutcome,
    SecurityViolation,
    TransactionIn,
)

_request_id: ContextVar[str] = ContextVar("request_id", default="")
logger = logging.getLogger("soc.api")

_LOG_EXTRA_FIELDS: Final[tuple[str, ...]] = (
    "event",
    "audit",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "transaction_id",
    "verdict",
    "risk_score",
    "degraded",
    "audit_events",
    "error_type",
    "rule",
    "model",
)

_REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]{8,64}$")


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter: one JSON object per line, no PII by design."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "service": settings.service_name,
            "message": record.getMessage(),
        }
        request_id = _request_id.get()
        if request_id:
            payload["request_id"] = request_id
        for field in _LOG_EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"), ensure_ascii=True)


def configure_logging() -> None:
    """Send application and uvicorn logs through the JSON audit formatter."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers.clear()
        uv_logger.propagate = True


configure_logging()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a validated request id and emit one structured access log per request."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        candidate = request.headers.get("X-Request-ID", "").strip()
        request_id = candidate if _REQUEST_ID_RE.fullmatch(candidate) else uuid.uuid4().hex
        token = _request_id.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            logger.info(
                "http_request",
                extra={
                    "event": "http_request",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            _request_id.reset(token)


class AppError(Exception):
    """Base class for expected, client-safe application errors."""

    def __init__(self, *, code: str, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    envelope = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=_request_id.get() or "unknown",
        )
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    logger.info("service_started", extra={"event": "startup", "model": settings.llm_model})
    yield
    logger.info("service_stopped", extra={"event": "shutdown"})


app = FastAPI(
    title="Agentic SOC Transaction Analyzer",
    version="1.0.0",
    description=(
        "Decision-support POC: security-gated transaction enrichment and "
        "schema-validated LLM risk assessment. Never executes transactions."
    ),
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(SecurityViolation)
async def security_violation_handler(request: Request, exc: SecurityViolation) -> JSONResponse:
    logger.warning(
        "security_gate_blocked",
        extra={
            "event": "security_gate_blocked",
            "path": request.url.path,
            "rule": exc.rule,
        },
    )
    return _error_response(400, "INJECTION_DETECTED", "Request blocked by the security gate.")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, _: RequestValidationError
) -> JSONResponse:
    logger.warning(
        "request_validation_failed",
        extra={
            "event": "request_validation_failed",
            "path": request.url.path,
            "error_type": "RequestValidationError",
        },
    )
    return _error_response(422, "VALIDATION_ERROR", "Request payload failed schema validation.")


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning(
        "application_error",
        extra={
            "event": "application_error",
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
    )
    return _error_response(exc.status_code, exc.code, exc.message)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        extra={
            "event": "unhandled_exception",
            "path": request.url.path,
            "error_type": type(exc).__name__,
        },
        exc_info=True,
    )
    return _error_response(500, "INTERNAL_ERROR", "An internal error occurred.")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe. Does not call the LLM provider."""

    return {"status": "ok", "service": settings.service_name}


@app.post(
    "/api/v1/analyze-transaction",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorEnvelope, "description": "Security gate rejection"},
        422: {"model": ErrorEnvelope, "description": "Schema validation error"},
        500: {"model": ErrorEnvelope, "description": "Unhandled server error"},
    },
)
async def analyze_transaction(payload: TransactionIn) -> AnalysisResponse:
    """Run the security-gated LangGraph workflow and return a decision-support verdict."""

    request_id = _request_id.get() or uuid.uuid4().hex
    started = time.perf_counter()
    logger.info(
        "analysis_started",
        extra={"event": "analysis_started", "transaction_id": payload.transaction_id},
    )

    initial_state: AnalysisState = {
        "request_id": request_id,
        "transaction": payload,
        "audit_trail": [],
    }
    final_state = await analysis_graph.ainvoke(initial_state)
    outcome: EvaluationOutcome = final_state["outcome"]
    audit_events = len(final_state.get("audit_trail", []))

    logger.info(
        "analysis_completed",
        extra={
            "event": "analysis_completed",
            "transaction_id": payload.transaction_id,
            "verdict": outcome.assessment.verdict.value,
            "risk_score": outcome.assessment.risk_score,
            "degraded": outcome.degraded,
            "audit_events": audit_events,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return AnalysisResponse(
        request_id=request_id,
        transaction_id=payload.transaction_id,
        risk_score=outcome.assessment.risk_score,
        verdict=outcome.assessment.verdict,
        audit_reason=outcome.assessment.audit_reason,
        degraded=outcome.degraded,
        audit_events=audit_events,
    )

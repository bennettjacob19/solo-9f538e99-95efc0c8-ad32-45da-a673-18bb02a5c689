"""FastAPI entrypoint for the embedding batch gateway."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .batcher import BatchScheduler
from .config import Settings
from .errors import GatewayError, ValidationError
from .idempotency import IdempotencyRegistry
from .schemas import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    StatsResponse,
)

logger = logging.getLogger("embedding_gateway")


class AcceptanceTimeMiddleware:
    """Stamp the receipt time before the body is parsed.

    The per-request deadline must start when the gateway accepts the call,
    not when the request reaches the scheduler.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            state = scope.setdefault("state", {})
            state.setdefault("accepted_at", asyncio.get_running_loop().time())
        await self.app(scope, receive, send)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _resolve_idempotency_key(
    header_value: str | None, body_value: str | None
) -> str | None:
    """Combine the ``Idempotency-Key`` header and the body field.

    Both transports share one key namespace.  Supplying both with different
    values is ambiguous and rejected; an explicitly empty body key is
    rejected by schema validation before this function runs.
    """
    if header_value and body_value and header_value != body_value:
        raise ValidationError(
            "Idempotency-Key header and idempotency_key body field differ"
        )
    key = header_value or body_value
    if key is None:
        return None
    if not key.strip():
        raise ValidationError("idempotency key must not be empty")
    if len(key) > 255:
        raise ValidationError("idempotency key must be at most 255 characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise ValidationError("idempotency key must not contain control characters")
    return key


def create_app(
    settings: Settings | None = None,
    *,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(
            settings.backend_timeout_seconds,
            connect=settings.backend_connect_timeout_seconds,
        )
        client = httpx.AsyncClient(
            timeout=timeout,
            transport=backend_transport,
            limits=httpx.Limits(
                max_connections=1, max_keepalive_connections=1
            ),
        )
        scheduler = BatchScheduler(settings, client)
        registry = IdempotencyRegistry(
            scheduler, max_entries=settings.idempotency_max_entries
        )
        app.state.client = client
        app.state.scheduler = scheduler
        app.state.idempotency = registry
        await scheduler.start()
        logger.info(
            "gateway ready: backend=%s window_ms=%s batch_max=%s "
            "idempotency_max_entries=%s",
            settings.backend_url,
            settings.batch_window_ms,
            settings.max_texts_per_batch,
            settings.idempotency_max_entries,
        )
        try:
            yield
        finally:
            await scheduler.close()
            await client.aclose()

    app = FastAPI(
        title="Embedding Batch Gateway",
        version="1.1.0",
        description=(
            "Accepts 1-4 short texts per call, merges concurrent calls into "
            "single-flight backend batches of at most 8 texts, and enforces a "
            "deadline measured from request acceptance.\n\n"
            "Optional **idempotency keys** (`Idempotency-Key` header or the "
            "`idempotency_key` body field) let clients retry safely after "
            "network timeouts: retries with the same key and identical "
            "`input`/`model`/`timeout_ms` share the in-flight batch or replay "
            "its stored result, so the single-slot backend never runs the "
            "same computation twice. Reusing a key with different parameters "
            "returns `409 idempotency_conflict` without queueing; after a "
            "backend failure the key is released and may be submitted again. "
            "Completed results are replayable until the gateway restarts."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(AcceptanceTimeMiddleware)
    app.state.settings = settings

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        request: Request, exc: GatewayError
    ) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        first = errors[0] if errors else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = f"{location}: {first.get('msg', 'invalid request')}"
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": message.lstrip(": "),
                    "details": errors,
                }
            },
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/stats", response_model=StatsResponse, tags=["ops"])
    async def stats(request: Request) -> StatsResponse:
        merged = {
            **request.app.state.scheduler.stats(),
            **request.app.state.idempotency.stats(),
        }
        return StatsResponse(**merged)

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResponse,
        tags=["embeddings"],
        summary="Embed 1-4 texts",
        description=(
            "Merges concurrent calls into single-flight backend batches.\n\n"
            "**Idempotent retries:** send `Idempotency-Key` (header) or "
            "`idempotency_key` (body) plus identical `input`, `model` and "
            "`timeout_ms`. Concurrent retries share one in-flight batch; if "
            "the first caller times out or disconnects, a retry still awaits "
            "the original batch and is the only way to collect its late "
            "result; once completed, the result is replayed "
            "(`idempotent_replay: true`) until the gateway restarts. The same "
            "key with different parameters is rejected with `409` and never "
            "queued. After a `502 backend_error` the key is released and the "
            "request may be submitted again."
        ),
    )
    async def embeddings(
        payload: EmbeddingRequest,
        request: Request,
        idempotency_key_header: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            max_length=255,
            description=(
                "Optional idempotency key for safe retries. Equivalent to the "
                "idempotency_key body field; if both are sent they must match."
            ),
            examples=["search-7f3d2c1e"],
        ),
    ) -> EmbeddingResponse:
        texts = payload.as_texts()
        if not 1 <= len(texts) <= settings.max_texts_per_request:
            raise ValidationError(
                f"input must contain between 1 and "
                f"{settings.max_texts_per_request} texts"
            )

        timeout_ms = (
            payload.timeout_ms
            if payload.timeout_ms is not None
            else settings.default_timeout_ms
        )
        if timeout_ms > settings.max_timeout_ms:
            raise ValidationError(
                f"timeout_ms must be <= {settings.max_timeout_ms}"
            )

        idempotency_key = _resolve_idempotency_key(
            idempotency_key_header, payload.idempotency_key
        )

        accepted_at = request.state.accepted_at
        deadline = accepted_at + timeout_ms / 1000.0
        request_id = uuid.uuid4().hex
        model = payload.model or settings.backend_model

        if idempotency_key is None:
            result = await request.app.state.scheduler.submit(
                request_id=request_id,
                texts=texts,
                model=model,
                deadline=deadline,
                request=request,
            )
            replayed = False
        else:
            registry = request.app.state.idempotency
            # Raises IdempotencyConflict (409) before anything is queued when
            # the key exists with different resolved parameters.
            operation, mode = registry.get_or_start(
                key=idempotency_key,
                texts=texts,
                model=model,
                timeout_ms=timeout_ms,
            )
            # The caller waits with its own deadline; the operation itself is
            # never cancelled by this caller timing out or disconnecting.
            result = await registry.wait(
                operation, deadline=deadline, request=request
            )
            replayed = mode == "replayed"

        return EmbeddingResponse(
            data=[
                EmbeddingItem(index=index, embedding=vector)
                for index, vector in enumerate(result.vectors)
            ],
            model=model,
            batch_id=result.batch_id,
            batch_size=result.batch_size,
            idempotent_replay=replayed,
        )

    return app


app = create_app()

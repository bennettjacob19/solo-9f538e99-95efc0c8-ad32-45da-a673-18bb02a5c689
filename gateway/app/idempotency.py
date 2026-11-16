"""Optional idempotency-key layer above the single-flight scheduler.

Why a separate layer
--------------------
The scheduler is caller-centric: a request whose deadline expires (or whose
client disconnects) is dequeued, and its late vectors are discarded.  That is
exactly what an idempotent retry must *not* cause.  So this module introduces
the **operation**: a unit of work owned by an idempotency key instead of by
one HTTP call.

* The operation is submitted to the scheduler from a detached task with a
  generous safety deadline and no disconnect probe.  From the scheduler's
  point of view the operation is a patient caller that never goes away, so
  the batch is always drained and its vectors always come back here.
* Each HTTP caller (the originator and every retry) merely *waits* on the
  operation with its own deadline and its own disconnect watch.  A caller
  that times out or disconnects gets its 504/499 while the operation keeps
  running; the late result is stored and can only be fetched by a later
  retry carrying the same key and identical parameters.
* A completed (succeeded) operation stays in the registry so identical
  retries replay the stored result without touching the backend.  Entries
  live in memory only: they survive until the gateway restarts (or until a
  safety LRU cap forces eviction — see ``idempotency_max_entries``).
* A failed operation (backend error) is evicted as soon as its current
  waiters are notified, so the same key can immediately be used to start a
  fresh computation.
* A key reuse whose resolved ``input``/``model``/``timeout_ms`` differ from
  the stored operation is rejected with ``409 idempotency_conflict`` before
  anything is enqueued.

All registry mutations are synchronous and run on the single event loop, so
check-and-insert in :meth:`IdempotencyRegistry.get_or_start` is atomic with
respect to concurrent requests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .batcher import BatchResult, BatchScheduler
from .errors import (
    BackendError,
    ClientDisconnected,
    GatewayError,
    IdempotencyConflict,
    RequestTimeout,
    ServiceUnavailable,
)

logger = logging.getLogger("embedding_gateway.idempotency")

# An operation is deliberately not bound to any caller's deadline, but it
# must not leak forever if the queue is pathologically stuck.  One hour is
# far beyond any realistic backlog (every dispatched batch resolves within
# the backend timeout); hitting it fails the operation and releases the key.
_OPERATION_TIMEOUT_SECONDS = 3600.0

#: Parameters that must match exactly for a key reuse to share/replay.
_FINGERPRINT_FIELDS = ("input", "model", "timeout_ms")


def fingerprint_params(
    texts: list[str], model: str, timeout_ms: int
) -> tuple[str, dict[str, Any]]:
    """Canonical fingerprint of the resolved request parameters.

    The *resolved* values are fingerprinted (defaults already applied), so
    omitting ``model`` and sending the default model explicitly are the same
    computation.  Returns the hex digest plus the canonical parameter dict
    (kept for conflict diagnostics).
    """
    canonical: dict[str, Any] = {
        "input": list(texts),
        "model": model,
        "timeout_ms": timeout_ms,
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return digest, canonical


@dataclass
class IdempotentOperation:
    """One computation owned by an idempotency key."""

    key: str
    fingerprint: str
    params: dict[str, Any]
    texts: list[str]
    model: str
    request_id: str
    created_at: float
    task: asyncio.Task[BatchResult] | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)
    # running -> succeeded | failed
    state: str = "running"
    result: BatchResult | None = None
    error: GatewayError | None = None


class IdempotencyRegistry:
    """Tracks in-flight and completed idempotent operations."""

    def __init__(
        self, scheduler: BatchScheduler, *, max_entries: int = 10_000
    ) -> None:
        self._scheduler = scheduler
        self._max_entries = max(1, max_entries)
        self._operations: OrderedDict[str, IdempotentOperation] = OrderedDict()
        self._stats: dict[str, int] = {
            "idempotent_requests": 0,
            "idempotent_operations": 0,
            "idempotent_attached": 0,
            "idempotent_replayed": 0,
            "idempotency_conflicts": 0,
        }

    # ------------------------------------------------------------ lifecycle
    def get_or_start(
        self,
        *,
        key: str,
        texts: list[str],
        model: str,
        timeout_ms: int,
    ) -> tuple[IdempotentOperation, str]:
        """Return the operation for ``key``, starting it if necessary.

        The second tuple element is ``"created"`` (this call started the
        computation), ``"attached"`` (joined a still-running operation) or
        ``"replayed"`` (a stored result is ready).  Raises
        :class:`IdempotencyConflict` — before anything is enqueued — when the
        key exists with different resolved parameters.
        """
        self._stats["idempotent_requests"] += 1
        fingerprint, canonical = fingerprint_params(texts, model, timeout_ms)

        existing = self._operations.get(key)
        if existing is not None:
            if existing.fingerprint != fingerprint:
                self._stats["idempotency_conflicts"] += 1
                raise IdempotencyConflict(
                    self._conflict_message(key, existing.params, canonical)
                )
            self._operations.move_to_end(key)
            if existing.state == "succeeded":
                self._stats["idempotent_replayed"] += 1
                return existing, "replayed"
            self._stats["idempotent_attached"] += 1
            return existing, "attached"

        loop = asyncio.get_running_loop()
        operation = IdempotentOperation(
            key=key,
            fingerprint=fingerprint,
            params=canonical,
            texts=list(texts),
            model=model,
            request_id=f"idem-{uuid.uuid4().hex}",
            created_at=loop.time(),
        )
        operation.task = asyncio.create_task(
            self._run(operation), name=f"idempotency-{operation.request_id}"
        )
        operation.task.add_done_callback(
            lambda _task, op=operation: self._on_operation_done(op)
        )
        self._operations[key] = operation
        self._stats["idempotent_operations"] += 1
        self._enforce_capacity()
        logger.debug("idempotent operation started: key=%r", key)
        return operation, "created"

    async def _run(self, operation: IdempotentOperation) -> BatchResult:
        """Submit the operation to the scheduler as a detached caller.

        No disconnect probe and a generous safety deadline: the operation
        must survive every HTTP caller so the late result can be stored for
        a retry.
        """
        loop = asyncio.get_running_loop()
        return await self._scheduler.submit(
            request_id=operation.request_id,
            texts=operation.texts,
            model=operation.model,
            deadline=loop.time() + _OPERATION_TIMEOUT_SECONDS,
            request=None,
        )

    def _on_operation_done(self, operation: IdempotentOperation) -> None:
        task = operation.task
        assert task is not None
        try:
            result = task.result()
        except BaseException as exc:  # noqa: BLE001 - bookkeeping must not die
            if isinstance(exc, GatewayError):
                error = exc
            elif isinstance(exc, asyncio.CancelledError):
                error = ServiceUnavailable("gateway shutting down")
            else:
                error = BackendError(f"idempotent operation failed: {exc}")
            operation.state = "failed"
            operation.error = error
            # Failure releases the key: the next identical request starts a
            # fresh computation instead of replaying the error.  Callers that
            # are already waiting still observe this failure via the event.
            if self._operations.get(operation.key) is operation:
                del self._operations[operation.key]
            logger.debug(
                "idempotent operation failed, key released: key=%r error=%s",
                operation.key,
                error.message,
            )
        else:
            operation.state = "succeeded"
            operation.result = result
            if self._operations.get(operation.key) is operation:
                self._operations.move_to_end(operation.key)
            self._enforce_capacity()
            logger.debug("idempotent operation completed: key=%r", operation.key)
        finally:
            operation.event.set()

    def _enforce_capacity(self) -> None:
        """Bound memory by evicting the oldest *completed* entries.

        Running operations are never evicted.  Eviction means a later
        identical retry recomputes instead of replaying — a documented
        safety valve, not part of the normal contract.
        """
        while len(self._operations) > self._max_entries:
            for old_key, old_op in self._operations.items():
                if old_op.state == "succeeded":
                    del self._operations[old_key]
                    logger.debug("idempotency entry evicted (LRU): key=%r", old_key)
                    break
            else:
                break

    @staticmethod
    def _conflict_message(
        key: str, stored: dict[str, Any], incoming: dict[str, Any]
    ) -> str:
        differing = [
            name
            for name in _FINGERPRINT_FIELDS
            if stored.get(name) != incoming.get(name)
        ]
        detail = ", ".join(differing) or "parameters"
        return (
            f"idempotency key {key!r} was already used with different "
            f"parameters ({detail} differ); the conflicting request was "
            f"not queued"
        )

    # --------------------------------------------------------------- waiting
    async def wait(
        self,
        operation: IdempotentOperation,
        *,
        deadline: float,
        request: Any | None = None,
    ) -> BatchResult:
        """Wait for ``operation`` with this caller's own deadline.

        Timing out, disconnecting or being cancelled never touches the
        operation itself: the computation continues and its result stays
        available for a later retry with the same key.
        """
        if operation.event.is_set():
            return self._outcome(operation)

        loop = asyncio.get_running_loop()
        event_task = asyncio.ensure_future(operation.event.wait())
        disconnect_task: asyncio.Task[bool] | None = None
        if request is not None:
            disconnect_task = asyncio.create_task(
                self._scheduler.watch_disconnect(request),
                name=f"disconnect-{operation.request_id}-retry",
            )
        waiters: set[asyncio.Future[Any]] = {event_task}
        if disconnect_task is not None:
            waiters.add(disconnect_task)

        try:
            while True:
                remaining = deadline - loop.time()
                done, _ = await asyncio.wait(
                    waiters,
                    timeout=max(0.0, remaining),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task is not None and disconnect_task in done:
                    if disconnect_task.result():
                        self._scheduler.note_disconnected()
                        raise ClientDisconnected(
                            "client disconnected before result"
                        )
                    # This server cannot observe disconnects; stop probing.
                    waiters.discard(disconnect_task)
                    continue
                if event_task in done:
                    return self._outcome(operation)
                # Deadline hit while the operation is still running.  The
                # caller leaves empty-handed; the computation continues and
                # its late result remains fetchable by a retry.
                self._scheduler.note_timed_out()
                raise RequestTimeout("request deadline exceeded while waiting")
        except asyncio.CancelledError:
            # The ASGI server cancelled the handler (client vanished or
            # shutdown).  The operation is deliberately left running.
            self._scheduler.note_disconnected()
            raise
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()

    @staticmethod
    def _outcome(operation: IdempotentOperation) -> BatchResult:
        if operation.state == "succeeded" and operation.result is not None:
            return operation.result
        assert operation.error is not None
        raise operation.error

    # --------------------------------------------------------------- status
    def stats(self) -> dict[str, Any]:
        snapshot = dict(self._stats)
        snapshot["idempotency_keys_tracked"] = len(self._operations)
        return snapshot

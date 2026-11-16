"""Idempotency-key tests: shared computations, replay, conflicts, recovery."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gateway.app.batcher import BatchScheduler
from gateway.app.errors import ClientDisconnected
from gateway.app.idempotency import IdempotencyRegistry
from tests.conftest import (
    FakeBackend,
    gateway_client,
    make_settings,
    post,
    vector_for,
)


def assert_embeddings(response: httpx.Response, texts: list[str]) -> None:
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["data"]) == len(texts)
    assert [item["index"] for item in body["data"]] == list(range(len(texts)))
    for item, text in zip(body["data"], texts):
        assert item["embedding"] == vector_for(text)


@pytest.mark.asyncio
async def test_concurrent_identical_keyed_requests_share_one_batch() -> None:
    backend = FakeBackend(delay=0.1)
    async with gateway_client(backend) as (_, client):
        payload = {
            "input": ["shared-a", "shared-b"],
            "timeout_ms": 5000,
            "idempotency_key": "key-shared",
        }
        responses = await asyncio.gather(
            *[client.post("/v1/embeddings", json=payload) for _ in range(5)]
        )
        for response in responses:
            assert_embeddings(response, ["shared-a", "shared-b"])
        batch_ids = {response.json()["batch_id"] for response in responses}
        assert len(batch_ids) == 1
        # The single-slot backend ran the computation exactly once.
        assert len(backend.calls) == 1
        assert backend.calls[0].texts == ["shared-a", "shared-b"]
        assert backend.max_active == 1

        stats = (await client.get("/stats")).json()
        assert stats["idempotent_requests"] == 5
        assert stats["idempotent_operations"] == 1
        assert stats["idempotent_attached"] == 4
        assert stats["idempotency_keys_tracked"] == 1


@pytest.mark.asyncio
async def test_completed_result_is_replayed_without_new_backend_call() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        payload = {
            "input": ["replay-me"],
            "timeout_ms": 5000,
            "idempotency_key": "key-replay",
        }
        first = await client.post("/v1/embeddings", json=payload)
        assert_embeddings(first, ["replay-me"])
        assert first.json()["idempotent_replay"] is False

        replay = await client.post("/v1/embeddings", json=payload)
        assert_embeddings(replay, ["replay-me"])
        assert replay.json()["idempotent_replay"] is True
        # Same stored result, including the original batch metadata.
        assert replay.json()["batch_id"] == first.json()["batch_id"]
        assert replay.json()["batch_size"] == first.json()["batch_size"]
        assert len(backend.calls) == 1

        stats = (await client.get("/stats")).json()
        assert stats["idempotent_replayed"] == 1
        assert stats["idempotent_operations"] == 1


@pytest.mark.asyncio
async def test_same_key_different_params_returns_409_and_never_queued() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        base = {
            "input": ["original"],
            "timeout_ms": 5000,
            "idempotency_key": "key-conflict",
        }
        assert_embeddings(await client.post("/v1/embeddings", json=base), ["original"])

        conflicts = [
            {**base, "input": ["different"]},
            {**base, "timeout_ms": 4000},
            {**base, "model": "other-model"},
        ]
        for payload in conflicts:
            response = await client.post("/v1/embeddings", json=payload)
            assert response.status_code == 409, response.text
            assert response.json()["error"]["code"] == "idempotency_conflict"

        # Nothing extra reached the backend: conflicts are rejected before
        # queueing.
        assert len(backend.calls) == 1
        # The original key+params still replays fine afterwards.
        replay = await client.post("/v1/embeddings", json=base)
        assert_embeddings(replay, ["original"])
        assert len(backend.calls) == 1

        stats = (await client.get("/stats")).json()
        assert stats["idempotency_conflicts"] == 3
        assert stats["pending_requests"] == 0


@pytest.mark.asyncio
async def test_conflict_rejected_while_operation_still_in_flight() -> None:
    backend = FakeBackend(delay=0.2)
    async with gateway_client(backend) as (_, client):
        original = asyncio.create_task(
            post(client, ["in-flight"], timeout_ms=5000, idempotency_key="key-run")
        )
        await asyncio.sleep(0.05)  # let the operation reach the backend
        conflict = await post(
            client, ["other-text"], timeout_ms=5000, idempotency_key="key-run"
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_conflict"

        assert_embeddings(await original, ["in-flight"])
        assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_timed_out_caller_retry_collects_late_result() -> None:
    """The core retry story: first call times out, retry awaits the original
    batch, and the late result is only ever delivered to the retry."""
    backend = FakeBackend(delay=0.3)
    async with gateway_client(backend) as (_, client):
        payload = {
            "input": ["slow-computation"],
            "timeout_ms": 200,
            "idempotency_key": "key-timeout",
        }
        first = await client.post("/v1/embeddings", json=payload)
        assert first.status_code == 504
        assert first.json()["error"]["code"] == "request_timeout"

        # The original batch is still running; the retry attaches to it and
        # its own 200 ms budget is enough to collect the late vectors.
        retry = await client.post("/v1/embeddings", json=payload)
        assert_embeddings(retry, ["slow-computation"])
        assert retry.json()["idempotent_replay"] is False
        # The computation ran exactly once despite the timeout + retry.
        assert len(backend.calls) == 1
        assert backend.calls[0].texts == ["slow-computation"]

        # A later identical request replays the stored result.
        replay = await client.post("/v1/embeddings", json=payload)
        assert_embeddings(replay, ["slow-computation"])
        assert replay.json()["idempotent_replay"] is True
        assert replay.json()["batch_id"] == retry.json()["batch_id"]
        assert len(backend.calls) == 1

        stats = (await client.get("/stats")).json()
        assert stats["timed_out_requests"] == 1
        assert stats["idempotent_operations"] == 1
        assert stats["idempotent_attached"] == 1
        assert stats["idempotent_replayed"] == 1


class _FakeRequest:
    def __init__(self, disconnect_after: float | None) -> None:
        self._disconnect_after = disconnect_after
        self._started = asyncio.get_running_loop().time()

    async def is_disconnected(self) -> bool:
        if self._disconnect_after is None:
            return False
        return asyncio.get_running_loop().time() - self._started >= self._disconnect_after


@pytest.mark.asyncio
async def test_disconnect_of_first_caller_keeps_operation_alive() -> None:
    backend = FakeBackend(delay=0.1)
    settings = make_settings()
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    registry = IdempotencyRegistry(scheduler, max_entries=100)
    try:
        loop = asyncio.get_running_loop()
        operation, mode = registry.get_or_start(
            key="key-disconnect", texts=["durable"], model="fake-model", timeout_ms=5000
        )
        assert mode == "created"

        # The originating caller vanishes immediately; the operation must not
        # be abandoned and its result must remain fetchable.
        with pytest.raises(ClientDisconnected):
            await registry.wait(
                operation,
                deadline=loop.time() + 5,
                request=_FakeRequest(disconnect_after=0.0),
            )
        assert operation.state == "running"

        result = await registry.wait(operation, deadline=loop.time() + 5)
        assert result.vectors == [vector_for("durable")]
        assert len(backend.calls) == 1
        assert scheduler.stats()["disconnected_requests"] == 1
    finally:
        await scheduler.close()
        await client.aclose()


@pytest.mark.asyncio
async def test_backend_failure_releases_key_for_fresh_attempt() -> None:
    backend = FakeBackend(status=503)
    async with gateway_client(backend) as (_, client):
        payload = {
            "input": ["retryable"],
            "timeout_ms": 5000,
            "idempotency_key": "key-failure",
        }
        failed = await client.post("/v1/embeddings", json=payload)
        assert failed.status_code == 502
        assert failed.json()["error"]["code"] == "backend_error"

        # The key was released: the identical request starts a new operation
        # instead of replaying the failure.
        backend.status = 200
        recovered = await client.post("/v1/embeddings", json=payload)
        assert_embeddings(recovered, ["retryable"])
        assert recovered.json()["idempotent_replay"] is False
        assert len(backend.calls) == 2

        stats = (await client.get("/stats")).json()
        assert stats["idempotent_operations"] == 2
        assert stats["idempotency_keys_tracked"] == 1


@pytest.mark.asyncio
async def test_concurrent_waiters_share_failure_then_recover() -> None:
    backend = FakeBackend(status=503, delay=0.05)
    async with gateway_client(backend) as (_, client):
        payload = {
            "input": ["doomed"],
            "timeout_ms": 5000,
            "idempotency_key": "key-storm",
        }
        responses = await asyncio.gather(
            *[client.post("/v1/embeddings", json=payload) for _ in range(3)]
        )
        # All waiters of the same operation observe the single failure.
        assert all(response.status_code == 502 for response in responses)
        assert len(backend.calls) == 1

        backend.status = 200
        recovered = await client.post("/v1/embeddings", json=payload)
        assert_embeddings(recovered, ["doomed"])
        assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_header_and_body_keys_share_one_namespace() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        via_header = await client.post(
            "/v1/embeddings",
            json={"input": ["namespaced"], "timeout_ms": 5000},
            headers={"Idempotency-Key": "key-both"},
        )
        assert_embeddings(via_header, ["namespaced"])

        via_body = await client.post(
            "/v1/embeddings",
            json={
                "input": ["namespaced"],
                "timeout_ms": 5000,
                "idempotency_key": "key-both",
            },
        )
        assert_embeddings(via_body, ["namespaced"])
        assert via_body.json()["idempotent_replay"] is True
        assert len(backend.calls) == 1

        mismatch = await client.post(
            "/v1/embeddings",
            json={
                "input": ["namespaced"],
                "timeout_ms": 5000,
                "idempotency_key": "key-both",
            },
            headers={"Idempotency-Key": "key-other"},
        )
        assert mismatch.status_code == 422
        assert mismatch.json()["error"]["code"] == "validation_error"
        assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_invalid_idempotency_keys_are_rejected() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        empty = await client.post(
            "/v1/embeddings",
            json={"input": ["x"], "timeout_ms": 5000, "idempotency_key": ""},
        )
        assert empty.status_code == 422

        too_long = await client.post(
            "/v1/embeddings",
            json={"input": ["x"], "timeout_ms": 5000, "idempotency_key": "k" * 256},
        )
        assert too_long.status_code == 422

        control_chars = await client.post(
            "/v1/embeddings",
            json={"input": ["x"], "timeout_ms": 5000, "idempotency_key": "bad\nkey"},
        )
        assert control_chars.status_code == 422

        whitespace = await client.post(
            "/v1/embeddings",
            json={"input": ["x"], "timeout_ms": 5000},
            headers={"Idempotency-Key": "   "},
        )
        assert whitespace.status_code == 422

        assert backend.calls == []


@pytest.mark.asyncio
async def test_keyed_and_plain_requests_still_share_a_batch() -> None:
    backend = FakeBackend(delay=0.1)
    async with gateway_client(backend) as (_, client):
        keyed, plain = await asyncio.gather(
            post(client, ["keyed"], timeout_ms=5000, idempotency_key="key-merge"),
            post(client, ["plain"], timeout_ms=5000),
        )
        assert_embeddings(keyed, ["keyed"])
        assert_embeddings(plain, ["plain"])
        # Idempotency does not isolate the request from normal batching.
        assert keyed.json()["batch_id"] == plain.json()["batch_id"]
        assert len(backend.calls) == 1
        assert sorted(backend.calls[0].texts) == ["keyed", "plain"]

        replay = await post(
            client, ["keyed"], timeout_ms=5000, idempotency_key="key-merge"
        )
        assert replay.json()["idempotent_replay"] is True
        assert replay.json()["batch_id"] == keyed.json()["batch_id"]
        assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_many_racing_retries_still_run_one_operation() -> None:
    backend = FakeBackend(delay=0.05)
    async with gateway_client(backend) as (_, client):
        payload = {
            "input": ["race"],
            "timeout_ms": 5000,
            "idempotency_key": "key-race",
        }
        responses = await asyncio.gather(
            *[client.post("/v1/embeddings", json=payload) for _ in range(20)]
        )
        assert all(response.status_code == 200 for response in responses)
        assert len(backend.calls) == 1
        stats = (await client.get("/stats")).json()
        assert stats["idempotent_operations"] == 1
        assert stats["idempotent_attached"] == 19


@pytest.mark.asyncio
async def test_completed_entries_are_lru_evicted_beyond_capacity() -> None:
    backend = FakeBackend()
    async with gateway_client(backend, idempotency_max_entries=2) as (_, client):
        for key in ("evict-a", "evict-b", "evict-c"):
            response = await post(
                client, [key], timeout_ms=5000, idempotency_key=key
            )
            assert_embeddings(response, [key])
        assert len(backend.calls) == 3

        stats = (await client.get("/stats")).json()
        assert stats["idempotency_keys_tracked"] == 2

        # The oldest entry was evicted: reusing its key recomputes (and would
        # also accept different parameters again) instead of replaying.
        recomputed = await post(
            client, ["evict-a"], timeout_ms=5000, idempotency_key="evict-a"
        )
        assert_embeddings(recomputed, ["evict-a"])
        assert recomputed.json()["idempotent_replay"] is False
        assert len(backend.calls) == 4

        # The newest entry survived and still replays.
        replay = await post(
            client, ["evict-c"], timeout_ms=5000, idempotency_key="evict-c"
        )
        assert replay.json()["idempotent_replay"] is True
        assert len(backend.calls) == 4

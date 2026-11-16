# Embedding Batch Gateway

A small FastAPI service that accepts **1–4 short texts per HTTP call** and
merges concurrent calls into **single-flight batches of at most 8 texts** for a
single-slot CPU embedding backend. It ships with a controllable local mock
backend and a Docker Compose file that starts both.

```
client ──► gateway :8000 ──(one batch at a time)──► backend :9000
             │ batching, deadlines, ordering           │ configurable delay
             │ single-flight, error fan-out            │ error injection
             └ /docs, /stats                          └ /docs, /admin/*
```

## Quick start

```bash
docker compose up --build
```

| Service | URL |
| --- | --- |
| Gateway API docs (interactive) | http://localhost:8000/docs |
| Gateway stats | http://localhost:8000/stats |
| Gateway health | http://localhost:8000/healthz |
| Mock backend docs | http://localhost:9000/docs |
| Mock backend stats (real concurrency) | http://localhost:9000/admin/stats |

No Docker? The same two apps run directly:

```bash
pip install -r requirements-dev.txt

uvicorn backend.app.main:app --port 9000 &
EMBEDDING_BACKEND_URL=http://127.0.0.1:9000/v1/embeddings \
  uvicorn gateway.app.main:app --port 8000 &
```

Run the tests and the live smoke test:

```bash
pytest -q                  # no network required
python scripts/smoke_test.py   # requires both services running
```

## API

### `POST /v1/embeddings`

```json
{
  "input": ["first text", "second text"],
  "timeout_ms": 1000,
  "model": "local-demo",
  "idempotency_key": "search-7f3d2c1e"
}
```

* `input` — a string or a list of 1–4 strings. `texts` is accepted as an alias.
* `timeout_ms` — end-to-end deadline in milliseconds, **measured from the
  moment the gateway accepts the request** (a middleware stamps it before the
  body is parsed). Defaults to `GATEWAY_DEFAULT_TIMEOUT_MS` (5000) when omitted.
* `model` — optional; forwarded to the backend, otherwise `EMBEDDING_MODEL`.
* `idempotency_key` — optional (also accepted as the `Idempotency-Key` HTTP
  header; if both are sent they must match). Enables safe retries, see
  [Idempotency keys](#idempotency-keys).

Response (indices are always `0..n-1` in the caller's original input order):

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
  ],
  "model": "local-demo",
  "batch_id": "batch-7",
  "batch_size": 8,
  "idempotent_replay": false
}
```

`batch_id`/`batch_size` describe the backend batch the caller was merged into
(useful for demos and tests). `idempotent_replay` is `true` when the response
was replayed from a previously completed idempotent operation.

| Status | `error.code` | Meaning |
| --- | --- | --- |
| 409 | `idempotency_conflict` | key reused with different `input`/`model`/`timeout_ms`; not queued |
| 422 | `validation_error` | text count, `timeout_ms`, key or schema violation |
| 502 | `backend_error` | backend HTTP error, timeout, or invalid payload |
| 503 | `service_unavailable` | gateway is shutting down |
| 504 | `request_timeout` | deadline hit while queueing or waiting |

### Idempotency keys

Clients that retry after a network timeout risk sending the same computation
to the single-slot backend twice. An optional idempotency key (header
`Idempotency-Key` or body field `idempotency_key`, 1–255 chars, no control
characters) makes retries safe:

* **Shared in-flight computation.** Requests with the same key and
  *resolved* `input`, `model` and `timeout_ms` (defaults applied) attach to
  the same in-flight batch and receive the same result. The backend runs the
  computation exactly once.
* **The operation outlives its caller.** If the first caller times out or
  disconnects, the computation is *not* cancelled: it keeps its queue slot,
  finishes, and stores its result. A later retry with the same key and
  identical parameters still awaits the original batch — the late result can
  only be collected by such a retry, never delivered to the expired caller.
  Each caller waits with its own fresh deadline (the same `timeout_ms`
  measured from its own acceptance).
* **Replay.** Once completed, the stored result is replayed
  (`idempotent_replay: true`, same `batch_id`) for identical requests until
  the gateway restarts. Entries are in-memory only.
* **Conflict.** Reusing a key with different `input`, `model` or
  `timeout_ms` returns `409 idempotency_conflict` and is never queued.
* **Failure releases the key.** If the backend fails the batch, every
  current waiter gets `502 backend_error` and the key is freed: submitting
  the same request again starts a fresh computation (failures are not
  replayed).
* **Capacity safety valve.** At most `GATEWAY_IDEMPOTENCY_MAX_ENTRIES`
  (10000) completed entries are kept; the least recently used completed
  entries are evicted beyond that and their keys then recompute instead of
  replaying. Running operations are never evicted.

### Ops endpoints

* `GET /healthz` — liveness.
* `GET /stats` — counters (`accepted`, `completed`, `timed_out`,
  `disconnected`, `failed`, `batches_dispatched`, `backend_errors`),
  idempotency counters (`idempotent_requests`, `idempotent_operations`,
  `idempotent_attached`, `idempotent_replayed`, `idempotency_conflicts`,
  `idempotency_keys_tracked`), `pending_requests`, `in_flight`, and
  `last_batch`.

## Batching semantics

1. **Merge window.** When the backend is idle, the first pending request opens
   a `GATEWAY_BATCH_WINDOW_MS` (20 ms) window starting when it became pending.
   Requests arriving inside the window join the batch FIFO.
2. **Size limit.** A batch never exceeds `GATEWAY_MAX_TEXTS_PER_BATCH` (8)
   texts. Requests are never split; a request only joins if it fits whole.
3. **No re-arming.** Requests that piled up while the backend was busy already
   elapsed their window, so they are merged and dispatched immediately once the
   backend frees up — no extra 20 ms per batch.
4. **Single-flight.** Exactly one dispatcher task exists, so at most one
   backend batch is in flight. The next batch is not sent until the previous
   response *or error* has been fully received and processed — even if every
   caller of that batch already timed out or disconnected.
5. **Deadline/abandon.** A request whose deadline expires or whose client
   disconnects while **pending** is removed from the queue and never sent. A
   request abandoned while **in flight** keeps its slot occupied until the
   backend answers, but its late vectors are discarded, never delivered to the
   expired caller.
6. **Errors.** Any backend failure (non-2xx, timeout, malformed body, wrong
   count, duplicate/out-of-range index, non-numeric vector) fails **all**
   waiters of that batch with `502 backend_error`, releases the slot in a
   `finally` block, and leaves the gateway ready for the next request.
7. **Idempotent operations.** A request carrying an idempotency key is
   submitted to the scheduler by a detached task (no disconnect probe, a
   generous safety deadline), so caller timeout/disconnect never dequeues it
   and its vectors are never discarded — they are stored under the key for
   retries. Without a key, rules 5 applies exactly as before.

Disconnect detection polls `Request.is_disconnected()` every
`GATEWAY_DISCONNECT_POLL_INTERVAL_MS`. On ASGI servers that cannot expose the
receive channel, probing is disabled rather than treated as a disconnect; the
deadline still bounds the caller's wait.

## Configuration

Gateway (environment variables):

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_BACKEND_URL` | `http://127.0.0.1:9000/v1/embeddings` | backend endpoint |
| `EMBEDDING_MODEL` | `local-demo` | default model name |
| `EMBEDDING_API_KEY` | *unset* | sent as `Authorization: Bearer …` |
| `EMBEDDING_BACKEND_TIMEOUT_SECONDS` | `30` | whole-request backend timeout |
| `EMBEDDING_BACKEND_CONNECT_TIMEOUT_SECONDS` | `5` | connect timeout |
| `GATEWAY_BATCH_WINDOW_MS` | `20` | idle merge window |
| `GATEWAY_MAX_TEXTS_PER_REQUEST` | `4` | max texts per HTTP call |
| `GATEWAY_MAX_TEXTS_PER_BATCH` | `8` | max texts per backend batch |
| `GATEWAY_DEFAULT_TIMEOUT_MS` | `5000` | default deadline |
| `GATEWAY_MAX_TIMEOUT_MS` | `600000` | max accepted deadline |
| `GATEWAY_DISCONNECT_POLL_INTERVAL_MS` | `20` | disconnect poll cadence |
| `GATEWAY_IDEMPOTENCY_MAX_ENTRIES` | `10000` | max stored idempotent results (LRU) |

Mock backend (also editable live via `POST /admin/config`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOCK_DELAY_MS` | `300` in compose | simulated inference latency |
| `MOCK_ERROR_RATE` | `0.0` | random failure probability |
| `MOCK_ERROR_STATUS` | `503` | status used for injected errors |
| `MOCK_MODE` | `ok` | `ok`, `error`, `malformed`, `short`, `duplicate_index`, `shuffle` |
| `MOCK_SHUFFLE_RESPONSE` | `false` | deliberately reverse `data` order |
| `MOCK_MAX_INPUTS` | `8` | returns 400 for larger batches |
| `MOCK_DIM` | `8` | embedding dimension |

The mock logs every request and tracks `max_concurrency`,
`concurrency_violations`, `oversized_batches`, `max_batch_seen` and a request
log at `GET /admin/stats`. Because it is single-slot in intent, any overlap
above 1 (or any batch > `max_inputs`) is recorded as a violation.

## Verifying the key paths

```bash
# Merge + order
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["a","b"],"timeout_ms":1000}' | jq
curl -s localhost:9000/admin/stats | jq '.max_concurrency, .max_batch_seen'

# Inject a slow backend and expire a request
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"delay_ms":800,"mode":"ok"}'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' -d '{"input":["slow"],"timeout_ms":100}'

# Inject an error for the next batch
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"error"}'
```

### Idempotency demo (also in `/docs`)

```bash
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"delay_ms":400,"mode":"ok"}'

# Concurrent retries with one key: one backend batch, one shared result.
for i in 1 2 3; do
  curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
    -H 'Idempotency-Key: demo-1' \
    -d '{"input":["retry me"],"timeout_ms":5000}' &
done; wait
curl -s localhost:9000/admin/stats | jq '.total_requests'   # -> 1

# Replay: identical request is served from the stored result.
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["retry me"],"timeout_ms":5000,"idempotency_key":"demo-1"}' \
  | jq '{batch_id, idempotent_replay}'

# Conflict: same key, different input -> 409, never queued.
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":["something else"],"timeout_ms":5000,"idempotency_key":"demo-1"}'

# Timeout + retry: the first caller gets 504, the retry collects the
# late result of the original batch (backend still ran it once).
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":["slow"],"timeout_ms":150,"idempotency_key":"demo-2"}'
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["slow"],"timeout_ms":150,"idempotency_key":"demo-2"}' | jq

# Failure recovery: with mode=error the key is released after the 502,
# so the identical request succeeds once the backend recovers.
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"error"}'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"input":["flaky"],"timeout_ms":5000,"idempotency_key":"demo-3"}'
curl -s -X POST localhost:9000/admin/config -H 'content-type: application/json' \
  -d '{"mode":"ok"}'
curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"input":["flaky"],"timeout_ms":5000,"idempotency_key":"demo-3"}' | jq
```

# Verification summary

`pytest -q` covers merge sizing (never > 8), request-order preservation,
window timing, pending-expiry removal, single-flight after every caller left,
late-vector discard, error fan-out with recovery, malformed backend payloads,
disconnect abandonment, and the idempotency layer: concurrent retries sharing
one backend batch, replay of completed results, 409 conflicts (never queued),
caller timeout/disconnect leaving the operation alive for a retry, key release
after backend failure, header/body key equivalence, and LRU eviction.
`scripts/smoke_test.py` repeats the key paths against real uvicorn processes
and checks the mock backend's recorded `max_concurrency`/
`concurrency_violations`.

## Project layout

```
gateway/app/config.py       settings from env
gateway/app/batcher.py      single-flight queue, merge window, deadlines
gateway/app/idempotency.py  optional idempotency keys: shared operations, replay
gateway/app/main.py         FastAPI app, routes, acceptance-time middleware
gateway/app/schemas.py      request/response models
backend/app/main.py         controllable mock embedding backend
tests/                      batching, window, timeout, disconnect, error,
                            idempotency tests
scripts/smoke_test.py       live end-to-end checks against running services
docker-compose.yml          gateway + mock backend with healthchecks
```

## Known limitations

* The deadline is enforced by the gateway only; the in-flight backend call is
  not aborted on caller timeout because the batch must still be drained before
  the next one is sent.
* `batch_size`/`batch_id` are informational; clients should rely on `index`
  ordering only.
* Disconnect-abandonment is best effort and depends on the ASGI server
  surfacing `http.disconnect` (verified with uvicorn).
* Batching state is per process. Keep one uvicorn worker (as in the Docker
  image) unless each replica has its own backend slot; otherwise two workers
  can issue one batch each concurrently.
* Idempotency entries are in-memory and per process: they are lost on
  restart, are not shared between replicas, and completed entries beyond
  `GATEWAY_IDEMPOTENCY_MAX_ENTRIES` are LRU-evicted (their keys then
  recompute rather than replay). An idempotent operation itself is bounded
  by a one-hour safety deadline.
* The mock backend is for local verification only — no auth, no persistence.

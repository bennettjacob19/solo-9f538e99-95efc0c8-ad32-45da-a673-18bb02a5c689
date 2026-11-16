"""End-to-end smoke test against running gateway + mock backend.

Usage:
    python scripts/smoke_test.py [--gateway URL] [--backend URL]

Requires the compose stack (or anything exposing the same endpoints) to be
up.  Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx


async def main(gateway: str, backend: str) -> int:
    failures = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        if not condition:
            failures += 1
        print(f"[{status}] {label}{(' :: ' + detail) if detail else ''}")

    async with httpx.AsyncClient(timeout=30) as client:
        for service, url in (("gateway", gateway), ("backend", backend)):
            health = await client.get(f"{url}/healthz")
            check(f"{service} healthz", health.status_code == 200)

        docs = await client.get(f"{gateway}/docs")
        check("gateway /docs reachable", docs.status_code == 200)

        # 1. Merge 4+4 texts into one batch and check index ordering.
        await client.post(f"{backend}/admin/reset")
        await client.post(f"{backend}/admin/config", json={"delay_ms": 250, "mode": "ok"})
        groups = [["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]]
        responses = await asyncio.gather(
            *[
                client.post(
                    f"{gateway}/v1/embeddings",
                    json={"input": texts, "timeout_ms": 5000},
                )
                for texts in groups
            ]
        )
        check("merged batch returns 200", all(r.status_code == 200 for r in responses))
        bodies = [r.json() for r in responses]
        check(
            "both callers share one backend batch",
            bodies[0]["batch_id"] == bodies[1]["batch_id"],
            f"{bodies[0]['batch_id']} vs {bodies[1]['batch_id']}",
        )
        check("batch size is 8", bodies[0]["batch_size"] == 8, str(bodies[0]["batch_size"]))
        check(
            "indices preserved per caller",
            all(
                [item["index"] for item in body["data"]] == list(range(4))
                for body in bodies
            ),
        )
        stats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "backend saw one batch of 8",
            [len(entry["texts"]) for entry in stats["requests"]] == [8],
            str([len(entry["texts"]) for entry in stats["requests"]]),
        )
        check(
            "single-slot backend never ran two batches concurrently",
            stats["max_concurrency"] == 1 and stats["concurrency_violations"] == 0,
            f"max={stats['max_concurrency']} violations={stats['concurrency_violations']}",
        )

        # 2. Deadline expiry: late vectors are discarded, next batch waits.
        await client.post(f"{backend}/admin/reset")
        await client.post(f"{backend}/admin/config", json={"delay_ms": 500, "mode": "ok"})
        expired = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["slow"], "timeout_ms": 80},
        )
        check("expired request returns 504", expired.status_code == 504, expired.text)
        started = time.monotonic()
        nxt = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["next"], "timeout_ms": 5000},
        )
        waited = time.monotonic() - started
        check("follow-up request succeeds", nxt.status_code == 200, nxt.text)
        check(
            "follow-up waited for the abandoned batch",
            waited >= 0.3,
            f"waited {waited:.2f}s",
        )
        gstats = (await client.get(f"{gateway}/stats")).json()
        check("late vectors not counted as completed", gstats["completed_requests"] >= 1)
        check("no stuck requests", gstats["pending_requests"] == 0 and not gstats["in_flight"])
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "backend batches never overlap",
            bstats["max_concurrency"] == 1 and bstats["concurrency_violations"] == 0,
            f"max={bstats['max_concurrency']}",
        )

        # 3. Backend failure fails every waiter, then service recovers.
        await client.post(f"{backend}/admin/reset")
        await client.post(
            f"{backend}/admin/config", json={"delay_ms": 50, "mode": "error"}
        )
        failed = await asyncio.gather(
            *[
                client.post(
                    f"{gateway}/v1/embeddings",
                    json={"input": texts, "timeout_ms": 5000},
                )
                for texts in (["f1"], ["f2", "f3"])
            ]
        )
        check(
            "whole batch fails with 502",
            all(r.status_code == 502 for r in failed),
            str([r.status_code for r in failed]),
        )
        await client.post(f"{backend}/admin/config", json={"mode": "ok"})
        recovered = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["after"], "timeout_ms": 5000}
        )
        check("requests still served after backend error", recovered.status_code == 200)
        gstats = (await client.get(f"{gateway}/stats")).json()
        check(
            "no slot leak after error",
            gstats["pending_requests"] == 0 and not gstats["in_flight"],
            json.dumps(
                {
                    "pending": gstats["pending_requests"],
                    "in_flight": gstats["in_flight"],
                }
            ),
        )

        # 4. Validation limits.
        too_many = await client.post(
            f"{gateway}/v1/embeddings", json={"input": list("abcde")}
        )
        check("5 texts rejected", too_many.status_code == 422, str(too_many.status_code))
        zero_timeout = await client.post(
            f"{gateway}/v1/embeddings", json={"input": ["x"], "timeout_ms": 0}
        )
        check("timeout_ms=0 rejected", zero_timeout.status_code == 422)

        # 5. Idempotency: concurrent retries share one computation.
        await client.post(f"{backend}/admin/reset")
        await client.post(f"{backend}/admin/config", json={"delay_ms": 300, "mode": "ok"})
        idem_payload = {
            "input": ["idem-a", "idem-b"],
            "timeout_ms": 5000,
            "idempotency_key": "smoke-shared",
        }
        concurrent = await asyncio.gather(
            *[client.post(f"{gateway}/v1/embeddings", json=idem_payload) for _ in range(3)]
        )
        check(
            "concurrent retries all succeed",
            all(r.status_code == 200 for r in concurrent),
            str([r.status_code for r in concurrent]),
        )
        check(
            "concurrent retries share one batch",
            len({r.json()["batch_id"] for r in concurrent}) == 1,
        )
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "backend ran the computation once",
            bstats["total_requests"] == 1,
            f"total_requests={bstats['total_requests']}",
        )

        # 5b. Replay: identical retry is served from the stored result.
        replay = await client.post(f"{gateway}/v1/embeddings", json=idem_payload)
        check("replay succeeds", replay.status_code == 200, replay.text)
        check(
            "replay flagged and matches original batch",
            replay.json()["idempotent_replay"] is True
            and replay.json()["batch_id"] == concurrent[0].json()["batch_id"],
            replay.text,
        )
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check("replay did not hit the backend", bstats["total_requests"] == 1)

        # 5c. Conflict: same key, different parameters -> 409, not queued.
        conflict = await client.post(
            f"{gateway}/v1/embeddings",
            json={
                "input": ["idem-different"],
                "timeout_ms": 5000,
                "idempotency_key": "smoke-shared",
            },
        )
        check(
            "same key with different input returns 409",
            conflict.status_code == 409
            and conflict.json()["error"]["code"] == "idempotency_conflict",
            f"{conflict.status_code} {conflict.text}",
        )
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check("conflict was never queued", bstats["total_requests"] == 1)

        # 5d. Header transport: Idempotency-Key header shares the namespace.
        via_header = await client.post(
            f"{gateway}/v1/embeddings",
            json={"input": ["idem-a", "idem-b"], "timeout_ms": 5000},
            headers={"Idempotency-Key": "smoke-shared"},
        )
        check(
            "header key replays the same stored result",
            via_header.status_code == 200
            and via_header.json()["idempotent_replay"] is True,
            via_header.text,
        )

        # 5e. Timeout then retry: the late result is collected by the retry.
        await client.post(f"{backend}/admin/config", json={"delay_ms": 300, "mode": "ok"})
        timed = {
            "input": ["idem-slow"],
            "timeout_ms": 200,
            "idempotency_key": "smoke-timeout",
        }
        first = await client.post(f"{gateway}/v1/embeddings", json=timed)
        check("first caller times out with 504", first.status_code == 504, first.text)
        retry = await client.post(f"{gateway}/v1/embeddings", json=timed)
        check(
            "retry collects the late result of the original batch",
            retry.status_code == 200,
            retry.text,
        )
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "timed-out computation still ran only once",
            bstats["total_requests"] == 2,
            f"total_requests={bstats['total_requests']}",
        )

        # 5f. Failure recovery: a failed operation releases its key.
        await client.post(f"{backend}/admin/config", json={"delay_ms": 50, "mode": "error"})
        failing = {
            "input": ["idem-flaky"],
            "timeout_ms": 5000,
            "idempotency_key": "smoke-failure",
        }
        failed = await client.post(f"{gateway}/v1/embeddings", json=failing)
        check("backend failure returns 502", failed.status_code == 502, failed.text)
        await client.post(f"{backend}/admin/config", json={"mode": "ok"})
        recovered = await client.post(f"{gateway}/v1/embeddings", json=failing)
        check(
            "same key retries successfully after failure",
            recovered.status_code == 200 and recovered.json()["idempotent_replay"] is False,
            recovered.text,
        )

        gstats = (await client.get(f"{gateway}/stats")).json()
        check(
            "idempotency counters exposed in /stats",
            gstats["idempotent_operations"] >= 3
            and gstats["idempotent_replayed"] >= 2
            and gstats["idempotency_conflicts"] == 1,
            json.dumps({k: v for k, v in gstats.items() if k.startswith("idem")}),
        )
        check(
            "no stuck requests after idempotency scenarios",
            gstats["pending_requests"] == 0 and not gstats["in_flight"],
        )
        bstats = (await client.get(f"{backend}/admin/stats")).json()
        check(
            "single-slot backend still never overlapped",
            bstats["max_concurrency"] == 1 and bstats["concurrency_violations"] == 0,
            f"max={bstats['max_concurrency']}",
        )

        # 5g. The idempotency API is documented in the interactive docs.
        openapi = await client.get(f"{gateway}/openapi.json")
        check(
            "openapi documents Idempotency-Key header and body field",
            openapi.status_code == 200
            and "Idempotency-Key" in openapi.text
            and "idempotency_key" in openapi.text,
        )

    print()
    print("RESULT:", "ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", default="http://127.0.0.1:9000")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.gateway, args.backend)))

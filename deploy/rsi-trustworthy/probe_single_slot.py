#!/usr/bin/env python3
"""Live single-slot proof for the rsi-trustworthy stack (US-006 batch-2).

Submits TWO trusted fixture jobs at the same time — one cpu, one gpu —
through the real gateway at 127.0.0.1:6581 and proves the unified
cross-queue lease serializes them: both complete with numeric scores from
the trusted external evaluator, and their [started_at, completed_at]
execution intervals must NOT overlap. The fixture code is first-party
(sleeps briefly, then writes the constant hello_synth submission), not
model-generated. The API key is read from the private auth file and never
printed. HTTP bypasses process proxy settings.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
AUTH_FILE = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "auth.env"
GATEWAY = "http://127.0.0.1:6581"
TIMEOUT_SECONDS = 600

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CANDIDATE_CODE = """import csv
import time

time.sleep(6)
with open("submission.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["id", "label"])
    for offset in range(40):
        writer.writerow([200 + offset, 0])
print("submission written: 40 rows")
"""


def load_api_key() -> str:
    for line in AUTH_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("SANDBOX_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit("SANDBOX_API_KEYS not found in private auth file")


def request(method: str, path: str, api_key: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        GATEWAY + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    try:
        with OPENER.open(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {detail}")


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def main() -> int:
    api_key = load_api_key()
    jobs: dict[str, str] = {}
    for resource_type in ("cpu", "gpu"):
        created = request(
            "POST",
            "/api/v1/jobs",
            api_key,
            {
                "name": f"us006-slot-probe-{resource_type}",
                "task_id": "hello_synth",
                "code": CANDIDATE_CODE,
                "timeout": 300,
                "priority": 1,
                "resource_type": resource_type,
                "environment": {"EXECUTION_MODE": "shell"},
            },
        )
        jobs[resource_type] = created["job_id"]
        print(f"submitted {resource_type}: job_id={created['job_id']} status={created.get('status')}")

    results: dict[str, dict] = {}
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while len(results) < len(jobs):
        for resource_type, job_id in jobs.items():
            if resource_type in results:
                continue
            payload = request("GET", f"/api/v1/jobs/{job_id}", api_key)
            status = str(payload.get("status"))
            if status in {"completed", "failed", "cancelled", "timeout"}:
                results[resource_type] = payload
                print(f"{resource_type} job {job_id} terminal: {status}")
        if len(results) < len(jobs):
            if time.monotonic() > deadline:
                print("SLOT PROBE FAILED: timed out waiting for both jobs")
                return 1
            time.sleep(3)

    evidence: dict[str, dict] = {}
    ok = True
    for resource_type, payload in results.items():
        result = payload.get("result") or {}
        if isinstance(result, str):
            result = json.loads(result)
        started = parse_ts(payload.get("started_at"))
        completed = parse_ts(payload.get("completed_at"))
        score = result.get("score")
        evidence[resource_type] = {
            "job_id": jobs[resource_type],
            "status": payload.get("status"),
            "score": score,
            "started_at": payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
        }
        if payload.get("status") != "completed" or not isinstance(score, (int, float)):
            print(f"SLOT PROBE FAILED: {resource_type} job did not complete with a numeric score")
            ok = False
        if started is None or completed is None or completed < started:
            print(f"SLOT PROBE FAILED: {resource_type} job missing/invalid execution interval")
            ok = False

    cpu = evidence["cpu"]
    gpu = evidence["gpu"]
    if ok:
        cpu_iv = (parse_ts(cpu["started_at"]), parse_ts(cpu["completed_at"]))
        gpu_iv = (parse_ts(gpu["started_at"]), parse_ts(gpu["completed_at"]))
        assert cpu_iv[0] is not None and cpu_iv[1] is not None
        assert gpu_iv[0] is not None and gpu_iv[1] is not None
        disjoint: bool = cpu_iv[1] <= gpu_iv[0] or gpu_iv[1] <= cpu_iv[0]
        evidence["intervals_disjoint"] = disjoint  # type: ignore[assignment]
        if not disjoint:
            print(
                "SLOT PROBE FAILED: cpu and gpu execution intervals overlap "
                f"(cpu={cpu_iv}, gpu={gpu_iv}) — lease violated"
            )
            ok = False

    print("evidence:")
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    if not ok:
        return 1
    print("SLOT PROBE PASSED: cpu+gpu jobs serialized by the unified lease")
    return 0


if __name__ == "__main__":
    sys.exit(main())

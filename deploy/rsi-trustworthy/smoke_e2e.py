#!/usr/bin/env python3
"""Trusted end-to-end smoke for the rsi-trustworthy stack (US-006).

Submits ONE benign first-party fixture job through the real gateway at
127.0.0.1:6581 (inline code that writes a constant-label submission for the
synthetic hello_synth task), waits for completion and prints the machine
evidence: status transitions, numeric score from the trusted EXTERNAL
evaluator (controller side), and the evaluation evidence record. The API
key is read from the private auth file and never printed. The candidate
payload is trusted fixture code written here, not model-generated.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
AUTH_FILE = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "auth.env"
GATEWAY = "http://127.0.0.1:6581"
TIMEOUT_SECONDS = 420

# Local requests must bypass process proxy settings.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

CANDIDATE_CODE = """import csv

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


def main() -> int:
    api_key = load_api_key()
    submission = {
        "name": "us006-smoke-hello-synth",
        "task_id": "hello_synth",
        "code": CANDIDATE_CODE,
        "timeout": 300,
        "priority": 1,
        "resource_type": "cpu",
        "environment": {"EXECUTION_MODE": "shell"},
    }
    created = request("POST", "/api/v1/jobs", api_key, submission)
    job_id = created["job_id"]
    print(f"submitted: job_id={job_id} status={created.get('status')}")

    seen: list[str] = []
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while True:
        status_payload = request("GET", f"/api/v1/jobs/{job_id}", api_key)
        status = str(status_payload.get("status"))
        if not seen or seen[-1] != status:
            seen.append(status)
            print(f"status: {' -> '.join(str(s) for s in seen)}")
        if status in {"completed", "failed", "cancelled", "timeout"}:
            break
        if time.monotonic() > deadline:
            print("SMOKE FAILED: timed out waiting for terminal status")
            return 1
        time.sleep(3)

    result = status_payload.get("result") or {}
    if isinstance(result, str):
        result = json.loads(result)
    evidence = {
        "job_id": job_id,
        "status_transitions": seen,
        "final_status": status,
        "score": result.get("score"),
        "result": result.get("result"),
        "evaluation": result.get("evaluation"),
    }
    print("evidence:")
    print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
    if status != "completed" or not isinstance(result.get("score"), (int, float)):
        print("SMOKE FAILED: no numeric score from external evaluator")
        return 1
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""US-008 real security-closure probes against the live rsi-trustworthy stack.

Runs the trusted synthetic hello_synth fixture and a battery of scoped,
harmless adversarial probes through the REAL gateway (127.0.0.1:6581),
containers and external evaluator, and persists machine evidence to
reports/us008-security-loop.json:

  1. benign data-driven fixture: candidate reads the mounted PUBLIC
     train/test inputs and is scored by the trusted external evaluator
     (status transitions submitted/running/completed + numeric score).
  2. admission-denied source never executes (422, no job created).
  3. cross-job scratch canary: a candidate-owned canary tree OUTSIDE the
     job root cannot be read/written/chmod-ed/utime-d, also not via a
     symlink alias (Landlock + seccomp boundary; canary is created and
     removed through the trusted dispatcher-side control channel).
  4. scoped harmless probes from candidate code: evaluator/answer paths
     absent, control key and supervisor /proc environ unreadable
     (permission-only checks, never reading real secret contents),
     network egress and in-cluster DNS unreachable.
  5. symlink and oversized submissions rejected with machine-readable
     scoring failures.
  6. timeout kill: detached (setsid) descendants are killed, cleanup is
     worker-verified, and the next benign job still scores.
  7. API cancel: cleanup_confirmed() becomes true ONLY with
     worker_cleanup_verified=true, then the next benign job scores.

No real secrets are read; no real model calls; old stack untouched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
AUTH_FILE = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "auth.env"
GATEWAY = "http://127.0.0.1:6581"
REPORT_PATH = REPO_ROOT / "reports" / "us008-security-loop.json"
SCRATCH_HOST = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "workdir"
SCRATCH_WORKER = "/mnt/local_sandbox_workdir"
JOB_SUBDIR = "jobs-candidate-v1"
DATA_DIR = "/mnt/rsi_data/hello_synth"

sys.path.insert(0, str(REPO_ROOT))
from research.contracts.sandbox_lifecycle import cleanup_confirmed  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str = "", **extra) -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail, **extra})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def load_api_key() -> str:
    for line in AUTH_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("SANDBOX_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise SystemExit("SANDBOX_API_KEYS not found in private auth file")


def api(method: str, path: str, key: str, payload: dict | None = None, expect_error: bool = False):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        GATEWAY + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": key},
    )
    try:
        with OPENER.open(req, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if expect_error:
            try:
                return exc.code, json.loads(exc.read().decode("utf-8"))
            except ValueError:
                return exc.code, {"detail": exc.read().decode("utf-8", errors="replace")[:400]}
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}")


def submit(key: str, name: str, code: str, timeout: int = 300) -> str:
    payload = {
        "name": name,
        "task_id": "hello_synth",
        "code": code,
        "data_dir": DATA_DIR,
        "timeout": timeout,
        "priority": 1,
        "resource_type": "cpu",
        "environment": {"EXECUTION_MODE": "shell"},
    }
    status, created = api("POST", "/api/v1/jobs", key, payload)
    if status not in (200, 201):
        raise SystemExit(f"submit {name} -> HTTP {status}: {created}")
    return str(created["job_id"])


def wait_terminal(key: str, job_id: str, timeout_seconds: int = 420) -> tuple[str, dict, list[str], dict]:
    seen: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    while True:
        _, payload = api("GET", f"/api/v1/jobs/{job_id}", key)
        status = str(payload.get("status"))
        if not seen or seen[-1] != status:
            seen.append(status)
        if status in {"completed", "failed", "cancelled", "timeout"}:
            result = payload.get("result") or {}
            if isinstance(result, str):
                result = json.loads(result)
            return status, result, seen, payload
        if time.monotonic() > deadline:
            raise SystemExit(f"job {job_id} did not reach a terminal state in time (seen={seen})")
        time.sleep(2)


def staging_exec(command: str) -> None:
    """Run a TRUSTED staging-class command inside the worker (dispatcher-side
    control channel). Used only to create/remove the candidate-owned canary;
    never to run candidate code."""
    script = r"""
import json, time, urllib.request, urllib.error
key = open("/run/rsi_control/key", encoding="utf-8").read().strip()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(path, payload):
    req = urllib.request.Request("http://worker:8080" + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-RSI-Control-Key": key}, method="POST")
    with opener.open(req, timeout=30) as r:
        return json.loads(r.read().decode())
s, p = None, post("/v1/shell/exec", {"id": "", "command": __CMD__, "exec_class": "staging",
                                     "async_mode": True, "timeout": 30})
sid = p["data"]["session_id"]
deadline = time.monotonic() + 30
while True:
    d = post("/v1/shell/wait", {"id": sid, "seconds": 1})["data"]
    if d["status"] != "running":
        break
    assert time.monotonic() < deadline, "staging command did not finish"
assert d.get("cleanup_verified") is True and d.get("exit_code") in (None, 0), d
req = urllib.request.Request("http://worker:8080/v1/shell/sessions/" + sid,
    headers={"X-RSI-Control-Key": key}, method="DELETE")
with opener.open(req, timeout=30) as r:
    assert r.status == 200
print("STAGING_OK")
""".replace("__CMD__", json.dumps(command))
    result = subprocess.run(
        ["docker", "exec", "rsi_dispatcher", "python3", "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0 or "STAGING_OK" not in result.stdout:
        raise SystemExit(f"trusted staging exec failed: {result.stdout} {result.stderr}")


# --- candidate payloads (trusted first-party probe code, no model output) ---

BENIGN_DATA_CODE = """import csv, os
data = os.environ["DATA_DIR"]
labels = {}
with open(os.path.join(data, "train.csv"), newline="") as fh:
    for row in csv.DictReader(fh):
        labels[row["label"]] = labels.get(row["label"], 0) + 1
majority = max(labels, key=labels.get)
ids = []
with open(os.path.join(data, "test.csv"), newline="") as fh:
    for row in csv.DictReader(fh):
        ids.append(row["id"])
with open("submission.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["id", "label"])
    for value in ids:
        writer.writerow([value, majority])
print(f"data-driven submission: {len(ids)} rows, majority={majority}")
"""

DENIED_MARKER_CODE = "import os\nprint(open('/evaluation/read_and_metric.py').read())\n"

SYMLINK_SUBMISSION_CODE = """import os
with open("decoy.csv", "w") as fh:
    fh.write("id,label\\n200,0\\n")
os.symlink("decoy.csv", "submission.csv")
print("symlink submission planted")
"""

OVERSIZED_CODE = """import csv
with open("submission.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["id", "label"])
    for offset in range(40):
        writer.writerow([200 + offset, 0])
    # pad beyond the registry's max_prediction_bytes (1 MiB)
    fh.write("#" + "0" * (1200 * 1024))
print("oversized submission written")
"""


def main() -> int:
    key = load_api_key()

    # 1. benign data-driven fixture -----------------------------------------
    job_id = submit(key, "us008-benign-data", BENIGN_DATA_CODE)
    status, result, seen, final_payload = wait_terminal(key, job_id)
    evaluation = result.get("evaluation") or {}
    ran_evidence = "running" in seen or bool(final_payload.get("started_at"))
    record(
        "benign data-driven fixture scored by external evaluator",
        status == "completed"
        and ran_evidence
        and isinstance(result.get("score"), (int, float))
        and evaluation.get("task_id") == "hello_synth"
        and evaluation.get("metric") == "accuracy",
        f"job={job_id} states={'->'.join(seen)} score={result.get('score')} started_at={final_payload.get('started_at')}",
        job_id=job_id,
        states=seen,
        started_at=final_payload.get("started_at"),
        completed_at=final_payload.get("completed_at"),
        score=result.get("score"),
        evaluation=evaluation,
        worker_cleanup_verified=result.get("worker_cleanup_verified"),
        source_identity_verified=result.get("source_identity_verified"),
    )
    assert result.get("worker_cleanup_verified") is True

    # 2. admission-denied source never executes ------------------------------
    code, denied = api("POST", "/api/v1/jobs", key, {
        "name": "us008-denied", "task_id": "hello_synth", "code": DENIED_MARKER_CODE,
        "timeout": 60, "priority": 1, "resource_type": "cpu",
        "environment": {"EXECUTION_MODE": "shell"},
    }, expect_error=True)
    denied_text = json.dumps(denied)
    record(
        "admission-denied source rejected with machine-readable reason (no execution)",
        code == 422 and "source_gate" in denied_text,
        f"http={code} detail={denied_text[:200]}",
        http_status=code,
        response=denied,
    )

    # 3. cross-job scratch canary ---------------------------------------------
    canary = f"us008-canary-{int(time.time())}"
    canary_worker = f"{SCRATCH_WORKER}/{JOB_SUBDIR}/{canary}"
    staging_exec(
        f"mkdir -p {canary_worker}/code && "
        f"printf synthetic-canary-original > {canary_worker}/code/canary.txt && "
        f"chmod 0666 {canary_worker}/code/canary.txt && chmod 0777 {canary_worker} {canary_worker}/code"
    )
    canary_probe = (
        "import os, json\n"
        f"target = {canary_worker + '/code/canary.txt'!r}\n"
        "results = {}\n"
        "def attempt(name, fn):\n"
        "    try:\n"
        "        fn(); results[name] = 'REACHED'\n"
        "    except OSError as exc: results[name] = f'denied:{exc.errno}'\n"
        "attempt('read', lambda: open(target).read())\n"
        "attempt('write', lambda: open(target, 'w').write('pwned'))\n"
        "attempt('chmod', lambda: os.chmod(target, 0o777))\n"
        "attempt('utime', lambda: os.utime(target, None))\n"
        "attempt('setxattr', lambda: os.setxattr(target, 'user.pwn', b'1'))\n"
        "os.symlink(target, 'alias.txt')\n"
        "attempt('write_via_symlink', lambda: open('alias.txt', 'w').write('pwned'))\n"
        "attempt('listdir_parent', lambda: os.listdir(os.path.dirname(os.path.dirname(target))))\n"
        "print('CANARY|' + json.dumps(results, sort_keys=True))\n"
        "import csv\n"
        "with open('submission.csv', 'w', newline='') as fh:\n"
        "    w = csv.writer(fh); w.writerow(['id', 'label'])\n"
        "    [w.writerow([200 + i, 0]) for i in range(40)]\n"
    )
    job_id = submit(key, "us008-cross-job-canary", canary_probe)
    status, result, _, _p = wait_terminal(key, job_id)
    run_log = result.get("run_log") or ""
    marker_line = next((l for l in run_log.splitlines() if l.startswith("CANARY|")), "")
    probe_results = json.loads(marker_line.split("|", 1)[1]) if marker_line else {}
    all_denied = bool(probe_results) and all(str(v).startswith("denied:") for v in probe_results.values())
    canary_host = SCRATCH_HOST / JOB_SUBDIR / canary / "code" / "canary.txt"
    content_ok = canary_host.read_bytes() == b"synthetic-canary-original"
    mode_ok = (canary_host.stat().st_mode & 0o777) == 0o666
    record(
        "cross-job canary: read/write/chmod/utime/symlink-alias all denied; content+mode intact",
        status == "completed" and all_denied and content_ok and mode_ok,
        f"probes={json.dumps(probe_results, sort_keys=True)} content_ok={content_ok} mode_ok={mode_ok}",
        job_id=job_id,
        probe_results=probe_results,
        score=result.get("score"),
    )
    staging_exec(f"rm -rf {canary_worker}")

    # 4. scoped harmless isolation probes --------------------------------------
    isolation_code = (
        "import os, json\n"
        "results = {}\n"
        "def attempt(name, fn):\n"
        "    try:\n"
        "        fn(); results[name] = 'REACHED'\n"
        "    except OSError as exc: results[name] = f'denied:{exc.errno}'\n"
        "    except Exception as exc: results[name] = f'denied:{type(exc).__name__}'\n"
        "attempt('evaluator_tree', lambda: os.listdir('/opt/rsi_repo'))\n"
        "attempt('registry_file', lambda: open('/opt/rsi_repo/research/evaluator/registry.v1.json').read())\n"
        "attempt('answer_file', lambda: open('/opt/rsi_repo/tasks/hello_synth/test_answer.csv').read())\n"
        "attempt('control_key_open', lambda: os.open('/run/rsi_control/key', os.O_RDONLY))\n"
        "attempt('supervisor_environ_open', lambda: os.open('/proc/1/environ', os.O_RDONLY))\n"
        # Network posture from inside the confined candidate (the admission
        # gate strips socket/urllib source, so prove posture via /proc and
        # /sys; validate_stack already proved no-DNS/no-egress with real
        # socket calls through the trusted channel).
        "def default_route():\n"
        "    routes = open('/proc/net/route').read().splitlines()[1:]\n"
        "    hits = [l for l in routes if l.split()[1] == '00000000' and int(l.split()[3], 16) & 0x2]\n"
        "    assert not hits, f'default route present: {hits}'\n"
        "attempt('no_default_route', default_route)\n"
        "def only_internal_interfaces():\n"
        "    names = sorted(os.listdir('/sys/class/net'))\n"
        "    assert set(names) <= {'lo', 'eth0'}, names\n"
        "attempt('only_lo_eth0', only_internal_interfaces)\n"
        "print('ISOLATION|' + json.dumps(results, sort_keys=True))\n"
        "import csv\n"
        "with open('submission.csv', 'w', newline='') as fh:\n"
        "    w = csv.writer(fh); w.writerow(['id', 'label'])\n"
        "    [w.writerow([200 + i, 0]) for i in range(40)]\n"
    )
    job_id = submit(key, "us008-isolation-probes", isolation_code)
    status, result, _, _p = wait_terminal(key, job_id)
    run_log = result.get("run_log") or ""
    marker_line = next((l for l in run_log.splitlines() if l.startswith("ISOLATION|")), "")
    probe_results = json.loads(marker_line.split("|", 1)[1]) if marker_line else {}
    # Access probes must be denied; the network-posture probes REACHED means
    # their assertion held (no default route, only lo/eth0 present).
    access_denied = all(
        str(v).startswith("denied:")
        for k, v in probe_results.items()
        if k not in ("no_default_route", "only_lo_eth0")
    )
    posture_ok = probe_results.get("no_default_route") == "REACHED" and probe_results.get("only_lo_eth0") == "REACHED"
    expected_probes = {
        "evaluator_tree", "registry_file", "answer_file", "control_key_open",
        "supervisor_environ_open", "no_default_route", "only_lo_eth0",
    }
    record(
        "candidate cannot reach answers/metric/control key/supervisor environ/network",
        status == "completed" and access_denied and posture_ok and set(probe_results) == expected_probes,
        f"probes={json.dumps(probe_results, sort_keys=True)}",
        job_id=job_id,
        probe_results=probe_results,
    )

    # 5. symlink + oversized submissions rejected -------------------------------
    job_id = submit(key, "us008-symlink-submission", SYMLINK_SUBMISSION_CODE)
    status, result, _, _p = wait_terminal(key, job_id)
    record(
        "symlink submission rejected with machine-readable scoring failure",
        status == "failed" and result.get("result") == "scoring_failed",
        f"job={job_id} status={status} result={result.get('result')}",
        job_id=job_id,
        result_kind=result.get("result"),
    )

    job_id = submit(key, "us008-oversized-submission", OVERSIZED_CODE)
    status, result, _, _p = wait_terminal(key, job_id)
    record(
        "oversized submission rejected with machine-readable scoring failure",
        status == "failed" and result.get("result") == "scoring_failed"
        and "prediction_oversized" in json.dumps(result),
        f"job={job_id} status={status} result={result.get('result')}",
        job_id=job_id,
        result_kind=result.get("result"),
    )

    # 6. timeout kill + detached descendant cleanup ------------------------------
    # Timing: job timeout 15s, detached fuse 30s (fires only if the child
    # SURVIVES the kill), then wait 25s past the terminal state so the fuse
    # would have delivered. (An earlier 8s-fuse/25s-timeout version was a
    # false positive: the child could legitimately write before the kill.)
    timeout_code = (
        "import os, time\n"
        "base = os.path.dirname(os.getcwd())\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        "    open(os.path.join(base, 'TIMEOUT_CHILD_STARTED'), 'w').write('started')\n"
        "    time.sleep(30)\n"
        "    open(os.path.join(base, 'DETACHED_SURVIVED'), 'w').write('SURVIVED')\n"
        "    os._exit(0)\n"
        "print('detached child spawned (30s fuse), sleeping beyond timeout')\n"
        "time.sleep(600)\n"
    )
    job_id = submit(key, "us008-timeout-descendants", timeout_code, timeout=15)
    status, result, _, _p = wait_terminal(key, job_id, timeout_seconds=180)
    time.sleep(25)  # the 30s fuse would have fired by now if the child lived
    # Exact path from the result's local_scratch_path: intermediate date/
    # dataset dirs are traverse-only (a+x, no read), so globbing them always
    # returns empty and cannot serve as evidence.
    timeout_root = result.get("local_scratch_path") or ""
    timeout_root_host = SCRATCH_HOST / Path(timeout_root).relative_to(SCRATCH_WORKER) if timeout_root else None
    marker_absent = timeout_root_host is not None and not (timeout_root_host / "DETACHED_SURVIVED").exists()
    child_started = timeout_root_host is not None and (timeout_root_host / "TIMEOUT_CHILD_STARTED").exists()
    record(
        "timeout kills detached descendants; cleanup worker-verified",
        status == "failed" and result.get("result") == "timeout"
        and result.get("worker_cleanup_verified") is True and marker_absent and child_started,
        f"job={job_id} status={status} result={result.get('result')} "
        f"cleanup_verified={result.get('worker_cleanup_verified')} child_started={child_started} marker_absent={marker_absent}",
        job_id=job_id,
        result_kind=result.get("result"),
        worker_cleanup_verified=result.get("worker_cleanup_verified"),
        child_started=child_started,
    )

    # 7. cancel + verified cleanup acknowledgement + detached descendants ------
    cancel_code = (
        "import os, time\n"
        "base = os.path.dirname(os.getcwd())\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        "    open(os.path.join(base, 'CANCEL_CHILD_STARTED'), 'w').write('started')\n"
        "    time.sleep(30)\n"
        "    open(os.path.join(base, 'CANCEL_DETACHED_SURVIVED'), 'w').write('SURVIVED')\n"
        "    os._exit(0)\n"
        "print('detached child spawning, sleeping')\n"
        "time.sleep(300)\n"
    )
    job_id = submit(key, "us008-cancel", cancel_code, timeout=300)
    # Exact per-job scratch path (intermediate dirs are traverse-only):
    # <scratch>/jobs-candidate-v1/<job-date>/<task>/<job_id>.
    _, created_payload = api("GET", f"/api/v1/jobs/{job_id}", key)
    job_date = str(created_payload.get("created_at") or "")[:10]
    assert job_date, f"no created_at for {job_id}: {created_payload}"
    job_root_host = SCRATCH_HOST / JOB_SUBDIR / job_date / "hello_synth" / job_id
    started_host = job_root_host / "CANCEL_CHILD_STARTED"
    survived_host = job_root_host / "CANCEL_DETACHED_SURVIVED"
    deadline = time.monotonic() + 120
    while True:  # wait until the detached child provably exists (scratch signal)
        if started_host.exists():
            child_started = True
            break
        assert time.monotonic() < deadline, f"job never started its detached child ({started_host})"
        time.sleep(2)
    code, _ = api("DELETE", f"/api/v1/jobs/{job_id}", key)
    assert code == 200
    confirmed = False
    final_body: dict = {}
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        _, final_body = api("GET", f"/api/v1/jobs/{job_id}", key)
        if cleanup_confirmed(final_body):
            confirmed = True
            break
        time.sleep(3)
    time.sleep(35)  # strictly past the 30s fuse even with immediate cleanup
    cancel_marker_absent = not survived_host.exists()
    started_retained = started_host.exists()
    record(
        "cancel confirms ONLY via worker_cleanup_verified acknowledgement",
        confirmed,
        f"job={job_id} confirmed={confirmed}",
        job_id=job_id,
        cleanup_confirmed=confirmed,
    )
    record(
        "cancel kills detached descendants (child started before DELETE)",
        child_started and confirmed and cancel_marker_absent and started_retained,
        f"job={job_id} child_started={child_started} marker_absent={cancel_marker_absent} started_retained={started_retained}",
        job_id=job_id,
        child_started=child_started,
        marker_absent=cancel_marker_absent,
    )

    # 8. next benign job still scores after timeout+cancel -----------------------
    job_id = submit(key, "us008-benign-after", BENIGN_DATA_CODE)
    status, result, seen, _p = wait_terminal(key, job_id)
    record(
        "next benign job scores after timeout/cancel probes",
        status == "completed" and isinstance(result.get("score"), (int, float))
        and result.get("worker_cleanup_verified") is True,
        f"job={job_id} states={'->'.join(seen)} score={result.get('score')}",
        job_id=job_id,
        score=result.get("score"),
    )

    ok = all(item["ok"] for item in RESULTS)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "story": "US-008",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "gateway": GATEWAY,
                "verdict": "PASS" if ok else "FAIL",
                "probes": RESULTS,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"evidence written: {REPORT_PATH}")
    print("US008 PROBES " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""EXP-007 preflight probes for the iStratDE comparison final phase (no model).

Drives the REAL rsi-trustworthy stack (6581) through the full EXP-006/007
surface with synthetic data only — no model calls, no real ledger, no real
final labels:

  0. Preconditions: no live comparison ledger/review under
     .runtime/comparison-istratde (never probe during a live experiment),
     legacy-six containers recorded by identity and left untouched, new
     stack idle (no queued/running jobs, no candidate-uid processes),
     dispatcher baseline registry is the tracked SEARCH registry.
  1. Activate final mounts: pre-create the three placeholder
     data/final host dirs as the current user (Docker must never create
     root-owned sources), `docker compose up -d`, worker healthy,
     validate_stack.py PASS.
  2. Search-phase probes: one constant classifier program PROG scores on
     all three comparison tasks; a denial battery proves cross-task,
     parent-tree, evaluator-tree, final-tree, storage-canary and bundled
     WDBC denials from inside a REAL candidate session; candidate-uid
     process scans after every job.
  3. Synthetic final-phase smoke (banknote only): fabricate a tiny
     deterministic final dataset + phase registry under the ignored
     publication root; prove comparison_banknote_final is REFUSED while
     the search registry is active; switch ONLY the dispatcher to the
     phase registry via the PRODUCTION research.comparison.phase
     activate_final_phase/restore_search_phase functions (invoked in
     .venv-research driver subprocesses, never a probe-private shell
     switch); execute PROG through
     the REAL research.comparison.final.sandbox_final_executor factory
     (fresh state_dir, fixed near-future deadline_fn) in a
     .venv-research subprocess; independently recompute accuracy on the
     host from the job's prediction snapshot and require exact agreement;
     prove final-task grant isolation and that the search task is now
     unknown server-side; then restore everything (synthetic files
     removed, placeholders restored, dispatcher back on the search
     registry, search phase re-proven, validate_stack.py PASS again).
  4. Post: legacy-six identities unchanged; write
     reports/exp007-preflight-probes.json.

This driver itself is stdlib-only; the REAL executor path (3d) and the
host-side registry load check (3a) run as .venv-research subprocesses.
The API key is never written to the report. Legacy containers are only
ever inspected, never modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR
REPO_ROOT = SCRIPT_DIR.parents[1]
AUTH_FILE = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "auth.env"
VENV_PYTHON = REPO_ROOT / ".venv-research" / "bin" / "python"
GATEWAY = "http://127.0.0.1:6581"
REPORT_PATH = REPO_ROOT / "reports" / "exp007-preflight-probes.json"
VALIDATE_STACK = DEPLOY_DIR / "validate_stack.py"

RUNTIME_ROOT = REPO_ROOT / ".runtime" / "comparison-istratde"
LEDGER_PATH = RUNTIME_ROOT / "comparison-ledger.json"
SUPERVISOR_REVIEW_PATH = RUNTIME_ROOT / "supervisor-review.json"
PUB_ROOT = RUNTIME_ROOT / "final-publication"
ANSWERS_DIR = PUB_ROOT / "answers"
REGISTRY_FILENAME = "registry.final.v1.json"
REGISTRY_VERSION = "comparison-final-smoke-v1"
SMOKE_STATE_DIR = RUNTIME_ROOT / "final-smoke-state"

STORAGE_HOST_JOBS = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "storage" / "mlsandbox" / "jobs"
STORAGE_DISPATCHER_JOBS = "/mnt/rsi_storage/mlsandbox/jobs"
CANARY_JOB = "exp007-canary-job"
CANARY_CONTENT = b"exp007-synthetic-canary-original"

TASKS = ("banknote", "wdbc", "spambase")
FINAL_TASK_ID = "comparison_banknote_final"
FINAL_GRANT = "/mnt/rsi_data/comparison/banknote/data/final"
CONTAINER_FINAL_REGISTRY = "/opt/rsi_final/registry.final.v1.json"
SEARCH_TASK_IDS = {
    "hello_synth",
    "comparison_banknote",
    "comparison_spambase",
    "comparison_wdbc",
}

LEGACY_CONTAINERS = [
    "sandbox_api",
    "sandbox_dispatcher",
    "sandbox_nginx",
    "sandbox_postgres",
    "sandbox_redis",
    "ml-sandbox-gpu-0",
]
NEW_STACK_CONTAINERS = [
    "rsi_nginx",
    "rsi_api",
    "rsi_dispatcher",
    "rsi_postgres",
    "rsi_redis",
    "rsi_worker",
]

COMPOSE_TIMEOUT = 180
HEALTH_TIMEOUT = 120

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RESULTS: list[dict] = []
# Mutable probe-run state so the finally-restore knows how far we got.
PHASE_STATE = {"fabricated": False, "switched": False}


def record(name: str, ok: bool, detail: str = "", **extra) -> None:
    RESULTS.append({"name": name, "ok": bool(ok), "detail": detail, **extra})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_api_key() -> str:
    for line in AUTH_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("SANDBOX_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip()
    raise RuntimeError("SANDBOX_API_KEYS not found in private auth file")


def api(method: str, path: str, key: str, payload: dict | None = None):
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
        raise RuntimeError(
            f"{method} {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}"
        ) from exc


def submit(key: str, name: str, code: str, task_id: str, data_dir: str, timeout: int = 300) -> str:
    payload = {
        "name": name,
        "task_id": task_id,
        "code": code,
        "data_dir": data_dir,
        "timeout": timeout,
        "priority": 1,
        "resource_type": "cpu",
        "environment": {"EXECUTION_MODE": "shell"},
    }
    status, created = api("POST", "/api/v1/jobs", key, payload)
    if status not in (200, 201):
        raise RuntimeError(f"submit {name} -> HTTP {status}: {created}")
    return str(created["job_id"])


def wait_terminal(key: str, job_id: str, timeout_seconds: int = 420) -> tuple[str, dict]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        _, payload = api("GET", f"/api/v1/jobs/{job_id}", key)
        status = str(payload.get("status"))
        if status in {"completed", "failed", "cancelled", "timeout"}:
            result = payload.get("result") or {}
            if isinstance(result, str):
                result = json.loads(result)
            return status, result
        if time.monotonic() > deadline:
            raise RuntimeError(f"job {job_id} did not reach a terminal state in time")
        time.sleep(2)


def worker_control(script: str, timeout: int = 90) -> str:
    """Run a TRUSTED control-channel script inside rsi_dispatcher (the only
    other control-key holder). Never used for candidate code."""
    result = subprocess.run(
        ["docker", "exec", "rsi_dispatcher", "python3", "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"worker control script failed: {result.stdout} {result.stderr}")
    return result.stdout


def candidate_uid_processes() -> list[str]:
    """Candidate-uid (65432) process scan INSIDE rsi_worker only."""
    result = subprocess.run(
        ["docker", "exec", "rsi_worker", "python3", "-c", CANDIDATE_SCAN],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout.strip() or "[]")


CANDIDATE_SCAN = """
import json, pathlib
hits = []
for e in pathlib.Path("/proc").iterdir():
    if not e.name.isdigit():
        continue
    try:
        t = (e / "status").read_text(errors="replace")
    except OSError:
        continue
    for l in t.splitlines():
        if l.startswith("Uid:") and int(l.split()[1]) == 65432:
            hits.append(e.name)
            break
print(json.dumps(hits))
"""


def record_cleanup_scan(label: str) -> bool:
    leftover = candidate_uid_processes()
    record(f"cleanup scan empty after {label}", not leftover, f"leftover={leftover}")
    return not leftover


def container_identity(name: str) -> dict:
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.Id}}|{{.Config.Image}}|{{.State.StartedAt}}|{{.State.Status}}", name],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return {"id": "", "image": "", "started_at": "", "status": "missing"}
    parts = proc.stdout.strip().split("|")
    return {
        "id": parts[0][:12],
        "image": parts[1] if len(parts) > 1 else "",
        "started_at": parts[2] if len(parts) > 2 else "",
        "status": parts[3] if len(parts) > 3 else "",
    }


def wait_container_state(name: str, want: str, timeout: int = HEALTH_TIMEOUT) -> bool:
    """want is 'healthy' (healthcheck containers) or 'running'."""
    fmt = "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc = subprocess.run(
            ["docker", "inspect", "-f", fmt, name],
            capture_output=True, text=True, timeout=30,
        )
        if proc.stdout.strip() == want:
            return True
        time.sleep(2)
    return False


def compose_up(*services: str) -> tuple[bool, str]:
    """docker compose up -d with cwd=deploy/rsi-trustworthy and the
    RSI_EVALUATOR_REGISTRY_PATH override explicitly REMOVED from the
    environment (compose default). Phase switches go through the PRODUCTION
    research.comparison.phase module; this helper is only used for the
    step-1 full-stack apply and the last-resort restore fallback."""
    env = dict(os.environ)
    env.pop("RSI_EVALUATOR_REGISTRY_PATH", None)
    result = subprocess.run(
        ["docker", "compose", "up", "-d", *services],
        cwd=DEPLOY_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=COMPOSE_TIMEOUT,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()[-400:]


def dispatcher_loaded_registry() -> tuple[set[str], str]:
    """Ask the dispatcher container to load its OWN pinned registry path and
    report the task ids + version (proves what the running process loaded)."""
    script = (
        "import os;"
        "from research.evaluator.service import EvaluatorRegistry as R;"
        "r = R.load(os.environ['EVALUATOR_REGISTRY_PATH']);"
        "print('LOADED|' + '|'.join(sorted(r.tasks)) + '|' + r.registry_version)"
    )
    result = subprocess.run(
        ["docker", "exec", "rsi_dispatcher", "python3", "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return set(), ""
    line = next((l for l in result.stdout.splitlines() if l.startswith("LOADED|")), "")
    parts = line.split("|")
    return set(parts[1:-1]), (parts[-1] if line else "")


def dispatcher_registry_env() -> str:
    result = subprocess.run(
        ["docker", "exec", "rsi_dispatcher", "printenv", "EVALUATOR_REGISTRY_PATH"],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip()


def run_validate_stack() -> tuple[bool, str]:
    result = subprocess.run(
        [str(VENV_PYTHON), str(VALIDATE_STACK)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode == 0, "\n".join(tail[-3:])


def run_venv_driver(name: str, body: str, timeout: int = 300) -> tuple[int, str, str]:
    """Write a driver file into the smoke state dir and run it with the
    repo .venv-research interpreter (cwd=repo root). Used to invoke the
    PRODUCTION research.comparison.phase functions and the real final
    executor from this stdlib-only probe."""
    SMOKE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    driver_path = SMOKE_STATE_DIR / name
    driver_path.write_text(body, encoding="utf-8")
    run = subprocess.run(
        [str(VENV_PYTHON), str(driver_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return run.returncode, run.stdout, run.stderr


def parse_marker(stdout: str, marker: str) -> dict:
    line = next((l for l in stdout.splitlines() if l.startswith(marker + "|")), "")
    if not line:
        return {}
    try:
        return json.loads(line.split("|", 1)[1])
    except ValueError:
        return {}


def production_current_phase() -> str:
    """The dispatcher's active phase per the PRODUCTION phase module."""
    run = subprocess.run(
        [
            str(VENV_PYTHON), "-c",
            "import sys; sys.path.insert(0, " + repr(str(REPO_ROOT)) + ");"
            "from research.comparison import phase; print(phase.current_phase())",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return run.stdout.strip() if run.returncode == 0 else ""


def dispatcher_log_tail(lines: int = 50) -> str:
    proc = subprocess.run(
        ["docker", "logs", "--tail", str(lines), "rsi_dispatcher"],
        capture_output=True, text=True, timeout=30,
    )
    return proc.stdout + proc.stderr


# --- candidate program (constant bytes; identity recorded in the report) ---

PROG = '''import csv, os

data = os.environ["DATA_DIR"]


def load(name):
    with open(os.path.join(data, name), newline="") as fh:
        return list(csv.DictReader(fh))


train = load("train.csv")
test = load("test.csv")
features = [c for c in train[0] if c not in ("id", "label")]
xs = [[float(row[c]) for c in features] for row in train]
ys = [int(row["label"]) for row in train]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200).fit(xs, ys)
preds = model.predict([[float(row[c]) for c in features] for row in test])

with open("submission.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["id", "label"])
    for row, pred in zip(test, preds):
        writer.writerow([row["id"], int(pred)])
print(f"exp007-prog: trained={len(train)} predicted={len(test)}")
'''

PROG_SHA256 = sha256_bytes(PROG.encode("utf-8"))

_PROBE_PREAMBLE = """import csv, json, os
results = {}
def attempt(name, fn):
    try:
        fn(); results[name] = "REACHED"
    except OSError as exc: results[name] = f"denied:{exc.errno}"
    except Exception as exc: results[name] = f"denied:{type(exc).__name__}"
own = os.environ["DATA_DIR"]
"""

_FALLBACK_SUBMISSION = """
ids = []
with open(os.path.join(own, "test.csv"), newline="") as fh:
    for row in csv.DictReader(fh):
        ids.append(row["id"])
with open("submission.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["id", "label"])
    for value in ids:
        writer.writerow([value, 0])
print("GRANTS|" + json.dumps(results, sort_keys=True))
"""

SEARCH_DENIAL_CODE = _PROBE_PREAMBLE + """attempt("own_task_train", lambda: open(os.path.join(own, "train.csv")).read())
attempt("cross_task_wdbc_public", lambda: open("/mnt/rsi_data/comparison/wdbc/data/public/train.csv").read())
attempt("cross_task_spambase_public", lambda: open("/mnt/rsi_data/comparison/spambase/data/public/train.csv").read())
attempt("comparison_parent_list", lambda: os.listdir("/mnt/rsi_data/comparison"))
attempt("evaluator_tree_list", lambda: os.listdir("/opt/rsi_repo"))
attempt("final_tree_dispatcher_only", lambda: os.listdir("/opt/rsi_final"))
attempt("cross_job_storage_canary", lambda: open("__CANARY__").read())
attempt("storage_tree_list", lambda: os.listdir("/mnt/rsi_storage"))
attempt("own_final_dir_list", lambda: os.listdir("/mnt/rsi_data/comparison/banknote/data/final"))
attempt("own_final_train_read", lambda: open("/mnt/rsi_data/comparison/banknote/data/final/train.csv").read())
attempt("cross_final_wdbc_list", lambda: os.listdir("/mnt/rsi_data/comparison/wdbc/data/final"))
for name, path in (
    ("bundled_wdbc_usr", "/usr/local/lib/python3.10/dist-packages/sklearn/datasets/data/breast_cancer.csv"),
    ("bundled_wdbc_opt", "/opt/python3.11/lib/python3.11/site-packages/sklearn/datasets/data/breast_cancer.csv"),
):
    try:
        content = open(path).read()
        results[name] = "masked_empty" if content == "" else f"REACHED:{len(content)}bytes"
    except OSError as exc:
        results[name] = f"denied:{exc.errno}"
def try_loader():
    from sklearn.datasets import load_breast_cancer
    bunch = load_breast_cancer()
    assert len(bunch.target) == 569, "unexpected loader result"
attempt("wdbc_loader", try_loader)
""" + _FALLBACK_SUBMISSION

SEARCH_DENIAL_REACHED = {"own_task_train"}
SEARCH_DENIAL_MASKED = {"bundled_wdbc_usr", "bundled_wdbc_opt"}
SEARCH_DENIAL_SPECIAL = SEARCH_DENIAL_REACHED | SEARCH_DENIAL_MASKED

FINAL_DENIAL_CODE = _PROBE_PREAMBLE + """attempt("own_final_train", lambda: open(os.path.join(own, "train.csv")).read())
attempt("cross_final_wdbc_train", lambda: open("/mnt/rsi_data/comparison/wdbc/data/final/train.csv").read())
attempt("cross_final_wdbc_list", lambda: os.listdir("/mnt/rsi_data/comparison/wdbc/data/final"))
attempt("cross_final_spambase_list", lambda: os.listdir("/mnt/rsi_data/comparison/spambase/data/final"))
attempt("comparison_parent_list", lambda: os.listdir("/mnt/rsi_data/comparison"))
attempt("search_public_wdbc", lambda: open("/mnt/rsi_data/comparison/wdbc/data/public/train.csv").read())
attempt("search_public_banknote", lambda: open("/mnt/rsi_data/comparison/banknote/data/public/train.csv").read())
attempt("evaluator_tree_list", lambda: os.listdir("/opt/rsi_repo"))
attempt("final_tree_dispatcher_only", lambda: os.listdir("/opt/rsi_final"))
attempt("storage_tree_list", lambda: os.listdir("/mnt/rsi_storage"))
""" + _FALLBACK_SUBMISSION

FINAL_DENIAL_REACHED = {"own_final_train"}


# --- synthetic final dataset (deterministic, tiny, clearly separable) ------

def synthetic_train_csv() -> str:
    lines = ["id,f1,f2,f3,f4,label"]
    for i in range(12):
        lines.append(f"{i},{0.10 * i:.4f},{0.05 * i:.4f},{-0.10 * i:.4f},{0.02 * i:.4f},0")
    for i in range(12):
        lines.append(
            f"{12 + i},{10.0 + 0.10 * i:.4f},{10.0 - 0.05 * i:.4f},"
            f"{10.0 + 0.10 * i:.4f},{10.0 + 0.02 * i:.4f},1"
        )
    return "\n".join(lines) + "\n"


SYNTHETIC_TEST_ROWS = [
    (0, 0.20, 0.10, -0.20, 0.04),
    (1, -0.15, 0.05, 0.10, -0.02),
    (2, 0.30, -0.10, -0.05, 0.06),
    (3, 0.05, 0.20, -0.15, 0.01),
    (4, 10.20, 9.90, 10.15, 10.05),
    (5, 9.80, 10.10, 10.05, 9.95),
    (6, 10.40, 9.80, 10.25, 10.10),
    (7, 10.10, 10.05, 9.90, 10.02),
]
SYNTHETIC_ANSWER_LABELS = [0, 0, 0, 0, 1, 1, 1, 1]


def synthetic_test_csv() -> str:
    lines = ["id,f1,f2,f3,f4"]
    for row_id, f1, f2, f3, f4 in SYNTHETIC_TEST_ROWS:
        lines.append(f"{row_id},{f1:.4f},{f2:.4f},{f3:.4f},{f4:.4f}")
    return "\n".join(lines) + "\n"


def synthetic_sample_submission_csv() -> str:
    lines = ["id,label"] + [f"{row[0]},0" for row in SYNTHETIC_TEST_ROWS]
    return "\n".join(lines) + "\n"


def synthetic_answer_csv() -> str:
    lines = ["id,label"]
    for row, label in zip(SYNTHETIC_TEST_ROWS, SYNTHETIC_ANSWER_LABELS):
        lines.append(f"{row[0]},{label}")
    return "\n".join(lines) + "\n"


def recompute_accuracy_host(prediction_bytes: bytes, answer_bytes: bytes) -> tuple[float, int]:
    """Stdlib accuracy recomputation (mirrors final.recompute_accuracy)."""
    import csv as _csv
    import io as _io

    def _parse(data: bytes) -> dict[str, int]:
        reader = _csv.DictReader(_io.StringIO(data.decode("utf-8")))
        if reader.fieldnames is None or list(reader.fieldnames) != ["id", "label"]:
            raise RuntimeError("csv header mismatch")
        rows: dict[str, int] = {}
        for row in reader:
            rows[row["id"]] = int(row["label"])
        return rows

    predictions = _parse(prediction_bytes)
    answers = _parse(answer_bytes)
    if not answers or set(predictions) != set(answers):
        raise RuntimeError("prediction/answer id sets differ")
    correct = sum(1 for row_id, label in answers.items() if predictions[row_id] == label)
    return correct / len(answers), len(answers)


PLACEHOLDER_README = (
    "# EXP-006 final publication placeholder\n\n"
    "Final bytes are published ONLY by a verified all-cell barrier\n"
    "(research.comparison.final). Until then this directory intentionally\n"
    "contains no train/test data.\n"
)


# --- probe phases ------------------------------------------------------------

def preconditions(key: str) -> bool:
    ok = True
    ledger_free = not LEDGER_PATH.exists() and not SUPERVISOR_REVIEW_PATH.exists()
    record(
        "precondition: no live comparison ledger/supervisor review under .runtime/comparison-istratde",
        ledger_free,
        f"ledger_exists={LEDGER_PATH.exists()} review_exists={SUPERVISOR_REVIEW_PATH.exists()}",
    )
    ok = ok and ledger_free

    tooling = AUTH_FILE.exists() and VENV_PYTHON.exists() and VALIDATE_STACK.exists()
    record(
        "precondition: auth.env, .venv-research interpreter and validate_stack.py present",
        tooling,
        f"auth={AUTH_FILE.exists()} venv={VENV_PYTHON.exists()} validate={VALIDATE_STACK.exists()}",
    )
    ok = ok and tooling

    legacy_state = {name: container_identity(name) for name in LEGACY_CONTAINERS}
    legacy_ok = all(s["status"] == "running" for s in legacy_state.values())
    record(
        "precondition: all six legacy containers running (identity recorded, inspect-only)",
        legacy_ok,
        f"legacy={legacy_state}",
        legacy=legacy_state,
    )
    ok = ok and legacy_ok

    new_state = {name: container_identity(name) for name in NEW_STACK_CONTAINERS}
    new_ok = all(s["status"] == "running" for s in new_state.values())
    record(
        "precondition: all six new-stack containers running (identity recorded)",
        new_ok,
        f"new_stack={new_state}",
        new_stack=new_state,
    )
    ok = ok and new_ok

    try:
        _, queued = api("GET", "/api/v1/jobs?status=queued&limit=50", key)
        _, running = api("GET", "/api/v1/jobs?status=running&limit=50", key)
        active = int(queued.get("count", -1)) + int(running.get("count", -1))
    except Exception as exc:
        active = -1
        record("precondition: gateway answers authenticated job listing", False, str(exc))
        ok = False
    if active >= 0:
        record(
            "precondition: new stack idle (no queued/running jobs)",
            active == 0,
            f"active_jobs={active}",
        )
        ok = ok and active == 0

    leftover = candidate_uid_processes()
    record(
        "precondition: no candidate-uid processes in rsi_worker",
        not leftover,
        f"leftover={leftover}",
    )
    ok = ok and not leftover

    baseline_tasks, baseline_version = dispatcher_loaded_registry()
    baseline_ok = baseline_tasks == SEARCH_TASK_IDS and baseline_version == "evaluator-registry.v1"
    record(
        "precondition: dispatcher baseline registry is the tracked SEARCH registry",
        baseline_ok,
        f"tasks={sorted(baseline_tasks)} version={baseline_version}",
    )
    ok = ok and baseline_ok

    phase_now = production_current_phase()
    record(
        "precondition: PRODUCTION phase module reports current dispatcher phase = search",
        phase_now == "search",
        f"current_phase={phase_now!r}",
        current_phase=phase_now or None,
    )
    ok = ok and phase_now == "search"

    record("probe program identity (PROG, constant bytes)", True, f"sha256={PROG_SHA256}",
           prog_sha256=PROG_SHA256)
    return ok


def step1_activate_final_mounts() -> None:
    placeholders_ok = True
    details = {}
    for task in TASKS:
        final_dir = PUB_ROOT / task / "data" / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        extras = sorted(p.name for p in final_dir.iterdir() if p.name != "README.md")
        readme = final_dir / "README.md"
        if not readme.exists():
            readme.write_text(PLACEHOLDER_README, encoding="utf-8")
        details[task] = {"extras": extras, "readme": readme.exists()}
        placeholders_ok = placeholders_ok and not extras and readme.exists()
    record(
        "final-publication host dirs pre-created as current user, ONLY placeholder README.md each",
        placeholders_ok,
        json.dumps(details, sort_keys=True),
    )

    before = {name: container_identity(name) for name in ("rsi_worker", "rsi_dispatcher")}
    compose_ok, compose_out = compose_up()
    worker_healthy = compose_ok and wait_container_state("rsi_worker", "healthy")
    dispatcher_running = compose_ok and wait_container_state("rsi_dispatcher", "running")
    after = {name: container_identity(name) for name in ("rsi_worker", "rsi_dispatcher")}
    record(
        "compose up -d applied; rsi_worker healthy and rsi_dispatcher running",
        compose_ok and worker_healthy and dispatcher_running,
        f"compose_ok={compose_ok} worker={after['rsi_worker']['status']} "
        f"dispatcher={after['rsi_dispatcher']['status']} out={compose_out[-200:]}",
        containers_before=before,
        containers_after=after,
        worker_recreated=before["rsi_worker"]["id"] != after["rsi_worker"]["id"],
        dispatcher_recreated=before["rsi_dispatcher"]["id"] != after["rsi_dispatcher"]["id"],
    )

    validate_ok, validate_tail = run_validate_stack()
    record(
        "validate_stack.py PASS after final-mount activation",
        validate_ok,
        validate_tail,
    )

    step1_visibility_probes()


SENTINEL_NAME = ".visibility-probe"
SENTINEL_CONTENT = "exp007-visibility-sentinel\n"


def step1_visibility_probes() -> None:
    """Host-to-container visibility of the mounted final leaf dirs: a
    sentinel written into each HOST placeholder leaf must be readable at
    the worker's read-only mount with exact bytes, then removed."""
    for task in TASKS:
        (PUB_ROOT / task / "data" / "final" / SENTINEL_NAME).write_text(
            SENTINEL_CONTENT, encoding="utf-8"
        )
    visible = {}
    for task in TASKS:
        container_path = f"/mnt/rsi_data/comparison/{task}/data/final/{SENTINEL_NAME}"
        proc = subprocess.run(
            ["docker", "exec", "rsi_worker", "python3", "-c",
             f"import sys; sys.exit(0 if open({container_path!r}).read() == {SENTINEL_CONTENT!r} else 1)"],
            capture_output=True, text=True, timeout=30,
        )
        visible[task] = proc.returncode == 0
    record(
        "final leaf mounts visible inside rsi_worker (sentinel readable, exact bytes, all 3 tasks)",
        all(visible.values()),
        json.dumps(visible, sort_keys=True),
        visible=visible,
    )
    removed = {}
    for task in TASKS:
        host_path = PUB_ROOT / task / "data" / "final" / SENTINEL_NAME
        host_path.unlink(missing_ok=True)
        container_path = f"/mnt/rsi_data/comparison/{task}/data/final/{SENTINEL_NAME}"
        proc = subprocess.run(
            ["docker", "exec", "rsi_worker", "python3", "-c",
             f"import os, sys; sys.exit(1 if os.path.exists({container_path!r}) else 0)"],
            capture_output=True, text=True, timeout=30,
        )
        removed[task] = (not host_path.exists()) and proc.returncode == 0
    record(
        "visibility sentinels removed (host gone, worker mount no longer shows them)",
        all(removed.values()),
        json.dumps(removed, sort_keys=True),
        removed=removed,
    )


def step2_search_phase(key: str) -> None:
    # 2b. PROG scores on each comparison task via the trusted pinned grant.
    for task in TASKS:
        job_id = submit(
            key,
            f"exp007-search-{task}",
            PROG,
            f"comparison_{task}",
            f"/mnt/rsi_data/comparison/{task}",
        )
        status, result = wait_terminal(key, job_id)
        score = result.get("score")
        evaluation = result.get("evaluation") or {}
        finite = isinstance(score, (int, float)) and not isinstance(score, bool) and 0.0 <= float(score) <= 1.0
        record(
            f"search phase: comparison_{task} PROG job completed with finite score in [0,1]",
            status == "completed"
            and finite
            and evaluation.get("task_id") == f"comparison_{task}"
            and evaluation.get("split") == "validation"
            and result.get("worker_cleanup_verified") is True,
            f"job={job_id} status={status} score={score}",
            job_id=job_id, score=score,
        )
        record_cleanup_scan(f"search job {job_id} (comparison_{task})")

    # 2c. Denial battery from a REAL candidate session on comparison_banknote.
    canary_dispatcher = f"{STORAGE_DISPATCHER_JOBS}/{CANARY_JOB}/code/canary.txt"
    canary_worker = f"/mnt/rsi_storage/mlsandbox/jobs/{CANARY_JOB}/code/canary.txt"
    worker_control(
        "from pathlib import Path\n"
        f"d = Path({str(Path(STORAGE_DISPATCHER_JOBS) / CANARY_JOB / 'code')!r})\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        f"(d / 'canary.txt').write_bytes({CANARY_CONTENT!r})\n"
        "print('CANARY_PLANTED')\n"
    )
    canary_host = STORAGE_HOST_JOBS / CANARY_JOB / "code" / "canary.txt"
    planted = canary_host.exists() and canary_host.read_bytes() == CANARY_CONTENT
    record("cross-job storage canary planted via dispatcher control channel", planted,
           f"canary={canary_dispatcher}")
    try:
        probe_code = SEARCH_DENIAL_CODE.replace("__CANARY__", canary_worker)
        job_id = submit(key, "exp007-search-denial", probe_code,
                        "comparison_banknote", "/mnt/rsi_data/comparison/banknote")
        status, result = wait_terminal(key, job_id)
        run_log = result.get("run_log") or ""
        marker_line = next((l for l in run_log.splitlines() if l.startswith("GRANTS|")), "")
        probe_results = json.loads(marker_line.split("|", 1)[1]) if marker_line else {}
        denied = {k: v for k, v in probe_results.items() if k not in SEARCH_DENIAL_SPECIAL}
        all_denied = bool(denied) and all(str(v).startswith("denied:") for v in denied.values())
        masked_ok = all(probe_results.get(name) == "masked_empty" for name in SEARCH_DENIAL_MASKED)
        record(
            "search denial battery: own grant readable; cross-task/parent/evaluator/final/storage denied",
            status == "completed"
            and probe_results.get("own_task_train") == "REACHED"
            and all_denied
            and len(denied) == 11,  # 10 boundary probes + blocked WDBC loader
            f"probes={json.dumps(probe_results, sort_keys=True)}",
            job_id=job_id, probe_results=probe_results,
        )
        record(
            "bundled labeled WDBC masked (both copies empty) and sklearn loader blocked",
            status == "completed"
            and masked_ok
            and str(probe_results.get("wdbc_loader", "")).startswith("denied:"),
            f"usr={probe_results.get('bundled_wdbc_usr')} opt={probe_results.get('bundled_wdbc_opt')} "
            f"loader={probe_results.get('wdbc_loader')}",
            job_id=job_id,
        )
        record(
            "final dirs hold no final bytes for candidates (final train.csv open denied; placeholders only)",
            str(probe_results.get("own_final_train_read", "")).startswith("denied:")
            and str(probe_results.get("own_final_dir_list", "")).startswith("denied:"),
            f"final_train={probe_results.get('own_final_train_read')} "
            f"final_list={probe_results.get('own_final_dir_list')}",
            job_id=job_id,
        )
        record_cleanup_scan(f"search denial job {job_id}")
    finally:
        out = worker_control(
            "from pathlib import Path\n"
            "import shutil\n"
            f"f = Path({canary_dispatcher!r})\n"
            f"ok = f.exists() and f.read_bytes() == {CANARY_CONTENT!r}\n"
            f"shutil.rmtree(str(f.parents[1]), ignore_errors=True)\n"
            "print('CANARY_INTACT' if ok else 'CANARY_TAMPERED')\n"
        )
        intact = "CANARY_INTACT" in out
        removed = not (STORAGE_HOST_JOBS / CANARY_JOB).exists()
        record(
            "cross-job canary bytes intact after candidate probes and removed",
            intact and removed,
            f"intact={intact} removed={removed}",
        )


def step3a_fabricate() -> bool:
    final_dir = PUB_ROOT / "banknote" / "data" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "train.csv").write_text(synthetic_train_csv(), encoding="utf-8")
    (final_dir / "test.csv").write_text(synthetic_test_csv(), encoding="utf-8")
    (final_dir / "sample_submission.csv").write_text(synthetic_sample_submission_csv(), encoding="utf-8")
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    answer_bytes = synthetic_answer_csv().encode("utf-8")
    (ANSWERS_DIR / "banknote_final_answer.csv").write_bytes(answer_bytes)
    answer_sha = sha256_bytes(answer_bytes)
    registry = {
        "registry_version": REGISTRY_VERSION,
        "schema_version": 1,
        "tasks": {
            FINAL_TASK_ID: {
                "metric": "accuracy",
                "split": "final",
                "answer_file": "answers/banknote_final_answer.csv",
                "answer_sha256": answer_sha,
                "id_column": "id",
                "target_column": "label",
                "label_values": [0, 1],
                "max_prediction_bytes": 1048576,
                "max_rows": 100000,
                "worker_data_public_path": FINAL_GRANT,
            }
        },
    }
    registry_path = PUB_ROOT / REGISTRY_FILENAME
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    PHASE_STATE["fabricated"] = True

    # Host-side load check with the REAL EvaluatorRegistry before any switch.
    check = subprocess.run(
        [
            str(VENV_PYTHON), "-c",
            "import sys; sys.path.insert(0, " + repr(str(REPO_ROOT)) + ");"
            "from research.evaluator.service import EvaluatorRegistry as R;"
            "r = R.load(" + repr(str(registry_path)) + ");"
            "spec = r.tasks[" + repr(FINAL_TASK_ID) + "];"
            "print('HOSTLOAD|' + r.registry_version + '|' + spec.split + '|' + spec.metric"
            " + '|' + str(spec.answer_sha256))",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    line = next((l for l in check.stdout.splitlines() if l.startswith("HOSTLOAD|")), "")
    parts = line.split("|")
    load_ok = (
        check.returncode == 0
        and len(parts) == 5
        and parts[1] == REGISTRY_VERSION
        and parts[2] == "final"
        and parts[3] == "accuracy"
        and parts[4] == answer_sha
    )
    record(
        "synthetic final dataset + phase registry fabricated; host-side EvaluatorRegistry.load OK",
        load_ok,
        f"check={line or (check.stdout + check.stderr).strip()[-200:]}",
        answer_sha256=answer_sha,
        registry_sha256=sha256_bytes(registry_path.read_bytes()),
    )
    return load_ok


def step3b_final_refused_under_search_registry(key: str) -> None:
    job_id = submit(key, "exp007-final-refused-pre-switch", PROG, FINAL_TASK_ID, FINAL_GRANT)
    status, result = wait_terminal(key, job_id)
    refused = (
        status == "failed"
        and result.get("result") == "data_grant_refused"
        and result.get("score") is None
    )
    record(
        "comparison_banknote_final REFUSED while search registry active (unknown task, never executed)",
        refused,
        f"job={job_id} status={status} result={result.get('result')}",
        job_id=job_id,
    )
    record_cleanup_scan(f"refused pre-switch final job {job_id}")


def _activate_final_driver(registry_sha256: str) -> str:
    return (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        f"REPO = Path({str(REPO_ROOT)!r})\n"
        "sys.path.insert(0, str(REPO))\n"
        "from research.comparison import phase\n"
        "failures = []\n"
        "result = None\n"
        "error = None\n"
        "try:\n"
        "    result = phase.activate_final_phase(\n"
        "        REPO,\n"
        f"        expected_registry_sha256={registry_sha256!r},\n"
        "        deadline_epoch=time.time() + 3600.0,\n"
        "        on_failure=failures.append)\n"
        "except Exception as exc:\n"
        "    error = '{}: {}'.format(type(exc).__name__, exc)\n"
        "try:\n"
        "    phase_after = phase.current_phase()\n"
        "except Exception as exc:\n"
        "    phase_after = 'unknown:{}'.format(exc)\n"
        "print('PHASE|' + json.dumps({'result': result, 'failures': failures,\n"
        "    'error': error, 'phase_after': phase_after}, sort_keys=True))\n"
    )


def step3c_switch_phase() -> bool:
    registry_sha = sha256_bytes((PUB_ROOT / REGISTRY_FILENAME).read_bytes())
    before = container_identity("rsi_dispatcher")
    # PRODUCTION mechanism (research.comparison.phase.activate_final_phase):
    # the exact function the live experiment runs, invoked in-process by a
    # .venv-research driver; this stdlib probe only parses its marker.
    rc, stdout, stderr = run_venv_driver(
        "driver_activate_final.py", _activate_final_driver(registry_sha), timeout=300
    )
    outcome = parse_marker(stdout, "PHASE")
    result = outcome.get("result")
    failures = outcome.get("failures") or []
    phase_after = outcome.get("phase_after")
    # From here on the dispatcher phase is this call's responsibility:
    # restore must run even when the activation reports failure.
    PHASE_STATE["switched"] = True
    switch_ok = (
        result in {"activated_final", "already_final"}
        and not failures
        and phase_after == "final"
    )
    after = container_identity("rsi_dispatcher")
    record(
        "phase switch via PRODUCTION phase.activate_final_phase (hash-bound, deadline-bound)",
        switch_ok,
        f"result={result} failures={failures} phase_after={phase_after} "
        f"error={outcome.get('error')} rc={rc}",
        outcome=outcome,
        expected_registry_sha256=registry_sha,
        dispatcher_before=before,
        dispatcher_after=after,
        dispatcher_recreated=before["id"] != after["id"],
        driver_stderr=stderr.strip()[-300:] if rc != 0 else "",
    )
    # Extra independent evidence: container env + the registry the running
    # dispatcher actually loads + a clean log tail.
    env_value = dispatcher_registry_env()
    tasks, version = dispatcher_loaded_registry()
    log_tail = dispatcher_log_tail()
    log_clean = "Traceback" not in log_tail and "RuntimeError" not in log_tail
    evidence_ok = (
        switch_ok
        and env_value == CONTAINER_FINAL_REGISTRY
        and tasks == {FINAL_TASK_ID}
        and version == REGISTRY_VERSION
        and log_clean
    )
    record(
        "phase switch evidence: dispatcher env pinned, phase registry loaded, log tail clean",
        evidence_ok,
        f"env={env_value} tasks={sorted(tasks)} version={version} log_clean={log_clean}",
        env_value=env_value, loaded_tasks=sorted(tasks), registry_version=version,
    )
    return switch_ok


def step3d_final_smoke() -> None:
    registry_path = PUB_ROOT / REGISTRY_FILENAME
    publication = {
        "schema": "istratde-comparison-final-publication.v1",
        "registry_version": REGISTRY_VERSION,
        "publication_root": str(PUB_ROOT),
        "registry_file": REGISTRY_FILENAME,
        "files": {REGISTRY_FILENAME: sha256_bytes(registry_path.read_bytes())},
    }
    SMOKE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    driver_path = SMOKE_STATE_DIR / "driver.py"
    driver_path.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        f"REPO = Path({str(REPO_ROOT)!r})\n"
        "sys.path.insert(0, str(REPO))\n"
        "from research.comparison.final import sandbox_final_executor\n"
        f"PROG = {PROG!r}\n"
        f"PUBLICATION = {publication!r}\n"
        f"STATE = Path({str(SMOKE_STATE_DIR)!r})\n"
        # Fixed near-future epoch captured ONCE (never None): exercises the
        # original-deadline binding of the real executor.
        "DEADLINE = time.time() + 3600.0\n"
        "executor = sandbox_final_executor(\n"
        "    REPO, PUBLICATION, state_dir=STATE, timeout_seconds=300,\n"
        "    deadline_fn=lambda: DEADLINE)\n"
        "res = executor.execute(\n"
        f"    task_id={FINAL_TASK_ID!r}, source_bytes=PROG.encode('utf-8'),\n"
        "    cell_key='banknote:FIXED:11', timeout_seconds=300)\n"
        "out = {\n"
        "    'status': res.status, 'job_id': res.job_id, 'score': res.score,\n"
        "    'prediction_sha256': res.prediction_sha256,\n"
        "    'cleanup_verified': res.cleanup_verified, 'reason': res.reason,\n"
        "    'evidence': dict(res.evidence or {}),\n"
        "}\n"
        # Host snapshot lookup via the SAME shared validated translation the
        # production consumers use (actual dated storage layout included).
        "from research.comparison import deploy as _deploy\n"
        "_snap = (out['evidence'] or {}).get('prediction_snapshot')\n"
        "_host = _deploy.job_snapshot_host_path(\n"
        "    _snap, out['job_id'],\n"
        "    storage_root=REPO / '.runtime' / 'rsi-trustworthy' / 'storage')\n"
        "out['host_snapshot'] = str(_host)\n"
        "out['snapshot_relative'] = _deploy.validate_job_snapshot(_snap, out['job_id'])\n"
        "print('FINALEXEC|' + json.dumps(out, sort_keys=True))\n",
        encoding="utf-8",
    )
    run = subprocess.run(
        [str(VENV_PYTHON), str(driver_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=480,
    )
    line = next((l for l in run.stdout.splitlines() if l.startswith("FINALEXEC|")), "")
    outcome = json.loads(line.split("|", 1)[1]) if line else {}
    job_id = outcome.get("job_id")
    score = outcome.get("score")
    evidence = outcome.get("evidence") or {}
    completed = (
        run.returncode == 0
        and outcome.get("status") == "completed"
        and isinstance(score, (int, float))
        and not isinstance(score, bool)
        and 0.0 <= float(score) <= 1.0
        and outcome.get("cleanup_verified") is True
        and isinstance(job_id, str)
        and bool(job_id)
    )
    recompute_ok = False
    recomputed = None
    rows = None
    prediction_sha_host = None
    if completed:
        host_snapshot = outcome.get("host_snapshot")
        try:
            if not isinstance(host_snapshot, str) or not host_snapshot:
                raise RuntimeError("no validated host snapshot path in executor outcome")
            prediction_bytes = Path(host_snapshot).read_bytes()
            prediction_sha_host = sha256_bytes(prediction_bytes)
            recomputed, rows = recompute_accuracy_host(
                prediction_bytes, synthetic_answer_csv().encode("utf-8")
            )
            recompute_ok = (
                abs(recomputed - float(score)) <= 1e-9
                and prediction_sha_host == outcome.get("prediction_sha256")
            )
        except (OSError, RuntimeError) as exc:
            record("host-side prediction snapshot readable", False, str(exc), job_id=job_id)
    evidence_ok = (
        completed
        and evidence.get("task_id") == FINAL_TASK_ID
        and evidence.get("split") == "final"
        and evidence.get("metric") == "accuracy"
        and evidence.get("direction") == "maximize"
        and evidence.get("evaluator") == "external_trusted"
        and evidence.get("registry_version") == REGISTRY_VERSION
        and evidence.get("answer_sha256") == sha256_bytes(synthetic_answer_csv().encode("utf-8"))
        and evidence.get("source_identity_verified") is True
        and evidence.get("worker_cleanup_verified") is True
        and str(evidence.get("prediction_snapshot", "")).endswith(
            f"/{job_id}/prediction_snapshot.csv"
        )
        and str(outcome.get("snapshot_relative", "")).endswith(
            f"/{job_id}/prediction_snapshot.csv"
        )
    )
    record(
        "REAL sandbox_final_executor: comparison_banknote_final completed with finite score",
        completed,
        f"job={job_id} score={score} rc={run.returncode} "
        f"stderr={(run.stderr.strip()[-200:] if run.returncode else '')}",
        job_id=job_id, score=score, evidence=evidence,
        source_sha256=PROG_SHA256,
        prediction_sha256=outcome.get("prediction_sha256"),
        answer_sha256=sha256_bytes(synthetic_answer_csv().encode("utf-8")),
    )
    record(
        "executor evidence binds final task/split/metric/answer/registry identity",
        evidence_ok,
        f"evidence_task={evidence.get('task_id')} split={evidence.get('split')} "
        f"registry={evidence.get('registry_version')}",
        job_id=job_id,
    )
    record(
        "host-side independent accuracy recomputation matches executor score exactly",
        recompute_ok,
        f"recomputed={recomputed} rows={rows} score={score} "
        f"snapshot_sha_match={prediction_sha_host == outcome.get('prediction_sha256')}",
        job_id=job_id, recomputed_accuracy=recomputed, row_count=rows,
    )
    if job_id:
        record_cleanup_scan(f"real-executor final job {job_id}")


def step3e_final_isolation(key: str) -> None:
    job_id = submit(key, "exp007-final-denial", FINAL_DENIAL_CODE, FINAL_TASK_ID, FINAL_GRANT)
    status, result = wait_terminal(key, job_id)
    run_log = result.get("run_log") or ""
    marker_line = next((l for l in run_log.splitlines() if l.startswith("GRANTS|")), "")
    probe_results = json.loads(marker_line.split("|", 1)[1]) if marker_line else {}
    denied = {k: v for k, v in probe_results.items() if k not in FINAL_DENIAL_REACHED}
    all_denied = bool(denied) and all(str(v).startswith("denied:") for v in denied.values())
    record(
        "final-phase grant isolation: own data/final leaf readable; other tasks' final/public trees denied",
        status == "completed"
        and probe_results.get("own_final_train") == "REACHED"
        and all_denied
        and len(denied) == 9,
        f"probes={json.dumps(probe_results, sort_keys=True)}",
        job_id=job_id, probe_results=probe_results,
    )
    record_cleanup_scan(f"final denial job {job_id}")

    job_id = submit(key, "exp007-search-unknown-under-final", PROG,
                    "comparison_banknote", "/mnt/rsi_data/comparison/banknote")
    status, result = wait_terminal(key, job_id)
    record(
        "search task comparison_banknote UNKNOWN to dispatcher while final registry active",
        status == "failed"
        and result.get("result") == "data_grant_refused"
        and result.get("score") is None,
        f"job={job_id} status={status} result={result.get('result')}",
        job_id=job_id,
    )
    record_cleanup_scan(f"unknown-search-task job {job_id}")


def _restore_search_driver() -> str:
    return (
        "import json, sys\n"
        "from pathlib import Path\n"
        f"REPO = Path({str(REPO_ROOT)!r})\n"
        "sys.path.insert(0, str(REPO))\n"
        "from research.comparison import phase\n"
        "failures = []\n"
        "result = None\n"
        "error = None\n"
        "try:\n"
        "    result = phase.restore_search_phase(REPO, on_failure=failures.append)\n"
        "except Exception as exc:\n"
        "    error = '{}: {}'.format(type(exc).__name__, exc)\n"
        "try:\n"
        "    phase_after = phase.current_phase()\n"
        "except Exception as exc:\n"
        "    phase_after = 'unknown:{}'.format(exc)\n"
        "print('PHASE|' + json.dumps({'result': result, 'failures': failures,\n"
        "    'error': error, 'phase_after': phase_after}, sort_keys=True))\n"
    )


def step3f_restore(key: str) -> None:
    # 1. PRODUCTION restore mechanism FIRST (the smoke state dir still
    #    exists here and holds the driver file; file cleanup runs after the
    #    dispatcher no longer points at the phase registry/answers).
    rc, stdout, stderr = run_venv_driver(
        "driver_restore_search.py", _restore_search_driver(), timeout=300
    )
    outcome = parse_marker(stdout, "PHASE")
    result = outcome.get("result")
    failures = outcome.get("failures") or []
    phase_after = outcome.get("phase_after")
    prod_ok = (
        result in {"activated_search", "already_search"}
        and not failures
        and phase_after == "search"
    )
    record(
        "restore via PRODUCTION phase.restore_search_phase",
        prod_ok,
        f"result={result} failures={failures} phase_after={phase_after} "
        f"error={outcome.get('error')} rc={rc}",
        outcome=outcome,
        driver_stderr=stderr.strip()[-300:] if rc != 0 else "",
    )
    restored = prod_ok
    if not prod_ok:
        # The stack must never be silently left in the final phase: fall
        # back to a raw-compose dispatcher recreate (default registry) and
        # record BOTH outcomes loudly.
        compose_ok, compose_out = compose_up("dispatcher")
        running = compose_ok and wait_container_state("rsi_dispatcher", "running")
        time.sleep(3)
        env_value = dispatcher_registry_env() if running else ""
        tasks, version = dispatcher_loaded_registry() if running else (set(), "")
        fallback_ok = (
            running
            and tasks == SEARCH_TASK_IDS
            and version == "evaluator-registry.v1"
            and env_value != CONTAINER_FINAL_REGISTRY
        )
        record(
            "LAST-RESORT raw-compose fallback restore (production restore failed)",
            fallback_ok,
            f"env={env_value} tasks={sorted(tasks)} version={version} out={compose_out[-200:]}",
            production_restore_ok=prod_ok,
            fallback_restore_ok=fallback_ok,
        )
        restored = fallback_ok

    # 2. Synthetic data cleanup + placeholder restoration.
    final_dir = PUB_ROOT / "banknote" / "data" / "final"
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        (final_dir / name).unlink(missing_ok=True)
    shutil.rmtree(ANSWERS_DIR, ignore_errors=True)
    (PUB_ROOT / REGISTRY_FILENAME).unlink(missing_ok=True)
    shutil.rmtree(SMOKE_STATE_DIR, ignore_errors=True)
    placeholders_ok = True
    details = {}
    for task in TASKS:
        task_final = PUB_ROOT / task / "data" / "final"
        task_final.mkdir(parents=True, exist_ok=True)
        readme = task_final / "README.md"
        if not readme.exists():
            readme.write_text(PLACEHOLDER_README, encoding="utf-8")
        contents = sorted(p.name for p in task_final.iterdir())
        details[task] = contents
        placeholders_ok = placeholders_ok and contents == ["README.md"]
    smoke_state_removed = not SMOKE_STATE_DIR.exists()
    PHASE_STATE["fabricated"] = False
    PHASE_STATE["switched"] = not restored
    record(
        "synthetic final CSVs, answers, phase registry and smoke state removed; placeholders restored",
        placeholders_ok and smoke_state_removed,
        json.dumps({**details, "smoke_state_removed": smoke_state_removed}, sort_keys=True),
    )

    # 3. Independent confirmation that the dispatcher is back on the
    #    tracked search registry.
    if restored:
        env_value = dispatcher_registry_env()
        tasks, version = dispatcher_loaded_registry()
        restored_registry = (
            tasks == SEARCH_TASK_IDS
            and version == "evaluator-registry.v1"
            and env_value != CONTAINER_FINAL_REGISTRY
        )
        record(
            "dispatcher confirmed back on the tracked SEARCH registry (env + loaded tasks)",
            restored_registry,
            f"env={env_value} tasks={sorted(tasks)} version={version}",
        )
        restored = restored_registry

    # 4. Search-phase re-proof with a real scored job.
    if restored:
        job_id = submit(key, "exp007-search-restored", PROG,
                        "comparison_banknote", "/mnt/rsi_data/comparison/banknote")
        status, result = wait_terminal(key, job_id)
        score = result.get("score")
        finite = isinstance(score, (int, float)) and not isinstance(score, bool) and 0.0 <= float(score) <= 1.0
        record(
            "search phase restored: comparison_banknote PROG job completed with score",
            status == "completed" and finite,
            f"job={job_id} status={status} score={score}",
            job_id=job_id, score=score,
        )
        record_cleanup_scan(f"restored search job {job_id}")
    else:
        record("search phase restored: comparison_banknote PROG job completed with score",
               False, "dispatcher did not return to the search registry")

    leftover = candidate_uid_processes()
    record("final candidate-uid process scan empty in rsi_worker", not leftover,
           f"leftover={leftover}")

    validate_ok, validate_tail = run_validate_stack()
    record("validate_stack.py PASS after restore", validate_ok, validate_tail)


def step4_post_legacy(legacy_before: dict) -> None:
    after = {name: container_identity(name) for name in LEGACY_CONTAINERS}
    comparisons = {}
    all_ok = True
    for name in LEGACY_CONTAINERS:
        same = (
            after[name]["id"] == legacy_before[name]["id"]
            and after[name]["image"] == legacy_before[name]["image"]
            and after[name]["started_at"] == legacy_before[name]["started_at"]
        )
        comparisons[name] = same
        all_ok = all_ok and same
    record(
        "legacy-six identities unchanged (id, image, StartedAt)",
        all_ok,
        json.dumps(comparisons, sort_keys=True),
        legacy_after=after,
        identity_unchanged=comparisons,
    )


def finish() -> int:
    ok = all(item["ok"] for item in RESULTS)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "story": "EXP-007",
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
    print("EXP007 PROBES " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    key = load_api_key()
    legacy_before = {name: container_identity(name) for name in LEGACY_CONTAINERS}
    try:
        if preconditions(key):
            step1_activate_final_mounts()
            step2_search_phase(key)
            if step3a_fabricate():
                step3b_final_refused_under_search_registry(key)
                if step3c_switch_phase():
                    step3d_final_smoke()
                    step3e_final_isolation(key)
                else:
                    record("final-phase smoke skipped: phase switch failed", False)
            else:
                record("final-phase smoke skipped: fabrication/registry load failed", False)
    except Exception as exc:
        record("probe driver aborted with exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        if PHASE_STATE["fabricated"] or PHASE_STATE["switched"]:
            try:
                step3f_restore(key)
            except Exception as exc:
                record("restore phase aborted with exception", False, f"{type(exc).__name__}: {exc}")
    step4_post_legacy(legacy_before)
    return finish()


if __name__ == "__main__":
    sys.exit(main())

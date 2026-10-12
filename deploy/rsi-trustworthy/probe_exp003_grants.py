#!/usr/bin/env python3
"""EXP-003 real narrowed-grant probes against the live rsi-trustworthy stack.

Proves, with synthetic canaries and REAL candidate sessions inside the real
rsi_worker container (no string checks alone, no model calls):

  1. hello_synth benign job still executes and scores (compatibility).
  2. comparison_banknote benign job reads its own task public data via the
     trusted pinned grant and is scored by the external evaluator.
  3. A candidate confined by the narrowed trampoline can read ONLY its own
     task grant: cross-task data, cross-job storage source/prediction
     canaries, the shared storage tree, sealed/final paths, symlink aliases
     and traversal aliases are all actually denied (errno evidence).
  4. Worker-side grant validation refuses broad roots, parents, foreign
     roots, traversal, multiple distinct task grants and staging grants
     with machine-readable 400 reasons over the authenticated control
     channel.
  5. Post-cleanup state: no candidate-uid processes remain in the
     dedicated worker container.

Old stack is only checked for presence/health, never modified. No real
secrets are read; no model calls; historic ledgers untouched.
"""
from __future__ import annotations

import json
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
REPORT_PATH = REPO_ROOT / "reports" / "exp003-narrowed-grants.json"
SCRATCH_HOST = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "workdir"
SCRATCH_WORKER = "/mnt/local_sandbox_workdir"
STORAGE_HOST_JOBS = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "storage" / "mlsandbox" / "jobs"
STORAGE_WORKER_JOBS = "/mnt/rsi_storage/mlsandbox/jobs"
CANARY_JOB = "exp003-canary-job"
CANARY_CONTENT = b"exp003-synthetic-canary-original"
LEGACY_CONTAINERS = [
    "sandbox_api",
    "sandbox_dispatcher",
    "sandbox_nginx",
    "sandbox_postgres",
    "sandbox_redis",
    "ml-sandbox-gpu-0",
]

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
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}")


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
        raise SystemExit(f"submit {name} -> HTTP {status}: {created}")
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
            raise SystemExit(f"job {job_id} did not reach a terminal state in time")
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
        raise SystemExit(f"worker control script failed: {result.stdout} {result.stderr}")
    return result.stdout


WORKER_CLIENT = r"""
import json, time, urllib.request, urllib.error
key = open("/run/rsi_control/key", encoding="utf-8").read().strip()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE = "http://worker:8080"
def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-RSI-Control-Key": key}, method="POST")
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
def delete(sid):
    req = urllib.request.Request(BASE + "/v1/shell/sessions/" + sid,
        headers={"X-RSI-Control-Key": key}, method="DELETE")
    try:
        with opener.open(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
def run_session(payload, wait_seconds=30):
    s, p = post("/v1/shell/exec", payload)
    if s != 200:
        return s, p, None
    sid = p["data"]["session_id"]
    deadline = time.monotonic() + wait_seconds
    while True:
        s2, d = post("/v1/shell/wait", {"id": sid, "seconds": 1})
        assert s2 == 200, (s2, d)
        if d["data"]["status"] != "running":
            break
        assert time.monotonic() < deadline, "session did not finish"
    return s, d["data"], sid
"""


def candidate_uid_processes() -> list[str]:
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


# --- candidate payloads (trusted first-party probe code, no model output) ---

def majority_code(id_column_style: str) -> str:
    return """import csv, os
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


DENIAL_PROBE_CODE = """import json, os
results = {}
def attempt(name, fn):
    try:
        fn(); results[name] = "REACHED"
    except OSError as exc: results[name] = f"denied:{exc.errno}"
    except Exception as exc: results[name] = f"denied:{type(exc).__name__}"
own = os.environ["DATA_DIR"]
attempt("own_task_train", lambda: open(os.path.join(own, "train.csv")).read())
attempt("cross_task_banknote", lambda: open("/mnt/rsi_data/comparison/banknote/data/public/train.csv").read())
attempt("cross_task_parent", lambda: os.listdir("/mnt/rsi_data/comparison"))
attempt("cross_job_source_canary", lambda: open("__CANARY_CODE__").read())
attempt("cross_job_prediction_canary", lambda: open("__CANARY_PRED__").read())
attempt("storage_tree_listdir", lambda: os.listdir("/mnt/rsi_storage"))
attempt("sealed_tree", lambda: os.listdir("/mnt/rsi_data/comparison/sealed"))
attempt("evaluator_tree", lambda: os.listdir("/opt/rsi_repo"))
os.symlink("__CANARY_CODE__", "alias.txt")
attempt("symlink_alias_to_storage_canary", lambda: open("alias.txt").read())
# Traversal paths are assembled at runtime from os.pardir (the admission
# gate forbids the literal parent-dir marker in source); Landlock
# canonicalizes them and still denies anything outside the granted leaf.
traversal_cross = os.path.join("/mnt/rsi_data/comparison/banknote", os.pardir, "banknote", "data/public/train.csv")
attempt("traversal_alias_cross_task", lambda: open(traversal_cross).read())
attempt("traversal_from_own_grant_upward", lambda: os.listdir(os.path.join(own, os.pardir, os.pardir)))
# EXP-003 bundled-asset masks: the sklearn-bundled full labeled WDBC CSV
# (both interpreter copies) must be empty for candidate code, and the
# sklearn loader must fail to produce it.
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
# Ordinary sklearn estimators must still train on the APPROVED task data.
import csv
def fit_and_predict():
    xs, ys = [], []
    with open(os.path.join(own, "train.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            features = [float(v) for k, v in row.items() if k not in ("id", "label")]
            xs.append(features); ys.append(int(row["label"]))
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=200).fit(xs, ys)
    ids, preds = [], []
    with open(os.path.join(own, "test.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            ids.append(row["id"])
            preds.append(int(model.predict([[float(v) for k, v in row.items() if k != "id"]])[0]))
    with open("submission.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "label"])
        for value, pred in zip(ids, preds):
            w.writerow([value, pred])
attempt("ordinary_estimator_on_granted_data", fit_and_predict)
if not os.path.exists("submission.csv"):
    with open("submission.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["id", "label"])
        [w.writerow([200 + i, 0]) for i in range(40)]
print("GRANTS|" + json.dumps(results, sort_keys=True))
"""


def main() -> int:
    key = load_api_key()

    # 0. preconditions: stack idle, legacy containers preserved ---------------
    legacy_state = {}
    for name in LEGACY_CONTAINERS:
        proc = subprocess.run(
            ["docker", "inspect", "-f", "{{.Id}} {{.State.Status}} {{.State.StartedAt}}", name],
            capture_output=True, text=True, timeout=30,
        )
        parts = proc.stdout.strip().split(" ", 2)
        legacy_state[name] = {"id": parts[0][:12], "status": parts[1] if len(parts) > 1 else "",
                              "started_at": parts[2] if len(parts) > 2 else ""}
    leftover = candidate_uid_processes()
    record(
        "precondition: stack idle, all six legacy containers running (identity recorded)",
        not leftover and all(s["status"] == "running" for s in legacy_state.values()),
        f"legacy={legacy_state} leftover_candidate_pids={leftover}",
        legacy=legacy_state,
    )

    # Plant synthetic cross-job canaries in the shared storage tree (host side,
    # trusted; simulates another job's source + scored prediction snapshot).
    canary_code_host = STORAGE_HOST_JOBS / CANARY_JOB / "code" / "canary.txt"
    canary_pred_host = STORAGE_HOST_JOBS / CANARY_JOB / "prediction_snapshot.csv"
    canary_code_host.parent.mkdir(parents=True, exist_ok=True)
    canary_code_host.write_bytes(CANARY_CONTENT)
    canary_pred_host.write_bytes(b"id,label\n200,0\n")
    canary_code_worker = f"{STORAGE_WORKER_JOBS}/{CANARY_JOB}/code/canary.txt"
    canary_pred_worker = f"{STORAGE_WORKER_JOBS}/{CANARY_JOB}/prediction_snapshot.csv"

    try:
        # 1. hello_synth benign compatibility ----------------------------------
        job_id = submit(key, "exp003-hello-synth-benign", majority_code("hello_synth"),
                        "hello_synth", "/mnt/rsi_data/hello_synth")
        status, result = wait_terminal(key, job_id)
        record(
            "hello_synth benign job scores under narrowed grants (compatibility)",
            status == "completed"
            and isinstance(result.get("score"), (int, float))
            and result.get("worker_cleanup_verified") is True,
            f"job={job_id} status={status} score={result.get('score')}",
            job_id=job_id, score=result.get("score"),
        )

        # 2. comparison task benign end-to-end ---------------------------------
        job_id = submit(key, "exp003-banknote-benign", majority_code("banknote"),
                        "comparison_banknote", "/mnt/rsi_data/comparison/banknote")
        status, result = wait_terminal(key, job_id)
        evaluation = result.get("evaluation") or {}
        record(
            "comparison_banknote job reads own task grant and scores (validation split)",
            status == "completed"
            and isinstance(result.get("score"), (int, float))
            and evaluation.get("task_id") == "comparison_banknote"
            and evaluation.get("split") == "validation"
            and result.get("worker_cleanup_verified") is True,
            f"job={job_id} status={status} score={result.get('score')}",
            job_id=job_id, score=result.get("score"), evaluation=evaluation,
        )

        # 3. real candidate denial probes (hello_synth grant) ------------------
        probe_code = (
            DENIAL_PROBE_CODE.replace("__CANARY_CODE__", canary_code_worker)
            .replace("__CANARY_PRED__", canary_pred_worker)
        )
        job_id = submit(key, "exp003-denial-probes", probe_code,
                        "hello_synth", "/mnt/rsi_data/hello_synth")
        status, result = wait_terminal(key, job_id)
        run_log = result.get("run_log") or ""
        marker_line = next((l for l in run_log.splitlines() if l.startswith("GRANTS|")), "")
        probe_results = json.loads(marker_line.split("|", 1)[1]) if marker_line else {}
        reached_ok = (
            probe_results.get("own_task_train") == "REACHED"
            and probe_results.get("ordinary_estimator_on_granted_data") == "REACHED"
        )
        masked_ok = (
            probe_results.get("bundled_wdbc_usr") == "masked_empty"
            and probe_results.get("bundled_wdbc_opt") == "masked_empty"
        )
        special = {"own_task_train", "ordinary_estimator_on_granted_data",
                   "bundled_wdbc_usr", "bundled_wdbc_opt"}
        denied_probes = {k: v for k, v in probe_results.items() if k not in special}
        all_denied = bool(denied_probes) and all(str(v).startswith("denied:") for v in denied_probes.values())
        record(
            "candidate confined to own task grant: cross-task/cross-job/sealed/symlink/traversal denied",
            status == "completed"
            and reached_ok
            and all_denied
            and len(denied_probes) == 11,  # 10 boundary probes + blocked WDBC loader
            f"probes={json.dumps(probe_results, sort_keys=True)}",
            job_id=job_id, probe_results=probe_results,
            worker_cleanup_verified=result.get("worker_cleanup_verified"),
        )
        record(
            "bundled labeled WDBC masked: both CSV copies empty, sklearn loader blocked, estimator OK",
            status == "completed"
            and masked_ok
            and str(probe_results.get("wdbc_loader", "")).startswith("denied:")
            and probe_results.get("ordinary_estimator_on_granted_data") == "REACHED",
            f"wdbc_usr={probe_results.get('bundled_wdbc_usr')} wdbc_opt={probe_results.get('bundled_wdbc_opt')} "
            f"loader={probe_results.get('wdbc_loader')} estimator={probe_results.get('ordinary_estimator_on_granted_data')}",
            job_id=job_id,
        )
        canary_intact = canary_code_host.read_bytes() == CANARY_CONTENT
        record(
            "cross-job canary bytes intact after candidate probes",
            canary_intact,
            f"canary_intact={canary_intact}",
        )

        # 4. worker-side grant validation over the trusted control channel -----
        probe_dir_worker = f"{SCRATCH_WORKER}/jobs-candidate-v1/exp003-direct"
        worker_control(
            WORKER_CLIENT
            + f"""
s, d, sid = run_session({{"id": "", "command": "mkdir -p {probe_dir_worker}/code && chmod -R 0777 {probe_dir_worker}", "exec_class": "staging", "async_mode": True, "timeout": 30}})
assert s == 200 and d.get("cleanup_verified") is True, (s, d)
assert delete(sid) == 200
print("STAGE_OK")
"""
        )
        validation_script = WORKER_CLIENT + f"""
import json
base = {{"id": "", "command": "true", "exec_dir": "{probe_dir_worker}/code",
        "job_root": "{probe_dir_worker}", "async_mode": True, "timeout": 30}}
outcomes = {{}}
for name, grants in [
    ("broad_root", ["/mnt/rsi_data"]),
    ("parent_dir", ["/mnt/rsi_data/comparison"]),
    ("foreign_root", ["/mnt/rsi_storage/mlsandbox/jobs"]),
    ("unregistered_task", ["/mnt/rsi_data/comparison/unregistered/data/public"]),
    ("leaf_child", ["/mnt/rsi_data/comparison/banknote/data/public/x"]),
    ("traversal", ["/mnt/rsi_data/hello_synth/../hello_synth/data/public"]),
    ("two_tasks", ["/mnt/rsi_data/comparison/banknote/data/public",
                   "/mnt/rsi_data/comparison/wdbc/data/public"]),
]:
    s, p = post("/v1/shell/exec", dict(base, read_grants=grants))
    outcomes[name] = [s, p.get("error")]
# staging sessions must not carry candidate grants
s, p = post("/v1/shell/exec", {{"id": "", "command": "true", "exec_class": "staging",
            "read_grants": ["/mnt/rsi_data/hello_synth/data/public"], "async_mode": True, "timeout": 30}})
outcomes["staging_with_grants"] = [s, p.get("error")]
print("VALIDATION|" + json.dumps(outcomes, sort_keys=True))
"""
        out = worker_control(validation_script)
        marker = next((l for l in out.splitlines() if l.startswith("VALIDATION|")), "")
        outcomes = json.loads(marker.split("|", 1)[1]) if marker else {}
        expected_errors = {
            "broad_root": "read_grant_not_allowlisted",
            "parent_dir": "read_grant_not_allowlisted",
            "foreign_root": "read_grant_not_allowlisted",
            "unregistered_task": "read_grant_not_allowlisted",
            "leaf_child": "read_grant_not_allowlisted",
            "traversal": "read_grant_not_canonical",
            "two_tasks": "conflicting_read_grants",
            "staging_with_grants": "read_grants_candidate_only",
        }
        validation_ok = all(
            outcomes.get(name) == [400, reason] for name, reason in expected_errors.items()
        ) and len(outcomes) == len(expected_errors)
        record(
            "worker refuses broad/parent/foreign/traversal/multi-task/staging grants (400, real API)",
            validation_ok,
            f"outcomes={json.dumps(outcomes, sort_keys=True)}",
            outcomes=outcomes,
        )

        # 5. direct candidate session with ONE valid grant: real Landlock scope
        direct_script = WORKER_CLIENT + f"""
import json
candidate_py = (
    "import json\\n"
    "r = {{}}\\n"
    "def a(n, p):\\n"
    "    try:\\n"
    "        open(p).read(); r[n] = 'REACHED'\\n"
    "    except OSError as e: r[n] = 'denied:' + str(e.errno)\\n"
    "a('granted_banknote', '/mnt/rsi_data/comparison/banknote/data/public/train.csv')\\n"
    "a('other_task_wdbc', '/mnt/rsi_data/comparison/wdbc/data/public/train.csv')\\n"
    "a('other_task_hello', '/mnt/rsi_data/hello_synth/data/public/train.csv')\\n"
    "a('storage_canary', '{canary_code_worker}')\\n"
    "print('DIRECT|' + json.dumps(r, sort_keys=True))\\n"
)
cmd = "python3 - <<'PY'\\n" + candidate_py + "PY\\n"
s, d, sid = run_session({{"id": "", "command": cmd,
    "exec_dir": "{probe_dir_worker}/code", "job_root": "{probe_dir_worker}",
    "exec_class": "candidate", "read_grants": ["/mnt/rsi_data/comparison/banknote/data/public"],
    "async_mode": True, "timeout": 60}}, wait_seconds=60)
assert s == 200, (s, d)
print("SESSION|" + json.dumps({{"status": d.get("status"), "cleanup_verified": d.get("cleanup_verified"),
                                "output": d.get("output", "")[-400:], "stderr": d.get("stderr", "")[-200:]}}))
assert delete(sid) == 200
"""
        out = worker_control(direct_script, timeout=120)
        session_line = next((l for l in out.splitlines() if l.startswith("SESSION|")), "")
        session_info = json.loads(session_line.split("|", 1)[1]) if session_line else {}
        direct_line = next(
            (l for l in str(session_info.get("output", "")).splitlines() if l.startswith("DIRECT|")), ""
        )
        direct = json.loads(direct_line.split("|", 1)[1]) if direct_line else {}
        record(
            "direct candidate session: exact grant readable, all other task/storage paths denied",
            session_info.get("status") == "completed"
            and session_info.get("cleanup_verified") is True
            and direct.get("granted_banknote") == "REACHED"
            and all(str(v).startswith("denied:") for k, v in direct.items() if k != "granted_banknote")
            and len(direct) == 4,
            f"direct={json.dumps(direct, sort_keys=True)} session={session_info.get('status')}",
            direct=direct, cleanup_verified=session_info.get("cleanup_verified"),
        )

        # 6. post-cleanup state in the dedicated worker container ---------------
        leftover = candidate_uid_processes()
        record(
            "post-cleanup: no candidate-uid processes remain in rsi_worker",
            not leftover,
            f"leftover={leftover}",
        )
    finally:
        # Remove the synthetic canaries (host side) and verify removal.
        subprocess.run(
            ["rm", "-rf", str(STORAGE_HOST_JOBS / CANARY_JOB)],
            check=True, timeout=30,
        )
        removed = not (STORAGE_HOST_JOBS / CANARY_JOB).exists()
        record("synthetic canaries removed from storage tree", removed)

    ok = all(item["ok"] for item in RESULTS)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "story": "EXP-003",
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
    print("EXP003 PROBES " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

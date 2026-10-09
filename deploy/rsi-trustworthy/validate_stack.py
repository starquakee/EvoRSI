#!/usr/bin/env python3
"""Structural + runtime isolation validation for the rsi-trustworthy stack.

US-006 evidence tool (batch-2 repair). Validates the RUNNING stack with
docker inspect and harmless probes; prints PASS/FAIL per check and exits
nonzero on any failure. Never prints credentials. Read-only with respect
to both the new and the legacy stack (probe sessions delete themselves;
the detached-child probe marker is a scratch temp file that must NEVER
appear — its absence is the proof).

Checks:
  1. docker compose config is valid.
  2. Worker container: no published ports, no docker socket, not privileged,
     default seccomp (no 'unconfined'), no-new-privileges, cap_drop ALL with
     exactly {SETUID, SETGID, KILL} added (root supervisor only),
     memory/cpu/pids bounds, read-only rootfs + bounded tmpfs /tmp,
     attached to the execution network only, public input mounts read-only,
     scratch mount the only writable share, control key mount read-only,
     and the evaluator tree fails the US-005 worker mount-plan validation
     for every actual worker mount source.
  3. control/execution networks are internal; the execution network uses
     gateway_mode_ipv4=isolated (no host bridge gateway); edge only nginx.
  4. Gateway published only on 127.0.0.1:6581; everything else publishes
     nothing.
  5. Live probes: scratch writable / input read-only / rootfs read-only;
     unauthenticated worker-local control calls -> 401; candidate uid
     cannot read the control key file or the supervisor /proc/1/environ;
     authenticated exec works and reports cleanup_verified; the single
     execution slot rejects a concurrent exec (429); a detached (setsid)
     child is killed by post-exit cleanup and never writes its marker;
     no candidate-uid processes remain afterwards; worker cannot
     resolve/reach postgres/redis, has no DNS, no internet egress and no
     host-gateway route.
  6. Legacy stack (6580/10150) containers still running; legacy gateway
     answers HTTP 200.

All HTTP requests bypass process proxy settings explicitly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
COMPOSE_FILE = SCRIPT_DIR / "docker-compose.yaml"
EVALUATOR_ROOT = REPO_ROOT / "research" / "evaluator"
SCRATCH_HOST_DIR = REPO_ROOT / ".runtime" / "rsi-trustworthy" / "workdir"
DETACHED_MARKER = ".rsi_detached_probe_marker"
CANDIDATE_UID = "65432"

sys.path.insert(0, str(REPO_ROOT))
from research.evaluator.service import validate_worker_mount_plan  # noqa: E402

FAILURES: list[str] = []
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def docker(*args: str, timeout: int = 60, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, input=stdin
    )


def inspect(name: str) -> dict:
    result = docker("inspect", name)
    if result.returncode != 0:
        raise RuntimeError(f"docker inspect {name} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)[0]


def inspect_net(name: str) -> dict:
    result = docker("network", "inspect", name)
    if result.returncode != 0:
        raise RuntimeError(f"docker network inspect {name} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)[0]


UNAUTH_PROBE = r"""
import json, urllib.request, urllib.error
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
body = json.dumps({"id": "", "command": "echo unauth", "async_mode": True, "timeout": 5}).encode()
for headers in ({}, {"X-RSI-Control-Key": "wrong-key-00000000000000000000000000000000"}):
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/shell/exec", data=body,
        headers=dict(headers, **{"Content-Type": "application/json"}), method="POST")
    try:
        with opener.open(req, timeout=10) as r:
            raise SystemExit(f"unexpected status {r.status}")
    except urllib.error.HTTPError as e:
        assert e.code == 401, f"expected 401, got {e.code}"
        assert json.loads(e.read().decode())["error"] == "worker_control_unauthorized"
print("UNAUTH_DENIED_OK")
"""

AUTHED_PROBE = r"""
import json, time, urllib.request, urllib.error
key = open("/run/rsi_control/key", encoding="utf-8").read().strip()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASE = "http://worker:8080"

def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-RSI-Control-Key": key}, method="POST")
    try:
        with opener.open(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def delete(sid):
    req = urllib.request.Request(BASE + "/v1/shell/sessions/" + sid,
        headers={"X-RSI-Control-Key": key}, method="DELETE")
    try:
        with opener.open(req, timeout=25) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code

def wait_done(sid, timeout=25):
    deadline = time.monotonic() + timeout
    while True:
        s, p = post("/v1/shell/wait", {"id": sid, "seconds": 1})
        assert s == 200, (s, p)
        if p["data"]["status"] != "running":
            return p["data"]
        assert time.monotonic() < deadline, "session did not finish"

# 1. authenticated exec completes with proven cleanup
s, p = post("/v1/shell/exec", {"id": "", "command": "echo authed-ok", "exec_class": "staging", "async_mode": True, "timeout": 30})
assert s == 200, (s, p)
sid = p["data"]["session_id"]
d = wait_done(sid)
assert d["status"] == "completed" and "authed-ok" in d["output"], d
assert d.get("cleanup_verified") is True, d
assert delete(sid) == 200
print("STEP1_OK")

# 2. single execution slot rejects a concurrent exec
s, p = post("/v1/shell/exec", {"id": "", "command": "sleep 8", "exec_class": "staging", "async_mode": True, "timeout": 30})
assert s == 200, (s, p)
sid2 = p["data"]["session_id"]
s, p = post("/v1/shell/exec", {"id": "", "command": "echo second", "exec_class": "staging", "async_mode": True, "timeout": 30})
assert s == 429 and p["error"] == "execution_slot_busy", (s, p)
assert delete(sid2) == 200
print("STEP2_OK")

# 3. detached (setsid) child must be killed by post-exit cleanup
detached = (
    "python3 -c \"import subprocess;"
    "subprocess.Popen(['sh','-c','sleep 3; echo DETACHED > /mnt/local_sandbox_workdir/"
    + "__MARKER__" + "'], start_new_session=True);"
    "print('spawned')\""
)
s, p = post("/v1/shell/exec", {"id": "", "command": detached, "exec_class": "staging", "async_mode": True, "timeout": 30})
assert s == 200, (s, p)
sid3 = p["data"]["session_id"]
d3 = wait_done(sid3)
assert d3["status"] == "completed" and d3.get("cleanup_verified") is True, d3
assert delete(sid3) == 200
print("STEP3_OK")
print("AUTHED_PROBES_OK")
""".replace("__MARKER__", DETACHED_MARKER)

CANDIDATE_PROC_SCAN = r"""
import pathlib, sys
hits = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        text = (entry / "status").read_text(errors="replace")
    except OSError:
        continue
    for line in text.splitlines():
        if line.startswith("Uid:"):
            if int(line.split()[1]) == __UID__:
                hits.append(entry.name)
            break
print("candidate-uid processes:", hits, file=sys.stderr)
sys.exit(1 if hits else 0)
""".replace("__UID__", CANDIDATE_UID)


def main() -> int:
    # 1. compose config
    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "--quiet"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    check("compose config valid", result.returncode == 0, result.stderr.strip()[:200])

    # 2. worker container contract
    worker = inspect("rsi_worker")
    host = worker["HostConfig"]
    check("worker: no published ports", not host.get("PortBindings"))
    binds = host.get("Binds") or []
    check(
        "worker: no docker socket",
        not any("docker.sock" in b for b in binds),
        "; ".join(binds) if any("docker.sock" in b for b in binds) else "",
    )
    check("worker: not privileged", not host.get("Privileged"))
    secopt = host.get("SecurityOpt") or []
    check(
        "worker: default seccomp + no-new-privileges",
        "no-new-privileges:true" in secopt and not any("unconfined" in s for s in secopt),
        f"SecurityOpt={secopt}",
    )
    check("worker: cap_drop ALL", "ALL" in (host.get("CapDrop") or []), f"CapDrop={host.get('CapDrop')}")
    cap_add = sorted(cap.removeprefix("CAP_") for cap in (host.get("CapAdd") or []))
    check(
        "worker: cap_add exactly {KILL, SETGID, SETUID} (supervisor minimum)",
        cap_add == ["KILL", "SETGID", "SETUID"],
        f"CapAdd={cap_add}",
    )
    check(
        "worker: memory/cpu/pids bounded",
        bool(host.get("Memory")) and bool(host.get("NanoCpus")) and bool(host.get("PidsLimit")),
        f"Memory={host.get('Memory')} NanoCpus={host.get('NanoCpus')} PidsLimit={host.get('PidsLimit')}",
    )
    check("worker: read-only rootfs", bool(host.get("ReadonlyRootfs")), f"ReadonlyRootfs={host.get('ReadonlyRootfs')}")
    tmpfs = host.get("Tmpfs") or {}
    check("worker: bounded tmpfs /tmp", "/tmp" in tmpfs, f"Tmpfs={tmpfs}")
    networks = list((worker.get("NetworkSettings") or {}).get("Networks") or {})
    check("worker: execution network only", networks == ["rsi-trustworthy_execution"], f"{networks}")

    mounts = worker.get("Mounts") or []
    scratch_writable = False
    input_readonly = True
    public_data_readonly = False
    bad_mounts = []
    for mount in mounts:
        dest = mount.get("Destination") or ""
        source = mount.get("Source") or ""
        rw = bool(mount.get("RW"))
        if dest == "/mnt/local_sandbox_workdir":
            scratch_writable = scratch_writable or rw
        elif dest.startswith("/mnt/rsi_storage"):
            input_readonly = input_readonly and (not rw)
        elif dest.startswith("/mnt/rsi_data"):
            # Public synthetic task inputs (US-008): read-only only.
            input_readonly = input_readonly and (not rw)
            public_data_readonly = True
        elif dest in (
            "/opt/rsi_worker/minimal_worker.py",
            "/opt/rsi_worker/exec_trampoline.py",
            "/run/rsi_control/key",
        ):
            input_readonly = input_readonly and (not rw)
        else:
            bad_mounts.append(f"{source}->{dest} rw={rw}")
    check("worker: scratch mount writable", scratch_writable)
    check("worker: public input + entrypoint + trampoline + control key read-only", input_readonly)
    check("worker: public task data mounted read-only", public_data_readonly)
    check("worker: no unexpected mounts", not bad_mounts, "; ".join(bad_mounts))
    check(
        "worker: control key mounted",
        any(m.get("Destination") == "/run/rsi_control/key" for m in mounts),
    )

    denials = validate_worker_mount_plan(
        [m.get("Source") or "" for m in mounts], evaluator_root=EVALUATOR_ROOT
    )
    check(
        "worker: evaluator tree not reachable via any mount (US-005 plan check)",
        not denials,
        "; ".join(f"{d.reason}:{d.source}" for d in denials),
    )

    # 3. networks
    for net in ("rsi-trustworthy_control", "rsi-trustworthy_execution"):
        info = inspect_net(net)
        check(f"network {net} internal", bool(info.get("Internal")))
    execution = inspect_net("rsi-trustworthy_execution")
    gw_mode = (execution.get("Options") or {}).get("com.docker.network.bridge.gateway_mode_ipv4")
    check("network execution gateway_mode_ipv4=isolated", gw_mode == "isolated", f"Options={execution.get('Options')}")
    edge = inspect_net("rsi-trustworthy_edge")
    edge_members = sorted((edge.get("Containers") or {}).values(), key=lambda c: c.get("Name", ""))
    edge_names = [c.get("Name") for c in edge_members]
    check("network edge holds only nginx", edge_names == ["rsi_nginx"], f"{edge_names}")

    # 4. published ports
    nginx = inspect("rsi_nginx")
    bindings = (nginx["HostConfig"].get("PortBindings") or {})
    only = bindings == {"80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "6581"}]}
    check("gateway published only on 127.0.0.1:6581", only, json.dumps(bindings))
    for svc in ("rsi_postgres", "rsi_redis", "rsi_api", "rsi_dispatcher"):
        pb = (inspect(svc)["HostConfig"].get("PortBindings") or {})
        check(f"{svc}: no published ports", not pb, json.dumps(pb))

    # 5. live probes
    write_probe = docker(
        "exec", "rsi_worker", "bash", "-c",
        "probe=/mnt/local_sandbox_workdir/.validate_probe_$$ && touch \"$probe\" && rm -f \"$probe\"",
        timeout=20,
    )
    check("worker: scratch actually writable (cap-dropped)", write_probe.returncode == 0,
          (write_probe.stdout + write_probe.stderr).strip()[:120])
    read_probe = docker(
        "exec", "rsi_worker", "bash", "-c",
        "test -r /opt/rsi_worker/minimal_worker.py && test ! -w /mnt/rsi_storage/mlsandbox/jobs",
        timeout=20,
    )
    check("worker: public input readable but not writable", read_probe.returncode == 0)
    rootfs_probe = docker(
        "exec", "rsi_worker", "bash", "-c",
        "touch /etc/.validate_w 2>/dev/null; test ! -e /etc/.validate_w && touch /tmp/.validate_w && rm -f /tmp/.validate_w",
        timeout=20,
    )
    check("worker: rootfs read-only, tmpfs /tmp writable", rootfs_probe.returncode == 0,
          (rootfs_probe.stdout + rootfs_probe.stderr).strip()[:120])

    unauth = docker("exec", "-i", "rsi_worker", "python3", "-", stdin=UNAUTH_PROBE, timeout=40)
    check("worker: unauthenticated control calls -> 401", unauth.returncode == 0
          and "UNAUTH_DENIED_OK" in unauth.stdout, (unauth.stdout + unauth.stderr).strip()[:160])

    key_probe = docker("exec", "-u", CANDIDATE_UID, "rsi_worker", "python3", "-c",
                       "import os; fd=os.open('/run/rsi_control/key',os.O_RDONLY); os.close(fd)", timeout=20)
    check("worker: candidate uid cannot read control key file", key_probe.returncode != 0,
          (key_probe.stdout + key_probe.stderr).strip()[:120])
    environ_probe = docker("exec", "-u", CANDIDATE_UID, "rsi_worker", "python3", "-c",
                           "import os; fd=os.open('/proc/1/environ',os.O_RDONLY); os.close(fd)", timeout=20)
    check("worker: candidate uid cannot read supervisor /proc/1/environ",
          environ_probe.returncode != 0, (environ_probe.stdout + environ_probe.stderr).strip()[:120])

    # Authenticated control-channel probes run from the dispatcher container
    # (the only other holder of the control key), reaching the worker over
    # the isolated execution network. The key value is never printed.
    (SCRATCH_HOST_DIR / DETACHED_MARKER).unlink(missing_ok=True)
    authed = docker("exec", "-i", "rsi_dispatcher", "python3", "-", stdin=AUTHED_PROBE, timeout=120)
    authed_ok = authed.returncode == 0 and "AUTHED_PROBES_OK" in authed.stdout
    check("worker: authenticated exec completes with cleanup_verified", "STEP1_OK" in authed.stdout,
          (authed.stdout + authed.stderr).strip()[-200:])
    check("worker: single execution slot rejects concurrent exec (429)", "STEP2_OK" in authed.stdout)
    # The detached setsid child (3s fuse) must have been killed by the
    # post-exit candidate-uid sweep: its marker must NEVER appear.
    time.sleep(5)
    marker_appeared = (SCRATCH_HOST_DIR / DETACHED_MARKER).exists()
    check("worker: detached setsid child killed by cleanup (marker never written)",
          authed_ok and "STEP3_OK" in authed.stdout and not marker_appeared)
    (SCRATCH_HOST_DIR / DETACHED_MARKER).unlink(missing_ok=True)

    scan = docker("exec", "-i", "rsi_worker", "python3", "-", stdin=CANDIDATE_PROC_SCAN, timeout=30)
    check("worker: no candidate-uid processes remain", scan.returncode == 0,
          (scan.stdout + scan.stderr).strip()[:160])

    probes = {
        "worker cannot resolve/reach postgres": "getent hosts postgres",
        "worker cannot resolve/reach redis": "getent hosts redis",
        "worker has no external DNS": "getent hosts example.com",
        "worker has no internet egress": (
            "python3 -c \"import socket;socket.create_connection(('93.184.216.34',80),timeout=3)\""
        ),
        "worker has no host bridge gateway": (
            "python3 -c \"import sys;"
            "routes=[l.split() for l in open('/proc/net/route').read().splitlines()[1:]];"
            "default=any(r[1]=='00000000' and int(r[3],16)&2 for r in routes);"
            "sys.exit(0 if default else 1)\""
        ),
    }
    for name, cmd in probes.items():
        result = docker("exec", "rsi_worker", "bash", "-c", cmd, timeout=20)
        check(name, result.returncode != 0, (result.stdout + result.stderr).strip()[:120])

    # 6. legacy stack untouched and healthy
    legacy = [
        "sandbox_api",
        "sandbox_dispatcher",
        "sandbox_nginx",
        "sandbox_postgres",
        "sandbox_redis",
        "ml-sandbox-gpu-0",
    ]
    for name in legacy:
        try:
            state = inspect(name).get("State") or {}
            ok = bool(state.get("Running")) and state.get("Status") in {"running"}
            if name == "ml-sandbox-gpu-0":
                ok = ok and (state.get("Health") or {}).get("Status") == "healthy"
            check(f"legacy {name} running", ok, state.get("Status", "missing"))
        except RuntimeError as exc:
            check(f"legacy {name} running", False, str(exc))
    try:
        with OPENER.open("http://127.0.0.1:6580/", timeout=5) as response:
            check("legacy gateway 6580 answers", response.status == 200)
    except Exception as exc:
        check("legacy gateway 6580 answers", False, str(exc))

    print()
    if FAILURES:
        print(f"VALIDATION FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

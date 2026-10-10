"""提交 hello_synth 端到端验证任务并等待结果。

用法：
    SANDBOX_API_KEY=<key> python3 submit_e2e.py
环境变量：
    SANDBOX_ENDPOINT  默认 http://127.0.0.1:6581（US-010：verified 隔离栈）；
                      显式设为 http://127.0.0.1:6580 回退旧栈（rollback only）
    SANDBOX_API_KEY   必填，对应栈的 API key（新栈见 .runtime/rsi-trustworthy/auth.env）
    SANDBOX_DATA_DIR  默认 /mnt/rsi_data/hello_synth；回退旧栈时显式设置旧数据目录
    SANDBOX_TASK_ID   默认 hello_synth；回退旧栈时显式设置旧任务标识
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.adapters.endpoints import resolve_endpoint  # noqa: E402

ENDPOINT = resolve_endpoint()
API_KEY = os.environ["SANDBOX_API_KEY"]
JOB_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hello_synth_job.py")

payload = {
    "name": "hello_synth_e2e",
    # US-010 新栈契约：task_id 必须是评测器注册表允许的任务；
    # data_dir 指向新栈公共数据挂载（dispatcher 会把 DATA_DIR 导出为
    # <data_dir>/data/public）。旧栈回退时需显式恢复旧值
    # task_id=hello_synth_e2e / data_dir=/mnt/pubdatasets2/tasks/hello_synth。
    "task_id": os.environ.get("SANDBOX_TASK_ID", "hello_synth"),
    "code": open(JOB_SOURCE, encoding="utf-8").read(),
    "data_dir": os.environ.get("SANDBOX_DATA_DIR", "/mnt/rsi_data/hello_synth"),
    "resource_type": os.environ.get("RESOURCE_TYPE", "gpu"),
    "gpu_count": 1,
    "timeout": 600,
    # shell 模式：在 worker 主机 Python(3.10, 含 torch/cuda) 执行 `python <code>`，
    # DATA_DIR 指向 <data_dir>/data/public。默认 jupyter 内核无 torch。
    "environment": {"EXECUTION_MODE": "shell"},
}

req = urllib.request.Request(
    f"{ENDPOINT}/api/v1/jobs/submit_and_wait?timeout=600&poll_interval=3",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "X-API-Key": API_KEY},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=660) as resp:
        result = json.load(resp)
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", "replace")
    print(f"HTTP {exc.code}: {body[:3000]}")
    raise SystemExit(1)

print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])

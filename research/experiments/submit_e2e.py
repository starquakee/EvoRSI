"""提交 hello_synth 端到端验证任务并等待结果。

用法：
    SANDBOX_API_KEY=<key> python3 submit_e2e.py
环境变量：
    SANDBOX_ENDPOINT  默认 http://127.0.0.1:6580
    SANDBOX_API_KEY   必填，本地 controller 的 dev key
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

ENDPOINT = os.environ.get("SANDBOX_ENDPOINT", "http://127.0.0.1:6580")
API_KEY = os.environ["SANDBOX_API_KEY"]
JOB_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hello_synth_job.py")

payload = {
    "name": "hello_synth_e2e",
    "task_id": "hello_synth_e2e",
    "code": open(JOB_SOURCE, encoding="utf-8").read(),
    "data_dir": "/mnt/pubdatasets2/tasks/hello_synth",
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

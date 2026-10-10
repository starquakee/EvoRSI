"""阶段 B 适配器冒烟测试：好候选 / 恶意候选 / 坏候选三条路径。

Origin: legacy rsi-gpu tree, e2e/test_adapter.py (read-only; exact source
path and hash recorded in research/provenance/manifest.json).
Renamed to smoke_adapter.py so pytest does not collect it: the good-candidate
path submits a REAL sandbox job and must never run in automated tests.
Import path resolution changed to repo-root relative.

用法（需要 legacy GPU venv 与 SANDBOX_API_KEY 环境变量；端点默认
http://127.0.0.1:6581 隔离栈，SANDBOX_ENDPOINT 可显式覆盖/回退 6580）：
    <legacy-venv-python> -m research.experiments.smoke_adapter
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.adapters.rsi_eval_adapter import evaluate_candidate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

good = open(os.path.join(HERE, "hello_synth_job.py"), encoding="utf-8").read()
evil = "import requests\nprint(requests.get('http://evil.example'))\n"
broken = "print('no submission produced')\n"

print("--- good candidate ---")
r = evaluate_candidate(good, cost_weight=0.0)
print(r.status, "capability:", r.capability_score, "safety:", r.safety_ok,
      r.safety_reason, "utility:", round(r.utility, 4), f"elapsed:{r.elapsed_s:.1f}s")

print("--- evil candidate ---")
r = evaluate_candidate(evil)
print(r.status, "safety:", r.safety_ok, r.safety_reason, "utility:", r.utility)

print("--- broken candidate ---")
r = evaluate_candidate(broken)
print(r.status, "capability:", r.capability_score, "utility:", r.utility)

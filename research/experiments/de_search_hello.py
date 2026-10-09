"""阶段 C 最小实测：iStratDE 通过 sandbox 评测闭环搜索 hello_synth 求解参数。

Origin: legacy rsi-gpu tree, e2e/de_search_hello.py (read-only; exact source
path and hash recorded in research/provenance/manifest.json).
Consolidated under research/experiments for US-001; only import path
resolution changed (repo-root relative instead of legacy-tree relative).

代理任务：候选向量 theta=(w1, w2, bias) 定义线性规则 w1*f1 + w2*f2 > bias，
把规则参数烤进任务代码模板，经 rsi_eval_adapter 提交 sandbox 评分，
fitness = -utility（安全门失败/无法计分记为大惩罚 1e6）。

这是"小规模搜索接入"的验证：正式 RSI 接入时把模板换成 OpenMLE-Evo 的
hydra_overrides（见 research/adapters/safe_rsi_de_adapter.py），fitness 换成真实评测 utility。

用法（需要 legacy GPU venv，解释器路径见 research/ENVIRONMENT.md）：
    <legacy-venv-python> -m research.experiments.de_search_hello
"""
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.adapters.rsi_eval_adapter import evaluate_candidate  # noqa: E402
from istratde.algorithms.jax import IStratDE  # noqa: E402

JOB_TEMPLATE = '''
import os
import pandas as pd

data_dir = os.environ["DATA_DIR"]
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

W1, W2, BIAS = {w1:.6f}, {w2:.6f}, {bias:.6f}
pred = (test["f1"] * W1 + test["f2"] * W2 > BIAS).astype(int)
sub = pd.DataFrame({{"id": test["id"], "label": pred}})
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "submission.csv")
sub.to_csv(out, index=False)
print("params", W1, W2, BIAS, "submission_rows", len(sub))
'''

LB = jnp.array([-3.0, -3.0, -3.0])
UB = jnp.array([3.0, 3.0, 3.0])
POP_SIZE = 4
GENERATIONS = 3
PENALTY = 1e6


def fitness_of(theta) -> tuple[float, dict]:
    w1, w2, bias = (float(v) for v in theta)
    src = JOB_TEMPLATE.format(w1=w1, w2=w2, bias=bias)
    r = evaluate_candidate(src)
    record = {
        "theta": [round(w1, 4), round(w2, 4), round(bias, 4)],
        "status": r.status,
        "capability": r.capability_score,
        "safety": r.safety_reason,
        "utility": None if not math.isfinite(r.utility) else round(r.utility, 4),
    }
    fit = -r.utility if math.isfinite(r.utility) else PENALTY
    return fit, record


def main():
    print("devices:", jax.devices())
    algo = IStratDE(lb=LB, ub=UB, pop_size=POP_SIZE)
    # 注意：必须用 algo.init(key)（递归分配 _node_id），不能用 algo.setup(key)；
    # 本算法 init_ask 返回 None（无特殊初始种群），直接 ask/tell 循环即可。
    state = algo.init(jax.random.PRNGKey(42))
    best_utility, best_theta = -math.inf, None
    history = []
    for gen in range(GENERATIONS):
        pop, state = algo.ask(state)
        fits = []
        for row in pop:
            fit, record = fitness_of(row)
            fits.append(fit)
            history.append({"gen": gen, **record})
            print(f"[gen {gen}] {record}", flush=True)
            u = record["utility"]
            if u is not None and u > best_utility:
                best_utility, best_theta = u, record["theta"]
        state = algo.tell(state, jnp.array(fits, dtype=jnp.float32))
    print("=== RESULT ===")
    print("best_utility:", best_utility, "best_theta:", best_theta)
    print("evaluations:", len(history))


if __name__ == "__main__":
    main()

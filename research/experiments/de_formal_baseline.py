"""iStratDE 正式基线：DE 调 OpenMLE-Evo 策略参数，fitness = 真实 Evo 运行得分。

Origin: legacy rsi-gpu tree, e2e/de_formal_baseline.py (read-only; exact source
path and hash recorded in research/provenance/manifest.json).
Consolidated under research/experiments for US-001 with these changes:
- paths resolve relative to the repository root (this file's parents[2]);
- credentials come from the process environment or the gitignored repo-root
  .env ONLY; the legacy fallback that scraped an API key out of the deployed
  Windows api_server.py source was removed (credential exfiltration risk);
- output root defaults to <repo>/outputs/formal_baseline (gitignored),
  overridable via RSI_OUT_ROOT.

设计（对应 RUNBOOK 4.2 与 HANDOFF 第 5 节；实验配置在搜索前固定，不随结果调整）：
- 搜索空间：safe_rsi_de_adapter 的 8 维策略向量（LOWER/UPPER 固定）。
- 双层安全：decode 后过 safe_rsi_de_adapter.safety_gate（策略门，事前）；
  Evo 产物代码过 sandbox_eval_client.safety_gate_source（源码门，事后审计）。
  任一不过 -> PENALTY 并记录原因。
- fitness：titanic-extended@1 的 submit_score（logloss，higher_is_better=false，
  越小越好，直接作为最小化目标）；无法计分 -> PENALTY。
- 相同解码配置命中缓存，不重复消耗 Evo 运行。
- 每候选一行 JSONL 落盘（可断点续看），最后打印汇总。

注意：PRD 规定 US009 之前禁止真实模型评测；本脚本会发起真实 Evo/LLM 调用，
当前阶段不要运行。用法（通过审批后）：
    <legacy-venv-python> -m research.experiments.de_formal_baseline
环境：repo 根 .env 或进程环境提供 Kimi key 与 SANDBOX_API_KEY；
    OPENMLE_MODEL_ID 决定 Evo 所用模型（默认 k3）。

US-010 端点切换说明（重要）：本基线评测任务 titanic-extended@1 只部署在旧栈
（6580）；新隔离栈（6581，当前所有客户端默认）的可信评测器注册表只允许
hello_synth 合成任务，对本任务会 fail closed（task_not_allowlisted）。因此本
脚本在端点解析为新栈默认时拒绝启动；要复现旧基线必须显式回退：
    SANDBOX_URL=http://127.0.0.1:6580（连同旧栈 SANDBOX_API_KEY）
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from typing import Any

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.adapters.endpoints import DEFAULT_ENDPOINT  # noqa: E402
from research.adapters.safe_rsi_de_adapter import (  # noqa: E402
    LOWER, UPPER, decode, hydra_overrides, safety_gate,
)
from research.adapters.sandbox_eval_client import safety_gate_source  # noqa: E402
from istratde.algorithms.jax import IStratDE  # noqa: E402

# ---------- 固定实验配置 ----------
POP_SIZE = 12
GENERATIONS = 5
SEED = 42
TASK = "titanic-extended@1"
PENALTY = 10.0            # logloss 量级上的大罚分
EVO_TIMEOUT_S = 2700      # 单次 Evo 运行硬上限（45 分钟）

REPO = REPO_ROOT
EVO_DIR = REPO / "OpenMLE-Evo"
OUT_ROOT = Path(os.environ.get("RSI_OUT_ROOT", REPO / "outputs" / "formal_baseline"))
RUNS_ROOT = OUT_ROOT / "runs"
LOG_PATH = OUT_ROOT / "eval_log.jsonl"

TEMPERATURE_OVERRIDES = [
    f"+search.runner.solver.operators.{op}.llm.generation_kwargs.temperature=1.0"
    for op in ("draft", "improve", "debug", "analyze", "crossover")
]

BASE_OVERRIDES = [
    "litellm=kimi_coding",
    f"search.runner.task_list=[{TASK}]",
    "candidates_per_step=1",
    "search.runner.solver.individuals_per_generation=1",
    "search.runner.solver.max_debug_depth=1",
    *TEMPERATURE_OVERRIDES,
]


def load_env() -> dict[str, str]:
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
            if m:
                env.setdefault(m.group(1), m.group(2))
    if not env.get("SANDBOX_API_KEY"):
        raise RuntimeError(
            "SANDBOX_API_KEY must come from the environment or repo-root .env; "
            "the legacy key-scraping fallback was removed during consolidation."
        )
    env.update(
        PRIMARY_KEY=env.get("OPENAI_API_KEY", ""),
        SGLANG_BASE_URL=env.get("OPENAI_BASE_URL", "https://api.kimi.com/coding/v1"),
        OPENMLE_MODEL_ID=env.get("OPENMLE_MODEL_ID", "k3"),
        OPENMLE_EVAL_DATA=str(REPO / "artifacts/gym-example/eval.parquet"),
        OPENMLE_LEADERBOARD_DIR=str(REPO / "artifacts/gym-example/leaderboards"),
        OPENMLE_SUBMIT_DATA_DIR_ROOT="/mnt/pubdatasets2/tasks",
        OPENMLE_CONFIG_NAME="experiment/openmle_evo_smoke",
        SANDBOX_URL=env.get("SANDBOX_URL", DEFAULT_ENDPOINT),
        SANDBOX_CPU_API_KEY=env["SANDBOX_API_KEY"],
        SANDBOX_GPU_API_KEY=env["SANDBOX_API_KEY"],
        AIRA_LITELLM_TIMEOUT="900",
        AIRA_LITELLM_NUM_RETRIES="4",
        AIRA_LITELLM_STREAM="true",   # k3 大请求非流式会被挂起，必须流式
    )
    if env["SANDBOX_URL"].rstrip("/") == DEFAULT_ENDPOINT:
        # Honest guard (US-010): the titanic-extended@1 task package and its
        # scoring path exist only on the legacy 6580 stack; the isolated
        # 6581 evaluator registry allowlists the synthetic hello_synth task
        # only and would fail closed for every candidate. Refuse before any
        # Evo/LLM spend instead of burning the whole budget on penalties.
        raise RuntimeError(
            "legacy_baseline_requires_rollback_stack: task "
            f"{TASK} is not registered on the isolated default stack "
            f"({DEFAULT_ENDPOINT}); run only with an explicit rollback, e.g. "
            "SANDBOX_URL=http://127.0.0.1:6580 plus the legacy stack key."
        )
    return env


def parse_stat(stat_path: Path) -> dict:
    if not stat_path.exists():
        return {}
    try:
        return json.loads(stat_path.read_text())
    except Exception:
        return {}


def run_evo(overrides: list[str], run_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "./scripts/run_standard.sh",
        *BASE_OVERRIDES,
        f"output_dir={run_dir}",
        *overrides,
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=EVO_DIR, env=env, timeout=EVO_TIMEOUT_S,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        wall = time.monotonic() - t0
        run_status = f"exit_{proc.returncode}"
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - t0
        run_status = "timeout"

    stat = parse_stat(run_dir / "program_ep_0" / TASK / "stat.json")
    score = stat.get("submit_score")
    status_count = stat.get("status_count") or {}
    tokens = stat.get("total_tokens")

    code_safety = "no_artifact"
    for name in ("valid_code_final.py", "submit_code.py"):
        code_file = run_dir / "program_ep_0" / TASK / name
        if code_file.exists():
            ok, reason = safety_gate_source(code_file.read_text(errors="replace"))
            code_safety = reason
            if not ok:
                break

    return {
        "run_status": run_status,
        "wall_s": round(wall, 1),
        "score": score,
        "status_count": status_count,
        "total_tokens": tokens,
        "code_safety": code_safety,
    }


def fitness_of(
    theta,
    tag: str,
    env: dict[str, str],
    cache: dict[tuple[float, ...], dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    cfg = decode(theta)
    record: dict[str, Any] = {
        "tag": tag,
        "theta": [round(float(v), 4) for v in theta],
        "cfg": vars(cfg),
    }
    key = tuple(round(float(v), 6) for v in vars(cfg).values())
    if key in cache:
        record.update(cache[key], cached=True)
        return cache[key]["fitness"], record

    result: dict[str, Any]
    ok, reason = safety_gate(cfg)
    if not ok:
        result = {"fitness": PENALTY, "policy_safety": reason, "run_status": "rejected"}
        record.update(result)
        cache[key] = result
        return PENALTY, record

    result = run_evo(hydra_overrides(cfg), RUNS_ROOT / tag, env)
    score = result.get("score")
    scored = (
        result["run_status"].startswith("exit_")
        and isinstance(score, (int, float))
        and (result.get("status_count") or {}).get("success", 0) > 0
        and result["code_safety"] in ("ok", "no_artifact")
    )
    fitness = float(score) if isinstance(score, (int, float)) and scored else PENALTY
    result.update(fitness=fitness, policy_safety="ok")
    record.update(result)
    cache[key] = result
    return fitness, record


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = load_env()
    print(f"model={env['OPENMLE_MODEL_ID']} pop={POP_SIZE} gens={GENERATIONS} seed={SEED}", flush=True)

    algo = IStratDE(lb=LOWER, ub=UPPER, pop_size=POP_SIZE)
    state = algo.init(jax.random.PRNGKey(SEED))
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"fitness": math.inf}

    for gen in range(GENERATIONS):
        pop, state = algo.ask(state)
        fits = []
        for i, row in enumerate(pop):
            tag = f"g{gen:02d}_i{i:02d}"
            fit, record = fitness_of(list(row), tag, env, cache)
            fits.append(fit)
            history.append(record)
            with LOG_PATH.open("a") as f:
                f.write(json.dumps({"gen": gen, **record}, ensure_ascii=False) + "\n")
            print(
                f"[gen {gen} #{i}] fit={fit:.4f} score={record.get('score')} "
                f"run={record.get('run_status')} wall={record.get('wall_s', '-')}s "
                f"tokens={record.get('total_tokens', '-')}",
                flush=True,
            )
            if fit < best["fitness"]:
                best = {"fitness": fit, **record}
        state = algo.tell(state, jnp.array(fits, dtype=jnp.float32))

    summary = {
        "config": {"pop": POP_SIZE, "gens": GENERATIONS, "seed": SEED, "task": TASK,
                   "model": env["OPENMLE_MODEL_ID"]},
        "evaluations": len(history),
        "unique_configs": len(cache),
        "scored": sum(1 for h in history if h.get("fitness", PENALTY) < PENALTY),
        "best": best,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

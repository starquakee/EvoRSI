"""RSI 单任务评测适配器：候选代码 -> 源码安全门 -> sandbox 评分 -> utility。

Origin: legacy rsi-gpu tree, rsi_eval_adapter.py (read-only; exact source
path and hash recorded in research/provenance/manifest.json).
Consolidated under research/adapters for US-001; only the import was changed
to a package-relative import.

安全门与评测契约固定在优化器之外；DE/RSI 只提供候选代码，不能修改本模块。
指标口径（对应 HANDOFF 第 5 节）：
    capability_score = sandbox 返回的 score
    safety_score     = 源码安全门（二元，1/0）
    cost_score       = elapsed_s（单次评测耗时，可扩展 token/GPU 时）
    utility          = 门控后的综合分，安全失败或无法计分一律 -inf
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .sandbox_eval_client import safety_gate_source, submit_and_wait


@dataclass(frozen=True)
class EvalResult:
    job_id: str | None
    status: str            # completed / failed / rejected / error
    capability_score: float | None
    safety_ok: bool
    safety_reason: str
    elapsed_s: float
    utility: float
    raw: dict = field(default_factory=dict)


def evaluate_candidate(
    source: str,
    *,
    cost_weight: float = 0.0,
    cost_reference_s: float = 300.0,
    **submit_kwargs,
) -> EvalResult:
    """评估单个候选代码。

    门控规则（阈值在实验前固定，不随搜索结果调整）：
        安全门失败            -> utility = -inf, status=rejected
        任务未 completed/无分 -> utility = -inf
        否则 utility = capability_score - cost_weight * (elapsed_s / cost_reference_s)
    """
    t0 = time.monotonic()
    ok, reason = safety_gate_source(source)
    if not ok:
        return EvalResult(
            job_id=None, status="rejected", capability_score=None,
            safety_ok=False, safety_reason=reason, elapsed_s=0.0, utility=-math.inf,
        )
    try:
        res = submit_and_wait(source, **submit_kwargs)
    except Exception as exc:  # noqa: BLE001
        return EvalResult(
            job_id=None, status="error", capability_score=None,
            safety_ok=True, safety_reason=f"submit_error:{type(exc).__name__}:{exc}",
            elapsed_s=time.monotonic() - t0, utility=-math.inf,
        )
    elapsed = time.monotonic() - t0
    if res.status != "completed" or res.score is None:
        return EvalResult(
            job_id=res.job_id, status=res.status, capability_score=res.score,
            safety_ok=True, safety_reason="job_not_scored",
            elapsed_s=elapsed, utility=-math.inf, raw=res.raw,
        )
    utility = float(res.score) - cost_weight * (elapsed / cost_reference_s)
    return EvalResult(
        job_id=res.job_id, status=res.status, capability_score=float(res.score),
        safety_ok=True, safety_reason="ok", elapsed_s=elapsed,
        utility=utility, raw=res.raw,
    )

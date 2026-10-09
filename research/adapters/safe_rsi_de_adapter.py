"""Strategy-vector bridge for JAX DE -> OpenMLE-Evo.

Origin: legacy rsi-gpu tree, safe_rsi_de_adapter.py (read-only; exact source
path and hash recorded in research/provenance/manifest.json).
Consolidated under research/adapters for US-001; behavior unchanged.

The vector controls search policy only. Safety bounds are fixed outside the optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class StrategyConfig:
    crossover_prob: float
    score_weight: float
    delta_weight: float
    novelty_weight: float
    max_related_cards: int
    ancestor_k: int
    sibling_k: int
    max_debug_depth: int


# [crossover, score, delta, novelty, cards, ancestors, siblings, debug_depth]
LOWER = jnp.array([0.05, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
UPPER = jnp.array([0.95, 2.0, 2.0, 2.0, 8.0, 6.0, 6.0, 6.0])


def decode(theta: Any) -> StrategyConfig:
    """Decode one bounded vector into an OpenMLE-Evo policy."""
    x = jnp.clip(jnp.asarray(theta, dtype=jnp.float32), LOWER, UPPER)
    return StrategyConfig(
        crossover_prob=float(x[0]),
        score_weight=float(x[1]),
        delta_weight=float(x[2]),
        novelty_weight=float(x[3]),
        max_related_cards=int(jnp.rint(x[4])),
        ancestor_k=int(jnp.rint(x[5])),
        sibling_k=int(jnp.rint(x[6])),
        max_debug_depth=int(jnp.rint(x[7])),
    )


def safety_gate(cfg: StrategyConfig) -> tuple[bool, str]:
    """Fixed safety gate; the optimizer cannot change this policy."""
    if cfg.max_debug_depth > 6:
        return False, "debug_depth_limit"
    if cfg.max_related_cards > 8:
        return False, "memory_context_limit"
    if cfg.score_weight + cfg.delta_weight + cfg.novelty_weight <= 0:
        return False, "empty_parent_selection_weights"
    return True, "ok"


def hydra_overrides(cfg: StrategyConfig) -> list[str]:
    """Convert a safe policy into OpenMLE-Evo Hydra overrides."""
    ok, reason = safety_gate(cfg)
    if not ok:
        raise ValueError(reason)
    return [
        f"search.runner.solver.crossover_prob={cfg.crossover_prob:.6f}",
        f"search.runner.solver.max_debug_depth={cfg.max_debug_depth}",
        f"search.runner.solver.experience.parent_selection.weights.score={cfg.score_weight:.6f}",
        f"search.runner.solver.experience.parent_selection.weights.delta={cfg.delta_weight:.6f}",
        f"search.runner.solver.experience.parent_selection.weights.novelty={cfg.novelty_weight:.6f}",
        f"search.runner.solver.experience.prompt_memory.max_related_cards={cfg.max_related_cards}",
        f"search.runner.solver.experience.prompt_memory.improve.ancestor_k={cfg.ancestor_k}",
        f"search.runner.solver.experience.prompt_memory.improve.sibling_k={cfg.sibling_k}",
    ]


def decode_batch(population: Any) -> list[StrategyConfig]:
    return [decode(row) for row in population]


if __name__ == "__main__":
    cfg = decode([0.5, 1.0, 0.4, 0.25, 3, 2, 2, 4])
    print(cfg)
    print("gate", safety_gate(cfg))
    print("overrides")
    print("\n".join(hydra_overrides(cfg)))

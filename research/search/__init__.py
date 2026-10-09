"""Acceptance-search evidence helpers (US-007).

- ``validation_config`` — the INNER Evo validation experiment contract
  (machine-readable constraint names) plus the OUTER iStratDE documentation
  helper. INNER generations are Evo solver generations; OUTER generations
  are DE generations over config vectors.
- ``search_vector`` — the retained/dropped search dims with bounds, hydra
  paths, real-effect documentation and the drop rationale, plus decode,
  safety gate and hydra-override rendering for the retained vector.
- ``budget_transport`` — the one-layer budget guard for model transports
  (reserve before every call, fail closed on unknown usage, never nested).
"""
from research.search.budget_transport import (
    TransportGuardConfig,
    extract_provider_usage,
    install_budget_guard,
    make_request_guard,
)
from research.search.search_vector import (
    DIMS,
    DROP_RATIONALE,
    DROPPED_DIMS,
    RETAINED_DIMS,
    SearchDim,
    by_name,
    decode_retained,
    hydra_overrides,
    mapping_explanation,
    safety_gate,
)
from research.search.validation_config import (
    InnerEvoValidationConfig,
    ValidationConfigError,
    outer_de_bounds,
    validate_inner_evo_config,
)

__all__ = [
    "DIMS",
    "DROP_RATIONALE",
    "DROPPED_DIMS",
    "RETAINED_DIMS",
    "InnerEvoValidationConfig",
    "SearchDim",
    "TransportGuardConfig",
    "ValidationConfigError",
    "by_name",
    "decode_retained",
    "extract_provider_usage",
    "hydra_overrides",
    "install_budget_guard",
    "make_request_guard",
    "mapping_explanation",
    "outer_de_bounds",
    "safety_gate",
    "validate_inner_evo_config",
]

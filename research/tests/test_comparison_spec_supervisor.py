"""Supervisor regressions: immutable, type-strict frozen experiment specs."""
from dataclasses import replace
from typing import Any

import pytest

from research.comparison.protocol import (
    ComparisonProtocol, ComparisonSpecError, FIXED_THETA,
)
from research.comparison.specs import TaskBinding


@pytest.mark.parametrize("field,value", [
    ("configs_per_cell", 8.0),
    ("inner_generations", 3.0),
    ("main_candidates_per_generation", 2.0),
    ("split_seed_base", 20261010.0),
    ("search_seeds", (11.0, 22.0, 33.0)),
])
def test_protocol_rejects_float_values_for_integer_fields(field: str, value: Any) -> None:
    # Equality with a pin is insufficient: these values later reach range(),
    # seed initializers and persisted identities with different JSON types.
    with pytest.raises(ComparisonSpecError):
        ComparisonProtocol(**{field: value})


def test_fixed_theta_cannot_be_a_mutable_list() -> None:
    # Frozen dataclass alone does not freeze nested containers. Keeping this
    # caller-owned list would allow protocol mutation after validation.
    mutable: Any = list(FIXED_THETA)
    with pytest.raises(ComparisonSpecError):
        ComparisonProtocol(fixed_theta=mutable)


@pytest.mark.parametrize("field,value", [("rows", 1372.0), ("features", 4.0)])
def test_task_shape_is_integer_typed(field: str, value: Any) -> None:
    binding = TaskBinding(
        task_id="banknote", split_seed=20261277, rows=1372, features=4,
        train_csv_sha256="1" * 64, validation_features_sha256="2" * 64,
        validation_labels_sha256="3" * 64, final_features_sha256="4" * 64,
        final_labels_sha256="5" * 64,
    )
    with pytest.raises(ComparisonSpecError):
        replace(binding, **{field: value})

"""Offline contract tests for research/comparison/specs.py (EXP-001).

Pins the typed spec validation (task binding hashes, method/seed/commit /
protocol binding), the canonical identities (any tampering changes them)
and the deterministic, method-independent inner-seed derivation.
"""
from __future__ import annotations

import hashlib

import pytest

from research.comparison.protocol import (
    CONFIGS_PER_CELL,
    METHODS,
    SEARCH_SEEDS,
    TASKS,
    CellBudget,
    ComparisonSpecError,
    ModelProfile,
    canonical_protocol,
)
from research.comparison.specs import (
    CellSpec,
    InnerRunSpec,
    TaskBinding,
    derive_inner_seed,
    iter_cells,
    iter_inner_runs,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"

# Canonical synthetic task content; every binding hash below is the REAL
# sha256 of these bytes, so fixtures that materialize the files can match
# the frozen bindings genuinely (trainval refit: validation ids offset
# past the train ids).
SYNTHETIC_TRAIN_CSV = b"id,f1,label\n0,0.4,0\n1,0.5,1\n"
SYNTHETIC_VALIDATION_FEATURES_CSV = b"id,f1\n0,0.6\n"
SYNTHETIC_VALIDATION_LABELS_CSV = b"id,label\n0,0\n"
SYNTHETIC_FINAL_FEATURES_CSV = b"id,f1\n10,0.7\n11,0.8\n12,0.9\n"
SYNTHETIC_FINAL_LABELS_CSV = b"id,label\n10,0\n11,1\n12,1\n"

_HASHES = {
    "train_csv_sha256": hashlib.sha256(SYNTHETIC_TRAIN_CSV).hexdigest(),
    "validation_features_sha256": hashlib.sha256(
        SYNTHETIC_VALIDATION_FEATURES_CSV
    ).hexdigest(),
    "validation_labels_sha256": hashlib.sha256(
        SYNTHETIC_VALIDATION_LABELS_CSV
    ).hexdigest(),
    "final_features_sha256": hashlib.sha256(SYNTHETIC_FINAL_FEATURES_CSV).hexdigest(),
    "final_labels_sha256": hashlib.sha256(SYNTHETIC_FINAL_LABELS_CSV).hexdigest(),
}


def make_binding(task_id: str, **overrides) -> TaskBinding:
    from research.comparison.protocol import TASKS_BY_ID, split_seed

    pin = TASKS_BY_ID[task_id]
    fields = {
        "task_id": task_id,
        "split_seed": split_seed(pin),
        "rows": pin.rows,
        "features": pin.features,
        **_HASHES,
        **overrides,
    }
    return TaskBinding(**fields)


def make_bindings() -> dict[str, TaskBinding]:
    return {task.task_id: make_binding(task.task_id) for task in TASKS}


def make_cell(
    task_id: str = "banknote", method: str = "FIXED", search_seed: int = 11
) -> CellSpec:
    return CellSpec(
        task_binding=make_binding(task_id),
        method=method,
        search_seed=search_seed,
        model_profile=ModelProfile(),
        source_commit=COMMIT,
        cell_budget=CellBudget(),
        protocol_identity=canonical_protocol().identity(),
    )


class TestTaskBinding:
    def test_valid_binding_constructs(self):
        binding = make_binding("wdbc")
        assert binding.split_seed == 20261010 + 17
        assert len(binding.identity()) == 64

    @pytest.mark.parametrize(
        ("overrides", "constraint"),
        [
            ({"split_seed": 20261010}, "split_seed_mismatch"),
            ({"split_seed": True}, "split_seed_mismatch"),
            ({"rows": 1373}, "task_shape_mismatch"),
            ({"features": 5}, "task_shape_mismatch"),
            ({"train_csv_sha256": "a" * 63}, "task_hash_invalid"),
            ({"train_csv_sha256": "A" * 64}, "task_hash_invalid"),
            ({"train_csv_sha256": "g" * 64}, "task_hash_invalid"),
            ({"validation_features_sha256": 7}, "task_hash_invalid"),
            ({"validation_labels_sha256": ""}, "task_hash_invalid"),
            ({"final_features_sha256": None}, "task_hash_invalid"),
            ({"final_labels_sha256": "f" * 65}, "task_hash_invalid"),
        ],
    )
    def test_invalid_binding_raises_named_constraint(self, overrides, constraint):
        with pytest.raises(ComparisonSpecError, match=constraint):
            make_binding("banknote", **overrides)

    def test_unknown_task_refused(self):
        pin_fields = {
            "task_id": "iris",
            "split_seed": 20261010 + 999,
            "rows": 150,
            "features": 4,
            **_HASHES,
        }
        with pytest.raises(ComparisonSpecError, match="unknown_task"):
            TaskBinding(**pin_fields)

    def test_hash_tampering_changes_identity(self):
        original = make_binding("banknote")
        tampered = make_binding("banknote", final_labels_sha256="0" * 64)
        assert original.identity() != tampered.identity()


class TestCellSpec:
    def test_valid_cell_constructs(self):
        cell = make_cell()
        assert cell.cell_key() == "banknote:11:FIXED"
        assert len(cell.identity()) == 64

    @pytest.mark.parametrize("method", ["fixed", "istratde", "RAND", ""])
    def test_unknown_method_refused(self, method):
        with pytest.raises(ComparisonSpecError, match="unknown_method"):
            make_cell(method=method)

    @pytest.mark.parametrize("seed", [10, 12, True, "11", 11.0])
    def test_unknown_search_seed_refused(self, seed):
        with pytest.raises(ComparisonSpecError, match="unknown_search_seed"):
            make_cell(search_seed=seed)

    @pytest.mark.parametrize(
        "commit", ["xyz", "0" * 39, "0" * 41, "G" * 40, 7, ""]
    )
    def test_source_commit_validation(self, commit):
        with pytest.raises(ComparisonSpecError, match="source_commit_invalid"):
            CellSpec(
                task_binding=make_binding("banknote"),
                method="FIXED",
                search_seed=11,
                model_profile=ModelProfile(),
                source_commit=commit,
                cell_budget=CellBudget(),
                protocol_identity=canonical_protocol().identity(),
            )

    def test_protocol_identity_mismatch_refused(self):
        with pytest.raises(
            ComparisonSpecError, match="protocol_identity_mismatch"
        ):
            CellSpec(
                task_binding=make_binding("banknote"),
                method="FIXED",
                search_seed=11,
                model_profile=ModelProfile(),
                source_commit=COMMIT,
                cell_budget=CellBudget(),
                protocol_identity="0" * 64,
            )

    def test_non_pinned_profile_or_budget_refused(self):
        with pytest.raises(
            ComparisonSpecError, match="model_profile_not_pinned"
        ):
            CellSpec(
                task_binding=make_binding("banknote"),
                method="FIXED",
                search_seed=11,
                model_profile=ModelProfile(temperature=0.5),
                source_commit=COMMIT,
                cell_budget=CellBudget(),
                protocol_identity=canonical_protocol().identity(),
            )
        with pytest.raises(ComparisonSpecError, match="cell_budget_not_pinned"):
            CellSpec(
                task_binding=make_binding("banknote"),
                method="FIXED",
                search_seed=11,
                model_profile=ModelProfile(),
                source_commit=COMMIT,
                cell_budget=CellBudget(max_requests=63),
                protocol_identity=canonical_protocol().identity(),
            )

    def test_identity_binds_all_fields(self):
        cell = make_cell()
        assert make_cell(method="RANDOM").identity() != cell.identity()
        assert make_cell(search_seed=22).identity() != cell.identity()
        assert make_cell(task_id="wdbc").identity() != cell.identity()
        tampered_binding_cell = CellSpec(
            task_binding=make_binding("banknote", train_csv_sha256="0" * 64),
            method="FIXED",
            search_seed=11,
            model_profile=ModelProfile(),
            source_commit=COMMIT,
            cell_budget=CellBudget(),
            protocol_identity=canonical_protocol().identity(),
        )
        assert tampered_binding_cell.identity() != cell.identity()
        other_commit = "f" * 40
        assert (
            CellSpec(
                task_binding=make_binding("banknote"),
                method="FIXED",
                search_seed=11,
                model_profile=ModelProfile(),
                source_commit=other_commit,
                cell_budget=CellBudget(),
                protocol_identity=canonical_protocol().identity(),
            ).identity()
            != cell.identity()
        )


class TestInnerRunSpec:
    def test_valid_run_constructs(self):
        run = make_cell().inner_run(3)
        assert run.run_key() == "banknote:11:FIXED:3"
        assert run.inner_seed == derive_inner_seed("banknote", 11, 3)

    @pytest.mark.parametrize("index", [-1, 8, 100, True, 1.5, "0"])
    def test_config_index_bounds(self, index):
        with pytest.raises(
            ComparisonSpecError, match="config_index_out_of_range"
        ):
            make_cell().inner_run(index)

    def test_seed_is_derived_never_chosen(self):
        cell = make_cell()
        with pytest.raises(ComparisonSpecError, match="inner_seed_mismatch"):
            InnerRunSpec(cell=cell, config_index=0, inner_seed=12345)
        with pytest.raises(ComparisonSpecError, match="inner_seed_mismatch"):
            InnerRunSpec(cell=cell, config_index=0, inner_seed=True)

    def test_inner_seed_method_independent_at_same_position(self):
        seeds = {
            method: make_cell(method=method).inner_run(4).inner_seed
            for method in METHODS
        }
        assert len(set(seeds.values())) == 1

    def test_inner_seed_varies_by_position_task_and_search_seed(self):
        base = derive_inner_seed("banknote", 11, 0)
        assert derive_inner_seed("banknote", 11, 1) != base
        assert derive_inner_seed("banknote", 22, 0) != base
        assert derive_inner_seed("wdbc", 11, 0) != base
        assert derive_inner_seed("banknote", 11, 0) == base  # deterministic
        assert 0 <= base < 2**32


class TestEnumeration:
    def test_iter_cells_yields_27_in_canonical_order(self):
        cells = iter_cells(make_bindings(), COMMIT)
        assert len(cells) == 27
        assert cells[0].cell_key() == "banknote:11:FIXED"
        assert cells[-1].cell_key() == "spambase:33:ISTRATDE"
        # Canonical order: task (PRD order) -> search seed -> method.
        expected = [
            f"{task.task_id}:{seed}:{method}"
            for task in TASKS
            for seed in SEARCH_SEEDS
            for method in METHODS
        ]
        assert [cell.cell_key() for cell in cells] == expected
        assert len({cell.identity() for cell in cells}) == 27

    def test_iter_inner_runs_yields_216_unique_runs(self):
        runs = iter_inner_runs(iter_cells(make_bindings(), COMMIT))
        assert len(runs) == 216
        assert len({run.run_key() for run in runs}) == 216
        assert len({run.identity() for run in runs}) == 216
        # Method-independent by design: each (task, search seed, config)
        # position's inner seed is shared by exactly the three methods.
        assert len({run.inner_seed for run in runs}) == 72
        for run in runs:
            assert 0 <= run.config_index < CONFIGS_PER_CELL

    def test_iter_cells_requires_exactly_one_binding_per_task(self):
        bindings = make_bindings()
        del bindings["wdbc"]
        with pytest.raises(
            ComparisonSpecError, match="task_bindings_incomplete"
        ):
            iter_cells(bindings, COMMIT)
        extra = {**make_bindings(), "iris": make_binding("banknote")}
        with pytest.raises(
            ComparisonSpecError, match="task_bindings_incomplete"
        ):
            iter_cells(extra, COMMIT)

    def test_iter_cells_rejects_miskeyed_binding(self):
        bindings = make_bindings()
        bindings["wdbc"] = make_binding("banknote")
        with pytest.raises(
            ComparisonSpecError, match="task_binding_key_mismatch"
        ):
            iter_cells(bindings, COMMIT)

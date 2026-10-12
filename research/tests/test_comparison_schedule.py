"""Contract tests for the frozen 216-position schedule (EXP-006).

Pins the deterministic rotation, the initial-before-feedback ordering, the
frozen outer seed derivations, and fail-closed freeze/load semantics.
"""
from __future__ import annotations

import json

import pytest

from research.comparison.protocol import (
    CONFIGS_PER_CELL,
    METHODS,
    NUM_INNER_RUNS,
    SEARCH_SEEDS,
    TASKS,
    ComparisonSpecError,
    canonical_protocol,
)
from research.comparison.schedule import (
    INITIAL_BATCH_CONFIGS,
    build_schedule,
    config_batch,
    derive_native_seed,
    derive_random_remainder_seed,
    freeze_schedule,
    load_schedule,
    render_schedule,
    schedule_identity,
    schedule_payload,
)
from research.comparison.specs import iter_cells

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _bindings() -> dict:
    from research.tests.test_comparison_specs import make_bindings

    return make_bindings()


def _cells(commit: str = COMMIT):
    return iter_cells(_bindings(), commit)


def test_schedule_has_216_unique_positions():
    positions = build_schedule(_cells())
    assert len(positions) == NUM_INNER_RUNS == 216
    assert [p.position for p in positions] == list(range(216))
    assert len({p.run_key for p in positions}) == 216


def test_schedule_covers_exact_roster_once_per_config():
    positions = build_schedule(_cells())
    per_cell: dict[str, list[int]] = {}
    for pos in positions:
        per_cell.setdefault(pos.cell_key, []).append(pos.config_index)
    assert len(per_cell) == 27
    for cell_key, indices in per_cell.items():
        assert sorted(indices) == list(range(CONFIGS_PER_CELL)), cell_key


def test_rotation_covers_all_methods_and_is_deterministic():
    positions = build_schedule(_cells())
    # Each (config index, task/seed block) slot runs all three methods.
    for config_index in range(CONFIGS_PER_CELL):
        slot = [p for p in positions if p.config_index == config_index]
        assert len(slot) == 27
        for i in range(0, 27, 3):
            methods = {p.cell_key.rsplit(":", 1)[1] for p in slot[i : i + 3]}
            assert methods == set(METHODS)
    # Rotation actually rotates: the method order of a slot is not constant.
    first_block_orders = set()
    for config_index in range(CONFIGS_PER_CELL):
        slot = [p for p in positions if p.config_index == config_index][:3]
        first_block_orders.add(tuple(p.cell_key.rsplit(":", 1)[1] for p in slot))
    assert len(first_block_orders) > 1
    # Build is deterministic.
    assert [p.to_dict() for p in build_schedule(_cells())] == [
        p.to_dict() for p in positions
    ]


def test_initial_batch_precedes_feedback_batch_per_cell():
    positions = build_schedule(_cells())
    per_cell: dict[str, list] = {}
    for pos in positions:
        per_cell.setdefault(pos.cell_key, []).append(pos)
    for cell_key, cell_positions in per_cell.items():
        by_index = {p.config_index: p.position for p in cell_positions}
        assert sorted(by_index) == list(range(8))
        # Per-cell launch order is config-index order.
        assert [p.position for p in sorted(cell_positions, key=lambda p: p.position)] == [
            by_index[i] for i in range(8)
        ]
        assert max(by_index[i] for i in range(INITIAL_BATCH_CONFIGS)) < min(
            by_index[i] for i in range(INITIAL_BATCH_CONFIGS, CONFIGS_PER_CELL)
        ), cell_key


def test_batch_labels():
    assert config_batch(0) == "initial"
    assert config_batch(3) == "initial"
    assert config_batch(4) == "feedback"
    assert config_batch(7) == "feedback"
    with pytest.raises(ComparisonSpecError, match="config_index_out_of_range"):
        config_batch(8)
    with pytest.raises(ComparisonSpecError, match="config_index_out_of_range"):
        config_batch(True)


def test_outer_seed_derivations_method_independent_and_frozen():
    for task in TASKS:
        for seed in SEARCH_SEEDS:
            native = derive_native_seed(task.task_id, seed)
            remainder = derive_random_remainder_seed(task.task_id, seed)
            assert 0 <= native < 2**32
            assert 0 <= remainder < 2**32
            # Frozen payload records the same values.
    payload = schedule_payload(build_schedule(_cells()), COMMIT)
    assert payload["protocol_identity"] == canonical_protocol().identity()
    assert payload["source_commit"] == COMMIT
    for task in TASKS:
        for seed in SEARCH_SEEDS:
            key = f"{task.task_id}:{seed}"
            assert payload["native_seeds"][key] == derive_native_seed(task.task_id, seed)
            assert payload["random_remainder_seeds"][key] == derive_random_remainder_seed(
                task.task_id, seed
            )
    # 9 distinct native seeds and 9 distinct remainder seeds.
    assert len(set(payload["native_seeds"].values())) == 9
    assert len(set(payload["random_remainder_seeds"].values())) == 9
    with pytest.raises(ComparisonSpecError, match="unknown_task"):
        derive_native_seed("nope", 11)
    with pytest.raises(ComparisonSpecError, match="unknown_search_seed"):
        derive_native_seed("banknote", 12)


def test_freeze_and_load_roundtrip(tmp_path):
    cells = _cells()
    path = tmp_path / "schedule.json"
    positions = freeze_schedule(path, cells)
    assert load_schedule(path, cells) == positions
    # Idempotent refreeze with identical bytes is accepted.
    assert freeze_schedule(path, cells) == positions


def test_freeze_refuses_different_bytes(tmp_path):
    cells = _cells()
    path = tmp_path / "schedule.json"
    freeze_schedule(path, cells)
    other_commit = "f" * 40
    with pytest.raises(ComparisonSpecError, match="schedule_already_frozen_mismatch"):
        freeze_schedule(path, _cells(other_commit))


def test_load_rejects_tampered_schedule(tmp_path):
    cells = _cells()
    path = tmp_path / "schedule.json"
    freeze_schedule(path, cells)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Swap two positions: order tampering must be detected.
    payload["positions"][0], payload["positions"][1] = (
        payload["positions"][1],
        payload["positions"][0],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ComparisonSpecError, match="schedule_mismatch"):
        load_schedule(path, cells)


def test_load_rejects_tampered_seed(tmp_path):
    cells = _cells()
    path = tmp_path / "schedule.json"
    freeze_schedule(path, cells)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["native_seeds"]["banknote:11"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ComparisonSpecError, match="schedule_mismatch"):
        load_schedule(path, cells)


def test_load_rejects_missing_and_malformed(tmp_path):
    cells = _cells()
    with pytest.raises(ComparisonSpecError, match="schedule_missing"):
        load_schedule(tmp_path / "absent.json", cells)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ComparisonSpecError, match="schedule_malformed"):
        load_schedule(bad, cells)


def test_roster_validation(tmp_path):
    cells = list(_cells())
    with pytest.raises(ComparisonSpecError, match="schedule_roster_invalid"):
        build_schedule(cells[:-1])
    with pytest.raises(ComparisonSpecError, match="schedule_roster_invalid"):
        build_schedule(cells + [cells[0]])
    mixed = cells[:-1] + [iter_cells(_bindings(), "f" * 40)[-1]]
    with pytest.raises(ComparisonSpecError, match="source_commit_not_uniform"):
        freeze_schedule(tmp_path / "mixed.json", mixed)


def test_schedule_identity_binds_commit_and_positions():
    cells = _cells()
    positions = build_schedule(cells)
    identity = schedule_identity(positions, COMMIT)
    assert identity != schedule_identity(positions, "f" * 40)
    swapped = [positions[1], positions[0], *positions[2:]]
    swapped = tuple(
        type(p)(position=i, cell_key=p.cell_key, config_index=p.config_index, batch=p.batch)
        for i, p in enumerate(swapped)
    )
    assert identity != schedule_identity(swapped, COMMIT)
    # Render is byte-stable.
    assert render_schedule(positions, COMMIT) == render_schedule(
        build_schedule(_cells()), COMMIT
    )

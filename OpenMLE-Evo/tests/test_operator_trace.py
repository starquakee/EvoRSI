"""US-007 operator-trace evidence tests (offline, deterministic, seeded).

Every test runs the REAL ``Evolutionary.search`` loop (generation mode) with
fake GenericLLMs and a fake task (see ``operator_trace_harness``); the trace
is written by the real ``OperatorTraceWriter`` through the solver's
``operator_tracer`` hook. No model/API/sandbox calls happen anywhere.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path

import numpy
import pytest

import operator_trace_harness as h
from dojo.solvers.evo.operator_trace import (
    OperatorTraceWriter,
    sha256_hex,
    validate_operator_event,
)


def _run_search(tmp_path, monkeypatch, *, responder=None, evaluator=None,
                overrides=None, seed=h.DEFAULT_SEED):
    # Fresh subdirectory per invocation: the checkpoint dir and the
    # append-only trace file must never leak between runs.
    run_dir = Path(tmp_path) / f"run-{_RUN_COUNTER[0]}"
    _RUN_COUNTER[0] += 1
    solver, llms = h.build_solver(
        run_dir, monkeypatch, responder=responder, overrides=overrides, seed=seed
    )
    trace_path = run_dir / "trace.jsonl"
    solver.operator_tracer = OperatorTraceWriter(trace_path)
    task = h.FakeTask(evaluator)
    solver.search(task, {})
    return solver, llms, task, h.read_trace(trace_path)


_RUN_COUNTER = [0]


def _non_root_nodes(solver):
    return [n for n in solver.journal.nodes if not solver.journal.is_root_node(n)]


def test_three_generations_two_candidates_real_trace(tmp_path, monkeypatch):
    solver, llms, task, events = _run_search(
        tmp_path,
        monkeypatch,
        overrides={
            "num_generations": 3,
            "individuals_per_generation": 2,
            "num_generations_till_crossover": 1,
            "crossover_prob": 0.5,
            "num_islands": 1,
            "step_limit": 12,
        },
    )

    non_root = _non_root_nodes(solver)
    # 1 root + exactly 6 candidate nodes (3 gens x 2 individuals, all valid,
    # no debug nodes because every candidate scored).
    assert len(solver.journal.nodes) == 7
    assert len(non_root) == 6
    # No artificial trace entries: one event per non-root journal node.
    assert len(events) == len(non_root)

    # Multiset of traced operators == multiset of journal operators_used[0].
    trace_ops = Counter(e["operator"] for e in events)
    journal_ops = Counter(n.operators_used[0] for n in non_root)
    assert trace_ops == journal_ops

    # Every traced child hash matches the node's actual code.
    by_id = {n.id: n for n in non_root}
    assert {e["child_node_id"] for e in events} == set(by_id)
    for event in events:
        node = by_id[event["child_node_id"]]
        assert event["child_code_sha256"] == sha256_hex(node.code)
        assert event["parent_node_ids"] == [p.id for p in node.parents]
        for parent, digest in zip(node.parents, event["parent_code_sha256"]):
            assert digest == sha256_hex(parent.code or "")

    # Generation 0 is draft-only; every event passes the fail-closed validator;
    # the config hash is stable across all events.
    assert all(e["operator"] == "draft" for e in events if e["generation_id"] == 0)
    assert sum(1 for e in events if e["generation_id"] == 0) == 2
    for event in events:
        validate_operator_event(event)
    assert len({e["config"]["config_sha256"] for e in events}) == 1

    # Deterministic metadata path only: analyze LLM never invoked.
    assert llms["analyze"].call_tracker == 0


def test_before_threshold_blocks_crossover(tmp_path, monkeypatch):
    # Threshold beyond the last generation: crossover never fires even though
    # crossover_prob is 1.0 and valid parents exist.
    _, _, _, events = _run_search(
        tmp_path,
        monkeypatch,
        overrides={
            "num_generations": 3,
            "individuals_per_generation": 2,
            "num_generations_till_crossover": 3,
            "crossover_prob": 1.0,
            "max_debug_depth": 0,
        },
    )
    assert events
    assert all(e["operator"] != "crossover" for e in events)
    assert any(e["operator"] == "improve" for e in events if e["generation_id"] >= 1)

    # Reachable threshold with prob 1.0: the coin flip always picks crossover
    # and both gen0 parents are available, so crossover MUST fire in gen >= 1.
    _, _, _, events2 = _run_search(
        tmp_path / "second",
        monkeypatch,
        overrides={
            "num_generations": 3,
            "individuals_per_generation": 2,
            "num_generations_till_crossover": 1,
            "crossover_prob": 1.0,
            "max_debug_depth": 0,
        },
    )
    crossover_events = [
        e for e in events2 if e["operator"] == "crossover" and e["generation_id"] >= 1
    ]
    assert crossover_events
    for event in crossover_events:
        assert len(event["parent_node_ids"]) == 2
        validate_operator_event(event)


def test_no_valid_parents_falls_back_to_draft(tmp_path, monkeypatch):
    # ALL candidates buggy: islands stay empty, so gen >= 1 always drafts via
    # the empty-population branch (last_parent_selection is None there).
    _, _, _, events = _run_search(
        tmp_path,
        monkeypatch,
        evaluator=lambda code, i: h.failure_result(),
        overrides={"num_generations": 3, "individuals_per_generation": 2,
                   "max_debug_depth": 0},
    )
    gen1_plus = [e for e in events if e["generation_id"] >= 1]
    assert gen1_plus
    assert all(e["operator"] == "draft" for e in gen1_plus)
    assert all(e["selection"] is None for e in gen1_plus)

    # Same "no eligible parent context" decision via the explicit reason path:
    # few_shot requires 2 parents for both improve and crossover while the
    # island holds exactly 1 valid node (only the first draft scores), so
    # sample_in_context records reason == "no_eligible_parent_context".
    def one_valid_evaluator(code, i):
        if "draft-1" in code:
            return h.success_result(0.5)
        return h.failure_result()

    _, _, _, events2 = _run_search(
        tmp_path,
        monkeypatch,
        evaluator=one_valid_evaluator,
        overrides={
            "num_generations": 3,
            "individuals_per_generation": 2,
            "max_debug_depth": 0,
            "crossover_prob": 1.0,
            "few_shot": {"improve": 2, "crossover": 2},
        },
    )
    gen1_plus2 = [e for e in events2 if e["generation_id"] >= 1]
    assert gen1_plus2
    assert all(e["operator"] == "draft" for e in gen1_plus2)
    reasons = {
        (e["selection"] or {}).get("reason") for e in gen1_plus2
    }
    assert reasons == {"no_eligible_parent_context"}


def test_crossover_parent_shortage_falls_back_to_improve(tmp_path, monkeypatch):
    # Exactly one valid gen0 parent (the second candidate fails and
    # max_debug_depth=0 keeps it out of the island). With crossover_prob=1.0
    # the coin flip always wants crossover, but the island has < 2 parents,
    # so the REAL fallback branch yields improve + the documented reason.
    # Improve children are scripted buggy, so the island stays at size 1 and
    # every later generation hits the same fallback branch.
    def shortage_responder(operator, call_index, query_data):
        if operator == "improve":
            code = (
                "FAIL = True\n"
                f"MARKER = \"improve-{call_index}\"\n"
            )
            return f"Improve plan.\n```python\n{code}```\n"
        return h.default_responder(operator, call_index, query_data)

    def one_valid_evaluator(code, i):
        if "draft-2" in code or "FAIL" in code:
            return h.failure_result()
        return h.success_result(0.5)

    _, _, _, events = _run_search(
        tmp_path,
        monkeypatch,
        responder=shortage_responder,
        evaluator=one_valid_evaluator,
        overrides={
            "num_generations": 3,
            "individuals_per_generation": 2,
            "num_generations_till_crossover": 1,
            "crossover_prob": 1.0,
            "max_debug_depth": 0,
        },
    )
    gen1_plus = [e for e in events if e["generation_id"] >= 1]
    assert gen1_plus
    assert all(e["operator"] == "improve" for e in gen1_plus)
    reasons = {
        (e["selection"] or {}).get("operator_fallback_reason") for e in gen1_plus
    }
    assert reasons == {"crossover_parent_shortage"}


def _family_score_responder(operator, call_index, query_data):
    """Drafts alternate lightgbm/sklearn families with scripted scores;
    improve emits family-less ("unknown") code whose score derives from the
    parent's family, so by gen2 the island families are lgbm x1, sklearn x1,
    unknown x2 and all three utility components (score/delta/novelty) rank
    the candidates differently:
    - score-only  -> improve-of-lightgbm node (0.91, top score)
    - delta-only  -> improve-of-sklearn node (+0.75 delta vs +0.01)
    - novelty-only -> a gen0 draft (novelty 1.0 vs 0.577 for unknown family)
    """
    if operator == "draft":
        if call_index % 2 == 1:
            code = (
                "import lightgbm\n"
                "SCORE = 0.9\n"
                f"MARKER = \"draft-{call_index}\"\n"
            )
        else:
            code = (
                "import sklearn\n"
                "SCORE = 0.1\n"
                f"MARKER = \"draft-{call_index}\"\n"
            )
        return f"Draft plan.\n```python\n{code}```\n"
    if operator == "improve":
        prev = str(query_data.get("prev_code") or "")
        if "lightgbm" in prev:
            score = 0.91  # small improvement over 0.9
        else:
            score = 0.85  # large improvement over 0.1
        code = (
            f"SCORE = {score}\n"
            f"MARKER = \"improve-{call_index}\"\n"
        )
        return f"Improve plan.\n```python\n{code}```\n"
    return h.default_responder(operator, call_index, query_data)


def _selected_parent_fingerprints(events):
    """Deterministic fingerprint of every improve/crossover parent choice:
    (method_family, fitness) per selected candidate of each traced event."""
    fingerprints = []
    for event in events:
        if event["operator"] not in ("improve", "crossover"):
            continue
        selection = event["selection"] or {}
        selected_ids = set(selection.get("selected_node_ids") or [])
        chosen = [
            (c["method_family_auto"], c["fitness"])
            for c in selection.get("candidates") or []
            if c["node_id"] in selected_ids
        ]
        fingerprints.append(tuple(sorted(chosen)))
    return fingerprints


def test_param_crossover_prob_changes_operator_branch(tmp_path, monkeypatch):
    overrides = {
        "num_generations": 3,
        "individuals_per_generation": 2,
        "num_generations_till_crossover": 1,
        "max_debug_depth": 0,
    }

    def counts(prob):
        _, _, _, events = _run_search(
            tmp_path / f"p{prob}",
            monkeypatch,
            overrides={**overrides, "crossover_prob": prob},
        )
        return Counter(
            e["operator"] for e in events if e["generation_id"] >= 1
        )

    counts_zero_1 = counts(0.0)
    counts_zero_2 = counts(0.0)
    counts_one_1 = counts(1.0)
    counts_one_2 = counts(1.0)

    # Deterministic under the same seed.
    assert counts_zero_1 == counts_zero_2
    assert counts_one_1 == counts_one_2
    # prob 0.0 -> improve only; prob 1.0 -> crossover present.
    assert counts_zero_1["crossover"] == 0
    assert counts_zero_1["improve"] > 0
    assert counts_one_1["crossover"] > 0
    assert counts_zero_1 != counts_one_1


def test_param_max_debug_depth_changes_debug_calls(tmp_path, monkeypatch):
    # Drafts always fail; the debug operator's code passes. So debug events
    # exist iff max_debug_depth > 0, and their count equals the real number
    # of debug attempts (each first debug attempt fixes its candidate).
    def responder(operator, call_index, query_data):
        if operator == "draft":
            code = (
                "FAIL = True\n"
                f"MARKER = \"draft-{call_index}\"\n"
            )
            return f"Draft plan.\n```python\n{code}```\n"
        if operator == "debug":
            code = (
                "SCORE = 0.5\n"
                f"MARKER = \"debug-{call_index}\"\n"
            )
            return f"Debug plan.\n```python\n{code}```\n"
        return h.default_responder(operator, call_index, query_data)

    _, llms0, _, events0 = _run_search(
        tmp_path / "depth0",
        monkeypatch,
        responder=responder,
        overrides={"num_generations": 2, "individuals_per_generation": 2,
                   "max_debug_depth": 0},
    )
    debug0 = [e for e in events0 if e["operator"] == "debug"]
    assert debug0 == []
    assert llms0["debug"].call_tracker == 0

    _, llms2, _, events2 = _run_search(
        tmp_path / "depth2",
        monkeypatch,
        responder=responder,
        overrides={"num_generations": 2, "individuals_per_generation": 2,
                   "max_debug_depth": 2},
    )
    debug2 = [e for e in events2 if e["operator"] == "debug"]
    assert len(debug2) > 0
    # Trace count == actual debug LLM attempts, one per traced debug node.
    assert len(debug2) == llms2["debug"].call_tracker
    for event in debug2:
        assert len(event["parent_node_ids"]) == 1
        validate_operator_event(event)


@pytest.mark.parametrize(
    "weights_a,weights_b",
    [
        ({"score": 1.0, "delta": 0.0, "novelty": 0.0},
         {"score": 0.0, "delta": 1.0, "novelty": 0.0}),
        ({"score": 1.0, "delta": 0.0, "novelty": 0.0},
         {"score": 0.0, "delta": 0.0, "novelty": 1.0}),
        ({"score": 0.0, "delta": 1.0, "novelty": 0.0},
         {"score": 0.0, "delta": 0.0, "novelty": 1.0}),
    ],
    ids=["score-vs-delta", "score-vs-novelty", "delta-vs-novelty"],
)
def test_param_parent_selection_weights_change_choice(
    tmp_path, monkeypatch, weights_a, weights_b
):
    """Each parent-selection weight (score/delta/novelty) provably changes
    which parent is sampled, deterministically.

    Candidate design (see _family_score_responder):
    - score: lightgbm line scores 0.9 -> 0.91, sklearn line 0.1 -> 0.85, so
      score-only ranks the lightgbm improve node first.
    - delta: sklearn improve node gains +0.75 vs +0.01 for lightgbm, so
      delta-only ranks the sklearn improve node first.
    - novelty: gen1 improve nodes carry no family imports ("unknown"), so by
      gen2 the island families are lgbm x1 / sklearn x1 / unknown x2 and
      novelty-only ranks the gen0 drafts (novelty 1.0) above the improve
      nodes (novelty 1/sqrt(3) ~ 0.577).
    The temperature is sharpened (0.01) so softmax sampling is essentially
    argmax, making the seeded choice differ between weight settings.
    """
    overrides = {
        "num_generations": 3,
        "individuals_per_generation": 2,
        "num_generations_till_crossover": 3,  # improve-only sampling (1 parent)
        "crossover_prob": 0.0,
        "max_debug_depth": 0,
        "initial_temp": 0.01,
        "final_temp": 0.01,
    }

    def run(weights):
        _, _, _, events = _run_search(
            tmp_path / f"w{len(weights)}{weights['score']}{weights['delta']}",
            monkeypatch,
            responder=_family_score_responder,
            overrides={
                **overrides,
                "experience": {
                    "enabled": True,
                    "prompt_memory": {"enabled": False},
                    "parent_selection": {"enabled": True, "weights": weights},
                },
            },
        )
        return events

    events_a_first = run(weights_a)
    events_a_second = run(weights_a)
    events_b = run(weights_b)

    fp_a_first = _selected_parent_fingerprints(events_a_first)
    fp_a_second = _selected_parent_fingerprints(events_a_second)
    fp_b = _selected_parent_fingerprints(events_b)

    assert fp_a_first, "expected at least one traced parent selection"
    # Deterministic under identical seeds/config.
    assert fp_a_first == fp_a_second
    # The weight change provably alters the sampled parents.
    assert fp_a_first != fp_b

    # The utility components that drove the difference are recorded in the
    # trace for every experienced selection.
    for event in events_a_first:
        selection = event["selection"] or {}
        if event["operator"] in ("improve", "crossover") and selection.get("enabled"):
            assert selection["candidates"]
            for candidate in selection["candidates"]:
                assert "score_component" in candidate
                assert "delta_component" in candidate
                assert "novelty_component" in candidate
                assert "probability" in candidate


class _StopSearch(RuntimeError):
    """Same exception path as single_task_runner.StopSearch: raised inside a
    guarded operator call, caught outside solver.search."""


def _install_stop_guards(solver, task, llm_call_counter):
    """Mirror single_task_runner.guard_operator_call (:1658-1669)."""

    def guard(fn):
        def wrapped(*args, **kwargs):
            if task.stop_requested:
                raise _StopSearch("Search budget reached.")
            return fn(*args, **kwargs)

        return wrapped

    solver._draft = guard(solver._draft)
    solver._improve = guard(solver._improve)
    solver._debug = guard(solver._debug)
    solver._crossover = guard(solver._crossover)


def test_cancellation_stops_and_checkpoints(tmp_path, monkeypatch):
    stop_after_evals = 3

    class StopTask(h.FakeTask):
        def step_task(self, state, code):
            state, result = super().step_task(state, code)
            if self.eval_count >= stop_after_evals:
                self.stop_requested = True
            return state, result

    solver, llms = h.build_solver(tmp_path, monkeypatch)
    trace_path = Path(tmp_path) / "trace.jsonl"
    solver.operator_tracer = OperatorTraceWriter(trace_path)
    task = StopTask()
    _install_stop_guards(solver, task, llms)

    llm_calls_at_stop = {}

    def snapshot_counts():
        return {name: llm.call_tracker for name, llm in llms.items()}

    original_draft = solver._draft

    with pytest.raises(_StopSearch):
        try:
            solver.search(task, {})
        except _StopSearch:
            llm_calls_at_stop.update(snapshot_counts())
            solver.save_checkpoint()  # runner catches StopSearch -> checkpoint
            raise

    # Search terminated early (before all 3 generations completed).
    assert solver.state.current_generation < solver.cfg.num_generations
    # No fake-LLM calls happened after the stop was requested.
    assert snapshot_counts() == llm_calls_at_stop
    # Checkpoint files exist and load.
    checkpoint_dir = Path(solver.cfg.checkpoint_path)
    assert (checkpoint_dir / "state.json").exists()
    assert (checkpoint_dir / "journal.jsonl").exists()
    events = h.read_trace(trace_path)
    non_root = _non_root_nodes(solver)
    # Trace events exist only for actually-created nodes.
    assert len(events) == len(non_root)
    for event in events:
        validate_operator_event(event)


def test_checkpoint_resume_continues_without_duplicate_trace(tmp_path, monkeypatch):
    solver, llms = h.build_solver(
        tmp_path, monkeypatch, overrides={"step_limit": 7}
    )
    trace_path = Path(tmp_path) / "trace.jsonl"
    solver.operator_tracer = OperatorTraceWriter(trace_path)
    task = h.FakeTask()
    solver.search(task, {})

    journal_size = len(solver.journal.nodes)
    journal_ops = [list(n.operators_used) for n in solver.journal.nodes]
    saved_step = solver.state.current_step
    assert journal_size == 7  # 1 root + 6 candidates
    assert saved_step == journal_size

    # Second instance loads the checkpoint and rebuilds the journal.
    solver2, llms2 = h.build_solver(tmp_path, monkeypatch)
    solver2.load_checkpoint()
    assert len(solver2.journal.nodes) == journal_size
    assert [list(n.operators_used) for n in solver2.journal.nodes] == journal_ops
    assert solver2.state.current_step == saved_step

    # Resuming with the runner-style budget guard stops immediately (the step
    # budget is spent): no new LLM calls and no re-emitted trace events.
    resume_trace = Path(tmp_path) / "trace-resume.jsonl"
    solver2.operator_tracer = OperatorTraceWriter(resume_trace)
    _install_stop_guards(solver2, task, llms2)
    task.stop_requested = True
    with pytest.raises(_StopSearch):
        solver2.search(task, {})
    assert all(llm.call_tracker == 0 for llm in llms2.values())
    assert h.read_trace(resume_trace) == []

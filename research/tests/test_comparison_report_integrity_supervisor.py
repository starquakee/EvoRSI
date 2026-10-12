"""Offline counterexamples: reporting cannot turn unverified claims into evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.comparison import report, review
from research.comparison.specs import iter_cells
from research.tests.test_comparison_orchestrate import ScriptedNative
from research.tests.test_comparison_specs import COMMIT, make_bindings


def _bytes_state(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()
    }


@pytest.mark.parametrize('explicit_provider', [False, True])
def test_bare_final_claims_never_establish_observed_full_matrix(
    tmp_path: Path, explicit_provider: bool,
) -> None:
    runtime = tmp_path / 'forged'
    runtime.mkdir()
    bindings = make_bindings()
    cells = iter_cells(bindings, COMMIT)
    (runtime / review.MANIFEST_FILENAME).write_text(json.dumps({
        'schema': review.MANIFEST_SCHEMA,
        'source_commit': COMMIT,
        'cells': {cell.cell_key(): cell.identity() for cell in cells},
    }))
    for cell in cells:
        slug = cell.cell_key().replace(':', '-')
        cell_dir = runtime / 'cells' / slug
        cell_dir.mkdir(parents=True)
        (cell_dir / 'cell.json').write_text(json.dumps({
            'terminal': {'status': 'frozen'}, 'configs': {},
        }))
        final_dir = runtime / 'final' / slug
        final_dir.mkdir(parents=True)
        (final_dir / 'final.json').write_text(json.dumps({
            'status': 'completed',
            'raw_score': 1.0 if cell.method == 'ISTRATDE' else 0.0,
        }))
    before = _bytes_state(runtime)
    try:
        result = report.build_full_report(
            tmp_path, runtime_dir=runtime, bindings=bindings,
            provider=ScriptedNative() if explicit_provider else None,
        )
    except report.ComparisonReportError:
        # An explicit refusal is also safe; the diagnostic bundle may
        # instead preserve unverified claims, with no accepted outcomes.
        pass
    else:
        assert not result['matrix']['complete']
        assert result['matrix']['observed_final_cells'] == 0
        assert result['conclusion']['conclusion'] == 'insufficient-evidence'
        for contrast in result['analysis']['contrasts'].values():
            assert contrast['observed_pairs'] == 0
            assert contrast['estimate'] is None
            assert contrast['interval'] is None
    assert _bytes_state(runtime) == before


def test_report_output_file_symlink_cannot_overwrite_original_evidence(tmp_path: Path) -> None:
    original = tmp_path / 'original-ledger.json'
    original.write_bytes(b'original ledger evidence\n')
    output = tmp_path / 'bundle'
    output.mkdir()
    (output / 'report.json').symlink_to(original)
    before = original.read_bytes()
    try:
        report.write_report_bundle({}, output, markdown='fixture report')
    except (report.ComparisonReportError, OSError):
        pass
    assert original.read_bytes() == before



@pytest.mark.parametrize('status', ['gate_rejected', 'completed'])
def test_zero_failure_utility_is_not_an_observed_validation_accuracy(status: str) -> None:
    cell = iter_cells(make_bindings(), COMMIT)[0]
    state = {'configs': {'0': {
        'status': status, 'position': 0, 'config_index': 0,
        'evaluated': True, 'raw_score': None, 'utility': 0.0,
    }}}
    curve = report._incumbent_curve(cell, state, {})
    assert all(point['validation_incumbent'] is None for point in curve)



# Reuse the existing evidence-complete synthetic fixture. These are
# protocol/pin tampering tests, not model or sandbox experiments.
from research.tests.test_comparison_audit import completed, _audit, _copy


@pytest.mark.parametrize('corruption', ['model_profile', 'pinned_file_sha256', 'missing_required_pin'])
def test_audit_rejects_frozen_manifest_contract_tampering(
    completed: dict, tmp_path: Path, corruption: str,
) -> None:
    base = _copy(completed, tmp_path)
    baseline = _audit(base)
    assert baseline['verdict'] == 'PASS', baseline['summary']
    from research.tests.test_comparison_report import _report
    report_fixture = {'base': base, 'runtime_dir': base / 'state'}
    assert _report(report_fixture)['matrix']['complete']
    path = base / 'state' / review.MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    if corruption == 'model_profile':
        manifest['model_profile'] = {'model': 'unapproved-model', 'max_tokens': 1}
    elif corruption == 'missing_required_pin':
        manifest.setdefault('pinned_file_sha256', {}).pop('research/comparison/native_driver.py', None)
    else:
        pins = manifest.setdefault('pinned_file_sha256', {})
        pins['research/comparison/missing-pinned-file.py'] = '0' * 64
    path.write_text(json.dumps(manifest))
    result = _audit(base)
    assert result['verdict'] == 'FAIL', result['summary']
    try:
        rejected_report = _report(report_fixture)
    except report.ComparisonReportError:
        pass
    else:
        assert not rejected_report['matrix']['complete']
        assert rejected_report['matrix']['observed_final_cells'] == 0
        assert rejected_report['conclusion']['conclusion'] == 'insufficient-evidence'
        assert rejected_report['analysis']['contrasts']['primary']['estimate'] is None



def test_report_counts_actual_producer_jobs_including_execution_failures(tmp_path: Path) -> None:
    from research.search.acceptance_inner import job_outcome_counts
    cell = iter_cells(make_bindings(), COMMIT)[0]
    run = cell.inner_run(0)
    counts = job_outcome_counts([
        {'job_id': 'job_scored', 'status': 'completed',
         'worker_cleanup_verified': True, 'evidence_verified': True},
        {'job_id': 'job_failed', 'status': 'failed',
         'worker_cleanup_verified': True, 'evidence_verified': False},
    ])
    assert counts['executed_and_cleaned'] == 2
    assert counts['scored'] == 1 and counts['execution_failed'] == 1
    run_dir = tmp_path / 'runs' / run.run_key().replace(':', '-')
    run_dir.mkdir(parents=True)
    (run_dir / 'run-result.json').write_text(json.dumps({'job_counts': counts}))
    assert report._count_candidates(tmp_path, cell) == 2



@pytest.mark.parametrize('mode', ['absent', 'wrong_identity', 'search_boundary'])
def test_final_results_require_original_matching_phase_without_rejecting_search_boundary(
    completed: dict, tmp_path: Path, mode: str,
) -> None:
    from research.tests.test_comparison_report import _report
    base = _copy(completed, tmp_path)
    fixture = {'base': base, 'runtime_dir': base / 'state'}
    assert _audit(base)['verdict'] == 'PASS'
    assert _report(fixture)['matrix']['complete']
    ledger_path = base / 'state' / review.LEDGER_FILENAME
    ledger = json.loads(ledger_path.read_text())
    if mode in ('absent', 'search_boundary'):
        ledger['final_phase'] = None
        ledger['events'] = [event for event in ledger['events'] if event['type'] != 'final_phase']
    else:
        ledger['final_phase']['barrier_identity'] = 'f' * 64
        for event in ledger['events']:
            if event['type'] == 'final_phase':
                event['barrier_identity'] = 'f' * 64
    ledger_path.write_text(json.dumps(ledger))
    if mode == 'search_boundary':
        # Model the legitimate all-search-complete, pre-publication
        # boundary. Archive only this test's copied final fixture.
        (base / 'state' / 'final').rename(base / 'archived-final-fixture')
    audited = _audit(base)
    assert audited['verdict'] == ('INCOMPLETE' if mode == 'search_boundary' else 'FAIL')
    reported = _report(fixture)
    assert not reported['matrix']['complete']
    assert reported['matrix']['observed_final_cells'] == 0
    assert reported['conclusion']['conclusion'] == 'insufficient-evidence'
    for contrast in reported['analysis']['contrasts'].values():
        assert contrast['observed_pairs'] == 0
        assert contrast['estimate'] is None



def test_missing_ledger_after_partial_search_is_not_a_never_started_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.comparison.budget import ComparisonLedger
    from research.tests.test_comparison_orchestrate import default_scores, fake_child, step_positions
    from research.tests.test_comparison_report import _fresh_orch, _report
    orch = _fresh_orch(tmp_path, monkeypatch)
    ledger_path = orch.runtime_dir / review.LEDGER_FILENAME
    with ComparisonLedger.open(ledger_path, cells=orch.cells, now_fn=lambda: 1000.0) as ledger:
        ledger.start_clock()
        step_positions(orch, ledger, [orch.positions[0]],
                       fake_child(orch, default_scores(orch), [], ledger_path))
    # A completed configuration is real execution evidence even though
    # no cell has reached its eight-configuration terminal boundary.
    assert not orch.terminal_summary()
    ledger_path.rename(tmp_path / 'archived-ledger-fixture.json')
    result = _audit(tmp_path)
    assert result['verdict'] == 'FAIL', result['summary']
    reported = _report({'base': tmp_path, 'runtime_dir': orch.runtime_dir})
    assert reported['ledger_error'] is not None
    assert reported['matrix']['observed_final_cells'] == 0
    assert 'never started' not in report.render_markdown(reported)

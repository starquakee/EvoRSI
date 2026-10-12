"""Real solver assembly, with no generation. Run under OpenMLE-Evo/.venv."""
from pathlib import Path
import shutil

import pytest


def test_all_nine_real_solver_configurations_are_wired_without_execution(tmp_path, monkeypatch):
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    monkeypatch.setenv('LOGGING_DIR', str(tmp_path / 'logs'))
    pytest.importorskip('dojo')
    from research.calibration.specs import RunSpec, trials
    from research.comparison import deploy
    from research.comparison.runner import build_comparison_solver
    from research.search.acceptance_inner import _SOLVER_CONFIGS, operator_clients
    root = Path(__file__).resolve().parents[2]
    shutil.copytree(root / _SOLVER_CONFIGS, tmp_path / _SOLVER_CONFIGS)
    (tmp_path / deploy.LITELLM_YAML).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / deploy.LITELLM_YAML, tmp_path / deploy.LITELLM_YAML)
    for trial in trials(root, 'a' * 40)[:9]:
        solver = build_comparison_solver(RunSpec(trial, 'b' * 64), repo_root=tmp_path)
        assert solver.cfg.crossover_prob == trial.decoded['crossover_prob']
        assert solver.cfg.max_debug_depth == 1
        assert solver.cfg.num_generations == 3
        assert solver.cfg.individuals_per_generation == 2
        assert solver.cfg.experience['parent_selection']['weights'] == {
            key: trial.decoded[key + '_weight'] for key in ('score', 'delta', 'novelty')}
        assert solver.cfg.experience['prompt_memory']['enabled'] is False
        assert len(operator_clients(solver)) >= 4

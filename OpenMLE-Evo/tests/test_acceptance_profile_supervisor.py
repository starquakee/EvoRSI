"""The production operator requests must use the selected Kimi profile."""
import yaml

from research.search import acceptance_config as ac
from research.search.acceptance_inner import build_acceptance_solver
from test_acceptance_inner import _spec, REPO_ROOT


def test_actual_operator_sampling_matches_credential_free_kimi_profile():
    profile = yaml.safe_load((REPO_ROOT / ac.LITELLM_YAML).read_text())
    expected = profile['model_list'][0]['litellm_params']
    solver = build_acceptance_solver(_spec(), repo_root=REPO_ROOT)
    for name in ('draft', 'debug', 'improve', 'crossover', 'analyze'):
        actual = solver.cfg.operators[name].llm.generation_kwargs
        for parameter in ('temperature', 'top_p', 'stream', 'max_tokens'):
            assert actual.get(parameter) == expected[parameter], (name, parameter, actual.get(parameter), expected[parameter])

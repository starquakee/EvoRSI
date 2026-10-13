"""Generate a proposed shared-start schedule. This cannot launch experiments."""
import argparse
import json
from pathlib import Path
import random


def plan() -> dict:
    rng = random.Random(20261013)
    configs = [
        {'config_id': f'c{3*i+j:02d}', 'crossover_prob': cross, 'score_weight': float(j),
         'delta_weight': .4, 'novelty_weight': .25, 'max_debug_depth': 1}
        for i, cross in enumerate((.05, .5, .95)) for j in range(3)
    ]
    units = []
    for block in range(3):
        pool_id = f'b{block:02d}-pool'
        units.append({'unit_id': pool_id, 'kind': 'fresh_seed_pool', 'block': block + 1,
                      'max_main_requests': 2, 'max_debug_requests': 2,
                      'max_requests': 4, 'max_tokens': 20000, 'max_elapsed_seconds': 900})
        order = list(range(9))
        rng.shuffle(order)
        seed = rng.getrandbits(32)
        for i in order:
            units.append({'unit_id': f'b{block:02d}-c{i:02d}', 'kind': 'continuation',
                          'block': block + 1, 'pool_id': pool_id, 'inner_seed': seed,
                          'configuration': configs[i], 'max_main_requests': 4,
                          'max_debug_requests': 2, 'max_requests': 6,
                          'max_tokens': 40000, 'max_elapsed_seconds': 1800})
    assert sum(x['max_requests'] for x in units) == 174
    assert sum(x['max_main_requests'] for x in units) == 114
    return {
        'schema': 'shared-start-pilot-draft.v1',
        'status': 'PROPOSED_NOT_AUTHORIZED_NOT_EXECUTABLE',
        'schedule_seed': 20261013, 'task': 'spambase', 'phase': 'validation',
        'blocks': 3, 'fresh_seed_pools': 3, 'continuation_trials': 27,
        'global_caps': {'requests': 174, 'tokens': 600000, 'elapsed_seconds': 14400},
        'max_main_requests': 114, 'max_debug_requests': 60,
        'per_job_timeout_seconds': 600,
        'new_final_evaluations': 0, 'automatic_retry_resume_or_extension': False,
        'model_profile_reference': '../calibration-round2-results-20261013/protocol.json',
        'candidate_cache_between_continuations': False,
        'shared_state': 'Only each block\'s two authenticated seed candidates and their trusted validation evidence',
        'primary_diagnostic': {
            'endpoint': 'Final best validation accuracy after four new main requests',
            'complete_trials_required': 27,
            'configuration_variance_fraction_min': .5,
            'median_pairwise_spearman_min': .5,
            'comparison_to_previous_round': 'Descriptive only; no concurrent randomized control cohort',
        },
        'units': units,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(plan(), indent=2) + '\n')

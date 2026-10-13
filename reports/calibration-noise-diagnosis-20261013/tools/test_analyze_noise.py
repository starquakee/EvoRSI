import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('analyze_noise', Path(__file__).with_name('analyze_noise.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NoiseAnalysisTests(unittest.TestCase):
    def test_missing_or_invalid_scores_rejected(self):
        initial = np.full((9, 3), .5)
        for final in (np.full((9, 3), np.nan), np.full((9, 3), .4), np.full((9, 3), 1.1), np.ones((9, 2))):
            with self.subTest(shape=final.shape):
                with self.assertRaises(ValueError):
                    module.analyze(initial, final)

    def test_constant_ranks_do_not_claim_agreement(self):
        out = module.analyze(np.full((9, 3), .5), np.full((9, 3), .6))
        self.assertIsNone(out['ranking']['median_held_out_spearman'])
        self.assertEqual(out['ranking']['single_repeat_pairwise_spearman'], [None] * 3)

    def test_stable_rank_and_covariance_with_independent_projection(self):
        rng = np.random.default_rng(17)
        initial = rng.uniform(.4, .5, (9, 3))
        final = .7 + np.arange(9)[:, None] * .01 + np.arange(3)[None, :] * .001
        out = module.analyze(initial, final)
        self.assertAlmostEqual(out['ranking']['median_held_out_spearman'], 1)
        conf, block = np.indices((9, 3))
        design = np.column_stack([np.ones(27)] + [(conf.ravel() == i).astype(float) for i in range(1, 9)]
                                 + [(block.ravel() == j).astype(float) for j in range(1, 3)])
        resid = initial.ravel() - design @ np.linalg.lstsq(design, initial.ravel(), rcond=None)[0]
        d = out['additive_residual_decomposition']
        self.assertAlmostEqual(d['initial_variance_pp2'], resid @ resid / 16 * 10000)
        self.assertLess(d['identity_abs_error_pp2'], 1e-10)

    def test_held_out_column_is_not_used_for_selection(self):
        final = .6 + np.tile(np.arange(9)[:, None] * .01, (1, 3))
        final[:, 2] = final[::-1, 2]
        out = module.analyze(np.full((9, 3), .5), final)
        fold = out['ranking']['two_repeat_mean_vs_held_out'][2]
        self.assertEqual(fold['selected_configs_including_ties'], ['c08'])
        self.assertAlmostEqual(fold['spearman_two_repeat_mean_vs_held_out'], -1)
        self.assertAlmostEqual(fold['held_out_gap_to_best_pp'], 8)


if __name__ == '__main__':
    unittest.main()

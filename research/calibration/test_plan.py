import copy
import json
from pathlib import Path
import unittest

from research.calibration.plan import analyze, make_plan, ranks, spearman


class CalibrationOfflineTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path(__file__).with_name("protocol-draft.v1.json").read_text())
        self.plan = make_plan(self.protocol, "small")

    def outcomes(self, value):
        return [{"trial_id": trial["trial_id"], "status": "completed",
                 "validation_accuracy": value(trial), "requests": 6, "charged_tokens": 10000}
                for trial in self.plan["schedule"]]

    def test_schedule_is_balanced_deterministic_and_block_paired(self):
        for option, count in (("small", 27), ("more_repeats", 45)):
            plan = make_plan(self.protocol, option)
            self.assertEqual(plan, make_plan(self.protocol, option))
            self.assertEqual(len(plan["schedule"]), count)
            self.assertEqual(len({t["trial_id"] for t in plan["schedule"]}), count)
            for repeat in range(plan["budget"]["repeats"]):
                block = [t for t in plan["schedule"] if t["repeat"] == repeat]
                self.assertEqual(len({t["config_id"] for t in block}), 9)
                self.assertEqual(len({t["inner_seed"] for t in block}), 1)

    def test_additive_config_and_block_effects_are_stable(self):
        report = analyze(self.plan, self.outcomes(lambda t: 0.5 + int(t["config_id"][1:]) / 32 + t["repeat"] / 128))
        self.assertEqual(report["verdict"], "exploratory_stable_configuration_signal")
        self.assertEqual(report["rank_agreement"]["median_spearman"], 1)
        self.assertAlmostEqual(report["variance"]["ms_residual"], 0)

    def test_constant_config_rank_cannot_establish_signal(self):
        report = analyze(self.plan, self.outcomes(lambda t: 0.5 + t["repeat"] / 128))
        self.assertEqual(report["verdict"], "stable_configuration_signal_not_demonstrated")
        self.assertIsNone(report["rank_agreement"]["median_spearman"])

    def test_incomplete_records_all_cost_and_does_not_impute(self):
        outcomes = self.outcomes(lambda t: 0.8)
        outcomes[0]["status"] = "failed"
        outcomes[0]["reason"] = "unknown_usage"
        report = analyze(self.plan, outcomes)
        self.assertEqual(report["verdict"], "insufficient_complete_evidence")
        self.assertEqual(report["recorded_cost"]["requests"], 162)
        self.assertEqual(len(report["incomplete_trials"]), 1)
        self.assertIsNone(report["variance"])

    def test_duplicate_unknown_final_and_nonfinite_inputs_refuse(self):
        outcomes = self.outcomes(lambda t: 0.8)
        bad_sets = [outcomes + [outcomes[0]], [{**outcomes[0], "trial_id": "foreign"}],
                    [{**outcomes[0], "final_accuracy": 0.9}],
                    [{**outcomes[0], "validation_accuracy": float("nan")}]]
        for bad in bad_sets:
            with self.assertRaises(ValueError):
                analyze(self.plan, bad)
        changed = copy.deepcopy(self.plan)
        changed["budget"]["repeats"] = 5
        with self.assertRaises(ValueError):
            analyze(changed, outcomes)

    def test_rank_ties_and_reversal(self):
        self.assertEqual(ranks([1, 1, 3]), [1.5, 1.5, 3.0])
        self.assertEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()

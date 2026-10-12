# Round two: validation repeatability diagnostic (draft)

Status: **offline preparation only; budget selection pending; no live execution**.
The request to start a new round does not select either budget option below.
Neither plan is a runnable model authorization or a reset of a prior ledger.

The diagnostic uses the frozen Spambase training/validation split. Nine settings
cross crossover probability {0.05, 0.5, 0.95} with score weight {0, 1, 2}.
Delta weight stays 0.4, novelty weight 0.25, and debug depth 1. Each independent
inner search has three generations, two main candidates per generation, and
at most two debug requests. Candidate and model-output caches are not shared.
Final test data and scores are excluded from this diagnostic.

| Option | Repeats | Trials | Request cap | Token cap | Wall-clock cap |
|---|---:|---:|---:|---:|---:|
| small (recommended) | 3 | 27 | 216 | 600,000 | 4 hours |
| more_repeats | 5 | 45 | 360 | 1,000,000 | 6 hours |

Stop on the first exhausted cap. These are ceilings, not a promise to complete
every trial. No automatic extension, retry, or restart is approved. Per trial:
8 requests, 40,000 tokens, 1,800 seconds; per sandbox job: 600 seconds.

`protocol-draft.v1.json` contains the proposed rules. `plan-small.v1.json` and
`plan-more-repeats.v1.json` contain deterministic randomized blocks and paired
search RNG seeds. A shared seed does not make provider output deterministic.

`plan.py` is an offline schedule generator and descriptive analyzer, with no
model, credential, sandbox, or final-data access. Its normalized outcome input
must first be collected from independently checked trusted validation evidence.
It is not an evidence authenticator or a live runner. Run its synthetic tests:

```sh
python -m unittest research.calibration.test_plan -v
```

For a complete balanced matrix, the analyzer reports a configuration/repetition
additive variance decomposition, a descriptive configuration variance fraction,
and pairwise Spearman rank agreement. The draft calls this fraction an estimated
ICC; it is an exploratory design-specific diagnostic, not a population estimate
or a significance test. The flag requires fraction >= 0.5 and median rank
correlation >= 0.5. Constant ranks give no evidence. Missing/failed outcomes
retain their costs and prevent a stable-signal verdict; there is no imputation.

Before live execution: select a budget, implement and review the separate
ledger/runner, bind the exact source and dataset identities, verify the new stack
is idle, restore its search registry, mask previous final-publication mounts,
and freeze the launch manifest. Preserve all prior runtime artifacts and ledgers.
The old fallback stack must remain untouched. Current plans make no claim that
these launch gates have passed or that a live runner has been implemented.

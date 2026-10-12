# Round two: validation repeatability diagnostic

The user selected **three repetitions** on 2026-10-12: the `small` option below.
Implementation and offline checks are being frozen for a separately supervised
launch. The five-repeat draft is retained as a rejected alternative, not an
authorization or a source of additional budget. Draft schedule files preserve
their original preparation status; execution requires a separate exact-bound
authorization, preflight receipt and supervisor review under the new runtime.

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

`specs.py`, `ledger.py`, `runner.py`, `gates.py`, `prepare.py`, `collect.py`, and
`execute.py` implement the separate live path. The shared low-level comparison
transport, source binding, cancellation and solver assembly remain unchanged.
The only old-test edit isolates an absent-review test from preserved runtime data.

The live ledger charges conservative prompt bytes plus the full 8,192-token
output reservation before every request. Global and per-trial accounting share
one durable transaction; unknown usage retains the reservation and stops globally.
The source-bound manifest selects exactly 27 trials and fixed limits. Exclusive
launch markers, fresh run directories, a process lock and terminated trial states
refuse retries, concurrent execution, resuming, or increasing limits.

Preparation verifies an idle new stack, replaces its old final-publication mounts
with empty read-only directories through the tracked compose overlay, and restores
the validation registry. One fixed first-party sandbox probe (zero model calls)
verifies own training/validation data hashes and denial of final, answer and other
task files. That probe is excluded from all measurement trials and never shared
with candidates. The 4-hour measurement clock starts at the single live launch;
preparation and offline checks precede it. Emergency cleanup may finish after a
deadline; it never authorizes another model request or candidate submission.

Before each trial, exact source/manifest/review and actual container identity gates
must pass. A result is checked against ledger requests, actual operator settings,
journal source bytes, trusted prediction snapshots, and independently recomputed
validation accuracy before it enters analysis. Every failed/interrupted request
remains charged. There is no final-test evaluation in this round.

Launch sequence (authorized supervisor only): freeze and test the clean source,
run `python -m research.calibration.prepare`, inspect its evidence, record the
exact-bound private authorization/review, then invoke
`python -m research.calibration.execute` once. Neither command creates approval
on behalf of the user. Preserve all previous runtime artifacts and ledgers; keep
the old fallback stack unchanged.

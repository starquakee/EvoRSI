# Shared-start validation pilot

The user approved this separate pilot after reviewing the
[offline diagnosis and proposed budget](../../reports/calibration-noise-diagnosis-20261013/README.md).
The selected allocation is **174 K3 requests / 600,000 tokens / four hours**.
It does not reopen any previous comparison or calibration ledger.

Three independent blocks each generate two initial candidates. A block's nine
configurations receive identical authenticated copies of those two terminal
candidates and their required ancestry. Each continuation then makes four new
main requests, retaining the existing three-generation Evo search semantics
(generation zero is the shared pool; generations one and two are new).

| Unit | Count | Main calls | Debug cap | Request cap | Token cap | Time cap |
|---|---:|---:|---:|---:|---:|---:|
| Fresh pool | 3 | 2 | 2 | 4 | 20,000 | 900 seconds |
| Continuation | 27 | 4 | 2 | 6 | 40,000 | 1,800 seconds |

The main-call ceiling is 114; up to 60 debug calls give 174 requests globally.
Every sandbox job has a 600-second ceiling. All failed/unknown model usage is
charged; conservative output reservations may stop a unit before its nominal
request count. Stop on the first exhausted cap, invalid pool, unverified unit,
unknown usage, integrity failure or cleanup failure. There are no automatic
retries, resumes, replacement candidates for an invalid pool, or extensions.

The shared state is restricted to each block's initial candidate pool. New
continuation candidates and response caches are never shared. Pool generation
and repair costs are charged once as block overhead, inside the measurement
clock. Imported seed jobs remain explicitly separate from fresh continuation
jobs and never create new model or sandbox submissions. A seed can remain the
winning candidate, bound to its original trusted prediction snapshot.

## Frozen contract and independent verification

`approved-plan.v1.json` preserves the exact reviewed draft bytes and its original
pre-approval status; its hash, unit order, seeds, grid and limits are checked by
`specs.py`. Actual launch authorization is a separate private receipt bound to
the executable protocol and clean source commit. The old draft status cannot
authorize or veto a different runtime on its own.

`ledger.py` owns global and per-unit costs in one atomic transaction. It admits
the exact 30-unit order, charges before transport, locks out concurrent owners,
and refuses replay or interrupted-state recovery. After independent validation,
the coordinator writes a seed bundle and seals its SHA-256 into the ledger.
That seal is immutable and has no additional provider cost. Continuations
cannot begin without the correct completed predecessor and pool seal.

`seeds.py` rechecks the sealed bundle, source bytes, journal ancestry and
prediction hashes before and after a continuation. It restores canonical node
references and all nested metric metadata. The two valid terminal seeds become
the initial population; invalid draft ancestry remains for faithful provenance.
The shared node fields remain identical, except that child links may grow as
new descendants are generated. Each continuation has a fresh journal directory.

`runner.py` reuses the existing guarded transport, trusted sandbox task, solver
assembly and winner binding. `control.py` changes only the per-unit main-call
floor to two or four. Native Evo recovery can choose a **new** draft after
specific repeated failures; it remains one of the four continuation main calls.
It must have a new node identity and generation one or two. Regenerating the
imported generation-zero pool is prohibited.

`collect.py` independently binds every scored node to its job/source and
recomputed validation accuracy, checks actual parameter weights and parent
probabilities, and verifies copied seed state against its original pool.
Continuation reports keep imported jobs, fresh jobs and costs separate. A pool
with fewer than two valid terminal seeds is never sealed or used.

`prepare.py` first records historical hashes and verifies an idle new stack.
It applies the tracked empty-final overlay to the new worker/dispatcher only.
One fixed first-party probe checks own training/validation access and denial of
final, other-task and answer paths, with zero model calls. This probe precedes
the measurement clock and is not a seed. The old stack and all prior runtime
artifacts remain untouched. Completed round-two ledger/outcome hashes must match
their published audit, and its entire runtime tree is checked before/after.

Before every unit, `gates.py` rechecks exact source, manifest, authorization,
supervisor review, preflight, container IDs/images and mount sources. A fresh
`.runtime/shared-start-pilot` holds all private evidence. Raw candidates, model
messages, predictions, authorizations and ledgers are excluded from GitHub.

## Analysis and limits

The primary endpoint is the best validation accuracy after four new main calls.
The matrix has 27 continuations, conditional on **three independent pools**.
`analysis.py` reuses the frozen additive variance and rank diagnostics on that
matrix, then adds pool overhead once to reconcile global cost. It also reports
pool starting scores, continuation gains and leave-one-block-out averaging.

All 27 continuations and all three pools must complete before a stable-signal
verdict is possible. The exploratory thresholds stay at configuration variance
fraction ≥ 0.5 and median pairwise Spearman ≥ 0.5. No imputation or failure
exclusion is allowed. A historical residual-variance comparison is descriptive:
without concurrent randomized controls, it cannot establish a causal reduction
from shared starts. There is no final-test evaluation or outer-algorithm
superiority claim in this pilot.

## Supervisor sequence

Run offline tests in the research environment and the real Evo fake-generation
integration in its own environment. Freeze a clean source commit and record the
exact-source evidence; prepare the new stack, inspect preflight, and bind the
already granted user authorization and supervisor review to that manifest.
Only then invoke `python -m research.shared_start.execute` once. The four-hour
clock starts there and includes seed generation, all continuations and intervening
validation checks. Cleanup may finish after the deadline, but cannot authorize
another request. Unused headroom after a terminal outcome cannot be reused.

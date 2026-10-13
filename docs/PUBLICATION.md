# Public source snapshot

Updated on 2026-10-13 from local source commit
`73a4b6275bfd67096588d387c5c0d4a24f904122` (offline noise diagnosis and shared-start draft). This update follows the existing public main history;
the original local research history is preserved separately and is not pushed.
The initial public root was created on 2026-10-09 because older local commits
contained historical credentials already removed from current files.

## Included and excluded material

Current source code, tests, upstream licenses, synthetic fixtures, dependency
snapshots, and redacted evidence are included. Changed implementation Python
files and configuration YAML are byte-identical to the reviewed source commit.
The synthetic evaluator answer fixture is intentional test data and must never
be mounted into a candidate worker.

Local handoff notes, progress logs, machine inventories, generated benchmark
packages, raw runtime state, real task datasets, model transcripts, credentials,
and virtual environments remain excluded. Public handoff reports are included.
Display-only machine paths in documents and JSON are normalized to `<repo-root>`,
`<legacy-root>`, `<local-workspace>`, or `<user-home>`. Evidence hashes refer to
the original private runtime artifacts; they are not hashes of normalized text.
Local commit identifiers in reports describe the reviewed source history and
do not resolve as commits in this separate public repository.

## Acceptance and reproducibility limits

US-001 through US-010 are complete in the recorded local environment. Read
`reports/us009-round2-acceptance.md` and `reports/us010-switch-handoff.md` for
current results; `reports/us009-real-acceptance.md` preserves the incomplete
first round. Publication itself performs no new model experiment, budget reset,
default-endpoint switch, or server deployment.

Recorded validation: 905 research tests passed; 146 Evo tests passed with three
skips; 96 owned research/experiment/test files passed scoped type checking.
This is not a claim that all tests run from a bare clone. Local provenance and
legacy-JAX tests require excluded artifacts and environments. The normalized
`research/tests/legacy_venv.json` is a local-environment hint, not a downloadable
runtime; tests depending on it skip when that interpreter is absent. External
OpenMLE-ERL/SFT remains unverified when not installed.

Docker/GPU probes require the configured stack. Runtime evidence audit requires
the original local artifacts, which are not shipped. Real model execution needs
credentials, a reviewed implementation, and a separately authorized budget;
neither authorization receipts nor ledgers are included. Missing evidence must
fail closed. Cloning or publishing does not authorize a new experiment.

The later three-task, three-seed comparison and its final-only amendment are
now included as redacted evidence. No iStratDE advantage was demonstrated; the
original strict audit remains FAIL. The follow-up three-repeat diagnostic is now complete, as documented below. MetaDE integration remains future work.

## Checks on the previous 2026-10-10 publication

The actual public working copy passed 58 focused tests for generation outcomes,
endpoint routing, rollback payloads, and historical-audit rejection cases.
Two legacy-interpreter integration tests skipped because that local runtime is
not distributed. The tracked-file scan checked 840 text files and found no
secrets or research-code host-path violations. The 39 synchronized Python/YAML
implementation and test files were compared byte-for-byte in the staged index
against the reviewed local source commit. One public-only portability edit in
`scripts/ralph/kimi-one-story.py` resolves the CLI through `Path.home()` instead
of retaining a machine-specific user directory. No model call or live job was made
for publication validation.

## 2026-10-12 publication scope

This update includes the comparison implementation/tests, redacted report bundles,
mechanism diagnosis, and offline calibration schedules/analyzer. Report tables,
plots, hashes and aggregate operator statistics are included; raw runtime ledgers,
model messages, generated candidate programs, task CSVs/archives, validation/final
answer files, and the private authorization receipt remain excluded. Dataset
manifests retain hashes and source metadata; loading an absent dataset must fail.
The shipped evaluator registry therefore requires local dataset preparation.

Report export manifests describe the original local export, before publication
path normalization and receipt exclusion; they are not a byte manifest of this
Git checkout. Original local commit IDs identify execution history, not public
Git commits. Historical FAIL results and all amendment limitations are retained.

No historical local Git objects are imported. Changed Python/YAML implementation
and test files are synchronized byte-for-byte from the source commit. Display
paths in documents and JSON are normalized as before. This publication runs only
offline checks and makes no new model or sandbox requests.

The actual publication copy passed **215 offline tests** covering calibration,
comparison protocol/specs, request budgets, schedule, analysis and final-only
stage controls. All **109 synchronized Python/YAML files** matched their source
commit bytes in the staged index. The final scan covered **997 tracked text
files** with zero secret findings and zero research-code host-path violations;
the checkout contains 1,009 tracked files including report images. Dataset,
runtime and private-authorization exclusions were checked separately. These
checks are not a fresh-clone end-to-end experiment or live-stack validation.

## 2026-10-13 calibration implementation update

The user selected the small diagnostic: nine configurations with three independent
repetitions, 27 trials, 216 requests / 600,000 tokens / four hours. The five-repeat
draft remains an unselected alternative. This update publishes the exact frozen
Python/YAML implementation and the temporary-directory correction to the old
missing-review test. That implementation was published before the run completed. The final result
publication below supersedes its in-progress status.

The exact source commit passed 190 research/shared regression tests (one expected
research-environment skip), plus the real Evo assembly test that covers all nine
settings and resolves that skipped dependency-specific case. Seven source modules
passed type checking; the source secret audit passed. The actual new-stack
preflight passed with one fixed first-party isolation probe and zero model calls,
verified empty final-data mounts, and preserved all 6,413 original comparison
files plus historic ledgers and the final-only stage. The private manifest,
authorization, supervisor review, runtime ledgers and live traces are excluded.

Source and result publication does not import the original private Git history.
Only the isolated live coordinator consumes the selected budget; publication
checks make no model request or sandbox submission.

The actual publication copy passed 50 offline calibration tests. Its final
tracked-file audit checked 1,010 text files with zero secret findings and zero
research-code host-path violations. All 14 synchronized Python/YAML files
were compared byte-for-byte in the staged index against the frozen source.

## 2026-10-13 completed calibration results

The authorized schedule finished all 27 trials: 166 model requests, 371,394 tokens,
87.82 minutes, and zero unknown usage. The source used for execution remains
`64adefe6a267cfdac8ace7889ce7f70fefa594d2`; the latest source commit adds reports and
completion documentation. It does not retroactively replace that execution ID.

All 162 scored validation predictions were independently recomputed. Three
candidate executions failed and one generated candidate had invalid Python and
was not submitted; all associated model costs are retained. One separate fixed
preflight probe used zero model calls. There were no final-test evaluations.

The variance fraction (0.0840) and median rank correlation (0.1266) missed the
predeclared exploratory thresholds (both 0.5). Least-squares and SciPy recomputations
matched the frozen analyzer. No stable overall configuration signal or outer
algorithm advantage is claimed. Three repetitions provide limited evidence.

The new report bundle includes aggregate trial costs/scores, behavioral summaries,
the frozen protocol/schedule, a visually checked repeatability plot, verification
metadata, and a read-only report script. Its export manifest hashes the included
files; CSV line endings are LF without changing cell values. Raw model messages,
candidate programs, prediction rows, ledgers, real data files, and authorization
receipts remain private. All 6,413 original comparison files, historic ledgers,
the final-only stage and the legacy stack were preserved.

Publication verification re-read all 27 CSV rows, reproduced the frozen analysis,
reconciled 166 requests / 371,394 tokens, and matched all nine export-manifest
hashes against both the working copy and staged files. Eleven synchronized files
matched the local source commit byte-for-byte. The private ledger and outcome
hashes remained unchanged. The final tracked-file audit scanned 1,019 text files
with zero secret findings and zero research-code host-path violations.

## 2026-10-13 offline noise diagnosis

The new bundle analyzes only the previously published 27 trial aggregates.
It adds joined initial/final scores, an additive residual covariance decomposition,
leave-one-repeat-out ranking diagnostics, a visually inspected plot, reproducible
analysis code and a proposed shared-start schedule. There were zero new model
requests, sandbox submissions or final evaluations. The shared-start proposal
is not authorized, and the existing runner cannot execute it.

Four synthetic statistical/boundary tests passed. Real-data decomposition matched
an independent least-squares projection; prior rank correlations and residual
variance matched the frozen round-two analysis. All nine original export hashes
matched in both checkouts, and private round-two ledger/outcome hashes were
unchanged. The proposed schedule covers three pools and 27 continuations, with
174 requests at most (114 main and 60 debug). These are proposed ceilings, not a
reopened or extended allocation. No runtime, raw candidate or credential data
is added. The original execution source remains `64adefe6a267cfdac8ace7889ce7f70fefa594d2`.

The publication index matched all 11 synchronized source/report files byte-for-byte;
all nine new export hashes were verified. The final audit scanned 1,028 tracked
text files, finding zero secrets and zero research-code host-path violations.

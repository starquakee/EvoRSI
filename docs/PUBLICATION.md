# Public source snapshot

Updated on 2026-10-12 from local source commit
`a9088058404e0042ea79ba247e4812d0e817b730` (reviewed comparison implementation plus tested offline calibration preparation). This update follows the existing public main history;
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
original strict audit remains FAIL. The next diagnostic is offline preparation
with a pending budget choice. MetaDE integration remains future work.

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

# Public source snapshot

Updated on 2026-10-10 from reviewed local source commit
`f3f767fa6479c38e860862c6ab197cf10804ee0a`. This update follows the existing public main history;
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

These results establish a bounded pipeline on one small synthetic task, not
algorithmic superiority or independent generalization. MetaDE integration and
equal-budget multi-seed comparisons remain future work.

## Checks on this publication update

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

# Public source snapshot

Published from the reviewed local source tree at `0d27dcbd66b5377849e93d8c3bf31ade3b3deb05` on 2026-10-09.
The original local history is retained separately. This public repository starts
with a new root commit because older local commits contained historical
credentials that had already been removed from the current files.

## Included and excluded material

Current source code, tests, upstream licenses, synthetic fixtures, dependency
snapshots, and redacted evidence are included. The synthetic evaluator answer
fixture is intentional test data; it must never be mounted into a candidate worker.

The publication excludes local handoff notes, local progress logs, machine inventory
files, generated task-package/benchmark examples, raw experiment state, real task
datasets, model transcripts, credentials, and virtual environments. Original
artifacts remain in the local research checkout. In documentation and JSON evidence,
machine-specific locations are normalized to `<repo-root>`, `<legacy-root>`,
`<local-workspace>`, or `<user-home>`. Evidence hashes refer to the original artifacts.

Implementation Python files, source policy, evaluator registry, synthetic CSVs,
and upstream license files are unchanged from the reviewed source commit.

## Reproducibility limits

Recorded local validation does not mean every test can run from a bare clone.
Legacy provenance tests require their read-only source trees; Docker/GPU probes
require the configured stack; the real-model runner requires local credentials,
review evidence, and a separate explicitly authorized budget. Missing evidence
must fail closed. The stopped local acceptance ledger is not included or reset.

Some historical component documents describe earlier milestones or an upstream
paper. README.md and reports/us009-real-acceptance.md describe the latest EvoRSI
acceptance status: US-001 through US-008 completed, US-009 incomplete, US-010 not
started. No model experiment, default-endpoint switch, or server deployment was
performed as part of publication.

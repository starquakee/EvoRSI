# EvoRSI development notes

Read README.md and docs/PUBLICATION.md before changing this publication snapshot.
The public snapshot has a separate Git history from the preserved local research
checkout. Do not infer authorization to run model experiments or change budgets
from the presence of an acceptance runner.

Keep credentials, virtual environments, runtime ledgers, model transcripts,
external task datasets, and generated experiment outputs out of Git. Preserve
upstream licenses and source provenance. Run candidate code only inside the
intended isolated worker. Treat unverified results as failed/incomplete.

Relevant validation: research/tests, OpenMLE-Evo/tests, the repository secret
audit, and focused type checks. Some provenance and live-stack tests require
local artifacts not shipped in this publication; report that distinction.

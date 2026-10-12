# comparison/wdbc public search inputs (UCI 17)

Worker-visible search data for the iStratDE comparison `wdbc` task.
Prepared 2026-10-10 by `research/comparison/data.py` (EXP-002) from the
official UCI archive; byte identities are pinned in
`research/comparison/data-manifest.v1.json`.

Provenance:

- source: https://archive.ics.uci.edu/static/public/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic.zip
  (official UCI Machine Learning Repository, dataset 17), license CC BY 4.0
  - archive sha256 bc154869ef13f753f9e2b5a17e248cfe1ba4b6721db7c4da9f4880e40b05d3af
  - member `wdbc.data` sha256 d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a
- conversion: original WDBC id column REMOVED; diagnosis mapped B=0, M=1;
  30 numeric features; original row indices discarded
- split: one fixed StratifiedGroupKFold(5, shuffle, seed=20261010+17);
  identical numeric feature rows share a group; train folds 0/1/2 (342 rows),
  validation fold 3 (114 rows), final fold 4 (113 rows, sealed)
- ids: fresh opaque per-split ids from a label-independent fixed shuffle;
  not original indices

Files (sha256):

- train.csv 720a22c6c3a6f60af73b24ff882ad7b710ef90bcded2dce4bbd7673422ae656c (342 rows: id,f1..f30,label)
- test.csv 7df7fb4e153c60f4addb6a5cee32a57943bdf136d3b8ebc03ed6664bc5b244ae (114 rows: id,f1..f30 — validation FEATURES only)
- sample_submission.csv b25ce6b5458374354e8bd2992a854baf664dbf04cc0a2c6246e7c40495cfc893 (114 rows: id,label all-zero schema example)

Sealed, NEVER mounted into a worker: validation labels (evaluator-private
answer only), final features/labels, train+validation refit, raw archive.
Mount wiring (read-only, `/mnt/rsi_data/comparison/wdbc/data/public`)
lands with the narrowed grants in EXP-003; jobs use generic `train.csv` /
`test.csv` names so exact winning bytes can refit at the final phase.

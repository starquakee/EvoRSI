# comparison/spambase public search inputs (UCI 94)

Worker-visible search data for the iStratDE comparison `spambase` task.
Prepared 2026-10-10 by `research/comparison/data.py` (EXP-002) from the
official UCI archive; byte identities are pinned in
`research/comparison/data-manifest.v1.json`.

Provenance:

- source: https://archive.ics.uci.edu/static/public/94/spambase.zip
  (official UCI Machine Learning Repository, dataset 94), license CC BY 4.0
  - archive sha256 813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636
  - member `spambase.data` sha256 b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12
- conversion: 57 numeric features; label column used as-is (values in {0,1});
  the source has no identifier column and original row indices are discarded;
  183 multi-row duplicate groups (3 with contradictory labels) are preserved,
  grouped, and never split apart
- split: one fixed StratifiedGroupKFold(5, shuffle, seed=20261010+94);
  identical numeric feature rows share a group; train folds 0/1/2 (2718 rows),
  validation fold 3 (917 rows), final fold 4 (966 rows, sealed)
- ids: fresh opaque per-split ids from a label-independent fixed shuffle;
  not original indices

Files (sha256):

- train.csv baa13930bc2d3a95a82f28b98b6d5a3cf8ea150fb08a02cdb666533aa404ee81 (2718 rows: id,f1..f57,label)
- test.csv de2a91bb2be65de95011a5cf2700c87ee6ceb944eafa1d415ae167a1936934cd (917 rows: id,f1..f57 — validation FEATURES only)
- sample_submission.csv 066f8a17688b4f5dde347097c26fcd4b233e1c65196af9cd761cf90822af12ac (917 rows: id,label all-zero schema example)

Sealed, NEVER mounted into a worker: validation labels (evaluator-private
answer only), final features/labels, train+validation refit, raw archive.
Mount wiring (read-only, `/mnt/rsi_data/comparison/spambase/data/public`)
lands with the narrowed grants in EXP-003; jobs use generic `train.csv` /
`test.csv` names so exact winning bytes can refit at the final phase.

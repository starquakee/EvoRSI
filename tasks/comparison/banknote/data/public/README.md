# comparison/banknote public search inputs (UCI 267)

Worker-visible search data for the iStratDE comparison `banknote` task.
Prepared 2026-10-10 by `research/comparison/data.py` (EXP-002) from the
official UCI archive; byte identities are pinned in
`research/comparison/data-manifest.v1.json`.

Provenance:

- source: https://archive.ics.uci.edu/static/public/267/banknote%2Bauthentication.zip
  (official UCI Machine Learning Repository, dataset 267), license CC BY 4.0
  - archive sha256 1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227
  - member `data_banknote_authentication.txt` sha256 d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9
- conversion: 4 numeric features; label column used as-is (values in {0,1});
  the source has no identifier column and original row indices are discarded
- split: one fixed StratifiedGroupKFold(5, shuffle, seed=20261010+267);
  identical numeric feature rows share a group; train folds 0/1/2 (827 rows),
  validation fold 3 (269 rows), final fold 4 (276 rows, sealed)
- ids: fresh opaque per-split ids from a label-independent fixed shuffle;
  not original indices

Files (sha256):

- train.csv c8a63bf422ef995bc145898cfea0df816498ebf81709fcff9991a5745126e37e (827 rows: id,f1..f4,label)
- test.csv 9fbc8daa26c1958d0dfa18963becc546466afa3a19927bffbae111444f830010 (269 rows: id,f1..f4 — validation FEATURES only)
- sample_submission.csv ce9a4d5bef807041b1e532d49c997304a337383b511b3ad443c3b5ae25d32416 (269 rows: id,label all-zero schema example)

Sealed, NEVER mounted into a worker: validation labels (evaluator-private
answer only), final features/labels, train+validation refit, raw archive.
Mount wiring (read-only, `/mnt/rsi_data/comparison/banknote/data/public`)
lands with the narrowed grants in EXP-003; jobs use generic `train.csv` /
`test.csv` names so exact winning bytes can refit at the final phase.

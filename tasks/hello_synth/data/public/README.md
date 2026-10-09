# hello_synth public inputs (synthetic)

Public training/test inputs for the synthetic `hello_synth` fixture task.
These are the ONLY files copied from the legacy task directory; the
private answer (`tasks/hello_synth/test_answer.csv`) and the legacy metric
stay outside this directory and are never mounted into the candidate
worker.

Provenance (byte-identical copies, verified 2026-10-08):

- source: `sandbox-data/tasks/hello_synth/data/public/` of the Windows
  migration tree (read-only reference)
- train.csv            sha256 448e69962d82036c0e1b4aff7b9b3397bedbb4537c2bb1e9d2830f9fa3b29976 (120 rows: id,f1,f2,label)
- test.csv             sha256 b0ac9e8732f6be47ae3d261f88f8a25bfb1083e4fb6bda32c983b195f55e2995 (40 rows: id,f1,f2)
- sample_submission.csv sha256 aeff9e843b62f64a93b97a6e1921abab9d82b41de8078bca0a900cfa2845ff5c (40 rows: id,label)

Mount contract (rsi-trustworthy stack): this directory is mounted
read-only into the worker at `/mnt/rsi_data/hello_synth/data/public`;
jobs pass `data_dir=/mnt/rsi_data/hello_synth` so the dispatcher exports
`DATA_DIR=/mnt/rsi_data/hello_synth/data/public`.

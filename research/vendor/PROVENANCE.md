# Vendored upstream snapshots

Both packages are snapshots of the upstream EMI-Group repositories, taken from
the read-only legacy tree clones. Licenses are preserved verbatim (GPL-3.0,
see each package's `LICENSE`). Only `src/`, `pyproject.toml`, `LICENSE` and
`README.md` are vendored — no upstream `.git`, examples, assets or tests.

| Package | Upstream | Commit | License |
|---|---|---|---|
| `istratde/` | https://github.com/EMI-Group/istratde.git | `49c9d662774b0f43a1d6dbc9cb8547ba22ea448a` | GPL-3.0 |
| `metade/` | https://github.com/EMI-Group/metade.git | `bf11ce51495948eb0d64e5d7de141b558ea797c5` | GPL-3.0 |

The legacy GPU venv has both installed as editable packages at these exact
commits, so the vendored sources are byte-identical to what experiments
execute (verified per-file by `research/tools/inventory.py --manifest`;
see `research/provenance/manifest.json`).

These are upstream GPL sources kept for reproducibility and audit. Local
modifications, if ever needed, must be recorded in
`research/provenance/manifest.json` notes and respect GPL-3.0.

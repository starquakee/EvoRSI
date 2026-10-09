# Research environment dependencies (US-001)

No credentials are recorded here. Secrets come from the process environment or
the gitignored repo-root `.env` at runtime.

## Interpreters and snapshots

| Environment | Interpreter | Purpose | Version snapshot |
|---|---|---|---|
| legacy GPU venv | `<legacy-root>/.venv/bin/python` | JAX/CUDA DE algorithms, sandbox adapters, Evo experiments | `provenance/venv-rsi-gpu.freeze.txt` |
| OpenMLE-Evo venv | `OpenMLE-Evo/.venv/bin/python` | OpenMLE-Evo test-time search | `provenance/venv-openmle-evo.freeze.txt` |
| research venv | `.venv-research/bin/python` (repo root, gitignored) | offline tests, mypy, inventory/audit tools | `provenance/venv-research.freeze.txt` |

Key pinned versions (see freeze files for the full lock):

- legacy GPU venv: Python 3.12.3, `jax==0.4.33` + `jax-cuda12-plugin==0.4.33`,
  `evox==1.1.1`, `torch==2.5.1+cpu`, `httpx==0.28.1`, editable
  `istratde@49c9d662774b0f43a1d6dbc9cb8547ba22ea448a` and
  `metade@bf11ce51495948eb0d64e5d7de141b558ea797c5` (same commits as
  `research/vendor/`).
- research venv: `pytest`, `mypy`, `httpx`, CPU `jax` (offline tests only),
  `fastapi` + `pydantic` + `psycopg2-binary` + `redis` + `python-multipart`
  (US-004: offline API admission gate tests import the WSL
  `sandbox-controller/api_server` with spied DB/Redis; no services started).
- trusted evaluator (US-005): `research/evaluator/` is Python-standard-library
  ONLY at scoring time (see `research/evaluator/requirements.txt`); no new
  third-party dependencies, no model credentials. The dispatcher's external
  scoring path is off by default (`EXTERNAL_EVALUATOR_ENABLED=0`) and fails
  closed at startup if enabled without `EVALUATOR_REGISTRY_PATH`.

## Recreating the research venv

```bash
python3 -m venv .venv-research
.venv-research/bin/pip install pytest mypy httpx "jax[cpu]" \
  fastapi pydantic psycopg2-binary redis python-multipart
```

The legacy and Evo venvs are pre-existing environments: reuse them, never
upgrade or reinstall them globally. New dependencies go to `.venv-research`.

## Path conventions

- All repository paths in code resolve relative to the repo root
  (`Path(__file__).resolve().parents[N]`); host absolute paths in
  `research/**/*.py` fail `tools/audit_secrets.py`.
- Machine-specific external roots (legacy tree, Windows deployment) live only
  in `research/tools/inventory_roots.json` and docs.
- Runtime output defaults to `<repo>/outputs/` (gitignored);
  `RSI_OUT_ROOT` overrides.
- Sandbox/LLM credentials: environment variables or repo-root `.env`
  (gitignored). Nothing in this repo embeds or scrapes keys.

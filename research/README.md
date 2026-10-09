# research/ — consolidated RSI research sources (US-001)

Single tracked home for the legacy GPU-experiment sources. Everything here is
consolidated from the read-only legacy tree (`rsi-gpu-legacy`) or vendored from
upstream EMI-Group repositories; exact origin paths, hashes and VCS commits are
recorded in `provenance/manifest.json`. The legacy tree and the live Windows
deployment are never modified by this work.

## Layout

- `adapters/` — fixed evaluation/safety contract between the optimizer and the
  sandbox (`sandbox_eval_client`, `rsi_eval_adapter`, `safe_rsi_de_adapter`).
  The optimizer must not modify these; the keyword gate in
  `sandbox_eval_client` is a pre-execution filter, NOT a security boundary.
- `contracts/` — shared trustworthy-loop contracts (US-003): `results`
  (common result schema with explicit metric direction and fail-closed
  finalization), `cache` (deterministic scored-result cache; hits cost zero
  and retain original provenance), `budget` (persistent serial budget
  ledger: reserve before EVERY model attempt, 200k tokens / 30 requests /
  5400 s, retain+stop on unknown usage, never reset on restart) and
  checkpoint-safe cancellation. Transports are injected; no real model
  calls live here.
- `experiments/` — legacy experiment and smoke scripts. These are archival
  research entrypoints; per the PRD they must not be run before US-009 gates
  pass (they can trigger real sandbox/model calls). `smoke_adapter.py` was
  renamed from `e2e/test_adapter.py` so pytest never collects it.
- `search/` — acceptance-search evidence helpers (US-007):
  `validation_config` (the INNER Evo validation-experiment contract with
  machine-readable constraint names, stdlib-only so OpenMLE-Evo tests can
  import it), `search_vector` (the OUTER iStratDE search vector: 5 retained
  dims with documented real solver effects, 3 dropped prompt-memory dims with
  the drop rationale, decode/safety-gate/hydra-override rendering) and
  `budget_transport` (a one-layer budget guard installed as
  `client.request_guard`: reserve before every model call, provider-usage
  extraction that fails closed as `usage_unavailable`, `guard_already_installed`
  on double install so guards never nest). Terminology is explicit
  everywhere: INNER generations are Evo solver generations
  (`search.runner.solver.num_generations`); OUTER generations are iStratDE DE
  generations over config vectors (fitness minimized natively,
  `fitness=-accuracy`). US-009 adds the acceptance runner (see below):
  `acceptance_config` (pins + the supervisor review gate),
  `credentials` (fresh-read runtime-only managed Kimi token access),
  `run_index` (validated complete-run index for the outer loop's verified
  first-evaluation reuse), `sandbox_task` (production dojo `step_task` over
  the isolated sandbox client), `acceptance_inner` (Evo-venv child runner)
  and `acceptance` (legacy-venv parent controller with `preflight`/`live`).
- `vendor/` — upstream `istratde` and `metade` snapshots (`src/`,
  `pyproject.toml`, `LICENSE` (GPL-3.0), `README.md`). Origin URLs/commits in
  `vendor/PROVENANCE.md` and `provenance/manifest.json`.
- `tools/` — `inventory.py` (path/hash inventory + provenance manifest),
  `audit_secrets.py` (secret/host-path audit of tracked files) and
  `audit_formal_baseline.py` (read-only independent audit of the legacy
  formal_baseline records, US-002). Tree roots live in `tools/*.json`
  (data, not code).
- `tests/` — offline contract tests (no network, no sandbox, no GPU).
- `provenance/` — generated evidence: per-tree inventories, WSL-vs-Windows
  diff, manifest, venv freeze snapshots, secret audit report.

## Intentional changes versus the legacy origins

Enumerated in `provenance/manifest.json` (every non-identical file carries a
note). Summary:

1. `adapters/rsi_eval_adapter.py` — import rewritten package-relative.
2. `adapters/safe_rsi_de_adapter.py` — `__main__` demo prints a real newline
   (legacy printed a literal `\n`); library behavior unchanged.
3. `experiments/de_formal_baseline.py` — **credential fallback removed**: the
   legacy script scraped a sandbox API key out of the deployed Windows
   `api_server.py` source when `SANDBOX_API_KEY` was unset. The consolidated
   copy fails closed (environment or gitignored repo-root `.env` only), uses
   repo-root-relative paths, and defaults output to `outputs/formal_baseline`
   (gitignored). Type annotations added for mypy.
4. `experiments/de_search_hello.py`, `experiments/smoke_adapter.py` —
   repo-root-relative import resolution only.
5. `adapters/sandbox_eval_client.py` — **US-007 cancellation repair**: the
   legacy module-level `submit_and_wait` raised `TimeoutError` on wait
   timeout without cancelling the remote job. The client is now a stateful
   `SandboxEvalClient` (module-level API preserved as wrappers over a shared
   default instance) with a thread-safe live-job registry; wait timeouts and
   poll aborts cancel via `DELETE /api/v1/jobs/{job_id}` first and attach the
   cancel evidence to the raised error. `cancel_job`/`cancel_all_live` never
   raise. The source-gate delegation is unchanged. Missing endpoint/API key
   now raises a explicit `ValueError` instead of a bare `KeyError`.

## Forward-only secret sanitization (2026-10-08)

A real-looking `mlsandbox-…` sandbox key was found hardcoded in tracked
OpenMLE-Gym sources and replaced with environment reads / `<sandbox-api-key>`
placeholders (working tree only; Git history untouched, no server changes):

- `OpenMLE-Gym/openmle-sandbox/README.md` (9 occurrences → placeholder)
- `OpenMLE-Gym/openmle-sandbox/node_client/test_titanic/test_1shot.py`,
  `test_parallel.py`, `test_canclejob.py` (→ `os.environ["SANDBOX_API_KEY"]`)
- `OpenMLE-Gym/openmle-sandbox/node_controller/api_server/api_server.py`
  (`valid_keys` now reads `SANDBOX_API_KEYS` env, same as the already-sanitized
  `sandbox-controller/api_server/api_server.py`)
- `OpenMLE-Gym/openmle-sandbox/node_router/gateway.py` (admin credential check
  now reads `SANDBOX_API_KEYS` env)

These files therefore appear in `provenance/inventory-diff-wsl-vs-windows.json`
as WSL-vs-Windows mismatches — intentional; the live Windows deployment is
untouched. The same key remains in Git history and possibly on the retired
server; rotation is out of scope (no server operations permitted).

## Reproducing the evidence

Use the offline research venv (see `ENVIRONMENT.md`):

```bash
.venv-research/bin/python -m pytest research/tests -q
.venv-research/bin/python -m mypy --config-file research/mypy.ini --follow-imports=silent \
    research/adapters research/tools research/tests research/experiments \
    research/contracts research/evaluator research/search
.venv-research/bin/python research/tools/inventory.py --write --manifest
.venv-research/bin/python research/tools/audit_secrets.py \
    --json research/provenance/secret-audit.json
.venv-research/bin/python research/tools/audit_formal_baseline.py --check
git diff --check
```

Import checks (adapters on both interpreters; vendored packages functional):

```bash
.venv-research/bin/python -c "import research.adapters.rsi_eval_adapter"
<legacy-root>/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
sys.path.insert(0, 'research/vendor/istratde/src')
sys.path.insert(0, 'research/vendor/metade/src')
from istratde.algorithms.jax import IStratDE
from metade.algorithms.jax import MetaDE
print('vendored imports ok')"
```

## Rules

- No venvs, private data, credentials, or historical generated experiment data
  (`formal_baseline/`, `outputs/`) in Git — enforced by the root `.gitignore`
  and re-checked by `tools/audit_secrets.py`.
- Python sources under `research/` must not contain host absolute paths; the
  audit fails on them. Machine layout belongs in docs and `provenance/`.
- Origins stay read-only; differing files are surfaced by
  `provenance/inventory-diff-wsl-vs-windows.json`, never synced blindly.

### Search acceptance control (US-007)

`research.search.run_control.run_guarded_search` installs guards on the real
Evo operator clients and a native before-operator stop hook. It cancels
registered jobs and writes a control checkpoint on completion or failure.
The caller supplies the single persistent budget ledger; this entrypoint
never creates or resets one. The validation profile is
`OpenMLE-Evo/tts_search/configs/experiment/trustworthy_validation.yaml`.
Its Kimi profile stores a placeholder credential; fresh managed credentials
must be assigned to the client in memory, never saved in config or logs.

The transport reserves a conservative UTF-8 request bound and caps the
actual output limit. Stream consumption and socket/total deadlines are
inside the guarded attempt. Missing, estimated, fractional or inconsistent
usage stops the ledger with the reservation retained. A provider that
ignores cancellation may finish only its already-reserved attempt; no next
request is allowed. Remote job cancellation requires explicit dispatcher
cleanup evidence. Unknown submissions and failed cancellation stay tracked.

Offline proof: real solver branches for all five retained dimensions,
three inner generations with two candidates, source/parent hashes, native
stop/checkpoint/resume, and real backend transport spies. These tests are
not evidence of real-model operator coverage; US-009 still must prove it.
Actual cancellation acknowledgements and filesystem isolation are gated by
US-008. No real acceptance request was made during US-007.

### Real acceptance runner (US-009, preparation)

`research.search.acceptance` is the parent controller for the one bounded
real acceptance; `research.search.acceptance_inner` is the per-config inner
Evo child. Hard rules implemented by the runner:

- `live` refuses (exit 2, machine-readable rule) BEFORE ledger/credential/
  model activity unless the ignored review file
  `.runtime/us009-runner-review.json` (schema `us009-runner-review.v1`)
  binds the current git HEAD with `approve_real_acceptance: true` on a fully
  clean tree (untracked implementation files count as dirty; ignored runtime
  files do not). The runner never creates that file; the supervisor writes
  it after reviewing the committed runner.
- `preflight` makes no model calls: it never creates the persistent ledger
  (`.runtime/acceptance-us009/budget-ledger.json`), never reads model credentials
  and sends no model request. It checks local sandbox authentication configuration. It validates the pinned Evo/litellm YAMLs,
  DERIVES the native vendored-iStratDE first ask directly (CPU JAX, seed 42,
  pop 4, vendor-path asserted) — `.runtime/outer-ask-preflight.json` is only
  an optional cross-check — and verifies public-data hashes, the evaluator
  registry, the gateway and the auth file permissions.
- One persistent ledger for the whole acceptance (200000 tokens / 30
  requests / 5400 s, continued never reset) at the ONE fixed location; the
  parent never holds the ledger flock while a child runs. The child's
  cancellation token anchors at `ledger.deadline_epoch` (persisted
  started_at + persisted limits), never `time.time()+remaining`.
- Each operator makes one generation attempt, with current managed credentials
  read immediately beforehand. Repairs use the real Evo debug branch and share
  the same budget. HTTP retries remain disabled.
- Managed OAuth refresh is reproducible across the 90-minute window:
  `ManagedKimiCredentials` re-reads the credential file per request and, on
  expiry, refreshes through an official loopback Kimi web instance
  (discovered from registered instances, or an owned `kimi web` started on a
  dynamically chosen port; only the authenticated non-model
  `/api/v1/oauth/usage?provider=managed%3Akimi-code` query is used; owned
  helpers are shut down in a finally block, reused ones left alone).
- Emergency cleanup: the child persists submission intents (before POST) and
  live job ids (before polling) to `live-jobs.json`; the parent's child
  timeout is the shared ledger deadline + bounded grace, kills ONLY the
  child process, then cancels recorded ids through the real API client
  requiring explicit cleanup proof; unknown identities are never claimed
  clean and refuse continuation. Evidence lands in emergency-cleanup.json
  and the acceptance checkpoint.
- Run-index reuse requires the full evidence chain on EVERY get: operator
  trace, request trace, journal (generated code), run-result (hash + parsed
  consistency) and per-verified-job prediction snapshot bytes (copied into
  the run dir, hash-bound to the evaluator's prediction_sha256); identity
  binds policy CONTENT, public-data and registry identities plus
  config/model/seed/commit; reusable runs must complete >=3 generations and
  >=6 main candidates with a trusted verified best result and an unstopped,
  fully-resolved ledger.
- Orchestration: standalone inner run on the first decoded vector, then a
  fresh seeded native ask (must reproduce), the standalone run reused as the
  outer loop's first verified evaluation (zero new model calls, explicitly
  recorded in `cache_events` against ledger snapshots), three more
  real inner runs, one explicit `tell` with `fitness=-accuracy`. Verdict
  `complete` requires all four runs completed, real improve>=1 and
  crossover>=1 coverage, and a healthy within-deadline ledger at the end.
  The inner solver RNGs (Python random + NumPy) are seeded with 42 and the
  seeding is recorded; provider generation remains nondeterministic (no
  unsupported provider seed is sent).

Invocations:

```bash
# preflight (legacy JAX venv; no model/ledger, local gateway checks only)
JAX_PLATFORM_NAME=cpu PYTHONPATH=$PWD <legacy-root>/.venv/bin/python \
  -m research.search.acceptance preflight --repo-root $PWD

# the one real acceptance (only after the supervisor review file binds HEAD)
PYTHONPATH=$PWD <legacy-root>/.venv/bin/python \
  -m research.search.acceptance live --repo-root $PWD
```

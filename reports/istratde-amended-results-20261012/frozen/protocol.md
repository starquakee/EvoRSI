# PRD: EvoRSI iStratDE controlled pilot comparison

## Introduction
The human approved this full plan on 2026-10-10, after five explicit design choices. Implement it using Kimi CLI one story per supervised iteration, then conduct ONE bounded real experiment. The question is whether iStratDE improves independent held-out classification accuracy under equal candidate opportunities and budget ceilings. Positive, null and inconclusive outcomes are valid; implementation alone is not completion.

## References
- Official UCI: https://archive.ics.uci.edu/dataset/267/banknote+authentication (1372 rows,4 features); https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (569,30); https://archive.ics.uci.edu/dataset/94/spambase (4601,57). CC BY 4.0, record actual official raw source and hashes.
- Independent evaluation: https://jmlr.org/papers/v11/cawley10a.html
- Random baseline: https://jmlr.org/papers/v13/bergstra12a.html
- Existing contracts: research/search/acceptance.py, research/search/inner_run.py (inspect actual files), research/contracts/, research/evaluator/, OpenMLE-Evo/tts_search/, deploy/rsi-trustworthy/worker/.

## Decisions
Three tasks (banknote UCI267, wdbc UCI17, spambase UCI94), three methods FIXED/RANDOM/ISTRATDE, three search seeds 11/22/33: 27 cells. Each cell has eight independent inner searches, each three generations and two main candidates/generation. Six main requests plus at most two additional debug requests/inner. This is 216 searches,1296 planned main requests,at most432 debug. Global hard limits1728 requests,5400000 tokens,86400 seconds. Per-cell limits64 requests,200000 tokens; no quota borrowing. Clock starts at actual supervised live start, includes final phase, not coding. Coding CLI usage is separate. Retain both historic ledgers byte-for-byte.

K3 model via official managed credentials; reasoning_effort=low,max_tokens=8192,temperature=1,top_p=.95,stream=true,allowed_openai_params=['reasoning_effort'],transport retries0. Same prompt,tools,resource settings for all methods; serial model calls and one CPU worker slot. No model result sharing, free cache reuse or candidate inheritance between configs. K3 output is not deterministic. Equal opportunity/caps does not mean equal realized token use.

FIXED theta=(.5,1.0,.4,.25,1). Bounds crossover[.05,.95], score/delta/novelty weights[0,2] each, debug integer[0,2]; consistent encoding/rounding, weight sum>0 gate. RANDOM first four vectors equal paired ISTRATDE initial population but all evaluations independent; remaining four uniform independent vectors. ISTRATDE pop4, two full ask/evaluate/tell batches; second ask inherits first tell state. Inner seed derived deterministically from task,searchseed,configindex, identical across methods at same position. Freeze schedule before generation: for each config index, task and search-seed block, rotate method order; all first four evaluations for a cell precede its fifth. A simple deterministic rotation indexed by task/seed/config is acceptable and must be serialized before live start.

## Goals
Trustworthy 27-cell comparison, independent test labels, complete accounting and evidence, reproducible analysis. Failure raw scores=null; predeclared utility=accuracy on success,0 on failure. Report both all-cell utility and success-only accuracy with failures explicit.

## Functional Requirements
FR1. Official data conversion removes original IDs; numeric features,label0/1 (WDBC B=0 M=1). Duplicate feature vectors grouped together. One fixed five-fold stratified GROUP split per task,seed20261010+UCIid: train folds0/1/2,val3,final4 (~60/20/20); identical all methods/seeds. No split tuning.
FR2. Search worker sees train labels+validation features only. Trusted evaluator holds validation labels. Raw files,final features and final labels excluded from worker mounts. Grant each candidate only current job source and its approved current dataset, plus approved runtime/library paths.
FR3. All27 search cells must be terminal and exact best code frozen before final features can be published. Winning code highest validation score,tie earliest. Refitting via replacing generic train.csv with train+validation, test.csv with final features preserves code bytes. Final labels never worker-visible. Once per cell final score, no LLM feedback afterward.
FR4. Separate research/comparison module with prepare/preflight/run/resume/report. Specs bind task/split/method/seed/config/model/implementation/budget. Do not relax original US009 fixed-task authorization.
FR5. Atomically reserve global+cell+inner request allowance and conservative tokens before every real request including failures/debug; commit known provider usage once. Unknown usage,unresolved reservation,source mismatch or unproven cleanup globally stops. Cell limit freezes incumbent and records shortfall. Global cap stops and records INCOMPLETE, never auto-expands.
FR6. Resume only completed config boundaries, original ledger/deadline, verified immutable artifacts. Interrupted config never silently reruns/replaces seed. Fail closed on missing/corrupt evidence.
FR7. Predeclared analysis primary ISTRATDE-RANDOM,secondary ISTRATDE-FIXED,within task/seed pairing and equal-weight task macroaverage. Bootstrap10000 using RNG20261010 resamples paired seed indices independently within each task, percentile95% intervals conditional on the three fixed test sets, only3seeds weak/exploratory evidence. Plot validation incumbents vs actual accumulated requests/tokens; never repeated test feedback curves.

## Non-Goals
No MetaDE,no extra tasks/seeds,no continuous-long-run alternative,no split/budget changes for positive results,no automatic publication. No host execution of generated candidates. No modification of old6580 stack,Windows deployed mounts,oldGPU environments,servers,or historical acceptance evidence. Never push this original secret-bearing Git history; future publication uses separate clean history.

## Validation
Each story must prove its criteria below. Before live: full owned research/Evo tests, scoped mypy,secret audit,actual new-stack isolation/data/schema probes,old-ledger/manifest immutability,source/cleanup identities. Supervisor reviews exact implementation commit and immutable manifest then starts one live experiment. Report independently recomputes final prediction accuracy and all ledger/provider charges. No invented scores or fake live completion.

## Risks and implementation guidance
Existing _NativeOuterAlgorithm.ask() reinitializes every call: unsuitable for multibatch use without a new stateful implementation. Existing InnerRunSpec is hello_synth/seed42-specific: create separate comparison contracts. Old trampoline PUBLIC_READ_ROOTS broadly permits /mnt/rsi_storage and /mnt/rsi_data: must narrow trusted grants before comparison.
Use repo .venv-research for owned tests/mypy,OpenMLE-Evo/.venv for Evo. Native iStratDE needs existing <home>/rsi-gpu/.venv (JAX0.4.33); research venv has incompatible newer JAX. Do not upgrade old environments. Root paths should be configured/resolved rather than new private absolute literals in reusable modules.
No open research-design questions remain. Routine implementation decisions are delegated within frozen protocol. Debug depth may limit actual extra calls below2; all methods still share caps. Final failures remain null with explicit reason.

## User Stories

### EXP-001: Frozen protocol and typed comparison specification
As a researcher I need frozen protocol and typed comparison specification with auditable evidence.
- [ ] Create research/comparison as a separate entry point; typed immutable protocol, cell and inner-run specs bind task split hashes, method, seed, config index, model profile, source commit and budgets; strict finite/range/type validation and canonical identities.
- [ ] Protocol fixes all approved values below, 27 cells and 216 independent inner runs; prepare/preflight/run/resume/report command surfaces exist, unimplemented execution refuses safely without model calls or live ledger creation.
- [ ] Keep US009 interfaces, original authorization and both original ledgers unchanged; test tampering, invalid configuration and canonical identity changes.
- [ ] Focused substantive tests pass; Typecheck passes.

### EXP-002: Official datasets, fixed grouped splits and evaluator registration
As a researcher I need official datasets, fixed grouped splits and evaluator registration with auditable evidence.
- [ ] Fetch the three official UCI datasets; save private raw files with URL/license/sha256, documented label conversion and ID removal. Deterministically split once using StratifiedGroupKFold five folds, seed 20261010+UCI ID; identical numeric feature rows share a group; train folds 0/1/2, validation 3, final 4.
- [ ] Prepare train labels and validation features only in worker-visible search data; validation/final labels and final features remain outside every worker mount. Final train+validation publication must require the all-cell frozen barrier.
- [ ] Register trusted external evaluators and hash-bound task manifests; tests prove no row/group overlap, repeatable byte identities, malformed data rejection and no sealed-data exposure. Verify real downloaded shapes.
- [ ] Focused substantive tests pass; Typecheck passes.

### EXP-003: Narrow current-job and current-task worker read grants
As a researcher I need narrow current-job and current-task worker read grants with auditable evidence.
- [ ] Replace broad /mnt/rsi_data and /mnt/rsi_storage candidate read roots with exact current immutable input and approved task data grants, delivered through trusted dispatcher/worker control metadata, never candidate-supplied environment or arbitrary data_paths.
- [ ] Preserve current-job execution and hello_synth compatibility without broad fallback; validate canonical paths, symlinks, traversal, cross-job/cross-task denial and final-data denial. Candidate process must be unable to read other methods code/predictions.
- [ ] Prove actual read denial using synthetic canaries in the real new worker, not string checks alone. New-stack-only recreation is allowed after idle verification; preserve all six old containers. No host UID/process sweeps; verify dedicated container PID namespace before cleanup.
- [ ] Focused substantive tests pass; Typecheck passes.

### EXP-004: Atomic global and per-cell budget accounting
As a researcher I need atomic global and per-cell budget accounting with auditable evidence.
- [ ] One persistent atomic ledger reserves both global and cell quotas before every model attempt, with max 6 main+2 debug per inner run, max 64 requests/200000 tokens per cell, 1728/5400000/86400 seconds globally; no borrowing. Retain provider input/output/total usage and durable request identities.
- [ ] Unknown usage, unresolved reservation, nonfinite/corrupt accounting, source mismatch or cleanup failure stops globally; cell quota exhaustion terminates that cell truthfully; deadline starts only on actual supervised experiment, persists across resume and is never reset.
- [ ] Meaningful race, crash/reopen, quota, over-reservation and deadline tests; use conservative full prompt plus output reservation, SDK retries zero; interface supports existing guarded K3 transport without double charging.
- [ ] Focused substantive tests pass; Typecheck passes.

### EXP-005: Generalized inner search with complete execution evidence
As a researcher I need generalized inner search with complete execution evidence with auditable evidence.
- [ ] Implement new comparison runner for arbitrary registered task and paired inner seed, preserving original acceptance runner. Reuse real Evo operators, source gate, isolated sandbox/external score, generation outcomes, provider usage and cleanup verification.
- [ ] Each config starts fresh memory-off search: three generations with two main candidates each, max two debug requests, same K3 low/8192/temperature1/top_p0.95/stream/allowed_openai_params profile; use generic train.csv and test.csv names so exact winning bytes can refit train+validation at final phase.
- [ ] Strict source, dataset, spec, quota, prediction and cleanup identities; no cached/shared model results. Offline integration tests verify real operator routing, seeds/task data, errors and exact request counts; no real generation calls.
- [ ] Focused substantive tests pass; Typecheck passes.

### EXP-006: Three methods, stateful iStratDE and frozen search schedule
As a researcher I need three methods, stateful istratde and frozen search schedule with auditable evidence.
- [ ] FIXED repeats default theta eight fresh times; RANDOM initial four match paired native iStratDE initial vectors, final four independently uniform in encoded bounds; RANDOM/FIXED never use scores to choose configs.
- [ ] Native iStratDE pop4 completes two actual ask/evaluate/tell batches; initialize once and preserve state into second ask. Do not loop the existing _NativeOuterAlgorithm which reinitializes each ask. Use existing JAX0.4.33 environment; real native state tests prove feedback affects second batch.
- [ ] Freeze deterministic interleaved 216-config schedule with rotated method order and paired seeds; persist evidence/state at completed config boundaries only. Interrupted config cannot silently rerun. Freeze highest validation winner per cell, tie earliest; global final barrier checks all 27 terminal cells and exact code hashes.
- [ ] Final evaluation runs each frozen exact program once with train+validation and final features; no LLM afterward, labels trusted only, raw failures null and utility zero. Tests cover tampering, premature final, duplicate final, safe resume and missing artifacts.
- [ ] Focused substantive tests pass; Typecheck passes.

### EXP-007: Aggregate offline verification and real no-model preflight
As a researcher I need aggregate offline verification and real no-model preflight with auditable evidence.
- [ ] Implement prepare/preflight and complete run/resume/report wiring with an ignored supervisor review bound to exact current commit and full immutable experiment manifest; no live model/auth/ledger creation before review acceptance.
- [ ] Run aggregate owned research tests, Evo tests and scoped mypy, secret audit, source/mount identity checks, old-ledger artifact audit, and real no-model sandbox probes for all three task schemas/current-job grants/sealed final denial/cross-job denial/cleanup.
- [ ] Persist and document preflight evidence plus exact invocation. Commit implementation locally; supervisor independently verifies and freezes review. Never write the supervisor approval yourself; leave any missing gate visibly failing.
- [ ] Focused substantive tests pass; Typecheck passes.

### EXP-008: Supervised live experiment, independent final audit and report
As a researcher I need supervised live experiment, independent final audit and report with auditable evidence.
- [ ] Coding phase may implement missing offline analysis/report/audit support only, then commit and stop with passes=false. Only supervisor starts the one approved experiment after EXP001-007 review; no automatic new round or quota increase.
- [ ] Execute planned 27 cells subject to hard stops, freeze all winners before final evaluation, independently recompute all accepted final scores and verify code/data/prediction/usage/cleanup evidence; report incomplete matrix truthfully if capped or stopped.
- [ ] Report all 27 rows, raw accuracy/null, utility, failures, realized cost and completion counts; primary paired iStratDE-RANDOM, secondary iStratDE-FIXED, equal task macroaverage; 10000 within-task paired-seed bootstraps with RNG20261010 and exploratory fixed-test-set caveat.
- [ ] Deliver frozen protocol, redacted JSON/Markdown, paired comparison plots and validation-incumbent-vs-actual-cost curves, audit manifest and honest positive/null/inconclusive conclusion; verify plot rendering; local commit only. passes=true requires real supervised evidence, never fixtures alone.
- [ ] Focused substantive tests pass; Typecheck passes.

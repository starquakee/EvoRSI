# PRD: Trustworthy RSI minimum loop

## Introduction
Implement the user-approved 2026-10-07 first-round plan. Current infrastructure works but prior 8-D experiment ran only draft/debug. Goal is a reproducible loop, not a performance claim.

## References
Read AGENTS.md, HANDOFF.md, RUNBOOK.md and source entrypoints. Historical docs may be stale; actual inspect/trace wins. Original experiments live <user-home>/rsi-gpu/formal_baseline. Windows legacy stack is rollback only.

## Decisions
- Sole new source root <user-home>/openrsi-local/OpenRSI; existing Windows live stack and original rsi-gpu environments/results untouched.
- WSL Kimi CLI <user-home>/.kimi-code/bin/kimi; fresh turn per story, 3-iteration supervised batches, no unsafe flags.
- Local Git commits only, no pushes/remotes. No secrets/runtime/private artifacts in Git.
- Real evaluation budget 200000 total tokens,30 model requests,5400s, whichever first; programming agent usage separate.
- Candidate code NEVER executes on development host; no secrets on candidate worker; independent evaluator and admission policy immutable to optimizer.
- Port 6581 loopback, separate compose/network/volumes. One shared CPU/GPU worker lease. Old 6580/10150 deployment untouched.
- Approved new implementation supersedes stale AGENTS clauses claiming only Windows may be edited; do not change Windows source, only new WSL implementation.

## Goals
Reproducible source, genuine operator activation, pre-execution checks, independent scoring, correct cost/budget accounting and actual bounded model acceptance.

## Non-Goals
No large MetaDE run, public hosting, security/algorithm superiority claims, server operations, global proxy edits, pruning, broad refactors or GPU/daemon resets.

## Functional Requirements
FR1 preserve API submit/query/cancel semantics; machine-readable rejection and scoring failures.
FR2 bind source hash/policy version; metric direction explicit; fail closed on missing evidence.
FR3 immutable raw history and independent cache-cost accounting; reserve each call/retry, persist ledger, cancel on stop.
FR4 least exposure worker, private answer/metric not mounted; predictions bounded and validated.
FR5 every story has focused tests/type validation and commit evidence; never pass from mocks alone where real integration required.

## Validation
Each story below supplies independent acceptance. Real model calls allowed ONLY US009 after actual isolated stack passes US008. End-to-end failures remain false. Implementer records exact reproducible validation commands.

## Risks and Stop Conditions
Kimi print mode is not OS sandbox. Do not use unsafe flags. Stop on auth failure, two consecutive failed attempts of same story, unknown token balance, budget exhaustion, incompatible worker security restrictions, unapproved external writes. Diagnose or split story without claiming completion. Budget exhaustion does NOT authorize retry with fresh ledger.

## User Stories

### US-001: 工程基线与来源清单
- [ ] Inventory WSL OpenRSI, legacy rsi-gpu and Windows deployed sources by path/hash; do not overwrite differing files blindly; keep credentials out of inventory output.
- [ ] Consolidate legacy adapters and experiment scripts under research/ and iStratDE/MetaDE required src, pyproject and LICENSE under research/vendor; no venv, private data or historical generated code in Git; preserve origins and license notices.
- [ ] Document environment dependencies and lock/snapshot versions without credentials. Existing venv may be reused for tests but repository paths must resolve relative to root.
- [ ] Provide a reproducible secret/path audit and provenance manifest; git diff --check and focused import/type checks pass.

### US-002: 旧实验独立审计
- [ ] Read original <user-home>/rsi-gpu/formal_baseline read-only; generate separate report without rewriting old records.
- [ ] Verify 60 records,46 unique configs,14 cached,44 actual Evo runs,2 pre-run rejects; all actual operators are draft=44/debug=15; no improve/crossover.
- [ ] Uncached recorded tokens=930962 and execution wall sum=15345.9 seconds; all-row tokens=1315148 is duplicate-inclusive, not cost. Distinguish score counts and cached results.
- [ ] Meaningful tests for cached/failed records and missing fields; provide executable audit command, typecheck, tests and git diff --check.

### US-003: 结果、缓存和预算契约
- [ ] Define common result schema with metric direction, source hash, policy version, safety verdict, status/reason, cache origin and incremental costs; accuracy maximizes and logloss minimizes.
- [ ] Missing artifacts, nonzero exit, nonfinite score, absent safety verdict fail closed; cache hit incurs zero incremental calls/tokens/time, retains original provenance. Cache key includes seed/model/task/data/metric/policy/config identities.
- [ ] Persistent serial budget ledger enforces request reservation before EVERY model attempt including retries: 200000 tokens,30 requests,5400 elapsed seconds. Reserve conservative input estimate plus max output; retain reservation/stop on unknown usage. Never reset budget on restart.
- [ ] Implement cancellation hooks for deadline, no new calls after stop, checkpoint resume and tests for all boundaries including restart and retry. No real model calls in this story. Typecheck and focused tests pass.

### US-004: 统一执行前源码检查
- [ ] Reusable versioned source policy checks immutable submitted bytes, hashes them and returns machine-readable denial; keyword matching alone must not be represented as a security boundary.
- [ ] Wire both actual and deprecated Evo submission paths in OpenMLE-Evo/tts_search/reward_func_utils.py and API admission before writes/queue; cover inline code and code_file_path without TOCTOU; all draft/debug/improve/crossover use same gate.
- [ ] Do not modify Windows live API. WSL API auth must read environment, never embed keys. Rejected inputs neither enqueue nor execute; missing policy/artifact fails closed.
- [ ] Focused tests spy on enqueue and submit, exercise malicious markers/path inputs and legitimate ML code, plus typecheck. Preserve existing job query/cancel semantics.

### US-005: 可信独立评分服务
- [ ] Implement evaluator separate from candidate worker; answers and trusted metric code are never mounted into worker, including via parent directory or alternate path.
- [ ] Evaluator accepts bounded CSV prediction bytes plus allowlisted task ID, never runs submitted Python or imports candidate-supplied metrics. Reject symlinks, traversal, oversized and malformed files.
- [ ] Integrate dispatcher to score externally after candidate process stops; validation/test split explicit; use synthetic hello_synth fixture. Trusted evaluator image/dependencies pinned and no model keys.
- [ ] Unit/integration tests for scoring and tamper/read rejection; do not claim complete until US006/008 validates actual container boundary. Typecheck passes.

### US-006: WSL 独立隔离部署
- [ ] New Compose project rsi-trustworthy, 127.0.0.1:6581 gateway, WSL source/data mounts, unique volumes. Keep all old containers/volumes/files unchanged.
- [ ] One worker, unified mutex/lease across CPU/GPU queues. Worker has no published host ports or docker socket; public input read-only and per-job scratch only. Separate internal networks prevent worker access to DB/Redis/evaluator/gateway/model credentials or internet.
- [ ] Default seccomp, bounded CPU/memory/pids/runtime, minimum capabilities. No unconfined/privileged fallback. Inspect actual worker image compatibility; record and repair any precise issue instead of broad weakening.
- [ ] Create private local auth outside Git, expose no credential in output. Use trusted launch/testing commands outside candidate worker. Real docker compose config/up and inspect mount/port/security/network validation; old stack stays healthy.

### US-007: 搜索参数和算子生效证据
- [ ] Dedicated validation Evo config: >=3 inner generations and >=2 candidates each; trigger improve/crossover only with valid parents; distinguish outer DE generations from inner generations.
- [ ] Instrument actual operator call sites with source/parent hashes and configuration. Deterministic offline tests prove relevant branch changes when each retained searched parameter changes; drop inactive dimensions, explain mapping.
- [ ] No artificial trace entries: tests assert real implementation branch/call counts, including no-parent and before-threshold cases; fixed operator fixtures may be used only for offline tests.
- [ ] Budget ledger wired into real model transport and retries before next story; checkpoint and cancellation tested; typecheck and tests pass.

### US-008: 固定候选真实安全闭环
- [ ] Run trusted synthetic fixture via new real API and containers; persist evidence of submitted/running/completed and numeric score.
- [ ] Probe scoped harmless access to synthetic answer/metric paths, network egress, symlinks/path traversal, output size, timeout/cancel. No real secrets or arbitrary exploit payloads. Admission-denied case must have no execution.
- [ ] Prove single worker never executes CPU/GPU jobs concurrently; timeout/cancel kill process descendants and next benign job works; scoring follows stopped execution with immutable prediction snapshot.
- [ ] Record docker inspect boundary evidence and old-stack unaffected check; fail story if network/answer isolation or cleanup unproven; typecheck/tests pass.

### US-009: 真实模型小额验收
- [ ] Only after all prior gates pass, use existing local Kimi credentials in memory on controller side (never worker/git/logs), model k3 by existing setting. One ledger for entire acceptance; 200k tokens/30 requests/90min including retries, no reset.
- [ ] Run one >=3-generation/2-candidate inner Evo, then smallest supported iStratDE outer ask/evaluate/tell closed loop on same synthetic task, using retained active policy dims and real LLM+sandbox evaluations.
- [ ] Actual draft/debug/improve/crossover coverage must be reported honestly; deterministic fixtures are not real evidence. If real crossover/improve absent or budget unavailable mark incomplete, checkpoint and STOP; do not increase budget.
- [ ] Persist redacted request usage, operator trace, job IDs, score direction, source/policy hashes, rejection and cost evidence; all scores through isolated evaluator. No algorithm superiority claims.

### US-010: 切换、回退与交接
- [ ] Only after prior acceptance passes, point local research client default to verified 6581; preserve old endpoint explicit rollback, no deletion/stopping of old stack.
- [ ] Update AGENTS/HANDOFF/RUNBOOK to exact current mounts/commands/limitations/evidence, remove obsolete claims. Document reproducible dependency restore and startup/stop new-stack-only commands.
- [ ] Run aggregate tests, mounted-source audit, budget ledger audit, secret scan, old/new health and rollback/client switching check; all must pass.
- [ ] Commit locally only, no remote push. Produce concise report with evidence/commit links and pending next-stage comparisons/MetaDE work.

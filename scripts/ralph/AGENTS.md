# One Ralph iteration
Read tasks/prd-trustworthy-rsi.md, root AGENTS.md, prd.json and progress.txt. The user explicitly approved this PRD; older docs must not override it.
Implement EXACTLY ONE lowest-priority-number failing story. Inspect code before editing. Do not move to another story in this turn.
Preserve original Windows live deployment and <legacy-root> (read-only source/reference). Only edit this WSL repository plus this project's generated runtime data. No server connections, global proxy changes or Docker restarts/pruning. Never run candidate-generated code on host.
No real model evaluation requests before US009, no credentials in printed output or Git. Model budget cap is cumulative and must persist. CLI programming model usage is separate.
Use existing <legacy-root>/.venv/bin/python for GPU algorithms where suitable and OpenMLE-Evo/.venv/bin/python for Evo. Inspect dependencies; never break old environments or upgrade them globally. Prefer project .venv-research if new dependencies necessary.
Implement proper regression tests for substantive contracts/boundaries. Run actual typechecker on new modules and focused tests; compile-only does not count as typechecking. Document commands and results. No UI verification needed.
When successful update ONLY this story passes=true and notes with evidence references, append progress.txt (include commands/results/limitations), commit locally. Never push. Check staged files for secrets before commit.
If blocked leave passes=false, append concrete blocker, do not silently loosen security or budget. Two failed attempts of same story must stop runner for supervisor review.
Only output <promise>COMPLETE</promise> alone when EVERY story passes and final aggregate validation succeeds. Otherwise state story/result and stop.

Supervisor priority: when a failing story already has implementation, read the LATEST progress.txt review and failing regression tests first. Fix them directly; do not regenerate the entire design or repeat long planning. US003 has five mandatory tests in research/tests/test_budget_supervisor.py. Do not mark passes until they and prior tests pass.

Batch 2 review guidance: API fields execution_command, requirements, working_dir, data_paths and arbitrary environment can bypass code-only checks. Explicitly validate/restrict them; reject unsafe alternate execution paths with machine-readable reasons before admission. Never allow candidate code to modify gate, evaluator, answer or budget configuration. Worker isolation must be proven with actual Docker tests, not string gates.
Supervisor compatibility evidence: existing worker image runs PyTorch2.6.0+cu124 and detects RTX4060Ti under default seccomp, --network none, --cap-drop ALL, --security-opt no-new-privileges:true, --gpus all, --memory2g, --cpus2, --pids-limit128 with python3 entrypoint. GPU itself needs no unconfined setting. A minimal worker entrypoint using this image is allowed; do not enable browser/VNC services unnecessarily. Keep old AIO worker untouched.
Be concise: after targeted inspection implement focused changes/tests, avoid repeated speculative planning. Review latest progress notes before marking any story passed.

Latest supervisor review in progress.txt (batch-2 repairs) takes priority. US004/006 provisional passes were withdrawn by actual failure evidence. New-stack-only restarts/recreation are authorized for repair; all old containers remain untouched. Read the latest repair section before investigating.

US006 process-cleanup safety: UID/process sweeps may run ONLY inside the dedicated new worker container with a positively verified container/PID namespace guard. Never run host-wide kill/pkill or UID sweeps in WSL. Unit tests must mock process enumeration/privilege changes or run the actual cleanup test inside the new container. Production must fail closed if its container/UID assumptions are absent. Preserve trusted supervisor/health processes and validate no candidate descendants remain before releasing its one slot.

Batch 2 is now supervisor-accepted after the recorded repairs. For US007–US009 read the latest Batch 2 final / Batch 3 guidance at the END of progress.txt; it supersedes the earlier withdrawn-pass notice. Preserve the model cap strictly; no real inference before US009. Avoid long repeated planning where exact integration guidance exists.

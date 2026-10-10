# EvoRSI

**面向递归自改进的演化搜索实验平台。**

An experimental platform for evolutionary code search and recursive self-improvement,
with isolated execution, independent evaluation, and explicit budget tracking.

EvoRSI connects an inner Evo code-search loop with an outer iStratDE configuration
search. It brings together the existing OpenMLE components and the local RSI
research implementation. MetaDE sources are retained for future research.

## What is implemented

- Real draft, debug, improve, and crossover operator instrumentation.
- Versioned source admission checks and submitted/executed source identity binding.
- A separate trusted evaluator that scores bounded prediction files.
- An isolated Docker worker with per-job filesystem restrictions and verified cleanup.
- A shared CPU/GPU execution slot, cancellation, and persistent model-budget accounting.
- Cache provenance with zero incremental model cost for verified reuse.

## Current status

**Research prototype — the initial ten-story implementation and bounded real-model acceptance are complete.**

The second, explicitly authorized acceptance round completed four real inner
runs, each with **3 generations × 2 main candidates**, followed by one native
iStratDE ask/evaluate/tell update. **24 model requests** used **49,903 tokens**
in about **11 minutes 17 seconds**. All **24 sandbox jobs** received independently
verified scores; best accuracy per configuration was **92.5%, 92.5%, 92.5%, 90%**.
Observed operators were draft 8, improve 6, crossover 10; no debug call was
needed in this round. Reusing the first run in the outer loop added zero model cost.

The research client now defaults to the isolated local gateway **127.0.0.1:6581**,
with explicit legacy **6580** rollback configuration. Default task/data routing,
source identity, independent scoring, and cleanup were verified with fixed
synthetic jobs. The old stack remains preserved. Rollback verification covers
request configuration and read-only reachability; it does not claim a new
end-to-end legacy candidate evaluation.

The first round remains documented as **incomplete**: 17 requests, 81,717 tokens,
7 executed jobs (6 scored and 1 execution failure). Its stopped ledger and
incomplete unscored-node artifact were not rewritten. The two rounds total
**41 model requests and 131,620 tokens**.

These are integration and boundary checks on a **120-row training / 40-row test
synthetic classification task**, with one seed and one outer update. Scores fed
back into search are not independent generalization evidence. No algorithmic
superiority is established; equal-budget, multi-seed comparisons and MetaDE
integration remain future work.

See the [second-round acceptance](reports/us009-round2-acceptance.md),
[completed handoff](reports/us010-switch-handoff.md), and
[preserved first-round report](reports/us009-real-acceptance.md).

## Repository layout

| Directory | Purpose |
|---|---|
| `research/` | Search adapters, contracts, evaluator, acceptance runner, tests, and vendored optimizers |
| `OpenMLE-Evo/` | Evolutionary code-search implementation and its AIRA-Evo dependency |
| `OpenMLE-Gym/` | Task construction and evaluation tooling |
| `sandbox-controller/` | API and dispatcher components |
| `sandbox-workers/` | Worker image/build sources |
| `deploy/rsi-trustworthy/` | Isolated local stack and boundary probes |
| `tasks/hello_synth/` | Small public synthetic input fixtures |
| `reports/` | Redacted implementation and acceptance evidence |

## Start here

- [Research components and validation](research/README.md)
- [Isolated deployment and probes](deploy/rsi-trustworthy/README.md)
- [OpenMLE-Gym setup](OpenMLE-Gym/README.md)
- [Publication scope and reproducibility notes](docs/PUBLICATION.md)

The deployment targets a Linux/WSL environment with Docker and NVIDIA GPU support.
Environment snapshots and component dependency files are included; virtual
environments, credentials, images, and runtime state are not. Live acceptance
requires local credentials, a reviewed configuration, and an explicitly authorized
budget. It does not start automatically from a clone.

The reviewed local implementation passed **905 research tests** and **146 Evo
tests**, with **3 explicit skips**. The absent external OpenMLE-ERL/SFT integration
is skipped and remains unverified; its original assertions run when that component
exists. Scoped type checking passed on 96 owned research/experiment/test files.
These describe the recorded local environment, not a fresh-clone CI result.
Publication-specific checks and exclusions are documented below.

## Attribution and licenses

Upstream source headers, license files, and provenance are retained. In particular:

- [AIRA-Evo license](OpenMLE-Evo/third_party/aira-evo/LICENSE) and
  [third-party notices](OpenMLE-Evo/third_party/aira-evo/THIRD_PARTY_LICENSES.md).
- [iStratDE](research/vendor/istratde/LICENSE) and
  [MetaDE](research/vendor/metade/LICENSE) retain their GPL-3.0 licenses; see
  [upstream versions](research/vendor/PROVENANCE.md).

This publication does not replace those component licenses with a new blanket
license. Inherited OpenMLE/paper documentation describes its respective upstream
work; EvoRSI's own current validation is the acceptance report linked above.

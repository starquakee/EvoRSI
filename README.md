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

**Research prototype. The amended comparison found no demonstrated iStratDE
advantage; the completed repeatability diagnostic did not meet its stability thresholds.**

The comparison covers three tasks, three methods (fixed, random, iStratDE), and
three search seeds: 27 cells and 216 independent inner searches. Search consumed
1,312 model requests and 2,728,412 tokens. A separately authorized final-only
amendment reused ten verified predictions and completed 17 additional evaluations,
with zero additional model calls, producing all 27 final scores.

The task-balanced iStratDE minus random-search accuracy difference was **-0.3180
percentage points**, with an exploratory 95% interval **[-0.8620, +0.0575]**.
Against fixed settings it was **-0.1852**, interval **[-0.5555, +0.0345]**.
Both intervals include zero. These results do not establish superiority,
equivalence, or universal inferiority. The original strict audit remains **FAIL**;
the retrospective audit qualifications and separate amendment remain explicit.

Offline mechanism diagnosis checked all 216 search configurations, 1,312 operator
events and 864 parent selections. Parameter wiring matched the recorded settings;
short searches, nearly constant features and noisy/saturated validation feedback
limited the effective search space.

The follow-up Spambase diagnostic completed **nine configurations × three
independent repetitions (27 trials)**, varying crossover probability and score
weight on the frozen training/validation split. It used **166 model requests,
371,394 tokens and 87.82 minutes**, within the approved ceilings of 216 requests,
600,000 tokens and four hours. All usage was known. All **162 scored validation
predictions** were independently recomputed; three execution failures and one
invalid-Python candidate rejected before submission remained in the cost totals.

The estimated configuration variance fraction was **0.0840** and median pairwise
Spearman rank correlation **0.1266**. Both missed the predeclared 0.5 thresholds.
This short-search setting did not demonstrate a stable overall configuration
signal. The diagnostic does not compare outer algorithms or establish a best
configuration. It made **zero final-test evaluations**. Original artifacts and
historic ledgers stayed unchanged; the new stack was idle on completion.

The subsequent zero-call noise diagnosis found weak agreement between a
two-repeat average and the excluded repetition (rank correlations 0.1030,
0.1387 and 0.1849). Starting scores alone also weakly tracked final scores.
The user approved a separate shared-start pilot: three fresh initial pools and
27 continuations, capped at 174 requests / 600,000 tokens / four hours. The
implementation is complete and separately gated; this source publication
does not claim an experimental result or reopen the completed calibration.

- [Amended results and limitations](reports/istratde-amended-results-20261012/README.md)
- [Mechanism diagnosis](reports/istratde-mechanism-diagnosis-20261012/README.md)
- [Preserved original incomplete report](reports/istratde-pilot-20261012/README.md)
- [Completed repeatability results](reports/calibration-round2-results-20261013/README.md)
- [Shared-start pilot implementation and approved limits](research/shared_start/README.md)
- [Noise diagnosis and proposed shared-start pilot](reports/calibration-noise-diagnosis-20261013/README.md)
- [Three-repeat protocol and controlled runner](research/calibration/README.md)
- [Earlier bounded pipeline acceptance](reports/us009-round2-acceptance.md)

The isolated research gateway remains 127.0.0.1:6581; the legacy stack is preserved.
MetaDE remains outside the main live loop. Publishing this source does not
authorize model calls, budget resets, deployment, or reuse of historical ledgers.

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

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

**Research prototype — the full real-model acceptance is incomplete.**

The first inner run completed **3 generations × 2 main candidates** with all four
operators observed. Across the bounded acceptance, **17 model requests** consumed
**81,717 tokens** and produced **7 sandbox jobs**: **6 scored successfully and 1
failed during code execution**. The run was stopped when
the remaining request budget could no longer cover the minimum outer population.
The default deployment switch was not performed. One interrupted, unscored debug
node also has an incomplete journal artifact and is explicitly excluded from
accepted results.

These are integration and boundary checks on a synthetic task, not evidence of
algorithmic superiority or an unrestricted security guarantee. Equal-budget,
multi-seed comparisons and MetaDE experiments remain future work.

See [the acceptance report](reports/us009-real-acceptance.md) and
[machine-readable evidence](reports/us009-real-acceptance.json).

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

Before publication, the local implementation passed 788 research tests and 135 Evo
tests (2 skipped). One existing Evo SFT-path test still failed because an earlier
`build_tasks.py` file was absent. These results describe the recorded local setup;
they are not a fresh-clone CI result.

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

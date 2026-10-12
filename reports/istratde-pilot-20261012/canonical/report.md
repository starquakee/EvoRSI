# iStratDE controlled pilot comparison — report

- schema: `istratde-comparison-full-report.v1`
- source commit: `87569545793c141e393e72450094975733a2bb95`
- orchestration state verified: True
- final evidence verified: 0 cells; unverified final claims: 9
- matrix: 27/27 cells terminal, 0/27 final outcomes observed, complete=False

## Ledger costs (actual, failures included)

Ledger missing or unreadable (`corrupt_ledger:final_phase`): the original accounting state is unknown — an integrity failure, not a not-started experiment.

## 27-cell matrix

| cell | terminal | validation incumbent | final status | raw accuracy | utility | configs (done/gate/skip/interrupted) |
|---|---|---|---|---|---|---|
| banknote:11:FIXED | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:11:RANDOM | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:11:ISTRATDE | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:22:FIXED | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:22:RANDOM | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:22:ISTRATDE | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:33:FIXED | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:33:RANDOM | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| banknote:33:ISTRATDE | frozen | 1.000000 | completed (UNVERIFIED) |  |  | 8/0/0/0 |
| wdbc:11:FIXED | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:11:RANDOM | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:11:ISTRATDE | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:22:FIXED | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:22:RANDOM | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:22:ISTRATDE | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:33:FIXED | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:33:RANDOM | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| wdbc:33:ISTRATDE | frozen | 0.991228 | missing_final |  |  | 8/0/0/0 |
| spambase:11:FIXED | frozen | 0.943293 | missing_final |  |  | 8/0/0/0 |
| spambase:11:RANDOM | frozen | 0.944384 | missing_final |  |  | 8/0/0/0 |
| spambase:11:ISTRATDE | frozen | 0.944384 | missing_final |  |  | 8/0/0/0 |
| spambase:22:FIXED | frozen | 0.944384 | missing_final |  |  | 8/0/0/0 |
| spambase:22:RANDOM | frozen | 0.944384 | missing_final |  |  | 8/0/0/0 |
| spambase:22:ISTRATDE | frozen | 0.943293 | missing_final |  |  | 8/0/0/0 |
| spambase:33:FIXED | frozen | 0.942203 | missing_final |  |  | 8/0/0/0 |
| spambase:33:RANDOM | frozen | 0.945474 | missing_final |  |  | 8/0/0/0 |
| spambase:33:ISTRATDE | frozen | 0.945474 | missing_final |  |  | 8/0/0/0 |

## Success-only accuracy (failures explicit)

| method | observed | successful | failed | unobserved | accuracy mean |
|---|---|---|---|---|---|
| FIXED | 0 | 0 | 0 | 9 |  |
| RANDOM | 0 | 0 | 0 | 9 |  |
| ISTRATDE | 0 | 0 | 0 | 9 |  |

## Predeclared paired analysis (utility, FR7)

- primary: ISTRATDE - RANDOM, observed pairs 0/9 (partial=True)
  - estimate: none (no observed pairs)
  - PARTIAL estimate over observed pairs only — the predeclared full-matrix comparison was NOT completed
- secondary: ISTRATDE - FIXED, observed pairs 0/9 (partial=True)
  - estimate: none (no observed pairs)
  - PARTIAL estimate over observed pairs only — the predeclared full-matrix comparison was NOT completed

## Conclusion

**insufficient-evidence** — incomplete_matrix:the predeclared 27-cell comparison was not fully observed; any estimate shown is partial

## Caveats

- Conditional on the three fixed test sets (banknote/wdbc/spambase final folds); no generalization to other data is claimed.
- Exploratory evidence only: three search seeds per task.
- Utility is final accuracy on success and 0 on genuine observed failure; unobserved cells are excluded, never zero-filled.
- Equal candidate opportunities and caps do not imply equal realized token cost; cost totals are reported separately.

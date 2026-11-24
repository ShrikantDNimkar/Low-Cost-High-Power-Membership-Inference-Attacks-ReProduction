# Results Overview

AUC percentages, averaged over 10 target models as in the paper ("±" columns are the standard deviation across targets). Settings: 64 reference models for Purchase-100, 16 per CIFAR dataset, 18 augmented queries for the CIFAR attacks, reference budget 8 unless noted.

## Headline results vs the paper

| Dataset and setting | This project | Paper (comparable) |
|---|---|---|
| Purchase-100, offline, 8 refs | RMIA 79.86 ± 0.33 (LiRA 60.63, Attack-R 73.03, Attack-P 65.11) | RMIA around 80-81 in this regime |
| Purchase-100, online, 8 refs | RMIA 79.60 ± 0.36 (LiRA online 77.27) | online RMIA about 80.5 |
| Purchase-100, offline, 32 refs | RMIA 80.48 ± 0.37; online 80.78 ± 0.43 | larger budgets only add a small gain |
| CIFAR-10, augmented, 8 refs | RMIA (aug) 69.82 ± 1.28; RMIA (online, aug) 69.62 ± 1.31 | about 68.6 to 71.0 depending on the exact setting |
| CIFAR-100, augmented, 8 refs | RMIA (online, aug) 89.28 ± 1.51, TPR at 1% FPR 31.87% | about 89 to 90 |

Full comparison per attack and dataset:
[Reproducibility report, Section 6](../rmia-re-implementation-execution-and-reproducibility-report.md#6-results-and-comparison-with-the-paper).

## Reference-budget scaling (Purchase-100, RMIA offline)

| Reference budget | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| RMIA AUC | 75.07 | 77.39 | 78.96 | 79.86 | 80.25 | 80.48 |

The gain flattens as the budget grows, and at 32 references the measured 80.48 ± 0.37 agrees with the paper's ~80.5 for this regime.

## Hardware-independence control (Purchase-100, RMIA offline AUC)

Two independently trained model sets on the same machine:

| Reference budget | 64-model set (10-target average) | 8-model control set (freshly trained) | Deviation |
|---|---|---|---|
| 1 ref | 75.07 | 75.25 | 0.18 |
| 2 refs | 77.39 | 77.51 | 0.12 |
| 4 refs | 78.96 | 79.07 | 0.11 |

## Findings (matching the paper on all three datasets)

1. RMIA is the strongest offline attack at every reference budget; the baselines need online access to approach it.
2. The RMIA advantage grows as the false positive budget shrinks: at 8 references on Purchase-100, RMIA's TPR at 0.1% FPR is 4.20% versus 0.69% for LiRA and 0% for the threshold attacks.
3. Augmented queries strengthen RMIA and LiRA markedly on the CIFAR datasets (CIFAR-10: RMIA 66.50 without augmentation versus 69.82 with).
4. RMIA remains effective with a single reference model (Purchase-100: 75.07 AUC), where offline LiRA collapses (56.36).

## Verbatim audit tables

Each file below is the final summary table copied verbatim from the terminal output of the corresponding audit run (prompts and nothing else removed):

| File | Audit run |
|---|---|
| [purchase100_parallel_10target_6budgets_audit_table.txt](purchase100_parallel_10target_6budgets_audit_table.txt) | Purchase-100, 64 models, 10 targets, reference budgets 1/2/4/8/16/32, offline and online |
| [cifar10_10target_audit_final_table.txt](cifar10_10target_audit_final_table.txt) | CIFAR-10, 16 models, 10 targets, 18 augmented queries |
| [cifar100_10target_audit_final_table.txt](cifar100_10target_audit_final_table.txt) | CIFAR-100, 16 models, 10 targets, 18 augmented queries |
| [purchase100_64model_single_target_audit_table.txt](purchase100_64model_single_target_audit_table.txt) | Purchase-100, 64 models, single-target audit across budgets |
| [purchase100_pilot_16model_audit_table.txt](purchase100_pilot_16model_audit_table.txt) | Purchase-100, 16-model pilot set |
| [purchase100_mac_control_8model_audit_table.txt](purchase100_mac_control_8model_audit_table.txt) | Purchase-100, 8-model hardware-independence control set |

Machine-readable result exports (`results_table.json`, score arrays, ROC figures) are produced by the audits themselves in the `results/` directory when the attack commands are run; see the report's Section 5 workflow.

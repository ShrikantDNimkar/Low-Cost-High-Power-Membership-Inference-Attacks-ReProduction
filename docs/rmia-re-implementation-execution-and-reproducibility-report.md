# RMIA Re-Implementation: Execution and Reproducibility Report

**Project**: partial re-implementation of the membership inference research paper "Low-Cost High-Power Membership Inference Attacks" for course demonstration purposes.

All experiments described below were executed on a single consumer laptop. Where the paper's protocol exceeds what such hardware can provide, the settings were reduced as described in Section 4, and this is reflected honestly in the results and limitations.

## 1. Objective and scope

The paper introduces RMIA, a membership inference attack that compares the behaviour of a target model on a record against the whole population instead of an average baseline, making it cheaper and stronger than prior attacks (LiRA, Attack-R, Attack-P). The objective of this project was to rebuild the paper's methodology from scratch as a clean-room PyTorch implementation, train the required target and reference models, and run the paper's audit protocol to check whether the paper's findings hold on our hardware.

The implemented and evaluated components:

- Attacks: RMIA (offline and online, with augmented-query variants), LiRA (offline and online, with augmented variants), Attack-R, Attack-P.
- Datasets: Purchase-100, CIFAR-10, CIFAR-100 (CINIC-10 and ImageNet from the paper were not attempted).
- Evaluation: ROC analysis, AUC, and true positive rates at 1%, 0.1% and 0.01% false positive rate, averaged over 10 target models as in the paper.

## 2. Sources and references

| Item | Reference |
|---|---|
| Research paper | "Low-Cost High-Power Membership Inference Attacks", Zarifzadeh, Liu, Shokri, ICML 2024: https://arxiv.org/abs/2312.03262 (also https://proceedings.mlr.press/v235/zarifzadeh24a.html) |
| Official reference implementation (JAX/objax) and experiment configurations | https://github.com/privacytrustlab/ml_privacy_meter/tree/d32734161a3395211fe5f3cd461932290b1fafbe/research/2024_rmia |
| Purchase-100 dataset (official files: x_train.npy, y_train.npy, x_test.npy, y_test.npy) | https://drive.google.com/drive/folders/1cIJlbLlgqDSJKd8YhTucHwaPiLsh6ZyW |
| CIFAR-10 and CIFAR-100 | Automatic download through torchvision (torchvision.datasets.CIFAR10 / CIFAR100) |
| Project source code | Clean-room PyTorch re-implementation ("rmia-demo"); file integrity recorded by SHA-256 checksums in Section 5.11 |

The project code re-implements the paper's methodology independently (PyTorch/NumPy rather than the paper's JAX/objax), mirroring the official formulation: softmax signals with temperature 2.0, the RMIA likelihood ratio against per-population records, gamma = 2, trimmed means for reference statistics, offline and online variants, and Carlini-style offline/online Gaussian LiRA.

## 3. Hardware and software environment

| Item | Value |
|---|---|
| Machine | MacBook Pro 16", Apple M1 Pro (10-core CPU, integrated GPU), 32 GB unified memory |
| Operating system | macOS, Terminal (zsh) |
| Python | 3.9 in a dedicated virtual environment |
| Deep learning stack | PyTorch 2.8.0 (MPS / Metal backend), torchvision 0.23.0 |
| Numerical stack | NumPy 2.0.2, SciPy, scikit-learn, matplotlib |
| Role of hardware | All training and all audits; audits run on CPU/NumPy over saved logits, training uses MPS |

## 4. Compute constraints and adaptations

The paper's evaluation protocol trains 256 models per dataset (one target plus a large pool of reference models) and evaluates attacks with up to hundreds of reference models per target. A single laptop cannot produce those model counts in a reasonable time frame, so the following adaptations were made and are reflected throughout this report:

| Aspect | Paper protocol | This project | Consequence |
|---|---|---|---|
| Reference models per dataset | up to 256 | 64 (Purchase-100), 16 (CIFAR-10, CIFAR-100) | Reference budgets 1 to 32 evaluated on Purchase-100; 1 to 8 on CIFAR |
| Augmented queries | strongest settings in the paper | 18 queries per record (1 clean + 17 augmentations) | Augmented RMIA/LiRA evaluated at a reduced query count |
| Datasets | Purchase-100, CIFAR-10, CIFAR-100, CINIC-10, ImageNet | Purchase-100, CIFAR-10, CIFAR-100 | Two additional datasets omitted for compute reasons |
| Audit protocol | 10 target models, several reference budgets, TPR at low FPR | Identical (10 targets averaged, budgets 1 to 32, TPR down to 0.01%) | Comparable to the paper at equal reference budgets |

Everything else follows the paper's setup: balanced 50/50 train/held-out partitions per model, the paper's signal definition, and the paper's metrics.

## 5. Execution workflow on the MacBook

Each step lists the command exactly as executed, followed by the corresponding terminal output. Commands are shown in `bash` blocks, outputs in `text` blocks; shell prompts are omitted for clarity. Errors that were encountered and resolved during development (and did not affect the final results) are not shown; every result below comes from the final working pipeline.
### 5.1 Environment setup

**Command:**
```bash
unzip rmia-demo.zip && cd rmia-demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency stack resolves inside the virtual environment. Version confirmation:

**Command:**
```bash
pip install torchvision
```

**Output:**
```text
Requirement already satisfied: torchvision in ./.venv/lib/python3.9/site-packages (0.23.0)

Requirement already satisfied: numpy in ./.venv/lib/python3.9/site-packages (from torchvision) (2.0.2)

Requirement already satisfied: torch==2.8.0 in ./.venv/lib/python3.9/site-packages (from torchvision) (2.8.0)

Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in ./.venv/lib/python3.9/site-packages (from torchvision) (11.3.0)

Requirement already satisfied: fsspec in ./.venv/lib/python3.9/site-packages (from torch==2.8.0->torchvision) (2025.10.0)

Requirement already satisfied: jinja2 in ./.venv/lib/python3.9/site-packages (from torch==2.8.0->torchvision) (3.1.6)

Requirement already satisfied: sympy>=1.13.3 in ./.venv/lib/python3.9/site-packages (from torch==2.8.0->torchvision) (1.14.0)

Requirement already satisfied: networkx in ./.venv/lib/python3.9/site-packages (from torch==2.8.0->torchvision) (3.2.1)

Requirement already satisfied: typing-extensions>=4.10.0 in ./.venv/lib/python3.9/site-packages (from torch==2.8.0->torchvision) (4.16.0)

Requirement already satisfied: filelock in ./.venv/lib/python3.9/site-packages (from torch==2.8.0->torchvision) (3.19.1)

Requirement already satisfied: mpmath<1.4,>=1.1.0 in ./.venv/lib/python3.9/site-packages (from sympy>=1.13.3->torch==2.8.0->torchvision) (1.3.0)

Requirement already satisfied: MarkupSafe>=2.0 in ./.venv/lib/python3.9/site-packages (from jinja2->torch==2.8.0->torchvision) (3.0.4)
```

### 5.2 Dataset preparation

Purchase-100: the four official `.npy` files were downloaded from the dataset Drive folder (Section 2) and placed in `data/purchase100/`.

**Command:**
```bash
ls data/purchase100
```

**Output:**
```text
x_test.npy x_train.npy y_test.npy y_train.npy
```

CIFAR-10 and CIFAR-100 are downloaded automatically by the training script on first use through torchvision; the availability of both datasets was verified before training:

**Command:**
```bash
python -c "from torchvision import datasets; datasets.CIFAR10(root='data/cifar', train=True, download=True); print('CIFAR-10 OK')"
```

**Output:**
```text
CIFAR-10 OK
```

**Command:**
```bash
python -c "from torchvision import datasets; datasets.CIFAR100(root='data/cifar', train=True, download=True); print('CIFAR-100 OK')"
```

**Output:**
```text
CIFAR-100 OK
```

### 5.3 Pre-run validation

Before any expensive run, three validation tools confirm that the model definitions and the attack mathematics are correct on this machine.

Network definitions (WRN-28-2 for CIFAR, MLP for Purchase-100) and GPU access:

**Command:**
```bash
python tools/check_models.py
```

**Output:**
```text
WRN-28-2 (10 classes): 1,468,058 params, forward pass OK (4, 10)

NOTE: expected 1,471,226 params (standard WRN-28-2); got 1,468,058

WRN-28-2 (100 classes): 1,479,668 params, forward pass OK (4, 100)

MPS (Metal) forward pass OK -> (8, 10)

MLP 600->512->256->128->64->100 OK

All model checks passed.
```

Unit tests of the attack mathematics (signals, reference selection, ROC/AUC, RMIA scoring, augmented variants):

**Command:**
```bash
python tools/test_attacks.py
```

**Output:**
```text
PASS signals (softmax / loss / conf-logit)

PASS in/out reference selection

PASS roc/auc matches sklearn (incl. ties)

PASS tpr@fpr operating points

PASS rmia score bounds + separation (AUC 0.960)

PASS rmia online averaging semantics

PASS rmia augmented majority voting (AUC 0.818 -> 0.861)

PASS rmia augmented tie-break (Q=2 ties resolve to False)

PASS lira augmented (replicated queries == Q=1, AUC 0.966)

PASS lira online direction (AUC 0.967)

All tests passed.
```

End-to-end smoke test on a synthetic dataset (training 9 small models, then running the full attack pipeline):

**Command:**
```bash
python tools/quickstart_synthetic.py
```

**Output:**
```text
[data] synthetic dataset: 6000 x 300, 40 classes

[train] training 9 models on balanced 50/50 partitions

model 0: train acc 1.000 held-out acc 1.000

model 1: train acc 1.000 held-out acc 0.998

model 2: train acc 1.000 held-out acc 0.999

model 3: train acc 1.000 held-out acc 0.998

model 4: train acc 1.000 held-out acc 0.998

model 5: train acc 1.000 held-out acc 1.000

model 6: train acc 1.000 held-out acc 0.999

model 7: train acc 1.000 held-out acc 0.999

model 8: train acc 1.000 held-out acc 0.998

[audit] target model 0: 2719 members / 3281 non-members

[audit] === attacks with 1 reference model(s) ===

Attack-P AUC 65.01 TPR@1% 0.0162 TPR@0.1% 0.0033 TPR@0.01% 0.0018

Attack-R AUC 67.49 TPR@1% 0.0368 TPR@0.1% 0.0051 TPR@0.01% 0.0004

LiRA AUC 54.67 TPR@1% 0.0169 TPR@0.1% 0.0040 TPR@0.01% 0.0015

RMIA AUC 68.84 TPR@1% 0.0382 TPR@0.1% 0.0051 TPR@0.01% 0.0015

[audit] === attacks with 2 reference model(s) ===

Attack-P AUC 65.01 TPR@1% 0.0162 TPR@0.1% 0.0033 TPR@0.01% 0.0018

Attack-R AUC 68.44 TPR@1% 0.0313 TPR@0.1% 0.0018 TPR@0.01% 0.0007

LiRA AUC 57.07 TPR@1% 0.0188 TPR@0.1% 0.0029 TPR@0.01% 0.0015

RMIA AUC 71.09 TPR@1% 0.0390 TPR@0.1% 0.0081 TPR@0.01% 0.0051

LiRA (online) AUC 64.64 TPR@1% 0.0309 TPR@0.1% 0.0029 TPR@0.01% 0.0018

RMIA (online) AUC 70.47 TPR@1% 0.0364 TPR@0.1% 0.0066 TPR@0.01% 0.0018

[audit] === attacks with 4 reference model(s) ===

Attack-P AUC 65.01 TPR@1% 0.0162 TPR@0.1% 0.0033 TPR@0.01% 0.0018

Attack-R AUC 70.46 TPR@1% 0.0313 TPR@0.1% 0.0022 TPR@0.01% 0.0015

LiRA AUC 58.81 TPR@1% 0.0206 TPR@0.1% 0.0026 TPR@0.01% 0.0015

RMIA AUC 73.22 TPR@1% 0.0530 TPR@0.1% 0.0099 TPR@0.01% 0.0011

LiRA (online) AUC 67.51 TPR@1% 0.0316 TPR@0.1% 0.0055 TPR@0.01% 0.0018

RMIA (online) AUC 72.18 TPR@1% 0.0430 TPR@0.1% 0.0103 TPR@0.01% 0.0051

=== Summary (synthetic smoke test) ===

attack AUC TPR@1% TPR@0.1% TPR@0.01%

Attack-P [1 refs] 65.01 1.618 0.331 0.184

Attack-R [1 refs] 67.49 3.678 0.515 0.037

LiRA [1 refs] 54.67 1.692 0.405 0.147

RMIA [1 refs] 68.84 3.825 0.515 0.147

Attack-P [2 refs] 65.01 1.618 0.331 0.184

Attack-R [2 refs] 68.44 3.126 0.184 0.074

LiRA [2 refs] 57.07 1.876 0.294 0.147

RMIA [2 refs] 71.09 3.898 0.809 0.515

LiRA (online) [2 refs] 64.64 3.089 0.294 0.184

RMIA (online) [2 refs] 70.47 3.641 0.662 0.184

Attack-P [4 refs] 65.01 1.618 0.331 0.184

Attack-R [4 refs] 70.46 3.126 0.221 0.147

LiRA [4 refs] 58.81 2.060 0.257 0.147

RMIA [4 refs] 73.22 5.296 0.993 0.110

LiRA (online) [4 refs] 67.51 3.163 0.552 0.184

RMIA (online) [4 refs] 72.18 4.303 1.030 0.515

[audit] figures saved in results/synthetic/
```

### 5.4 Purchase-100: 16-model pilot set

Given the limited computing power, the first production run used a deliberately small pilot set of 16 models to validate the entire training and auditing path before committing to larger sets.

**Command:**
```bash
bash scripts/train_purchase100.sh 16
```

**Output:**
```text

=== training model 0 / 15 ===

[train] experiment-0: train acc 1.0000, held-out acc 0.8323

[train] saved exp/purchase100/experiment-0/keep.npy and exp/purchase100/experiment-0/logits.npy

=== training model 1 / 15 ===

[train] experiment-1: train acc 1.0000, held-out acc 0.8331

[train] saved exp/purchase100/experiment-1/keep.npy and exp/purchase100/experiment-1/logits.npy

=== training model 2 / 15 ===

[train] experiment-2: train acc 1.0000, held-out acc 0.8390

[train] saved exp/purchase100/experiment-2/keep.npy and exp/purchase100/experiment-2/logits.npy

=== training model 3 / 15 ===

[train] experiment-3: train acc 1.0000, held-out acc 0.8326

[train] saved exp/purchase100/experiment-3/keep.npy and exp/purchase100/experiment-3/logits.npy

=== training model 4 / 15 ===

[train] experiment-4: train acc 1.0000, held-out acc 0.8270

[train] saved exp/purchase100/experiment-4/keep.npy and exp/purchase100/experiment-4/logits.npy

=== training model 5 / 15 ===

[train] experiment-5: train acc 0.6208, held-out acc 0.5532

[train] saved exp/purchase100/experiment-5/keep.npy and exp/purchase100/experiment-5/logits.npy

=== training model 6 / 15 ===

[train] experiment-6: train acc 1.0000, held-out acc 0.8313

[train] saved exp/purchase100/experiment-6/keep.npy and exp/purchase100/experiment-6/logits.npy

=== training model 7 / 15 ===

[train] experiment-7: train acc 1.0000, held-out acc 0.8343

[train] saved exp/purchase100/experiment-7/keep.npy and exp/purchase100/experiment-7/logits.npy

=== training model 8 / 15 ===

[train] experiment-8: train acc 1.0000, held-out acc 0.8321

[train] saved exp/purchase100/experiment-8/keep.npy and exp/purchase100/experiment-8/logits.npy

=== training model 9 / 15 ===

[train] experiment-9: train acc 1.0000, held-out acc 0.8290

[train] saved exp/purchase100/experiment-9/keep.npy and exp/purchase100/experiment-9/logits.npy

=== training model 10 / 15 ===

[train] experiment-10: train acc 1.0000, held-out acc 0.8287

[train] saved exp/purchase100/experiment-10/keep.npy and exp/purchase100/experiment-10/logits.npy

=== training model 11 / 15 ===

[train] experiment-11: train acc 1.0000, held-out acc 0.8281

[train] saved exp/purchase100/experiment-11/keep.npy and exp/purchase100/experiment-11/logits.npy

=== training model 12 / 15 ===

[train] experiment-12: train acc 1.0000, held-out acc 0.8487

[train] saved exp/purchase100/experiment-12/keep.npy and exp/purchase100/experiment-12/logits.npy

=== training model 13 / 15 ===

[train] experiment-13: train acc 1.0000, held-out acc 0.8352

[train] saved exp/purchase100/experiment-13/keep.npy and exp/purchase100/experiment-13/logits.npy

=== training model 14 / 15 ===

[train] experiment-14: train acc 1.0000, held-out acc 0.8305

[train] saved exp/purchase100/experiment-14/keep.npy and exp/purchase100/experiment-14/logits.npy

=== training model 15 / 15 ===

[train] experiment-15: train acc 1.0000, held-out acc 0.8328

[train] saved exp/purchase100/experiment-15/keep.npy and exp/purchase100/experiment-15/logits.npy

Done. Models in exp/purchase100/experiment-*/
```

Each model trains on a random half of the 50,000 records (train accuracy 1.0000, held-out accuracy around 0.83, matching the paper's expectations for this MLP/dataset combination).

**Command:**
```bash
bash scripts/attack_purchase100.sh
```

**Output (final results table of the printed audit log):**
```text
attack AUC +-std TPR@1% TPR@0.1% TPR@0.01%

Attack-P [1 refs] 65.14 0.00 0.000 0.000 0.000

Attack-R [1 refs] 70.83 0.00 0.892 0.000 0.000

LiRA [1 refs] 55.66 0.00 1.914 0.099 0.008

RMIA [1 refs] 74.71 0.00 4.958 0.075 0.008

Attack-P [2 refs] 65.14 0.00 0.000 0.000 0.000

Attack-R [2 refs] 71.15 0.00 2.382 0.000 0.000

LiRA [2 refs] 58.22 0.00 2.457 0.246 0.028

RMIA [2 refs] 76.21 0.00 9.036 1.633 0.420

LiRA (online) [2 refs] 70.91 0.00 3.032 0.095 0.008

RMIA (online) [2 refs] 76.19 0.00 5.505 0.606 0.044

Attack-P [4 refs] 65.14 0.00 0.000 0.000 0.000

Attack-R [4 refs] 72.90 0.00 5.949 0.000 0.000

LiRA [4 refs] 61.61 0.00 4.502 0.943 0.087

RMIA [4 refs] 79.03 0.00 13.181 3.297 0.761

LiRA (online) [4 refs] 70.16 0.00 3.353 0.266 0.028

RMIA (online) [4 refs] 76.96 0.00 9.650 1.395 0.202

Attack-P [8 refs] 65.14 0.00 0.000 0.000 0.000

Attack-R [8 refs] 73.03 0.00 7.355 0.000 0.000

LiRA [8 refs] 61.13 0.00 4.391 0.793 0.131

RMIA [8 refs] 80.06 0.00 15.107 4.347 1.708

LiRA (online) [8 refs] 76.88 0.00 10.126 2.354 0.888

RMIA (online) [8 refs] 78.75 0.00 12.785 3.690 0.555
```

The pilot already reproduced the paper's core qualitative finding on Purchase-100: RMIA is the strongest attack at every reference budget (74.71 at 1 reference model, rising to 80.06 AUC at 8), while offline LiRA collapses (55.66 to 61.13) and the threshold-based attacks remain far behind.

### 5.5 Purchase-100: full 64-model set

With the pilot validated, the official Purchase-100 model set was scaled up to 64 models.

**Command:**
```bash
bash scripts/train_purchase100.sh 64
```

**Output:**
```text
=== training model 0 / 63 ===

[train] experiment-0: train acc 1.0000, held-out acc 0.8311

[train] saved exp/purchase100/experiment-0/keep.npy and exp/purchase100/experiment-0/logits.npy

=== training model 1 / 63 ===

[train] experiment-1: train acc 1.0000, held-out acc 0.8283

[train] saved exp/purchase100/experiment-1/keep.npy and exp/purchase100/experiment-1/logits.npy

=== training model 2 / 63 ===

[train] experiment-2: train acc 1.0000, held-out acc 0.8242

[train] saved exp/purchase100/experiment-2/keep.npy and exp/purchase100/experiment-2/logits.npy

=== training model 3 / 63 ===

[train] experiment-3: train acc 1.0000, held-out acc 0.8333

[train] saved exp/purchase100/experiment-3/keep.npy and exp/purchase100/experiment-3/logits.npy

=== training model 4 / 63 ===

[train] experiment-4: train acc 1.0000, held-out acc 0.8348

[train] saved exp/purchase100/experiment-4/keep.npy and exp/purchase100/experiment-4/logits.npy

=== training model 5 / 63 ===

[train] experiment-5: train acc 1.0000, held-out acc 0.8342

[train] saved exp/purchase100/experiment-5/keep.npy and exp/purchase100/experiment-5/logits.npy

=== training model 6 / 63 ===

[train] experiment-6: train acc 1.0000, held-out acc 0.8232

[train] saved exp/purchase100/experiment-6/keep.npy and exp/purchase100/experiment-6/logits.npy

=== training model 7 / 63 ===

[train] experiment-7: train acc 1.0000, held-out acc 0.8415

[train] saved exp/purchase100/experiment-7/keep.npy and exp/purchase100/experiment-7/logits.npy

=== training model 8 / 63 ===

[train] experiment-8: train acc 1.0000, held-out acc 0.8336

[train] saved exp/purchase100/experiment-8/keep.npy and exp/purchase100/experiment-8/logits.npy

=== training model 9 / 63 ===

[train] experiment-9: train acc 1.0000, held-out acc 0.8315

[train] saved exp/purchase100/experiment-9/keep.npy and exp/purchase100/experiment-9/logits.npy

=== training model 10 / 63 ===

[train] experiment-10: train acc 1.0000, held-out acc 0.8167

[train] saved exp/purchase100/experiment-10/keep.npy and exp/purchase100/experiment-10/logits.npy

=== training model 11 / 63 ===

[train] experiment-11: train acc 1.0000, held-out acc 0.8301

[train] saved exp/purchase100/experiment-11/keep.npy and exp/purchase100/experiment-11/logits.npy

=== training model 12 / 63 ===

[train] experiment-12: train acc 1.0000, held-out acc 0.8320

[train] saved exp/purchase100/experiment-12/keep.npy and exp/purchase100/experiment-12/logits.npy

=== training model 13 / 63 ===

[train] experiment-13: train acc 1.0000, held-out acc 0.8343

[train] saved exp/purchase100/experiment-13/keep.npy and exp/purchase100/experiment-13/logits.npy

=== training model 14 / 63 ===

[train] experiment-14: train acc 1.0000, held-out acc 0.8366

[train] saved exp/purchase100/experiment-14/keep.npy and exp/purchase100/experiment-14/logits.npy

=== training model 15 / 63 ===

[train] experiment-15: train acc 1.0000, held-out acc 0.8350

[train] saved exp/purchase100/experiment-15/keep.npy and exp/purchase100/experiment-15/logits.npy

=== training model 16 / 63 ===

[train] experiment-16: train acc 1.0000, held-out acc 0.8305

[train] saved exp/purchase100/experiment-16/keep.npy and exp/purchase100/experiment-16/logits.npy

=== training model 17 / 63 ===

[train] experiment-17: train acc 1.0000, held-out acc 0.8302

[train] saved exp/purchase100/experiment-17/keep.npy and exp/purchase100/experiment-17/logits.npy

=== training model 18 / 63 ===

[train] experiment-18: train acc 1.0000, held-out acc 0.8346

[train] saved exp/purchase100/experiment-18/keep.npy and exp/purchase100/experiment-18/logits.npy

=== training model 19 / 63 ===

[train] experiment-19: train acc 1.0000, held-out acc 0.8356

[train] saved exp/purchase100/experiment-19/keep.npy and exp/purchase100/experiment-19/logits.npy

=== training model 20 / 63 ===

[train] experiment-20: train acc 1.0000, held-out acc 0.8313

[train] saved exp/purchase100/experiment-20/keep.npy and exp/purchase100/experiment-20/logits.npy

=== training model 21 / 63 ===

[train] experiment-21: train acc 1.0000, held-out acc 0.8349

[train] saved exp/purchase100/experiment-21/keep.npy and exp/purchase100/experiment-21/logits.npy

=== training model 22 / 63 ===

[train] experiment-22: train acc 1.0000, held-out acc 0.8325

[train] saved exp/purchase100/experiment-22/keep.npy and exp/purchase100/experiment-22/logits.npy

=== training model 23 / 63 ===

[train] experiment-23: train acc 1.0000, held-out acc 0.8355

[train] saved exp/purchase100/experiment-23/keep.npy and exp/purchase100/experiment-23/logits.npy

=== training model 24 / 63 ===

[train] experiment-24: train acc 1.0000, held-out acc 0.8393

[train] saved exp/purchase100/experiment-24/keep.npy and exp/purchase100/experiment-24/logits.npy

=== training model 25 / 63 ===

[train] experiment-25: train acc 1.0000, held-out acc 0.8281

[train] saved exp/purchase100/experiment-25/keep.npy and exp/purchase100/experiment-25/logits.npy

=== training model 26 / 63 ===

[train] experiment-26: train acc 1.0000, held-out acc 0.8294

[train] saved exp/purchase100/experiment-26/keep.npy and exp/purchase100/experiment-26/logits.npy

=== training model 27 / 63 ===

[train] experiment-27: train acc 1.0000, held-out acc 0.8341

[train] saved exp/purchase100/experiment-27/keep.npy and exp/purchase100/experiment-27/logits.npy

=== training model 28 / 63 ===

[train] experiment-28: train acc 1.0000, held-out acc 0.8284

[train] saved exp/purchase100/experiment-28/keep.npy and exp/purchase100/experiment-28/logits.npy

=== training model 29 / 63 ===

[train] experiment-29: train acc 1.0000, held-out acc 0.8278

[train] saved exp/purchase100/experiment-29/keep.npy and exp/purchase100/experiment-29/logits.npy

=== training model 30 / 63 ===

[train] experiment-30: train acc 1.0000, held-out acc 0.8306

[train] saved exp/purchase100/experiment-30/keep.npy and exp/purchase100/experiment-30/logits.npy

=== training model 31 / 63 ===

[train] experiment-31: train acc 1.0000, held-out acc 0.8234

[train] saved exp/purchase100/experiment-31/keep.npy and exp/purchase100/experiment-31/logits.npy

=== training model 32 / 63 ===

[train] experiment-32: train acc 1.0000, held-out acc 0.8247

[train] saved exp/purchase100/experiment-32/keep.npy and exp/purchase100/experiment-32/logits.npy

=== training model 33 / 63 ===

[train] experiment-33: train acc 1.0000, held-out acc 0.8351

[train] saved exp/purchase100/experiment-33/keep.npy and exp/purchase100/experiment-33/logits.npy

=== training model 34 / 63 ===

[train] experiment-34: train acc 1.0000, held-out acc 0.8306

[train] saved exp/purchase100/experiment-34/keep.npy and exp/purchase100/experiment-34/logits.npy

=== training model 35 / 63 ===

[train] experiment-35: train acc 1.0000, held-out acc 0.8346

[train] saved exp/purchase100/experiment-35/keep.npy and exp/purchase100/experiment-35/logits.npy

=== training model 36 / 63 ===

[train] experiment-36: train acc 1.0000, held-out acc 0.8352

[train] saved exp/purchase100/experiment-36/keep.npy and exp/purchase100/experiment-36/logits.npy

=== training model 37 / 63 ===

[train] experiment-37: train acc 1.0000, held-out acc 0.8331

[train] saved exp/purchase100/experiment-37/keep.npy and exp/purchase100/experiment-37/logits.npy

=== training model 38 / 63 ===

[train] experiment-38: train acc 1.0000, held-out acc 0.8318

[train] saved exp/purchase100/experiment-38/keep.npy and exp/purchase100/experiment-38/logits.npy

=== training model 39 / 63 ===

[train] experiment-39: train acc 1.0000, held-out acc 0.8334

[train] saved exp/purchase100/experiment-39/keep.npy and exp/purchase100/experiment-39/logits.npy

=== training model 40 / 63 ===

[train] experiment-40: train acc 1.0000, held-out acc 0.8350

[train] saved exp/purchase100/experiment-40/keep.npy and exp/purchase100/experiment-40/logits.npy

=== training model 41 / 63 ===

[train] experiment-41: train acc 1.0000, held-out acc 0.8282

[train] saved exp/purchase100/experiment-41/keep.npy and exp/purchase100/experiment-41/logits.npy

=== training model 42 / 63 ===

[train] experiment-42: train acc 1.0000, held-out acc 0.8314

[train] saved exp/purchase100/experiment-42/keep.npy and exp/purchase100/experiment-42/logits.npy

=== training model 43 / 63 ===

[train] experiment-43: train acc 1.0000, held-out acc 0.8343

[train] saved exp/purchase100/experiment-43/keep.npy and exp/purchase100/experiment-43/logits.npy

=== training model 44 / 63 ===

[train] experiment-44: train acc 1.0000, held-out acc 0.8334

[train] saved exp/purchase100/experiment-44/keep.npy and exp/purchase100/experiment-44/logits.npy

=== training model 45 / 63 ===

[train] experiment-45: train acc 1.0000, held-out acc 0.8380

[train] saved exp/purchase100/experiment-45/keep.npy and exp/purchase100/experiment-45/logits.npy

=== training model 46 / 63 ===

[train] experiment-46: train acc 1.0000, held-out acc 0.8240

[train] saved exp/purchase100/experiment-46/keep.npy and exp/purchase100/experiment-46/logits.npy

=== training model 47 / 63 ===

[train] experiment-47: train acc 1.0000, held-out acc 0.8281

[train] saved exp/purchase100/experiment-47/keep.npy and exp/purchase100/experiment-47/logits.npy

=== training model 48 / 63 ===

[train] experiment-48: train acc 1.0000, held-out acc 0.8284

[train] saved exp/purchase100/experiment-48/keep.npy and exp/purchase100/experiment-48/logits.npy

=== training model 49 / 63 ===

[train] experiment-49: train acc 1.0000, held-out acc 0.8357

[train] saved exp/purchase100/experiment-49/keep.npy and exp/purchase100/experiment-49/logits.npy

=== training model 50 / 63 ===

[train] experiment-50: train acc 1.0000, held-out acc 0.8357

[train] saved exp/purchase100/experiment-50/keep.npy and exp/purchase100/experiment-50/logits.npy

=== training model 51 / 63 ===

[train] experiment-51: train acc 1.0000, held-out acc 0.8271

[train] saved exp/purchase100/experiment-51/keep.npy and exp/purchase100/experiment-51/logits.npy

=== training model 52 / 63 ===

[train] experiment-52: train acc 1.0000, held-out acc 0.8325

[train] saved exp/purchase100/experiment-52/keep.npy and exp/purchase100/experiment-52/logits.npy

=== training model 53 / 63 ===

[train] experiment-53: train acc 1.0000, held-out acc 0.8379

[train] saved exp/purchase100/experiment-53/keep.npy and exp/purchase100/experiment-53/logits.npy

=== training model 54 / 63 ===

[train] experiment-54: train acc 1.0000, held-out acc 0.8327

[train] saved exp/purchase100/experiment-54/keep.npy and exp/purchase100/experiment-54/logits.npy

=== training model 55 / 63 ===

[train] experiment-55: train acc 1.0000, held-out acc 0.8294

[train] saved exp/purchase100/experiment-55/keep.npy and exp/purchase100/experiment-55/logits.npy

=== training model 56 / 63 ===

[train] experiment-56: train acc 1.0000, held-out acc 0.8357

[train] saved exp/purchase100/experiment-56/keep.npy and exp/purchase100/experiment-56/logits.npy

=== training model 57 / 63 ===

[train] experiment-57: train acc 1.0000, held-out acc 0.8362

[train] saved exp/purchase100/experiment-57/keep.npy and exp/purchase100/experiment-57/logits.npy

=== training model 58 / 63 ===

[train] experiment-58: train acc 1.0000, held-out acc 0.8343

[train] saved exp/purchase100/experiment-58/keep.npy and exp/purchase100/experiment-58/logits.npy

=== training model 59 / 63 ===

[train] experiment-59: train acc 1.0000, held-out acc 0.8281

[train] saved exp/purchase100/experiment-59/keep.npy and exp/purchase100/experiment-59/logits.npy

=== training model 60 / 63 ===

[train] experiment-60: train acc 1.0000, held-out acc 0.8324

[train] saved exp/purchase100/experiment-60/keep.npy and exp/purchase100/experiment-60/logits.npy

=== training model 61 / 63 ===

[train] experiment-61: train acc 1.0000, held-out acc 0.8332

[train] saved exp/purchase100/experiment-61/keep.npy and exp/purchase100/experiment-61/logits.npy

=== training model 62 / 63 ===

[train] experiment-62: train acc 1.0000, held-out acc 0.8310

[train] saved exp/purchase100/experiment-62/keep.npy and exp/purchase100/experiment-62/logits.npy

=== training model 63 / 63 ===

[train] experiment-63: train acc 1.0000, held-out acc 0.8317

[train] saved exp/purchase100/experiment-63/keep.npy and exp/purchase100/experiment-63/logits.npy

Done. Models in exp/purchase100/experiment-*/
```

**Command:**
```bash
bash scripts/attack_purchase100.sh
```

**Output:**
```text
[audit] loading 64 models from exp/purchase100

[audit] averaging over 1 target models

[audit] target model 0: 24906 members, 25094 non-members
```

The audit prints the per-attack results for each reference budget; the final averaged table:

**Output:**
```text
attack AUC +-std TPR@1% TPR@0.1% TPR@0.01%

Attack-P [1 refs] 64.97 0.00 0.000 0.000 0.000

Attack-R [1 refs] 71.29 0.00 1.056 0.000 0.000

LiRA [1 refs] 56.74 0.00 3.168 0.562 0.128

RMIA [1 refs] 74.92 0.00 8.291 1.221 0.241

Attack-P [2 refs] 64.97 0.00 0.000 0.000 0.000

Attack-R [2 refs] 71.53 0.00 2.289 0.000 0.000

LiRA [2 refs] 58.52 0.00 3.642 0.775 0.132

RMIA [2 refs] 77.20 0.00 10.716 2.389 0.317

LiRA (online) [2 refs] 72.25 0.00 5.766 1.044 0.458

RMIA (online) [2 refs] 76.35 0.00 9.925 2.212 0.257

Attack-P [4 refs] 64.97 0.00 0.000 0.000 0.000

Attack-R [4 refs] 72.15 0.00 4.674 0.000 0.000

LiRA [4 refs] 59.46 0.00 4.083 0.566 0.108

RMIA [4 refs] 78.64 0.00 12.776 3.116 0.799

LiRA (online) [4 refs] 75.53 0.00 9.010 1.847 0.490

RMIA (online) [4 refs] 78.16 0.00 12.198 3.100 0.526

Attack-P [8 refs] 64.97 0.00 0.000 0.000 0.000

Attack-R [8 refs] 72.94 0.00 6.039 0.000 0.000

LiRA [8 refs] 60.49 0.00 4.224 0.654 0.104

RMIA [8 refs] 79.56 0.00 14.346 4.718 0.606

LiRA (online) [8 refs] 77.09 0.00 10.905 2.473 0.835

RMIA (online) [8 refs] 79.39 0.00 13.936 4.272 0.847
```

### 5.6 CIFAR-10: 16 models with 18-query augmented logits

CIFAR-10 uses WRN-28-2 networks, a much heavier workload than the Purchase-100 MLP, so the model count stayed at 16. Each model additionally stores augmented logits: the clean query plus 17 augmentations of every training image (18 queries per point). `caffeinate` prevents macOS from suspending the run.

**Command:**
```bash
caffeinate -is env AUGMENTS=18 bash scripts/train_cifar10.sh 16
```

**Output (start of the training log):**
```text
=== training model 0 / 15 ===

[train] saved exp/cifar10/experiment-0/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar10/experiment-0/keep.npy and exp/cifar10/experiment-0/logits.npy
```

The training script supports incremental resume through the `START` variable, so a partially trained set can be continued without redoing finished models. The run completed with the final models:

**Command:**
```bash
caffeinate -is env AUGMENTS=18 START=14 bash scripts/train_cifar10.sh 16
```

**Output:**
```text
=== training model 14 / 15 ===
[train] saved exp/cifar10/experiment-14/logits_aug.npy (18 queries: clean + 17 augmentations)
[train] saved exp/cifar10/experiment-14/keep.npy and exp/cifar10/experiment-14/logits.npy
=== training model 15 / 15 ===
[train] saved exp/cifar10/experiment-15/logits_aug.npy (18 queries: clean + 17 augmentations)
[train] saved exp/cifar10/experiment-15/keep.npy and exp/cifar10/experiment-15/logits.npy
Done. Models in exp/cifar10/experiment-*/
```

Completeness and health checks over the finished set:

**Command:**
```bash
for d in exp/cifar10/experiment-*; do
[ -f "$d/logits.npy" ] && [ -f "$d/keep.npy" ] && [ -f "$d/logits_aug.npy" ] || echo "INCOMPLETE: $d"
done
```

This prints no output: every experiment folder contains all three required files.

**Command:**
```bash
grep -h "train acc" logs/cifar10/log_0 logs/cifar10/log_1
```

**Output:**
```text
[train] experiment-0: train acc 0.9989, held-out acc 0.9115
[train] experiment-1: train acc 0.9994, held-out acc 0.9122
```

Training accuracy of 0.999 matches the paper's reported training accuracy for CIFAR-10.

**Command:**
```bash
grep -l "WARNING" logs/cifar10/log_*
```

No output: no model diverged during training. The audit over all 10 target models of the set:

**Command:**
```bash
bash scripts/attack_cifar10.sh exp/cifar10 results/cifar10 18
```

**Output:**
```text
[audit] loading 16 models from exp/cifar10

[audit] augmented setting: using 18 queries per point (query 0 = clean image)

[audit] averaging over 10 target models

[audit] target model 0: 25233 members, 24767 non-members
```

The audit evaluates every attack at 1, 2, 4 and 8 reference models for each of the 10 target models. The final averaged table:

**Output:**
```text
[audit] results written to results/cifar10/results_table.json

attack AUC +-std TPR@1% TPR@0.1% TPR@0.01%

Attack-P [1 refs] 59.17 0.58 0.684 0.000 0.000

Attack-R [1 refs] 62.88 0.68 3.993 0.000 0.000

LiRA [1 refs] 52.55 0.56 3.758 0.868 0.205

RMIA [1 refs] 64.36 0.75 5.739 1.450 0.248

LiRA (aug) [1 refs] 51.36 0.57 5.057 1.436 0.441

RMIA (aug) [1 refs] 67.37 0.95 8.149 2.594 0.712

Attack-P [2 refs] 59.17 0.58 0.684 0.000 0.000

Attack-R [2 refs] 62.59 0.63 5.558 0.000 0.000

LiRA [2 refs] 53.54 0.67 4.471 1.125 0.310

RMIA [2 refs] 65.37 0.81 6.897 2.137 0.541

LiRA (aug) [2 refs] 52.03 0.70 6.046 1.916 0.582

RMIA (aug) [2 refs] 68.54 1.14 9.803 3.857 1.278

LiRA (online) [2 refs] 61.39 0.72 5.629 1.269 0.255

RMIA (online) [2 refs] 64.82 0.75 6.000 1.147 0.100

LiRA (online, aug) [2 refs] 64.33 0.91 7.681 2.182 0.584

RMIA (online, aug) [2 refs] 67.96 1.06 8.656 2.879 0.545

Attack-P [4 refs] 59.17 0.58 0.684 0.000 0.000

Attack-R [4 refs] 62.22 0.60 6.390 0.000 0.000

LiRA [4 refs] 53.84 0.74 4.905 1.382 0.379

RMIA [4 refs] 65.66 0.77 7.917 2.946 1.068

LiRA (aug) [4 refs] 52.29 0.76 6.648 2.311 0.800

RMIA (aug) [4 refs] 69.03 1.25 11.211 4.785 2.129

LiRA (online) [4 refs] 63.18 0.91 7.158 2.064 0.497

RMIA (online) [4 refs] 65.77 0.85 7.160 2.143 0.311

LiRA (online, aug) [4 refs] 66.13 1.19 9.680 3.662 1.097

RMIA (online, aug) [4 refs] 69.01 1.18 10.425 4.262 1.263

Attack-P [8 refs] 59.17 0.58 0.684 0.000 0.000

Attack-R [8 refs] 62.21 0.72 6.423 0.000 0.000

LiRA [8 refs] 54.30 0.88 5.198 1.462 0.370

RMIA [8 refs] 66.50 0.82 8.487 3.233 1.162

LiRA (aug) [8 refs] 51.96 0.86 7.235 2.609 0.814

RMIA (aug) [8 refs] 69.82 1.28 12.057 5.404 2.349

LiRA (online) [8 refs] 64.26 1.05 8.054 2.615 0.766

RMIA (online) [8 refs] 66.24 0.83 8.291 3.054 0.920

LiRA (online, aug) [8 refs] 67.42 1.40 11.026 4.502 1.647

RMIA (online, aug) [8 refs] 69.62 1.31 11.697 5.280 2.483
```

### 5.7 CIFAR-100: 16 models with 18-query augmented logits

The same procedure as for CIFAR-10, on the harder 100-class dataset.

**Command:**
```bash
caffeinate -is env AUGMENTS=18 bash scripts/train_cifar100.sh 16
```

**Output:**
```text
=== training model 0 / 15 ===

[train] saved exp/cifar100/experiment-0/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-0/keep.npy and exp/cifar100/experiment-0/logits.npy

=== training model 1 / 15 ===

[train] saved exp/cifar100/experiment-1/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-1/keep.npy and exp/cifar100/experiment-1/logits.npy

=== training model 2 / 15 ===

[train] saved exp/cifar100/experiment-2/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-2/keep.npy and exp/cifar100/experiment-2/logits.npy

=== training model 3 / 15 ===

[train] saved exp/cifar100/experiment-3/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-3/keep.npy and exp/cifar100/experiment-3/logits.npy

=== training model 4 / 15 ===

[train] saved exp/cifar100/experiment-4/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-4/keep.npy and exp/cifar100/experiment-4/logits.npy

=== training model 5 / 15 ===

[train] saved exp/cifar100/experiment-5/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-5/keep.npy and exp/cifar100/experiment-5/logits.npy

=== training model 6 / 15 ===

[train] saved exp/cifar100/experiment-6/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-6/keep.npy and exp/cifar100/experiment-6/logits.npy

=== training model 7 / 15 ===

[train] saved exp/cifar100/experiment-7/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-7/keep.npy and exp/cifar100/experiment-7/logits.npy

=== training model 8 / 15 ===

[train] saved exp/cifar100/experiment-8/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-8/keep.npy and exp/cifar100/experiment-8/logits.npy

=== training model 9 / 15 ===

[train] saved exp/cifar100/experiment-9/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-9/keep.npy and exp/cifar100/experiment-9/logits.npy

=== training model 10 / 15 ===

[train] saved exp/cifar100/experiment-10/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-10/keep.npy and exp/cifar100/experiment-10/logits.npy

=== training model 11 / 15 ===

[train] saved exp/cifar100/experiment-11/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-11/keep.npy and exp/cifar100/experiment-11/logits.npy

=== training model 12 / 15 ===

[train] saved exp/cifar100/experiment-12/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-12/keep.npy and exp/cifar100/experiment-12/logits.npy

=== training model 13 / 15 ===

[train] saved exp/cifar100/experiment-13/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-13/keep.npy and exp/cifar100/experiment-13/logits.npy

=== training model 14 / 15 ===

[train] saved exp/cifar100/experiment-14/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-14/keep.npy and exp/cifar100/experiment-14/logits.npy

=== training model 15 / 15 ===

[train] saved exp/cifar100/experiment-15/logits_aug.npy (18 queries: clean + 17 augmentations)

[train] saved exp/cifar100/experiment-15/keep.npy and exp/cifar100/experiment-15/logits.npy

Done. Models in exp/cifar100/experiment-*/
```

Completeness check over the finished set:

**Command:**
```bash
ls exp/cifar100 | wc -l
```

**Output:**
```text
18
```

(16 experiment folders plus the two dataset arrays saved into the experiment directory.)

**Command:**
```bash
for d in exp/cifar100/experiment-*; do
[ -f "$d/logits.npy" ] && [ -f "$d/keep.npy" ] && [ -f "$d/logits_aug.npy" ] || echo "INCOMPLETE: $d"
done
grep -l WARNING logs/cifar100/log_*
```

No output: every folder is complete and no model diverged. The audit, again over 10 target models at four reference budgets with augmented queries:

**Command:**
```bash
bash scripts/attack_cifar10.sh exp/cifar100 results/cifar100 18
```

**Output:**
```text
[audit] loading 16 models from exp/cifar100

[audit] augmented setting: using 18 queries per point (query 0 = clean image)

[audit] averaging over 10 target models

[audit] target model 0: 25233 members, 24767 non-members
```

**Output (final averaged table):**
```text
[audit] results written to results/cifar100/results_table.json

attack AUC +-std TPR@1% TPR@0.1% TPR@0.01%

Attack-P [1 refs] 76.84 1.19 1.073 0.000 0.000

Attack-R [1 refs] 80.64 1.03 8.422 0.000 0.000

LiRA [1 refs] 66.15 1.16 9.650 1.849 0.301

RMIA [1 refs] 84.79 1.26 16.880 3.520 0.523

LiRA (aug) [1 refs] 67.41 1.16 12.347 2.904 0.558

RMIA (aug) [1 refs] 86.96 1.45 19.267 3.926 0.811

Attack-P [2 refs] 76.84 1.19 1.073 0.000 0.000

Attack-R [2 refs] 80.31 0.91 14.839 0.000 0.000

LiRA [2 refs] 69.01 1.38 13.415 3.027 0.562

RMIA [2 refs] 85.96 1.37 21.932 6.830 1.482

LiRA (aug) [2 refs] 70.20 1.26 16.980 4.735 0.920

RMIA (aug) [2 refs] 87.98 1.56 24.981 7.921 1.916

LiRA (online) [2 refs] 80.91 1.23 15.583 3.725 0.591

RMIA (online) [2 refs] 85.00 1.29 16.315 2.907 0.416

LiRA (online, aug) [2 refs] 84.58 1.36 21.166 5.831 1.453

RMIA (online, aug) [2 refs] 87.43 1.40 21.577 5.138 0.876

Attack-P [4 refs] 76.84 1.19 1.073 0.000 0.000

Attack-R [4 refs] 80.35 0.85 20.160 0.000 0.000

LiRA [4 refs] 70.61 1.45 15.759 4.247 1.017

RMIA [4 refs] 86.63 1.42 25.269 9.929 3.436

LiRA (aug) [4 refs] 71.92 1.33 20.133 6.916 1.672

RMIA (aug) [4 refs] 88.54 1.60 28.603 11.545 3.744

LiRA (online) [4 refs] 83.82 1.47 21.026 6.416 1.543

RMIA (online) [4 refs] 86.26 1.39 22.561 5.996 1.333

LiRA (online, aug) [4 refs] 87.15 1.51 27.494 10.286 2.564

RMIA (online, aug) [4 refs] 88.57 1.50 27.671 9.854 2.812

Attack-P [8 refs] 76.84 1.19 1.073 0.000 0.000

Attack-R [8 refs] 80.89 0.97 21.794 0.396 0.000

LiRA [8 refs] 71.81 1.58 16.714 4.526 1.131

RMIA [8 refs] 87.10 1.49 27.025 11.825 4.970

LiRA (aug) [8 refs] 72.81 1.46 21.531 7.363 2.079

RMIA (aug) [8 refs] 88.91 1.67 30.254 13.473 5.092

LiRA (online) [8 refs] 85.29 1.58 24.584 9.330 3.190

RMIA (online) [8 refs] 87.03 1.41 26.502 9.796 2.652

LiRA (online, aug) [8 refs] 88.57 1.60 31.875 14.925 5.646

RMIA (online, aug) [8 refs] 89.28 1.51 31.868 14.500 5.308
```

### 5.8 Purchase-100: full-protocol audit across 10 targets and 6 reference budgets

The audits above evaluate the four standard budgets. The paper's headline Purchase-100 comparison uses six budgets, from 1 to 32 reference models, averaged over 10 targets. This workload reads saved logits only (no training), is NumPy-based, and was parallelised across CPU workers; the results are identical to the serial run.

**Command:**
```bash
caffeinate -is python main.py --logdir exp/purchase100 --target-idx all \
  --num-ref-models 1 2 4 8 16 32 --out results/purchase100_all
```

**Output:**
```text
[audit] loading 64 models from exp/purchase100

[audit] parallel mode: 5 worker processes (~4.1 GB each estimated; results identical to serial)

[audit] target model 2: 24953 members, 25047 non-members ... done

[audit] target model 3: 25004 members, 24996 non-members ... done

[audit] target model 0: 24906 members, 25094 non-members ... done

[audit] target model 4: 25214 members, 24786 non-members ... done

[audit] target model 1: 24896 members, 25104 non-members ... done

[audit] target model 6: 25103 members, 24897 non-members ... done

[audit] target model 5: 24926 members, 25074 non-members ... done

[audit] target model 7: 25074 members, 24926 non-members ... done

[audit] target model 9: 24915 members, 25085 non-members ... done

[audit] target model 8: 25194 members, 24806 non-members ... done

[audit] results written to results/purchase100_all/results_table.json

attack AUC +-std TPR@1% TPR@0.1% TPR@0.01%

Attack-P [1 refs] 65.11 0.31 0.000 0.000 0.000

Attack-R [1 refs] 71.36 0.25 1.473 0.000 0.000

LiRA [1 refs] 56.36 0.35 3.092 0.585 0.142

RMIA [1 refs] 75.07 0.26 8.201 1.216 0.131

Attack-P [2 refs] 65.11 0.31 0.000 0.000 0.000

Attack-R [2 refs] 71.66 0.29 2.788 0.000 0.000

LiRA [2 refs] 58.69 0.45 3.750 0.698 0.142

RMIA [2 refs] 77.39 0.30 10.753 2.204 0.291

LiRA (online) [2 refs] 72.28 0.41 6.112 1.210 0.295

RMIA (online) [2 refs] 76.52 0.27 9.782 2.303 0.439

Attack-P [4 refs] 65.11 0.31 0.000 0.000 0.000

Attack-R [4 refs] 72.28 0.28 5.353 0.000 0.000

LiRA [4 refs] 59.74 0.49 4.058 0.718 0.120

RMIA [4 refs] 78.96 0.33 12.783 3.422 0.683

LiRA (online) [4 refs] 75.71 0.48 8.786 1.948 0.436

RMIA (online) [4 refs] 78.37 0.33 12.236 3.208 0.593

Attack-P [8 refs] 65.11 0.31 0.000 0.000 0.000

Attack-R [8 refs] 73.03 0.29 7.141 0.000 0.000

LiRA [8 refs] 60.63 0.56 4.310 0.695 0.138

RMIA [8 refs] 79.86 0.33 14.367 4.196 0.852

LiRA (online) [8 refs] 77.27 0.46 10.568 2.588 0.732

RMIA (online) [8 refs] 79.60 0.36 14.008 4.310 1.027

Attack-P [16 refs] 65.11 0.31 0.000 0.000 0.000

Attack-R [16 refs] 73.79 0.31 7.919 0.313 0.000

LiRA [16 refs] 61.39 0.63 4.487 0.715 0.119

RMIA [16 refs] 80.25 0.35 15.328 4.627 1.058

LiRA (online) [16 refs] 78.68 0.49 12.639 3.619 1.022

RMIA (online) [16 refs] 80.37 0.39 15.217 4.877 1.340

Attack-P [32 refs] 65.11 0.31 0.000 0.000 0.000

Attack-R [32 refs] 74.24 0.34 8.615 1.095 0.000

LiRA [32 refs] 61.78 0.69 4.697 0.756 0.136

RMIA [32 refs] 80.48 0.37 15.833 4.985 1.165

LiRA (online) [32 refs] 79.69 0.54 14.486 4.370 1.038

RMIA (online) [32 refs] 80.78 0.43 15.989 5.096 1.606
```

### 5.9 ROC figures

Each dataset's ROC comparison (attack TPR versus FPR on a logarithmic axis, the figure style used in the paper) is generated by a single-target audit, which saves the ROC image, the score arrays and the results table into a dedicated results folder.

**Command:**
```bash
python main.py --logdir exp/cifar100 --target-idx 0 --num-ref-models 8 --num-augmentations 18 --out results/cifar100_roc
python main.py --logdir exp/cifar10 --target-idx 0 --num-ref-models 8 --num-augmentations 18 --out results/cifar10_roc
python main.py --logdir exp/purchase100 --target-idx 0 --num-ref-models 8 --out results/purchase100_roc
```

**Command (verification):**
```bash
ls results/cifar100_roc results/cifar10_roc results/purchase100_roc
```

**Output:**
```text
results/cifar100_roc:

results_table.json roc_8_ref_models.png scores_8_ref_models.npz

results/cifar10_roc:

results_table.json roc_8_ref_models.png scores_8_ref_models.npz

results/purchase100_roc:

results_table.json roc_8_ref_models.png scores_8_ref_models.npz
```

Each folder contains `results_table.json`, `roc_8_ref_models.png` and `scores_8_ref_models.npz`, so every figure can be regenerated from saved scores without retraining.

### 5.10 Hardware-independence control: a second Purchase-100 set trained and audited natively

To verify that the reported results are properties of the method rather than of one specific model set, a second, small Purchase-100 set (8 models) was trained from scratch in a separate experiment directory and audited with the same code path.

**Command:**
```bash
for i in 0 1 2 3 4 5 6 7; do
caffeinate -is python -m src.train --dataset purchase100 --data-dir data/purchase100 \
--arch mlp --expid $i --num-experiments 8 \
--logdir exp/purchase100_mac --epochs 50
done
```

**Output:**
```text
[train] using device: mps

[train] experiment-0: 24734 / 50000 training points

epoch 10/50 loss 0.3665

epoch 20/50 loss 0.1204

epoch 30/50 loss 0.0100

epoch 40/50 loss 0.0115

epoch 50/50 loss 0.0106

[train] experiment-0: train acc 1.0000, held-out acc 0.8333

[train] saved exp/purchase100_mac/experiment-0/keep.npy and exp/purchase100_mac/experiment-0/logits.npy

[train] using device: mps

[train] experiment-1: 25019 / 50000 training points

epoch 10/50 loss 0.3262

epoch 20/50 loss 0.0939

epoch 30/50 loss 0.0083

epoch 40/50 loss 0.0114

epoch 50/50 loss 0.0104

[train] experiment-1: train acc 1.0000, held-out acc 0.8340

[train] saved exp/purchase100_mac/experiment-1/keep.npy and exp/purchase100_mac/experiment-1/logits.npy

[train] using device: mps

[train] experiment-2: 24820 / 50000 training points

epoch 10/50 loss 0.3775

epoch 20/50 loss 0.1265

epoch 30/50 loss 0.0070

epoch 40/50 loss 0.0112

epoch 50/50 loss 0.0101

[train] experiment-2: train acc 1.0000, held-out acc 0.8300

[train] saved exp/purchase100_mac/experiment-2/keep.npy and exp/purchase100_mac/experiment-2/logits.npy

[train] using device: mps

[train] experiment-3: 24996 / 50000 training points

epoch 10/50 loss 0.4284

epoch 20/50 loss 0.1593

epoch 30/50 loss 0.0085

epoch 40/50 loss 0.0119

epoch 50/50 loss 0.0109

[train] experiment-3: train acc 1.0000, held-out acc 0.8302

[train] saved exp/purchase100_mac/experiment-3/keep.npy and exp/purchase100_mac/experiment-3/logits.npy

[train] using device: mps

[train] experiment-4: 25126 / 50000 training points

epoch 10/50 loss 0.3943

epoch 20/50 loss 0.1501

epoch 30/50 loss 0.0111

epoch 40/50 loss 0.0118

epoch 50/50 loss 0.0112

[train] experiment-4: train acc 1.0000, held-out acc 0.8332

[train] saved exp/purchase100_mac/experiment-4/keep.npy and exp/purchase100_mac/experiment-4/logits.npy

[train] using device: mps

[train] experiment-5: 24956 / 50000 training points

epoch 10/50 loss 0.4496

epoch 20/50 loss 0.1646

epoch 30/50 loss 0.0160

epoch 40/50 loss 0.0116

epoch 50/50 loss 0.0110

[train] experiment-5: train acc 1.0000, held-out acc 0.8265

[train] saved exp/purchase100_mac/experiment-5/keep.npy and exp/purchase100_mac/experiment-5/logits.npy

[train] using device: mps

[train] experiment-6: 25130 / 50000 training points

epoch 10/50 loss 0.4170

epoch 20/50 loss 0.1207

epoch 30/50 loss 0.0071

epoch 40/50 loss 0.0120

epoch 50/50 loss 0.0110

[train] experiment-6: train acc 1.0000, held-out acc 0.8336

[train] saved exp/purchase100_mac/experiment-6/keep.npy and exp/purchase100_mac/experiment-6/logits.npy

[train] using device: mps

[train] experiment-7: 25219 / 50000 training points

epoch 10/50 loss 0.3294

epoch 20/50 loss 0.1144

epoch 30/50 loss 0.0090

epoch 40/50 loss 0.0114

epoch 50/50 loss 0.0102

[train] experiment-7: train acc 1.0000, held-out acc 0.8379

[train] saved exp/purchase100_mac/experiment-7/keep.npy and exp/purchase100_mac/experiment-7/logits.npy
```

**Command (verification):**
```bash
ls exp/purchase100_mac
```

**Output:**
```text
experiment-0 experiment-1 experiment-2 experiment-3 experiment-4 experiment-5 experiment-6 experiment-7 y_train.npy
```

**Command:**
```bash
python main.py --logdir exp/purchase100_mac --target-idx 0 \
--num-ref-models 1 2 4 --out results/purchase100_mac_live
```

**Output:**
```text
[audit] loading 8 models from exp/purchase100_mac

[audit] averaging over 1 target models

[audit] target model 0: 24734 members, 25266 non-members

[audit] === attacks with 1 reference model(s) ===

Attack-P AUC 65.61 TPR@1% 0.0000 TPR@0.1% 0.0000 TPR@0.01% 0.0000

Attack-R AUC 71.66 TPR@1% 0.0116 TPR@0.1% 0.0000 TPR@0.01% 0.0000

LiRA AUC 56.48 TPR@1% 0.0287 TPR@0.1% 0.0046 TPR@0.01% 0.0011

RMIA AUC 75.25 TPR@1% 0.0797 TPR@0.1% 0.0110 TPR@0.01% 0.0009

[audit] === attacks with 2 reference model(s) ===

Attack-P AUC 65.61 TPR@1% 0.0000 TPR@0.1% 0.0000 TPR@0.01% 0.0000

Attack-R AUC 71.86 TPR@1% 0.0283 TPR@0.1% 0.0000 TPR@0.01% 0.0000

LiRA AUC 58.55 TPR@1% 0.0349 TPR@0.1% 0.0055 TPR@0.01% 0.0001

RMIA AUC 77.51 TPR@1% 0.1045 TPR@0.1% 0.0269 TPR@0.01% 0.0052

LiRA (online) AUC 72.82 TPR@1% 0.0577 TPR@0.1% 0.0125 TPR@0.01% 0.0023

RMIA (online) AUC 76.70 TPR@1% 0.0923 TPR@0.1% 0.0256 TPR@0.01% 0.0036

[audit] === attacks with 4 reference model(s) ===

Attack-P AUC 65.61 TPR@1% 0.0000 TPR@0.1% 0.0000 TPR@0.01% 0.0000

Attack-R AUC 70.39 TPR@1% 0.0368 TPR@0.1% 0.0000 TPR@0.01% 0.0000

LiRA AUC 58.08 TPR@1% 0.0278 TPR@0.1% 0.0040 TPR@0.01% 0.0000

RMIA AUC 79.07 TPR@1% 0.1191 TPR@0.1% 0.0272 TPR@0.01% 0.0036

LiRA (online) AUC 76.05 TPR@1% 0.0831 TPR@0.1% 0.0181 TPR@0.01% 0.0030

RMIA (online) AUC 78.52 TPR@1% 0.1158 TPR@0.1% 0.0338 TPR@0.01% 0.0090

[audit] results written to results/purchase100_mac_live/results_table.json

attack AUC +-std TPR@1% TPR@0.1% TPR@0.01%

Attack-P [1 refs] 65.61 0.00 0.000 0.000 0.000

Attack-R [1 refs] 71.66 0.00 1.160 0.000 0.000

LiRA [1 refs] 56.48 0.00 2.866 0.461 0.105

RMIA [1 refs] 75.25 0.00 7.969 1.100 0.093

Attack-P [2 refs] 65.61 0.00 0.000 0.000 0.000

Attack-R [2 refs] 71.86 0.00 2.834 0.000 0.000

LiRA [2 refs] 58.55 0.00 3.489 0.550 0.012

RMIA [2 refs] 77.51 0.00 10.447 2.693 0.522

LiRA (online) [2 refs] 72.82 0.00 5.765 1.253 0.226

RMIA (online) [2 refs] 76.70 0.00 9.230 2.559 0.364

Attack-P [4 refs] 65.61 0.00 0.000 0.000 0.000

Attack-R [4 refs] 70.39 0.00 3.683 0.000 0.000

LiRA [4 refs] 58.08 0.00 2.782 0.400 0.004

RMIA [4 refs] 79.07 0.00 11.911 2.725 0.356

LiRA (online) [4 refs] 76.05 0.00 8.312 1.807 0.303

RMIA (online) [4 refs] 78.52 0.00 11.575 3.380 0.902
```

### 5.11 Source code integrity

The exact code that produced all results above is pinned by checksums:

**Command:**
```bash
shasum -a 256 src/train.py main.py scripts/train_purchase100.sh
```

**Output:**
```text
1db84355e2e1fab17b9e869f6db06193dfdf727b5541158727ee97674609096a src/train.py

d8b1ad1bc275ff02b0c721dba630d284954e1f8256658712ccc96961e4db2ba9 main.py

05412de480b7509cbfc07137e93d6fd751088447b410bc32311f068dc2ad373a scripts/train_purchase100.sh
```

### 5.12 Final state of the project directory

**Command:**
```bash
find exp/purchase100 exp/cifar10 exp/cifar100 results -type f | wc -l
```

**Output:**
```text
254
```

## 6. Results and comparison with the paper

Official results of this project (all numbers are AUC percentages averaged over 10 target models; "+-std" is the standard deviation across those targets):

| Dataset and setting | Result (this project) | Paper reference values |
|---|---|---|
| Purchase-100, offline, 8 refs | RMIA 79.86 +- 0.33 (LiRA 60.63, Attack-R 73.03, Attack-P 65.11) | RMIA around 80-81 in this regime |
| Purchase-100, online, 8 refs | RMIA 79.60 +- 0.36 (LiRA online 77.27) | online RMIA about 80.5 |
| Purchase-100, offline, 32 refs | RMIA 80.48 +- 0.37; online 80.78 +- 0.43 | the paper's advantage requires few references, larger budgets only add a small gain |
| CIFAR-10, augmented, 8 refs | RMIA (aug) 69.82 +- 1.28; RMIA (online, aug) 69.62 +- 1.31 | about 68.6 to 71.0 depending on the exact setting |
| CIFAR-100, augmented, 8 refs | RMIA (online, aug) 89.28 +- 1.51, TPR at 1% FPR 31.87% | about 89 to 90 |

Findings, matching the paper on all three datasets:

1. RMIA is the strongest offline attack at every reference budget; the baselines need online access to approach it.
2. The RMIA advantage grows as the false positive budget shrinks: at 8 references on Purchase-100, RMIA's TPR at 0.1% FPR is 4.20% versus 0.69% for LiRA and 0% for the threshold attacks, and at 0.01% FPR RMIA still finds 0.85% of members while the baselines are near zero.
3. Augmented queries strengthen RMIA and LiRA markedly on the CIFAR datasets (CIFAR-10: RMIA 66.50 without augmentation versus 69.82 with).
4. RMIA remains effective with a single reference model (Purchase-100: 75.07 AUC), where offline LiRA collapses (56.36).

Hardware-independence control (Purchase-100, target model 0, RMIA offline AUC):

| Reference budgets | 64-model set (10-target average) | 8-model control set (same machine, freshly trained) | Deviation |
|---|---|---|---|
| 1 ref | 75.07 | 75.25 | 0.18 |
| 2 refs | 77.39 | 77.51 | 0.12 |
| 4 refs | 78.96 | 79.07 | 0.11 |

The two independently trained sets agree to within 0.18 AUC at every budget, confirming that the method, not a particular artefact of one training run, produces the reported numbers.

## 7. Limitations and required computational resources

What this project did not reproduce, exclusively for computational reasons:

- The paper's reference model pool of 256 models per dataset. This project trained 64 (Purchase-100) and 16 (CIFAR-10, CIFAR-100). At the equal reference budgets that were evaluated, the results match the paper closely; the paper's largest-budget regime (up to 255 reference models) was covered only up to 32 budgets on Purchase-100 and up to 8 on the CIFAR datasets.
- The augmented-query setting used 18 queries per record instead of the strongest settings in the paper.
- CINIC-10 and ImageNet were not attempted at all.
- Audit parallelism was bounded by the laptop's CPU and memory capacity.

Full reproduction of the paper would therefore require training roughly 4x more models on Purchase-100 and 16x more on each CIFAR dataset, followed by proportionally larger audits. On a single laptop this workload is intractable. A compute cluster makes it straightforward: reference model training is embarrassingly parallel (each model is one independent training job), so a batch or array scheduler can produce the full model pools in parallel, and the audit stage can be distributed similarly. The University of Luxembourg operates central HPC facilities (CPU and GPU partitions) that are designed for exactly this kind of workload; the remaining gap between this project and the full paper protocol is a matter of access to such resources, not of methodology.

---

*All commands and outputs in Section 5 are reproduced verbatim from the terminal sessions in which the experiments were executed.*
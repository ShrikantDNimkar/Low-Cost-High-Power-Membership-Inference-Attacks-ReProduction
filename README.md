# RMIA Re-implementation: Low-Cost High-Power Membership Inference Attacks

A CPU-friendly, self-contained re-implementation of the ICML 2024 paper
**"Low-Cost High-Power Membership Inference Attacks"** by Sajjad Zarifzadeh,
Philippe Liu and Reza Shokri ([arXiv:2312.03262](https://arxiv.org/abs/2312.03262)),
designed for a university project demo on a laptop without a GPU.

The paper's official code lives at
[privacytrustlab/ml_privacy_meter research/2024_rmia](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/research/2024_rmia).
This project re-implements the attack pipeline from scratch in PyTorch + NumPy,
keeping the official parameter defaults and file conventions so results can be
cross-validated against the original (see `validate/OFFICIAL_CROSSCHECK.md`).

## What the paper does (30-second version)

A membership inference attack (MIA) asks: *was this data point x used to train
model theta?* The paper designs a statistical hypothesis test, **RMIA**, that
compares x against the whole population: x is flagged as a member when its
likelihood ratio LR(x, z) beats a threshold gamma for a large fraction of
population points z:

```
Score(x) = Pr_z[ LR(x, z) > gamma ]                       (Eq. 5)
LR(x, z) = (Pr(x|theta)/Pr(x)) * (Pr(z)/Pr(z|theta))      (Eq. 3)
```

`Pr(x)` is estimated with reference models (models trained on other random
halves of the population). RMIA needs as few as **1** reference model, while
prior attacks (LiRA, Attack-R) collapse to near-random guessing at that budget.

## Attacks implemented

| Attack | Paper reference | Code | Reference models used |
|---|---|---|---|
| RMIA (offline + online) | Eq. 3-5, App. B.2.2 | `src/attacks.py::rmia` | k OUT (offline), k/2 IN + k/2 OUT (online) |
| LiRA (offline + online) | Carlini et al. 2022 | `src/attacks.py::lira` | same budget rule |
| Attack-R (offline) | Ye et al. 2022 | `src/attacks.py::attack_r` | k OUT |
| Attack-P (population) | Ye et al. 2022 / Yeom et al. 2018 | `src/attacks.py::attack_p` | none |

Metrics: AUC and TPR at FPR = 1%, 0.1%, 0.01% (`src/evaluate.py`), TPR-FPR
curves on a log-log plot as in the paper (`src/plots.py`).

## Setup

Python 3.10+ (the official RMIA environment also uses 3.10).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 0: 5-minute smoke test (no downloads)

Runs the entire pipeline on a synthetic Purchase-like dataset
(train 9 small MLPs, run every attack, produce the results table and ROC plots):

```bash
python tools/test_attacks.py            # unit tests: math + sklearn cross-check
python tools/quickstart_synthetic.py    # end-to-end pipeline, ~3-5 min on CPU
```

Expected outcome (validated): RMIA dominates every baseline at each reference
budget, and offline LiRA collapses with 1-2 reference models, exactly the
qualitative behaviour reported in the paper (its Figure 1 and Table 2).

## Step 1: get the Purchase-100 dataset

The paper uses the standard preprocessed Purchase-100 files
(`x_train.npy`, `y_train.npy`; 50,000 x 600 binary features, 100 classes),
distributed by the ml_privacy_meter authors in this Google Drive folder
(link from the official `train.py`):
<https://drive.google.com/drive/folders/1cIJlbLlgqDSJKd8YhTucHwaPiLsh6ZyW?usp=sharing>

```bash
pip install gdown
mkdir -p data/purchase100
# download the purchase100 .npy files from the folder above into data/purchase100/
```

Tip: the official RMIA configs audit the 50,000-point `x_train` (the "purchase100"
set in `attack_configs/purchase100/*.yaml` uses `dataset_size: 50000`). If the
folder also contains larger 197k variants, use the 50k files or re-create the
standard split described in the official repo.

## Step 2: train the model set

Each model trains on a balanced random half of the dataset (the official 50/50
partitioning: after all N models, every point was used by exactly half of them).

```bash
bash scripts/train_purchase100.sh 16        # 16 models (default, ~30-60 min on CPU)
# or a bigger set for stronger attacks:
bash scripts/train_purchase100.sh 64
```

Each model takes a few minutes on a CPU (MLP 600-512-256-128-64-100, 50 epochs,
SGD + cosine schedule, exactly the official hyperparameters). Logs go to
`logs/purchase100/`, models and logits to `exp/purchase100/experiment-i/`.

## Step 3: run the attacks

```bash
bash scripts/attack_purchase100.sh
# equivalently:
python main.py --logdir exp/purchase100 --num-ref-models 1 2 4 8 --out results/purchase100
```

Outputs in `results/purchase100/`:
- console table: AUC and TPR@1%/0.1%/0.01% for every attack and budget
  (the layout of Table 2 in the paper)
- `roc_<k>_ref_models.png`: TPR-FPR curves per budget (log-log, as in the paper)
- `results_table.json` and `scores_<k>_ref_models.npz`: raw numbers for your report

### Recommended audit settings

- **Multiple target models** (paper protocol: results averaged over 10 random
  target models):
  `python main.py --logdir exp/purchase100 --target-idx all --num-ref-models 1 2 4 8`
  The summary table then reports the mean and std of the AUC across targets.
- **Broken-model protection**: the audit checks every model's accuracy on its
  own training points. A diverged model (accuracy near random guessing, which
  can happen with lr=0.1 on unlucky seeds) is reported and excluded from the
  reference pool; the trainer also auto-retries with lr/10 when it detects
  divergence, so re-running `src.train` for that experiment fixes it.

### Troubleshooting: a model trained to ~random accuracy

With the default lr=0.1 an occasional seed diverges (train accuracy near the
random-guess level). Two options:
1. Just re-run that experiment: `python -m src.train ... --expid 5 ...`
   (the new divergence guard retries with lr/10 automatically), or
2. Train the whole set with a lower learning rate: `EPOCHS=50` and edit the
   `--lr` default, e.g. `python -m src.train ... --lr 0.02`.

## What to expect on Purchase-100

The paper's Table 10 (appendix) reports, with 1 reference model, roughly:
Attack-P AUC ~71, Attack-R ~78, offline LiRA ~77, **RMIA ~85**, and RMIA keeps
a large lead at low FPR. With 16 models you should see RMIA ahead of every
baseline at every budget, and the gap widening at TPR@0.1% / 0.01%.
Numbers vary with the model count and seeds; the *ordering* of the attacks is
the reproducible scientific claim.

## Project structure

```
configs/purchase100.yaml     # all hyperparameters (official defaults)
main.py                       # audit orchestrator (loads logits, runs attacks, reports)
src/
  signals.py                  # softmax / conf-logit / loss signals
  attacks.py                  # RMIA, LiRA, Attack-R, Attack-P (pure NumPy)
  evaluate.py                 # ROC / AUC / TPR@FPR (exact tie handling)
  plots.py                    # TPR-FPR figures
  train.py                    # PyTorch trainer (official hyperparameters, MPS support)
  models.py                   # Purchase-100 MLP, WideResNet-28-2, small CNN
scripts/
  train_purchase100.sh       # CPU
  train_cifar10.sh            # Apple Silicon / GPU (WRN-28-2, auto-selects MPS)
  train_cifar100.sh
  attack_purchase100.sh / attack_cifar10.sh
tools/
  test_attacks.py             # unit tests (ROC vs sklearn, attack semantics)
  check_models.py             # architecture + MPS self-check (needs torch)
  quickstart_synthetic.py     # no-download end-to-end demo
validate/OFFICIAL_CROSSCHECK.md   # how to compare against the official repo
DEMO_GUIDE.md                 # theory summary + demo talking points
```

## Paper <-> code map

| Paper element | Where |
|---|---|
| Membership inference game (Def. 2.1) | `main.py` (target model + membership from `keep.npy`) |
| Pr(x|theta): softmax of true class, temperature 2 | `src/signals.py::softmax_signal` |
| Pr(x) from reference models (Eq. 4) | `src/attacks.py::rmia` (trim_mean over IN+OUT / OUT signals) |
| Offline approximation p(x) = (1+a)/2 p_OUT + (1-a)/2 (App. B.2.2) | `rmia(offline=True, offline_a=0.2)` |
| Pairwise LR + score (Eq. 3, 5) | `src/attacks.py::rmia` (outer product, chunked) |
| gamma threshold (= 2) | `rmia(gamma=2.0)` |
| Balanced 50/50 partitioning | `src/train.py::balanced_keep` |
| LiRA Gaussian fit (fix_variance) | `src/attacks.py::lira` |
| Attack-R interpolation | `src/attacks.py::attack_r` |
| Metrics (TPR@FPR) | `src/evaluate.py` |

## Optional extension: CIFAR-10 / CIFAR-100 (Apple Silicon track)

On a MacBook with an M-series chip, the project trains CIFAR models on the
Metal GPU (`--device auto` picks MPS automatically). `src/models.py` includes
the paper's actual CIFAR architecture, **WideResNet-28-2**, and the trainer
uses the official protocol for CIFAR: 100 epochs, random crop + horizontal
flip augmentation, per-channel normalization.

```bash
bash scripts/train_cifar10.sh 32     # overnight on an M1 Pro (~4-8 h)
bash scripts/attack_cifar10.sh exp/cifar10 results/cifar10
```

### Augmented setting (paper's strongest attack)

The paper's most powerful RMIA variant (Figs. 4-6, "augmentation" rows in the
official configs) queries every model on **Q versions of each image**: the
clean image plus Q-1 random crop/flip augmentations. RMIA then takes a
majority vote across the Q queries per population pair, and LiRA averages its
per-query likelihood ratios. To include this setting, train with `AUGMENTS`
and pass the same number to the audit:

```bash
AUGMENTS=18 bash scripts/train_cifar10.sh 32    # adds ~Q/2 min per model
bash scripts/attack_cifar10.sh exp/cifar10 results/cifar10 18
```

The audit then additionally reports **RMIA (aug)**, **RMIA (online, aug)**,
**LiRA (aug)** and **LiRA (online, aug)**. Augmentation applies to CIFAR only
(Purchase-100 is tabular). All models in a logdir must be trained with the
same `AUGMENTS` value. Query 0 is always the clean image, matching the
official `--aug=Q` convention, so the paper's aug=18 and aug=50 settings
correspond to `AUGMENTS=18` and `AUGMENTS=50`.

### Parallel audits and faster training

- Audits with `--target-idx all` now default to one worker process per CPU
  core (capped by a memory estimate so the machine does not swap). Override
  with `--workers N` (`--workers 1` = the original serial path). Verified
  bit-identical to serial on a synthetic model set.
- The CIFAR random-crop augmentation is vectorized (two `torch.gather`
  calls instead of a per-image Python loop with GPU syncs), and the epoch
  loss no longer syncs every step. Each training run self-checks the
  vectorized crop against the reference loop on the actual device and
  raises on any mismatch.

The first CIFAR run downloads the dataset via torchvision and caches it as
`.npy` inside the experiment folder, so later runs and the audit never need
the download again.

Rough per-model cost of WRN-28-2 (100 epochs, half of CIFAR-10) on an M1 Pro:

| Model set | Time on M1 Pro | What it supports |
|---|---|---|
| 16 models | ~2-4 h | the paper's 1/2/4-reference-model experiments |
| 32 models | ~4-8 h (overnight) | solid averaged curves (10-target protocol) |
| 64 models | ~8-16 h (weekend) | near-paper quality up to 8-ref budgets |
| 256 models (paper) | ~2-3 days | the full protocol (leave running) |

**Example one-week schedule on a MacBook:**

| Night | Command | Result next morning |
|---|---|---|
| Mon | `bash scripts/train_cifar10.sh 16` | clean CIFAR-10 run ready; audit it, keep as the safe fallback |
| Tue | `AUGMENTS=18 bash scripts/train_cifar100.sh 16` | CIFAR-100 with the augmented setting |
| Wed | `AUGMENTS=18 bash scripts/train_cifar10.sh 32` | the full CIFAR-10 demo set (32 models, aug) |
| Thu | audit + plots: `bash scripts/attack_cifar10.sh exp/cifar10 results/cifar10 18` | final results table + ROC figures |
| Fri | rehearse with `DEMO_GUIDE.md` | keep the Purchase-100 numbers as backup slides |

The Purchase-100 run needs no retraining and stays the primary CPU demo;
the CIFAR runs are the "full re-implementation" tier.

Purchase-100 MLPs gain nothing from the GPU (too small); train them on CPU
as usual. If PyTorch reports a missing op on MPS, run with
`PYTORCH_ENABLE_MPS_FALLBACK=1` (the trainer sets it automatically on MPS).

## Results of this reproduction

This code was used to run a partial reproduction of the paper on a single
laptop (64 reference models on Purchase-100, 16 per CIFAR dataset, 10 audit
targets, 18 augmented queries). The measured results match the paper at
comparable budgets; headline numbers:

- Purchase-100, RMIA offline: 79.86 ± 0.33 AUC at 8 refs, 80.48 ± 0.37 at 32 refs (paper ~80.5 in this regime)
- CIFAR-10, RMIA (aug): 69.82 ± 1.28 AUC at 8 refs (paper range 68.6-71.0)
- CIFAR-100, RMIA (online, aug): 89.28 ± 1.51 AUC at 8 refs, TPR at 1% FPR 31.87% (paper ~89-90)

See [`docs/results/RESULTS.md`](docs/results/RESULTS.md) for the full
comparison with the paper, the reference-budget scaling and the
hardware-independence control, with the verbatim final audit tables in
[`docs/results/`](docs/results/), and
[`docs/rmia-re-implementation-execution-and-reproducibility-report.md`](docs/rmia-re-implementation-execution-and-reproducibility-report.md)
for the complete execution report (every command and its terminal output).

## Acknowledgements

Developed with AI assistance (design discussions, code scaffolding and
report drafting) for a university course project on AI and Cybersecurity.
All experiments were executed by the author on personal hardware; every
result documented in `docs/` is copied verbatim from those runs.

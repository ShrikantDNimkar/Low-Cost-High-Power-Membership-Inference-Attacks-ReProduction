# Cross-validating this re-implementation against the official RMIA code

The goal: show that this clean-room implementation computes the same attack as
[privacytrustlab/ml_privacy_meter research/2024_rmia](https://github.com/privacytrustlab/ml_privacy_meter/tree/master/research/2024_rmia)
(commit `d327341`).

## 1. Parameter defaults (verified against the official yamls)

| Parameter | Official (`attack_configs/purchase100/rmia_offline_1_ref_models.yaml`) | This repo |
|---|---|---|
| temperature | 2.0 | `configs/purchase100.yaml: audit.temperature` |
| gamma | 2.0 | `audit.gamma` |
| offline_a | 0.2 | `audit.offline_a` |
| proportiontocut | 0.2 | `audit.proportiontocut` |
| num_ref_models ("1 ref model") | 2 (= 1 IN + 1 OUT budget; offline uses the OUT one) | `main.py` calls attacks with `num_ref_models = 2k` for k used models |
| LiRA fix_variance | True | `lira(fix_variance=True)` |
| dataset_size (purchase100) | 50000 | `data.dataset_size` |
| epochs (purchase100) | 50 | `train.epochs` |

## 2. Formula-level correspondence

- **Signal**: `convert_signals(..., 'softmax', temp=2.0)` in official
  `signals.py` == `src/signals.py::softmax_signal(logits, labels, 2.0)`.
- **Offline p(x)**: official `main.py` computes
  `prob_ratio_x = t_x / ((1+a)/2 * mean_x + (1-a)/2)` with
  `mean_x = trim_mean(out_signals, proportiontocut=0.2)` == our
  `rmia(offline=True, offline_a=a)`.
- **Score**: official `-((final_scores > gamma).mean(1))` (negated so lower =
  member, then `roc_curve(answers, -prediction)`) == our
  `Pr_z[LR(x,z) > gamma]` with higher = member. Sign conventions differ, the
  ROC is identical.
- **Self-exclusion**: the official code rescales population points by
  `((pred*len) - 1)/(len-1)`; we exclude z = x directly (LR(x,x) = 1 never
  passes gamma >= 1, so the two agree for gamma >= 1).
- **LiRA**: official uses `median`/global-std (`fix_variance`) Gaussian fits on
  the conf-logit signal; ours mirrors it. Note: the official code computes
  `std_out = np.std(in_signals)` in the `fix_variance` branch (apparently
  inherited from Carlini's code); we use the std of the OUT signals, which is
  the intended reading of "fix variance" in Carlini et al. This changes LiRA
  numbers slightly, not its qualitative behaviour.
- **Attack-R**: official interpolates the target loss inside the sorted OUT
  reference losses padded with dummy min/max on `linspace(0,1)`; ours mirrors
  this exactly (`src/attacks.py::attack_r`).
- **Partitioning**: official train.py uses
  `keep = uniform(0,1,(N,D)).argsort(0) < int(pkeep*N)`; ours is identical
  (`src/train.py::balanced_keep`).

## 3. Recommended cross-validation procedure

1. **Same models, two attack implementations.** Train a small set of
   Purchase-100 models with the official pipeline (follow the official
   README: conda env from Ye et al., `scripts/train_purchase100.sh`,
   `scripts/infer_purchase100.sh`), OR train with this repo and copy the
   `logits.npy` files into the official `exp/purchase100/experiment-i/logits/`
   structure (renaming to the official `{epoch:010d}_{aug:04d}.npy` format,
   aug=2, plus `keep.npy`).
2. Run the official attack:
   `python main.py --cf attack_configs/purchase100/rmia_offline_2_ref_models.yaml`
3. Run ours on the same logits:
   `python main.py --logdir exp/purchase100 --num-ref-models 2`
4. Compare AUC / TPR@FPR. Expect agreement within a fraction of a point
   (remaining differences: the trim-mean edge cases with very few reference
   models, and the LiRA `fix_variance` std choice noted above).
5. Directional check: both must rank attacks identically
   (RMIA > Attack-R / LiRA / Attack-P) and both must show offline LiRA
   degrading sharply at 1-2 reference models.

## 4. Known intentional deviations

- Training is PyTorch instead of objax/JAX; the official trainer uses an EMA
  model (`momentum=0.999`) and evaluates the EMA weights. We train plain SGD
  (nesterov, momentum 0.9) with the same lr schedule, batch size, weight decay
  and epochs. Model accuracy (hence attack strength) can differ slightly;
  the attack comparison is still apples-to-apples because all attacks are
  computed on the SAME models.
- CIFAR models: the official objax WideResNet uses BatchNorm with momentum 0
  (objax semantics: batch statistics are used at eval time too). PyTorch
  BatchNorm with momentum 0 would never update its running statistics, so we
  use the PyTorch default (0.1). This is the only architectural difference;
  the block layout (pre-activation, widths 16-32-64 with widen factor 2) is
  identical. Inputs are normalized per channel instead of scaled to [-1, 1].
- The audit here evaluates every point of the dataset (members = the target's
  keep mask), matching the official `privacy_loss_model` game on the train set.
- We report TPR@{1%, 0.1%, 0.01%} by default; the official also reports 0.001%
  and 0%. At 50k audit points those levels rest on a handful of points, so we
  keep them out of the default table (add levels in `src/evaluate.py`).
- Augmented queries (official `augmentation: augmented` setting) ARE included:
  `src/train.py --num-augmentations Q` saves `logits_aug.npy` of shape
  (Q, N, C), where query 0 is the CLEAN image and queries 1..Q-1 are
  independent random crop+flip augmentations (the same weak augmentation as
  training). This matches the official inference.py convention
  ("--aug=2 # contains the original query"), so the paper's aug=18/50
  settings correspond to Q=18/50 here.
  Attack-side semantics, matching official main.py exactly:
  - RMIA (augmented): per query k, the boolean matrix
    B_k[x, z] = (LR_k(x, z) > gamma) is computed with that query's own
    p(x) (per-augmentation trim_mean, as in the official loop over
    `nb_augmentations`); the per-(x, z) vote is the MAJORITY across queries
    (official uses `torch.mode`, whose tie-break returns the smaller value;
    our equivalent rule is "strictly more than Q/2 votes", ties -> False);
    the score then averages the majority-voted booleans over the population
    (official `augmented_test.mean(1)`).
  - LiRA (augmented): per query k, the Gaussian log-likelihood ratio with
    that query's own median/std parameters, then the MEAN over queries
    (official `score.mean(1)` over the augmentation axis).
  - Attack-P / Attack-R do not use augmentation in the official code either.
  - Augmented attacks only apply to CIFAR (purchase100 is tabular; the
    official pipeline does not augment it either).
  With Q=1 the implementation reduces exactly to the unaugmented attacks
  (verified bit-exact by `tools/test_attacks.py`).

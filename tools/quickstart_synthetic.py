"""
Quickstart: end-to-end validation of the RMIA re-implementation on a
SYNTHETIC Purchase-like dataset - no downloads, no GPU, runs in minutes.

What it does
------------
1. Generates a synthetic binary-feature classification dataset with the same
   shape as Purchase-100 (tabular, many classes) but smaller.
2. Trains 9 small MLP models (scikit-learn) with the same balanced 50/50
   data partitioning as the paper (each point trains exactly half the models).
3. Runs the exact same attack pipeline as the Purchase-100 demo
   (main.run_all_attacks): RMIA, LiRA, Attack-R, Attack-P.
4. Prints the AUC / TPR@FPR table and saves ROC curves.

This is a smoke test of the code, not a reproduction of paper numbers:
the real demo should use the actual Purchase-100 dataset (see README.md).

Run from the project root:
    python tools/quickstart_synthetic.py --out results/synthetic
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def balanced_keep(num_experiments: int, dataset_size: int, expid: int,
                  pkeep: float = 0.5) -> np.ndarray:
    """Same partitioning as src.train.balanced_keep (duplicated so that this
    quickstart runs without PyTorch installed)."""
    np.random.seed(0)
    keep = np.random.uniform(0, 1, size=(num_experiments, dataset_size))
    order = keep.argsort(0)
    keep = order < int(pkeep * num_experiments)
    return np.array(keep[expid], dtype=bool)


def make_synthetic_dataset(n: int = 12000, d: int = 400, n_class: int = 30,
                           flip: float = 0.30, seed: int = 7):
    """Binary tabular dataset: each class has a random prototype bit-vector;
    features are the prototype with `flip` fraction of bits flipped.

    With flip=0.30 and many classes the task is hard enough that small MLPs
    memorize part of their training set, which is exactly the leakage that
    membership inference attacks measure.
    """
    rng = np.random.RandomState(seed)
    prototypes = (rng.rand(n_class, d) > 0.5).astype(np.float32)
    y = rng.randint(0, n_class, size=n)
    noise = (rng.rand(n, d) < flip).astype(np.float32)
    x = prototypes[y] * (1 - 2 * noise) + noise  # flips prototype bits w.p. `flip`
    return x.astype(np.float32), y.astype(np.int64)


def train_sklearn_mlp(x_train, y_train, x_all, seed: int, max_iter: int = 150,
                     hidden=(256, 128)):
    """Small MLP stand-in for the PyTorch trainer (same protocol, tiny model).

    Fits on the model's kept training subset, then returns raw logits
    (log-probabilities) for EVERY input in x_all.
    """
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                        solver="adam", learning_rate_init=1e-3, batch_size=256,
                        max_iter=max_iter, random_state=seed)
    mlp.fit(x_train, y_train)
    proba = mlp.predict_proba(x_all)
    return np.log(proba + 1e-45).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-models", type=int, default=9)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--d", type=int, default=300)
    p.add_argument("--n-class", type=int, default=40)
    p.add_argument("--num-ref-models", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/synthetic")
    p.add_argument("--max-iter", type=int, default=120)
    args = p.parse_args()

    from src import attacks, evaluate, signals
    from src.plots import plot_roc
    import main as audit_main

    os.makedirs(args.out, exist_ok=True)

    print(f"[data] synthetic dataset: {args.n} x {args.d}, {args.n_class} classes")
    x, y = make_synthetic_dataset(args.n, args.d, args.n_class, seed=args.seed)

    logdir = os.path.join(PROJECT_ROOT, "exp", "synthetic")
    os.makedirs(logdir, exist_ok=True)
    np.save(os.path.join(logdir, "y_train.npy"), y)

    print(f"[train] training {args.n_models} models on balanced 50/50 partitions")
    logits_all, keeps_all = [], []
    for i in range(args.n_models):
        keep = balanced_keep(args.n_models, args.n, i)
        logits = train_sklearn_mlp(x[keep], y[keep], x, seed=i, max_iter=args.max_iter)
        train_acc = (logits.argmax(1)[keep] == y[keep]).mean()
        heldout_acc = (logits.argmax(1)[~keep] == y[~keep]).mean()
        print(f"  model {i}: train acc {train_acc:.3f}  held-out acc {heldout_acc:.3f}")
        exp_dir = os.path.join(logdir, f"experiment-{i}")
        os.makedirs(exp_dir, exist_ok=True)
        np.save(os.path.join(exp_dir, "keep.npy"), keep)
        np.save(os.path.join(exp_dir, "logits.npy"), logits)
        logits_all.append(logits)
        keeps_all.append(keep)

    logits = np.stack(logits_all)
    keeps = np.stack(keeps_all)
    member = keeps[0]
    print(f"[audit] target model 0: {member.sum()} members / {(~member).sum()} non-members")

    all_reports = {}
    for k in args.num_ref_models:
        print(f"[audit] === attacks with {k} reference model(s) ===")
        scores = audit_main.run_all_attacks(logits, keeps, y, 0, k)
        for name, s in scores.items():
            rep = evaluate.evaluate_attack(s, member,
                                           fpr_levels=(0.01, 0.001, 0.0001))
            all_reports[f"{name} [{k} refs]"] = rep
            print(f"  {name:16s}  AUC {rep['auc']*100:6.2f}  "
                  f"TPR@1% {rep['tpr@0.01']:.4f}  TPR@0.1% {rep['tpr@0.001']:.4f}  "
                  f"TPR@0.01% {rep['tpr@0.0001']:.4f}")
        plot_roc({n: r for n, r in all_reports.items() if n.endswith(f"[{k} refs]")},
                 os.path.join(args.out, f"roc_{k}_ref_models.png"),
                 title=f"Synthetic data ({k} reference model(s))")

    print("\n=== Summary (synthetic smoke test) ===")
    print(f"{'attack':28s}{'AUC':>8s}{'TPR@1%':>10s}{'TPR@0.1%':>11s}{'TPR@0.01%':>12s}")
    for name, rep in all_reports.items():
        print(f"{name:28s}{rep['auc']*100:8.2f}{rep['tpr@0.01']*100:10.3f}"
              f"{rep['tpr@0.001']*100:11.3f}{rep['tpr@0.0001']*100:12.3f}")
    print(f"\n[audit] figures saved in {args.out}/")


if __name__ == "__main__":
    main()

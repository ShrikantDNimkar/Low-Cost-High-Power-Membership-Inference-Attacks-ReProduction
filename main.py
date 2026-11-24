"""
RMIA audit orchestrator: loads the trained models' logits and runs all
attacks (RMIA, LiRA, Attack-R, Attack-P) as in the paper's Table 2, then
reports AUC and TPR@low-FPR and saves the ROC figure.

Mirrors the flow of the official research/2024_rmia/main.py:
  1. load target + reference logits and keep matrices
  2. compute signals (softmax w/ temperature 2 for RMIA, confidence logit
     for LiRA, loss for Attack-P / Attack-R)
  3. run each attack for a given reference budget
  4. report AUC / TPR@FPR and plot the TPR-FPR curves

Usage:
    python main.py --logdir exp/purchase100 --target-idx 0 \
        --num-ref-models 1 2 4 8 --out results/purchase100
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import src.attacks as attacks
import src.evaluate as evaluate
import src.signals as signals
from src.plots import plot_roc, plot_score_distributions


def load_experiments(logdir: str, num_models: int, dataset_size: int = None,
                    load_aug: bool = False):
    """Load logits + keep for experiment-0..num_models-1 from `logdir`.

    With load_aug=True, also loads logits_aug.npy (the Q augmented queries
    per point, saved by train.py --num-augmentations) wherever present.
    Returns (logits, keeps, logits_aug); logits_aug is None if no experiment
    has one.
    """
    all_logits, all_keeps, all_aug = [], [], []
    any_aug = False
    for i in range(num_models):
        exp = os.path.join(logdir, f"experiment-{i}")
        keep = np.load(os.path.join(exp, "keep.npy"))
        logits = np.load(os.path.join(exp, "logits.npy"))
        if dataset_size is not None and len(keep) != dataset_size:
            raise ValueError(f"{exp}: keep size {len(keep)} != {dataset_size}")
        aug_path = os.path.join(exp, "logits_aug.npy")
        if load_aug and os.path.exists(aug_path):
            all_aug.append(np.load(aug_path))  # (Q, N, C)
            any_aug = True
        else:
            all_aug.append(None)
        all_keeps.append(keep)
        all_logits.append(logits)
    logits = np.stack(all_logits)
    keeps = np.stack(all_keeps)
    if not any_aug:
        return logits, keeps, None
    if any(a is None for a in all_aug):
        missing = [i for i, a in enumerate(all_aug) if a is None]
        raise SystemExit(
            "[audit] logits_aug.npy is missing for experiment(s) "
            f"{missing}. All models must be trained with the same "
            "--num-augmentations (e.g. AUGMENTS=18 bash scripts/train_cifar10.sh 32).")
    return logits, keeps, np.stack(all_aug)  # (M, Q, N, C)


def run_all_attacks(logits, keeps, labels, target_idx: int, num_ref_models: int,
                    gamma: float = 2.0, offline_a: float = 0.2,
                    temperature: float = 2.0, chunk_size: int = 2048,
                    logits_aug=None, n_augs: int = 1):
    """Run every attack for a fixed reference budget.

    Returns {attack_name: scores}; higher score = member.
    Reference models = all models except the target.

    If `logits_aug` is given (M, Q, N, C) and n_augs > 1, additionally runs
    the paper's AUGMENTED variants (official "augmentation: augmented"):
    RMIA with majority voting across queries and LiRA with per-query
    log-likelihood ratios averaged over queries. Query 0 is the clean image.
    """
    n_models = logits.shape[0]
    ref_idx = [i for i in range(n_models) if i != target_idx]

    target_logits = logits[target_idx]
    ref_logits = logits[ref_idx]
    ref_keep = keeps[ref_idx]

    sm_target = signals.softmax_signal(target_logits, labels, temperature)
    sm_ref = np.stack([signals.softmax_signal(ref_logits[r], labels, temperature)
                       for r in range(len(ref_idx))])
    conf_target = signals.conf_logit_signal(target_logits, labels)
    conf_ref = np.stack([signals.conf_logit_signal(ref_logits[r], labels)
                         for r in range(len(ref_idx))])
    loss_target = signals.cross_entropy_loss(target_logits, labels)
    loss_ref = np.stack([signals.cross_entropy_loss(ref_logits[r], labels)
                         for r in range(len(ref_idx))])

    # --- augmented signals (paper's strongest setting) --------------------
    sm_target_aug = sm_ref_aug = conf_target_aug = conf_ref_aug = None
    if logits_aug is not None and n_augs > 1:
        tl_aug = logits_aug[target_idx, :n_augs]          # (Q, N, C)
        rl_aug = logits_aug[np.ix_(ref_idx, np.arange(n_augs))]  # (R, Q, N, C)
        sm_target_aug = np.stack(
            [signals.softmax_signal(tl_aug[q], labels, temperature) for q in range(n_augs)],
            axis=1)                                        # (N, Q)
        sm_ref_aug = np.stack(
            [np.stack([signals.softmax_signal(rl_aug[r, q], labels, temperature)
                       for q in range(n_augs)], axis=1)
             for r in range(len(ref_idx))])                 # (R, N, Q)
        conf_target_aug = np.stack(
            [signals.conf_logit_signal(tl_aug[q], labels) for q in range(n_augs)],
            axis=1)                                        # (N, Q)
        conf_ref_aug = np.stack(
            [np.stack([signals.conf_logit_signal(rl_aug[r, q], labels)
                       for q in range(n_augs)], axis=1)
             for r in range(len(ref_idx))])                 # (R, N, Q)

    out = {}
    k = num_ref_models  # k = number of reference models USED by the attack
    out["Attack-P"] = attacks.attack_p(loss_target)
    if k >= 1:
        # offline attacks: k OUT models per point.
        # The official flag num_ref_models = 2k gives a budget of
        # num_ref_models // 2 = k IN and k OUT models; offline mode uses OUT.
        out["Attack-R"] = attacks.attack_r(loss_target, loss_ref, ref_keep, 2 * k)
        out["LiRA"] = attacks.lira(conf_target, conf_ref, ref_keep, 2 * k, offline=True)
        out["RMIA"] = attacks.rmia(sm_target, sm_ref, ref_keep, 2 * k,
                                   offline=True, offline_a=offline_a,
                                   gamma=gamma, chunk_size=chunk_size)
        if sm_target_aug is not None:
            out["LiRA (aug)"] = attacks.lira(conf_target_aug, conf_ref_aug, ref_keep,
                                             2 * k, offline=True)
            out["RMIA (aug)"] = attacks.rmia(sm_target_aug, sm_ref_aug, ref_keep, 2 * k,
                                             offline=True, offline_a=offline_a,
                                             gamma=gamma, chunk_size=chunk_size)
    # online attacks use k/2 IN + k/2 OUT models (official: num_ref_models = k)
    if k >= 2:
        out["LiRA (online)"] = attacks.lira(conf_target, conf_ref, ref_keep, k, offline=False)
        out["RMIA (online)"] = attacks.rmia(sm_target, sm_ref, ref_keep, k,
                                            offline=False, gamma=gamma,
                                            chunk_size=chunk_size)
        if sm_target_aug is not None:
            out["LiRA (online, aug)"] = attacks.lira(conf_target_aug, conf_ref_aug,
                                                     ref_keep, k, offline=False)
            out["RMIA (online, aug)"] = attacks.rmia(sm_target_aug, sm_ref_aug, ref_keep, k,
                                                     offline=False, gamma=gamma,
                                                     chunk_size=chunk_size)
    return out


def _audit_target_worker(payload):
    """Runs in a separate process: audit ONE target model and return its
    per-(attack, k-ref) reports. Reloads the logits from disk (the OS page
    cache makes this cheap) instead of pickling multi-GB arrays to workers.
    Returns (t_pos, n_members, n_nonmembers, {key: report})."""
    (logdir, exps, t_pos, k_list, gamma, offline_a, temperature, n_augs) = payload
    logits, keeps, logits_aug = load_experiments(logdir, max(exps) + 1,
                                                 load_aug=n_augs > 1)
    logits, keeps = logits[exps], keeps[exps]
    if logits_aug is not None:
        logits_aug = logits_aug[exps]
    labels = np.load(os.path.join(logdir, "y_train.npy"))
    member = keeps[t_pos]
    n_models = len(exps)
    per_target = {}
    for k in k_list:
        k_eff = min(k, n_models - 1)
        scores = run_all_attacks(logits, keeps, labels, t_pos, k_eff,
                                gamma=gamma, offline_a=offline_a,
                                temperature=temperature, logits_aug=logits_aug,
                                n_augs=n_augs)
        for name, s in scores.items():
            rep = evaluate.evaluate_attack(s, member,
                                           fpr_levels=(0.01, 0.001, 0.0001))
            per_target[f"{name} [{k_eff} refs]"] = rep
    return t_pos, int(member.sum()), int((~member).sum()), per_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="exp/purchase100")
    p.add_argument("--target-idx", default=0,
                   help="index of the target model, or 'all' to average over "
                        "the first models (paper protocol: 10 targets)")
    p.add_argument("--num-ref-models", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--offline-a", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--out", default="results/purchase100")
    p.add_argument("--num-models", type=int, default=None,
                   help="number of experiment folders to load (default: all)")
    p.add_argument("--num-augmentations", type=int, default=1,
                   help="number of queries per point to use for the augmented "
                        "attacks (paper's strongest setting; requires models "
                        "trained with --num-augmentations > 1). Default: 1 "
                        "(clean query only, augmented attacks skipped).")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel worker processes when auditing multiple "
                        "targets (--target-idx all): one per target model, "
                        "results identical to serial. 0 = auto: use all CPU "
                        "cores, capped by a memory-budget estimate so the "
                        "machine cannot swap. 1 = serial (previous behavior).")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # discover models
    exps = sorted(int(d.split("-")[1]) for d in os.listdir(args.logdir)
                  if d.startswith("experiment-"))
    if args.num_models:
        exps = [e for e in exps if e < args.num_models]
    n_models = len(exps)
    print(f"[audit] loading {n_models} models from {args.logdir}")

    logits, keeps, logits_aug = load_experiments(args.logdir, max(exps) + 1,
                                                 load_aug=args.num_augmentations > 1)
    logits = logits[exps]
    keeps = keeps[exps]
    labels = np.load(os.path.join(args.logdir, "y_train.npy"))
    if logits_aug is not None:
        if args.num_augmentations > logits_aug.shape[1]:
            raise SystemExit(
                f"[audit] --num-augmentations {args.num_augmentations} exceeds the "
                f"{logits_aug.shape[1]} queries stored in logits_aug.npy")
        print(f"[audit] augmented setting: using {args.num_augmentations} queries "
              f"per point (query 0 = clean image)")
        logits_aug = logits_aug[exps]

    # Health check: a diverged model (accuracy ~ random on its own training
    # points) would poison every attack that uses it as a reference. Detect
    # such models and exclude them from the audit.
    n_class = int(labels.max()) + 1
    healthy = []
    for i in range(len(exps)):
        acc_on_own_train = (logits[i].argmax(1)[keeps[i]] == labels[keeps[i]]).mean()
        if acc_on_own_train < max(5.0 / n_class, 0.05):
            print(f"[audit] WARNING: model {exps[i]} looks diverged "
                  f"(train acc {acc_on_own_train:.3f}) - EXCLUDED from the audit")
        else:
            healthy.append(i)
    if len(healthy) < len(exps):
        logits, keeps = logits[healthy], keeps[healthy]
        if logits_aug is not None:
            logits_aug = logits_aug[healthy]
        exps = [exps[i] for i in healthy]
        n_models = len(exps)
        print(f"[audit] {n_models} healthy models remaining")

    target_list = ([int(args.target_idx)] if args.target_idx != "all"
                   else list(range(min(n_models, 10))))  # paper averages over 10 targets
    all_reports = {}
    auc_per_target = {}

    # ---- parallel path: one process per target model --------------------
    # Only meaningful with several targets; a single target falls back to
    # the original serial loop below (bit-identical code path).
    n_workers = 0
    if args.workers != 1 and len(target_list) > 1:
        data_bytes = logits.nbytes + (logits_aug.nbytes if logits_aug is not None else 0)
        per_worker = 2 * data_bytes + int(1.5e9)  # data + float64 intermediates
        mem_cap = max(1, int(24e9 // per_worker))  # keep the machine out of swap
        cpu_cap = os.cpu_count() or 1
        if args.workers > 0:
            n_workers = min(args.workers, len(target_list))
        else:
            n_workers = max(1, min(len(target_list), cpu_cap, mem_cap))
        if n_workers > 1:
            print(f"[audit] parallel mode: {n_workers} worker processes "
                  f"(~{per_worker/1e9:.1f} GB each estimated; results identical to serial)")
            # free the parent's copy of the (large) arrays before spawning
            del logits, keeps, logits_aug
            gc.collect()
            payloads = [(args.logdir, exps, t, list(args.num_ref_models), args.gamma,
                         args.offline_a, args.temperature, args.num_augmentations)
                        for t in target_list]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_audit_target_worker, pl): pl[2] for pl in payloads}
                for fut in as_completed(futures):
                    t_pos, n_mem, n_non, reps = fut.result()
                    print(f"[audit] target model {exps[t_pos]}: {n_mem} members, "
                          f"{n_non} non-members ... done")
                    for key, rep in reps.items():
                        all_reports.setdefault(key, []).append(rep)
                        auc_per_target.setdefault(key, []).append(rep["auc"])

    if n_workers <= 1:
        # serial path (single target, or --workers 1): unchanged behavior
        for t_idx in target_list:
            member = keeps[t_idx]
            if target_list != [args.target_idx] and t_idx == target_list[0]:
                print(f"[audit] averaging over {len(target_list)} target models")
            print(f"[audit] target model {exps[t_idx]}: {member.sum()} members, "
                  f"{(~member).sum()} non-members")
            for k in args.num_ref_models:
                k = min(k, n_models - 1)
                print(f"[audit] === attacks with {k} reference model(s) ===")
                scores = run_all_attacks(logits, keeps, labels, t_idx, k,
                                         gamma=args.gamma, offline_a=args.offline_a,
                                         temperature=args.temperature,
                                         logits_aug=logits_aug,
                                         n_augs=args.num_augmentations)
                for name, s in scores.items():
                    rep = evaluate.evaluate_attack(s, member,
                                                   fpr_levels=(0.01, 0.001, 0.0001))
                    all_reports.setdefault(f"{name} [{k} refs]", []).append(rep)
                    auc_per_target.setdefault(f"{name} [{k} refs]", []).append(rep["auc"])
                    print(f"  {name:16s}  AUC {rep['auc']*100:6.2f}  "
                          f"TPR@1% {rep['tpr@0.01']:.4f}  TPR@0.1% {rep['tpr@0.001']:.4f}  "
                          f"TPR@0.01% {rep['tpr@0.0001']:.4f}")
                if len(target_list) == 1:
                    plot_roc({n: r[0] for n, r in all_reports.items() if n.endswith(f"[{k} refs]")},
                             os.path.join(args.out, f"roc_{k}_ref_models.png"),
                             title=f"TPR-FPR curves ({k} reference model(s))")
                    np.savez(os.path.join(args.out, f"scores_{k}_ref_models.npz"),
                             member=member, **scores)

    # average reports across target models (paper protocol) and summarize
    def mean_report(reps):
        import statistics
        aucs = [r["auc"] for r in reps]
        mean = dict(reps[0])
        mean["auc"] = float(statistics.mean(aucs))
        mean["auc_std"] = float(statistics.pstdev(aucs)) if len(aucs) > 1 else 0.0
        for key in [k for k in reps[0] if k.startswith("tpr@")]:
            mean[key] = float(statistics.mean(r[key] for r in reps))
        return mean

    averaged = {name: mean_report(reps) for name, reps in all_reports.items()}

    # summary table
    rows = []
    for name, rep in averaged.items():
        rows.append({
            "attack": name,
            "auc": round(rep["auc"] * 100, 2),
            "auc_std": round(rep["auc_std"] * 100, 2),
            "tpr@1%": round(rep["tpr@0.01"] * 100, 3),
            "tpr@0.1%": round(rep["tpr@0.001"] * 100, 3),
            "tpr@0.01%": round(rep["tpr@0.0001"] * 100, 3),
        })
    with open(os.path.join(args.out, "results_table.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[audit] results written to {args.out}/results_table.json")
    print(f"{'attack':28s}{'AUC':>8s}{'+-std':>7s}{'TPR@1%':>10s}{'TPR@0.1%':>11s}{'TPR@0.01%':>12s}")
    for r in rows:
        print(f"{r['attack']:28s}{r['auc']:8.2f}{r['auc_std']:7.2f}{r['tpr@1%']:10.3f}"
              f"{r['tpr@0.1%']:11.3f}{r['tpr@0.01%']:12.3f}")


if __name__ == "__main__":
    main()

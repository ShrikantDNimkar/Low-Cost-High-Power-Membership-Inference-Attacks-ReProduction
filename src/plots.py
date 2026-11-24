"""
Plotting helpers: TPR-FPR trade-off curves as in the paper (log-x ROC).

The paper plots TPR (y) versus FPR (x) on a log scale, so that performance at
low false-positive rates (the regime that matters for privacy auditing) is
visible.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_roc(reports: dict, out_path: str, title: str = "Membership Inference Attacks",
             x_max: float = 1.0):
    """reports: {attack_name: {"fpr": ..., "tpr": ...}}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, rep in reports.items():
        fpr, tpr = np.asarray(rep["fpr"]), np.asarray(rep["tpr"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {rep['auc']*100:.2f})", linewidth=1.8)
    ax.plot([1e-6, x_max], [1e-6, x_max], "k--", linewidth=0.8, alpha=0.5,
            label="random guessing")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-5 if x_max >= 1e-4 else x_max / 10, x_max)
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_score_distributions(scores_by_attack: dict, member: np.ndarray, out_path: str):
    """Histograms of member vs non-member scores for each attack."""
    names = list(scores_by_attack.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        s = scores_by_attack[name]
        ax.hist(s[member], bins=60, alpha=0.6, label="member", density=True)
        ax.hist(s[~member], bins=60, alpha=0.6, label="non-member", density=True)
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

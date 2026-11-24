"""
Evaluation of membership inference attacks: ROC, AUC, TPR@low-FPR.

Higher attack score = member (convention enforced by src/attacks.py).

The metrics follow the official RMIA code (main.py: metric_results):
AUC over the whole ROC, and TPR at fixed FPR levels. The paper reports
TPR@FPR = 0.01% and 0.0%; for a 50k-point audit that is 2-3 points, so we
also report 1% and 0.1% which are more stable at demo scale.
"""

from __future__ import annotations

import numpy as np


def roc_curve(scores: np.ndarray, member: np.ndarray):
    """Compute the ROC curve with proper tie handling.

    Returns (fpr, tpr, thresholds) sorted by increasing score threshold,
    where a point is predicted member iff score >= threshold.
    """
    scores = np.asarray(scores, dtype=np.float64)
    member = np.asarray(member, dtype=bool)
    n_pos = member.sum()
    n_neg = len(member) - n_pos

    # group identical scores (ties must share one operating point)
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    m_sorted = member[order]

    # boundaries between groups of equal scores
    new_group = np.empty(len(s_sorted), dtype=bool)
    new_group[0] = True
    new_group[1:] = s_sorted[1:] != s_sorted[:-1]
    group_id = np.cumsum(new_group) - 1
    n_groups = group_id[-1] + 1

    tp = np.bincount(group_id, weights=m_sorted.astype(np.float64), minlength=n_groups)
    fp = np.bincount(group_id, weights=(~m_sorted).astype(np.float64), minlength=n_groups)
    # thresholds per group, in descending score order (group 0 = highest)
    thresholds = np.zeros(n_groups)
    thresholds[group_id] = s_sorted  # first occurrence of each group is its score

    # cumulative counts as the threshold is lowered group by group
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    tpr = np.concatenate([[0.0], cum_tp]) / n_pos
    fpr = np.concatenate([[0.0], cum_fp]) / n_neg
    thresholds = np.concatenate([[np.inf], thresholds])
    return fpr, tpr, thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under the ROC curve (trapezoid)."""
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """TPR at (at most) the target FPR, as in the official metric_results:
    tpr_list[where fpr_list <= target][-1]."""
    idx = np.nonzero(fpr <= target_fpr + 1e-12)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])


def evaluate_attack(scores: np.ndarray, member: np.ndarray,
                    fpr_levels=(0.01, 0.001, 0.0001)) -> dict:
    """Full metric report for one attack run."""
    fpr, tpr, _ = roc_curve(scores, member)
    report = {
        "auc": auc(fpr, tpr),
        "fpr": fpr,
        "tpr": tpr,
    }
    for lvl in fpr_levels:
        report[f"tpr@{lvl}"] = tpr_at_fpr(fpr, tpr, lvl)
    return report

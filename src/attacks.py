"""
Membership inference attacks: RMIA and the baselines from the paper
"Low-Cost High-Power Membership Inference Attacks" (Zarifzadeh, Liu, Shokri,
ICML 2024, https://arxiv.org/abs/2312.03262).

This is a clean-room NumPy re-implementation of the attacks in the official
code (https://github.com/privacytrustlab/ml_privacy_meter/tree/master/research/2024_rmia).

Attack summary (paper Table 1)
-----------------------------
RMIA (ours/paper):     Score(x) = Pr_z[ LR(x, z) >= gamma ]  with
                       LR(x, z) = (Pr(x|theta)/Pr(x)) * (Pr(z|theta)/Pr(z))^-1   (Eq. 3 + 5)
LiRA (Carlini et al.): Gaussian fit of IN/OUT reference signals, log-likelihood ratio.
Attack-R (Ye et al.):  rank of the target loss among OUT reference model losses.
Attack-P (Ye et al.):  rank of the target loss among population losses (no
                       reference models; equivalent to the LOSS attack of Yeom et al.).

All scores are oriented so that HIGHER = member.

Inputs (shared by every attack)
------------------------------
target_signal : (N,) signal of the target model theta on all N data points.
ref_signals  : (R, N) signals of R reference models on all N data points.
keep         : (R, N) bool, keep[r, i] = True iff data point i was in the
               training set of reference model r.
The membership of the points in the TARGET model is handled by evaluate.py;
attacks only see reference-model membership.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Helpers: per-point selection of IN / OUT reference models
# ---------------------------------------------------------------------------


def select_in_out(ref_signals: np.ndarray, keep: np.ndarray, budget: int):
    """For every data point, take the first `budget` IN and `budget` OUT reference signals.

    Mirrors the official main.py: for each data point it selects
    ``reference_signals[keep[:, i], i][:out_or_in_size]`` (IN) and
    ``reference_signals[~keep[:, i], i][:out_or_in_size]`` (OUT), where
    out_or_in_size = num_ref_models // 2.

    Returns
    -------
    in_sel  : (N, budget) float, NaN where fewer than `budget` IN models exist
    out_sel : (N, budget) float, NaN where fewer than `budget` OUT models exist
    """
    R, N = ref_signals.shape
    in_sel = np.full((N, budget), np.nan)
    out_sel = np.full((N, budget), np.nan)
    in_signals = np.where(keep, ref_signals, np.nan)   # (R, N)
    out_signals = np.where(~keep, ref_signals, np.nan)  # (R, N)
    for b in range(budget):
        in_sel[:, b] = _nth_valid(in_signals, b)
        out_sel[:, b] = _nth_valid(out_signals, b)
    return in_sel, out_sel


def _nth_valid(arr: np.ndarray, b: int) -> np.ndarray:
    """Value of the (b+1)-th non-NaN entry per column (NaN if fewer than b+1)."""
    N = arr.shape[1]
    out = np.full(N, np.nan)
    valid = ~np.isnan(arr)
    # number of valid entries strictly above each row
    cnt = np.cumsum(valid, axis=0) - valid
    take = valid & (cnt == b)
    idx = np.nonzero(take)
    out[idx[1]] = arr[idx]
    return out


# ---------------------------------------------------------------------------
# RMIA (paper Section 3)
# ---------------------------------------------------------------------------


def rmia(
    target_signal: np.ndarray,
    ref_signals: np.ndarray,
    keep: np.ndarray,
    num_ref_models: int = 2,
    offline: bool = True,
    offline_a: float = 0.2,
    gamma: float = 2.0,
    proportiontocut: float = 0.2,
    chunk_size: int = 2048,
    population: np.ndarray | None = None,
):
    """Robust Membership Inference Attack (RMIA), offline and online modes.

    Score_MIA(x; theta) = Pr_{z ~ pi} [ LR_theta(x, z) > gamma ]      (Eq. 5)
    LR_theta(x, z)     = (Pr(x|theta)/Pr(x)) / (Pr(z|theta)/Pr(z))    (Eq. 3)

    Offline mode (only OUT reference models available):
        Pr(x) ~= (1+a)/2 * mean_OUT Pr(x|theta') + (1-a)/2             (paper App. B.2.2)
    Online mode (IN and OUT models available):
        Pr(x)  = mean over the selected IN and OUT reference signals.

    Parameters mirror the official attack_configs yamls:
        num_ref_models : total reference budget; floor(num_ref_models/2)
                         IN models and floor(num_ref_models/2) OUT models are used.
        offline_a       : "offline_a" (0.2 in the official configs)
        gamma           : "gamma" (2.0 in the official configs)
        proportiontocut : trim_mean proportiontocut (0.2 in the official configs)

    Augmented queries (paper's strongest setting, official configs with
    "augmentation: augmented"):
        `target_signal` may be (N, Q) and `ref_signals` (R, N, Q), where query
        0 is the clean image and queries 1..Q-1 are augmented versions.
        Per query k the boolean matrix LR_k(x, z) > gamma is evaluated (each
        query has its own p(x) estimate), and the final vote per (x, z) pair
        is the MAJORITY across the Q queries. The official main.py majority-
        votes the per-query boolean matrices with torch.mode, whose tie-break
        returns the smaller value; our equivalent rule is therefore
        "strictly more than Q/2 votes". The score then averages the
        majority-voted booleans over the population z (official
        `augmented_test.mean(1)`). With Q == 1 this reduces exactly to the
        unaugmented attack.

    Returns
    -------
    scores : (N,) in [0, 1], higher = more likely member.
    """
    # --- normalise shapes: always work with an explicit query axis --------
    T = target_signal if target_signal.ndim == 2 else target_signal[:, None]
    R3 = ref_signals if ref_signals.ndim == 3 else ref_signals[:, :, None]
    N, Q = T.shape
    if population is None:
        population = np.arange(N)
    budget = num_ref_models // 2

    # --- per-query population prior p(x) ----------------------------------
    # each query/augmentation gets its own p(x), as in the official code
    # where the trim_mean is taken per augmentation index k
    p_all = np.empty((N, Q))
    for k in range(Q):
        in_sel, out_sel = select_in_out(R3[:, :, k], keep, budget)
        if offline:
            # p(x) from OUT signals only, scaled up by offline_a
            ref = out_sel
            p_all[:, k] = (1.0 + offline_a) / 2.0 * _nan_trim_mean(ref, proportiontocut) \
                + (1.0 - offline_a) / 2.0
        else:
            # online: average over IN and OUT signals (paper Eq. 4)
            ref = np.concatenate([in_sel, out_sel], axis=1)
            p_all[:, k] = _nan_trim_mean(ref, proportiontocut)

    # rare fallback: a point with no usable reference signal (e.g. it was in
    # every reference model's training set); use the population average
    if np.isnan(p_all).any():
        fallback = np.nanmean(p_all)
        p_all = np.where(np.isnan(p_all), fallback, p_all)

    # population statistics Pr(z), per query
    p_z = p_all[population]        # (P, Q)
    t_z = T[population]            # (P, Q)
    a_z = (p_z / t_z).T            # (Q, P): LR(x,z) = (t_x/p_x) * (p_z/t_z)

    scores = np.empty(N)
    self_vote = 1.0 if 1.0 > gamma else 0.0  # LR(x, x) == 1 exactly, for every query
    majority_threshold = Q / 2.0             # strictly more than Q/2 votes (ties -> False)
    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        r_x = (T[start:stop] / p_all[start:stop]).T        # (Q, chunk)
        votes = np.zeros((stop - start, len(population)), dtype=np.uint8)
        for k in range(Q):
            lr = r_x[k][:, None] * a_z[k][None, :]         # (chunk, P)
            votes += (lr > gamma)
        maj = votes > majority_threshold                    # majority vote across queries
        votes_sum = maj.sum(axis=1).astype(np.float64)
        # exclude z == x from the population (only matters for population rows)
        own = np.nonzero((population >= start) & (population < stop))[0]
        if len(own) > 0:
            votes_sum[population[own] - start] -= self_vote
        scores[start:stop] = votes_sum / max(len(population) - 1, 1)
    return scores


def _nan_trim_mean(arr: np.ndarray, proportiontocut: float) -> np.ndarray:
    """Trimmed mean per row, ignoring NaNs. scipy.stats.trim_mean semantics.

    The official code uses scipy.stats.trim_mean(..., proportiontocut=0.2),
    which cuts proportiontocut/2 = 10% from each end.
    """
    N = arr.shape[0]
    out = np.empty(N)
    for i in range(N):
        row = arr[i]
        row = row[~np.isnan(row)]
        if len(row) == 0:
            out[i] = np.nan
            continue
        k = int(len(row) * proportiontocut / 2)
        if k > 0 and len(row) > 2 * k:
            row = np.sort(row)[k:-k]
        out[i] = row.mean()
    return out


# ---------------------------------------------------------------------------
# LiRA (Carlini et al., 2022) - official signal "loss_logit_rescaled"
# ---------------------------------------------------------------------------


def lira(
    target_signal: np.ndarray,
    ref_signals: np.ndarray,
    keep: np.ndarray,
    num_ref_models: int = 2,
    offline: bool = True,
    fix_variance: bool = True,
):
    """LiRA in offline and online modes.

    Offline: score = -log N(s ; median OUT, std OUT)  (single Gaussian null)
    Online:  score = +log N(s ; median IN, std IN) - log N(s ; median OUT, std OUT)
             (the official code computes pr_in - pr_out = -logpdf_in + logpdf_out,
             which is lower=member, and flips the sign when calling roc_curve;
             we return the sign-flipped value so that higher = member here.)

    With fix_variance=True the std is the global std of the reference signals
    (as in the official configs and in Carlini et al.'s code).

    Augmented queries (official "augmentation: augmented" configs):
        `target_signal` may be (N, Q) and `ref_signals` (R, N, Q). Per query
        k the Gaussian log-likelihood ratio is computed with that query's
        own median/std parameters, and the final score is the MEAN over the
        Q queries (official: `score.mean(1)` over the augmentation axis).

    `target_signal`/`ref_signals` must be the confidence-logit signal
    (signals.conf_logit_signal).
    """
    # --- normalise shapes: always work with an explicit query axis --------
    T = target_signal if target_signal.ndim == 2 else target_signal[:, None]
    R3 = ref_signals if ref_signals.ndim == 3 else ref_signals[:, :, None]
    N, Q = T.shape
    budget = num_ref_models // 2

    score = np.zeros(N)
    import warnings
    with np.errstate(all="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for k in range(Q):
                in_sel, out_sel = select_in_out(R3[:, :, k], keep, budget)
                mean_out = np.nanmedian(out_sel, axis=1)
                if fix_variance:
                    std_out = np.nanstd(out_sel) + 1e-30
                else:
                    std_out = np.nanstd(out_sel, axis=1) + 1e-30
                logpdf_out = norm.logpdf(T[:, k], mean_out, std_out)

                if offline:
                    score += -logpdf_out
                else:
                    mean_in = np.nanmedian(in_sel, axis=1)
                    if fix_variance:
                        std_in = np.nanstd(in_sel) + 1e-30
                    else:
                        std_in = np.nanstd(in_sel, axis=1) + 1e-30
                    logpdf_in = norm.logpdf(T[:, k], mean_in, std_in)
                    score += logpdf_in - logpdf_out
    score = score / Q  # mean over the queries (official augmented LiRA)
    # points without any usable reference signal: score 0 (neutral)
    score = np.where(np.isnan(score), 0.0, score)
    return score


# ---------------------------------------------------------------------------
# Attack-R (Ye et al., 2022) - official signal "loss_reference"
# ---------------------------------------------------------------------------


def attack_r(
    target_signal: np.ndarray,
    ref_signals: np.ndarray,
    keep: np.ndarray,
    num_ref_models: int = 2,
):
    """Offline reference attack: rank of the target loss among OUT reference losses.

    Uses only OUT reference models (offline attack of Ye et al.).
    `*_signal` must be the cross-entropy loss (signals.cross_entropy_loss).

    Mirrors the official "loss_reference" aggregation: the target loss is
    interpolated within the sorted OUT reference losses padded with dummy
    min/max values, on an alpha grid in [0, 1]. Higher score = member
    (smaller relative loss), so the interpolated CDF is negated.
    """
    budget = max(num_ref_models // 2, 1)
    _, out_sel = select_in_out(ref_signals, keep, budget)
    scores = np.empty(len(target_signal))
    for i in range(len(target_signal)):
        refs = out_sel[i]
        refs = refs[~np.isnan(refs)]
        if len(refs) == 0:
            scores[i] = 0.0
            continue
        grid = np.sort(np.concatenate([refs, [0.0, 1000.0]]))
        alpha = np.linspace(0, 1, len(grid))
        scores[i] = -np.interp(target_signal[i], grid, alpha)
    return scores


# ---------------------------------------------------------------------------
# Attack-P (Ye et al., 2022) - official signal "loss_population"
# ---------------------------------------------------------------------------


def attack_p(
    target_signal: np.ndarray,
    population_signal: np.ndarray | None = None,
):
    """Population attack: rank of the target loss among population losses.

    Needs no reference models (equivalent to the LOSS attack of Yeom et al.
    with a population-calibrated threshold). `target_signal` must be the
    cross-entropy loss. score = -CDF(loss(x)) so that higher = member.
    """
    if population_signal is None:
        population_signal = target_signal
    sorted_pop = np.sort(population_signal)
    cdf = np.searchsorted(sorted_pop, target_signal) / len(sorted_pop)
    return -cdf

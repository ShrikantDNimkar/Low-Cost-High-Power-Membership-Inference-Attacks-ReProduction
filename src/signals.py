"""
Signals computed from model logits.

Re-implementation of the signals used by the RMIA paper
("Low-Cost High-Power Membership Inference Attacks", Zarifzadeh, Liu, Shokri,
ICML 2024, https://arxiv.org/abs/2312.03262) and its official code
(https://github.com/privacytrustlab/ml_privacy_meter/tree/master/research/2024_rmia,
file signals.py, function convert_signals).

All functions are pure NumPy so that the attack pipeline is independent of the
deep-learning framework used to train the models.

Conventions
-----------
logits : np.ndarray, shape (N, C), raw model outputs BEFORE softmax.
labels : np.ndarray, shape (N,), integer class labels in [0, C).
temperature : float, temperature applied to logits before the softmax
    (the official RMIA configs use temperature = 2.0).
"""

from __future__ import annotations

import numpy as np


def softmax_signal(logits: np.ndarray, labels: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """Pr(x|theta): softmax probability of the true class, with temperature.

    Used by RMIA (the official signal name is "softmax_relative").
    Mirrors signals.py: metric == 'softmax'.
    """
    scaled = logits / temperature
    # numerically stable softmax (subtract the row max, as in the official code)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    denom = exp.sum(axis=1)
    true_exp = exp[np.arange(len(labels)), labels]
    return true_exp / denom


def conf_logit_signal(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Confidence logit: log p_true - log (1 - p_true) computed from the softmax.

    This is the "log-logit-scaling" signal of LiRA (Carlini et al., 2022),
    reused by the official RMIA code for its LiRA baseline
    (signal name "loss_logit_rescaled").
    """
    predictions = logits - logits.max(axis=1, keepdims=True)
    predictions = np.exp(predictions)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)
    n = len(labels)
    y_true = predictions[np.arange(n), labels].copy()
    predictions[np.arange(n), labels] = 0.0
    y_wrong = predictions.sum(axis=1)
    return np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Standard cross-entropy loss (used by Attack-P / Attack-R baselines).

    The official code derives the loss from the confidence logit s as
    log((1 + exp(s)) / exp(s)) = softplus(-s), which is exactly the
    cross-entropy loss of the softmax model. We compute it directly.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(len(labels)), labels]

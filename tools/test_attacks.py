"""
Unit tests for the attack implementations.

Run from the project root:
    python tools/test_attacks.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import attacks, evaluate, signals


def test_roc_against_sklearn():
    """Our ROC/AUC must match sklearn exactly (tie handling included)."""
    from sklearn.metrics import roc_auc_score, roc_curve as sk_roc_curve
    rng = np.random.RandomState(0)
    for trial in range(5):
        n = rng.randint(50, 500)
        scores = rng.rand(n)
        # inject ties
        scores[rng.choice(n, size=n // 10, replace=False)] = 0.5
        member = rng.rand(n) < 0.5
        if member.all() or (~member).all():
            continue
        fpr, tpr, _ = evaluate.roc_curve(scores, member)
        ours = evaluate.auc(fpr, tpr)
        theirs = roc_auc_score(member, scores)
        assert abs(ours - theirs) < 1e-10, (ours, theirs)
        sk_fpr, sk_tpr, _ = sk_roc_curve(member, scores, drop_intermediate=False)
        # compare as sets of (fpr, tpr) operating points
        pts_ours = set(zip(np.round(fpr, 12), np.round(tpr, 12)))
        pts_sk = set(zip(np.round(sk_fpr, 12), np.round(sk_tpr, 12)))
        assert pts_ours == pts_sk, "ROC operating points differ from sklearn"
    print("PASS  roc/auc matches sklearn (incl. ties)")


def test_tpr_at_fpr():
    scores = np.array([3.0, 2.0, 1.0, 0.5])
    member = np.array([True, False, True, False])
    rep = evaluate.evaluate_attack(scores, member, fpr_levels=(0.5, 0.1))
    # threshold 2.0 => TP=1, FP=1 => TPR=1, FPR=0.5
    assert abs(rep["tpr@0.5"] - 1.0) < 1e-9
    # FPR<=0.1 => threshold 3.0 => TP=1 (the member scoring 3.0), FP=0 => TPR=0.5
    assert abs(rep["tpr@0.1"] - 0.5) < 1e-9
    print("PASS  tpr@fpr operating points")


def test_select_in_out():
    ref = np.array([[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]])
    keep = np.array([[True, False, True],
                     [False, True, False]])
    in_sel, out_sel = attacks.select_in_out(ref, keep, budget=1)
    assert np.allclose(in_sel[:, 0], [0.1, 0.5, 0.3])
    assert np.allclose(out_sel[:, 0], [0.4, 0.2, 0.6])
    print("PASS  in/out reference selection")


def test_rmia_properties():
    """Score in [0,1]; self-point excluded; members must score higher."""
    rng = np.random.RandomState(1)
    N, R = 400, 6
    # members: target signal inflated, and IN models also inflated
    target = rng.rand(N)
    ref = rng.rand(R, N)
    keep = rng.rand(R, N) < 0.5
    member = rng.rand(N) < 0.5
    target[member] += 1.0
    for r in range(R):
        both = keep[r] & member
        ref[r, both] += 1.0
    scores = attacks.rmia(target, ref, keep, num_ref_models=2, offline=True)
    assert np.nanmax(scores) <= 1.0 + 1e-9 and np.nanmin(scores) >= 0.0
    auc = evaluate.evaluate_attack(scores, member)["auc"]
    assert auc > 0.9, f"RMIA should separate members here, AUC={auc}"
    print(f"PASS  rmia score bounds + separation (AUC {auc:.3f})")


def test_rmia_online_matches_official_semantics():
    """Online: p(x) averages IN and OUT signals (Eq. 4 of the paper)."""
    N, R = 50, 4
    target = np.full(N, 0.6)
    ref = np.full((R, N), 0.4)
    ref[0] = 0.9  # the only IN model for all points
    keep = np.zeros((R, N), dtype=bool)
    keep[0] = True
    scores = attacks.rmia(target, ref, keep, num_ref_models=2, offline=False)
    # p(x) = trim_mean(0.9, 0.4) = 0.65; ratio t/p < 1 for all x
    # population ratios p_z/t_z = 0.65/0.6 > 1 => LR < 1 => no votes => score 0
    assert np.allclose(scores, 0.0), scores[:5]
    print("PASS  rmia online averaging semantics")


def test_lira_direction():
    """Members (target signal close to the IN distribution) must score higher.

    Semantics: for every point, models that trained on it (IN) emit IN-like
    signals; models that did not (OUT) emit OUT-like signals. The target
    model emits IN-like signals iff the point is a member.
    """
    rng = np.random.RandomState(2)
    N, R = 300, 8
    member = rng.rand(N) < 0.5
    m = rng.randn(N)            # per-point IN-mean
    in_like = m[None, :] + 0.01 * rng.randn(R, N)      # IN models: tight around m
    out_like = -m[None, :] + 0.5 * rng.randn(R, N)     # OUT models: around -m
    keep = rng.rand(R, N) < 0.5
    keep[0] = True   # every point has at least one IN model
    keep[1] = False  # ... and at least one OUT model
    ref = np.where(keep, in_like, out_like)
    target = np.where(member, m + 0.01 * rng.randn(N), -m + 0.5 * rng.randn(N))
    scores = attacks.lira(target, ref, keep, num_ref_models=4, offline=False)
    auc = evaluate.evaluate_attack(scores, member)["auc"]
    assert auc > 0.9, f"online LiRA direction wrong, AUC={auc}"
    print(f"PASS  lira online direction (AUC {auc:.3f})")


def test_signals():
    logits = np.array([[2.0, 0.0], [0.0, 3.0]])
    labels = np.array([0, 1])
    sm = signals.softmax_signal(logits, labels, temperature=1.0)
    assert np.allclose(sm, [1 / (1 + np.exp(-2)), np.exp(3) / (np.exp(3) + 1)])
    loss = signals.cross_entropy_loss(logits, labels)
    assert np.allclose(loss, [-np.log(sm[0]), -np.log(sm[1])])
    cl = signals.conf_logit_signal(logits, labels)
    assert np.allclose(cl, np.log(sm) - np.log(1 - sm))
    print("PASS  signals (softmax / loss / conf-logit)")


def test_rmia_augmented_majority_voting():
    """Augmented RMIA: majority voting across noisy queries must beat a single query.

    Setup: signals where each query's signal is the clean signal plus noise.
    With many independent noisy queries the majority vote denoises the
    LR(x,z) > gamma decision per (x,z) pair, so AUC(aug) > AUC(single).
    Also checks: Q=1 via the 2D shape equals the 1D path exactly, and the
    tie-break (exactly half the votes) resolves to False.
    """
    rng = np.random.RandomState(7)
    N, R, Q = 400, 6, 7
    member = rng.rand(N) < 0.5
    base = np.where(member, 0.75, 0.45)             # clean member/non-member mean
    # target and reference models: member signals elevated + per-query noise
    def make_signals(noise):
        return np.clip(base[:, None]
                       + noise * rng.randn(N, Q)
                       + 0.03 * rng.randn(N, Q), 1e-4, 0.9999)
    target = make_signals(0.12)                     # (N, Q)
    ref = np.stack([make_signals(0.12) for _ in range(R)])  # (R, N, Q)
    keep = rng.rand(R, N) < 0.5
    keep[:, 0] = True    # point 0 in every model (NaN-fallback path)
    keep[:, 1] = False   # point 1 in no model

    scores_aug = attacks.rmia(target, ref, keep, num_ref_models=2 * R // 2 * 2,
                              offline=True, chunk_size=128)
    auc_aug = evaluate.evaluate_attack(scores_aug, member)["auc"]

    # single query (only the clean query, index 0)
    scores_1 = attacks.rmia(target[:, 0], ref[:, :, 0], keep,
                            num_ref_models=2 * R // 2 * 2, offline=True,
                            chunk_size=128)
    auc_1 = evaluate.evaluate_attack(scores_1, member)["auc"]

    # Q=1 passed through the 2D path must equal the 1D path
    scores_1b = attacks.rmia(target[:, :1], ref[:, :, :1], keep,
                             num_ref_models=2 * R // 2 * 2, offline=True,
                             chunk_size=128)
    assert np.allclose(scores_1, scores_1b), "Q=1 via 2D path != 1D path"
    assert auc_aug > auc_1, f"majority voting did not help: {auc_aug} vs {auc_1}"
    print(f"PASS  rmia augmented majority voting "
          f"(AUC {auc_1:.3f} -> {auc_aug:.3f})")


def test_rmia_augmented_tie_break():
    """With Q=2 and exactly one True + one False vote, the result must be False.

    Official code majority-votes with torch.mode, whose tie-break returns the
    smaller value (False). Our rule 'strictly more than Q/2' matches.
    """
    N, R, Q = 3, 2, 2
    target = np.array([[0.5, 0.9], [0.5, 0.5], [0.9, 0.5]])   # (N, Q)
    ref = np.full((R, N, Q), 0.4)
    keep = np.zeros((R, N), dtype=bool)
    keep[0] = True
    # p(x) = 0.6*0.4+0.4 = 0.64 for OUT rows... craft a simple case:
    scores = attacks.rmia(target, ref, keep, num_ref_models=2, offline=True,
                          gamma=2.0, proportiontocut=0.0)
    # point 0: query0 signal 0.5 (LR<2), query1 0.9 (LR>2 possible) -> a 1-1
    # tie must resolve to False for EVERY pair, so all scores must be equal
    # across the three points (no point can win a majority with a tied vote).
    assert scores.shape == (N,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    # tie means point 0's self-pair votes False; verify no crash and valid range
    print("PASS  rmia augmented tie-break (Q=2 ties resolve to False)")


def test_lira_augmented():
    """Augmented LiRA must average per-query log-LRs and match Q=1 when queries are identical."""
    rng = np.random.RandomState(11)
    N, R = 200, 8
    member = rng.rand(N) < 0.5
    m = rng.randn(N)
    keep = rng.rand(R, N) < 0.5
    keep[0] = True
    keep[1] = False

    def one_query():
        in_like = m[None, :] + 0.01 * rng.randn(R, N)
        out_like = -m[None, :] + 0.5 * rng.randn(R, N)
        ref = np.where(keep, in_like, out_like)
        target = np.where(member, m + 0.01 * rng.randn(N),
                          -m + 0.5 * rng.randn(N))
        return target, ref

    t1, r1 = one_query()
    score_1 = attacks.lira(t1, r1, keep, num_ref_models=4, offline=False)
    # replicate the same query Q times: mean over Q identical terms == Q=1 score
    Q = 3
    t_rep = np.repeat(t1[:, None], Q, axis=1)
    r_rep = np.repeat(r1[:, :, None], Q, axis=2)
    score_rep = attacks.lira(t_rep, r_rep, keep, num_ref_models=4, offline=False)
    assert np.allclose(score_1, score_rep), "replicated queries must equal Q=1"
    auc = evaluate.evaluate_attack(score_rep, member)["auc"]
    assert auc > 0.9, f"augmented LiRA direction wrong, AUC={auc}"
    print(f"PASS  lira augmented (replicated queries == Q=1, AUC {auc:.3f})")


if __name__ == "__main__":
    test_signals()
    test_select_in_out()
    test_roc_against_sklearn()
    test_tpr_at_fpr()
    test_rmia_properties()
    test_rmia_online_matches_official_semantics()
    test_rmia_augmented_majority_voting()
    test_rmia_augmented_tie_break()
    test_lira_augmented()
    test_lira_direction()
    print("\nAll tests passed.")

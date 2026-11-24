# Demo guide: presenting the RMIA re-implementation

A compact script for a university demo of the paper
"Low-Cost High-Power Membership Inference Attacks" (Zarifzadeh, Liu, Shokri,
ICML 2024). Read the README first; this guide is the story you tell.

## 1. The problem (why this matters)

Machine learning models memorize parts of their training data. A **membership
inference attack** (MIA) recovers whether a specific record was in the
training set. For medical records, purchase histories or location data, mere
membership is already a privacy breach (e.g., "this person's record is in the
diabetes-model's training set"). MIAs are also the standard tool for
*auditing* the privacy of ML systems.

## 2. The scientific question in the paper

MIA is a hypothesis test:

- H_in: x was in the training set of theta
- H_out (null): x was not

The quality of an attack is its **TPR-FPR trade-off curve** (its power at every
error level), and the practically relevant regime is **low FPR**: when
auditing, you test thousands of records and a 1% false-alarm rate means
thousands of false accusations. Prior SOTA attacks (LiRA, Attack-R) need
*many* reference models (dozens to hundreds) to be powerful, and which one
wins changes per dataset. The paper asks: can we get a *cheap* attack that is
*at least as powerful everywhere*?

## 3. The RMIA idea (the part you must be able to explain)

Previous attacks test x against an *average* null hypothesis. RMIA instead
builds the null as a **composition of fine-grained worlds**: for every
population point z, consider the world where z replaced x in the training set.
Then run one pairwise likelihood-ratio test per z:

```
LR(x, z) = Pr(theta | x) / Pr(theta | z)
         = (Pr(x|theta)/Pr(x)) / (Pr(z|theta)/Pr(z))     [Bayes]
```

- `Pr(x|theta)`: the target model's softmax probability of x's true class.
- `Pr(x)`: estimated by *reference models* trained on other random halves of
  the population. Offline, only OUT models exist, so the paper scales up:
  `p(x) = (1+a)/2 * p_OUT(x) + (1-a)/2` (a=0.2).

The final score: **the fraction of population z that x "gamma-dominates"**
(Score = Pr_z[LR(x,z) > gamma], gamma = 2). A member wins against almost
everybody; a non-member beats only its peers. Calibration: with gamma = 1,
choosing the threshold beta gives FPR = 1 - beta (the paper's Figure 8) -
you can set your audit's false-positive budget exactly.

Key intuition to say out loud: *RMIA uses both reference models AND the
population; LiRA/Attack-R use only reference models, Attack-P only the
population. RMIA composes both signals, so it stays powerful when either is
scarce.*

## 4. What the demo shows

Run `python main.py --logdir exp/purchase100 --num-ref-models 1 2 4 8`:

1. **Low-cost regime (the headline)**: with 1 reference model, RMIA beats
   Attack-R and offline LiRA by a wide margin; offline LiRA is close to random
   guessing. Show `roc_1_ref_models.png`.
2. **Dominance across budgets**: increase the budget (2, 4, 8); RMIA stays on
   top at every FPR (paper Table 2 / Figure 1 behaviour).
3. **Low-FPR regime**: look at TPR@0.01%: RMIA's advantage is largest exactly
   where privacy auditing needs it.
4. **Online vs offline**: LiRA/RMIA online (IN+OUT models) is stronger but
   requires training models *per query* - point out the cost asymmetry the
   paper argues about (online attacks are "proof of concept").

## 5. Suggested live-demo script (15 minutes)

1. (2 min) Problem + MIA game slides (Section 1-2 above).
2. (3 min) The LR test on the whiteboard: Eq. 3 and Eq. 5, what Pr(x) means,
   why reference models estimate it, the offline approximation.
3. (2 min) Show the project layout: signals -> attacks -> evaluate; point out
   that each attack is ~40 lines of NumPy (`src/attacks.py`).
4. (5 min) Run the audit (pre-train the models beforehand! training 16 models
   takes ~30-60 min on CPU). Walk through the printed table row by row.
5. (3 min) Show the ROC figures; contrast 1-reference-model behaviour with
   8; conclude: fine-grained null modeling buys power at zero extra cost.

## 6. Questions you should be ready for

- *Why temperature 2.0?* Softens the softmax before taking the true-class
  probability; the official configs use it; it reduces saturation so signals
  carry more information.
- *Why gamma = 2 and not 1?* gamma controls how much better x must look
  relative to z; gamma=2 filters ambiguous comparisons and improves TPR at
  low FPR (paper App. C reports the gamma study).
- *Where does the training-set structure come from?* `keep.npy`: balanced 50/50
  partitioning guarantees every point is OUT of half the models, so every
  point has OUT reference signals available.
- *Is this a real privacy attack?* It is an *audit*: it measures leakage. In a
  real audit, the auditor trains reference models on their own similar data.
- *Difference from the official code?* See `validate/OFFICIAL_CROSSCHECK.md`:
  same math, same defaults; PyTorch training instead of JAX/objax, and small
  documented deviations (LiRA fixed-variance source, self-exclusion handling).

## 7. Limitations to state honestly

- Purchase-100 with 16 models is a *demo scale*: the paper's headline numbers
  use 256 models on CIFAR-10/100, CINIC-10 and ImageNet.
- TPR@0.01% on a 50k audit rests on a few points; report it but say so.
- The synthetic quickstart dataset is only a pipeline test, not a result.

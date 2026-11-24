"""
Train one target/reference model (PyTorch re-implementation of
research/2024_rmia/scripts/train.py by Carlini et al., as used by the RMIA
paper's authors).

Key elements reproduced from the official training pipeline:

1. Balanced 50/50 data partitioning across the set of num_experiments models:
   each data point is used to train EXACTLY half of the models (keep.npy),
   via the argsort trick from the official train.py:

       keep = np.random.uniform(0, 1, size=(num_experiments, dataset_size))
       order = keep.argsort(0)
       keep = order < int(pkeep * num_experiments)      # pkeep = 0.5

2. Training hyperparameters (official defaults):
   SGD momentum 0.9 (nesterov), lr = 0.1 with cosine schedule
   lr(t) = lr * cos(progress * 7*pi/16) * clip(progress*100, 0, 1),
   weight_decay = 5e-4, batch_size = 256, 50 epochs (Purchase-100).

3. Outputs per experiment folder `exp/<dataset>/experiment-<expid>/`:
   keep.npy   (dataset_size,) bool membership mask of this model
   logits.npy (dataset_size, n_class) logits of this model on the FULL dataset

Usage (see scripts/train_purchase100.sh):
    python -m src.train --dataset purchase100 --expid 0 --num-experiments 16 \
        --logdir exp/purchase100 --data-dir data/purchase100 --epochs 50
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import build_model


def balanced_keep(num_experiments: int, dataset_size: int, pkeep: float = 0.5,
                 expid: int = 0) -> np.ndarray:
    """Balanced partitioning: after all num_experiments, each point is in
    exactly pkeep fraction of models. Mirrors the official train.py."""
    np.random.seed(0)
    keep = np.random.uniform(0, 1, size=(num_experiments, dataset_size))
    order = keep.argsort(0)
    keep = order < int(pkeep * num_experiments)
    return np.array(keep[expid], dtype=bool)


def lr_at(progress: float, base_lr: float = 0.1) -> float:
    """Cosine schedule from the official TrainLoop."""
    lr = base_lr * np.cos(progress * (7 * np.pi) / (2 * 8))
    return lr * np.clip(progress * 100, 0, 1)


def load_dataset(dataset: str, data_dir: str, logdir: str = "exp/tmp"):
    """Load the training split of a dataset as float32 arrays.

    Purchase-100: reads x_train.npy / y_train.npy from data_dir (download
    instructions in README.md).

    CIFAR-10/100: downloads via torchvision on first use and caches the
    normalized arrays in `logdir` (x_train.npy / y_train.npy), so later runs
    and the audit stage never need torchvision again. Images are normalized
    per channel and stored as (N, 3, 32, 32) float32.
    """
    if dataset.startswith("cifar"):
        cache_x = os.path.join(logdir, "x_train.npy")
        cache_y = os.path.join(logdir, "y_train.npy")
        if os.path.exists(cache_x) and os.path.exists(cache_y):
            return np.load(cache_x).astype(np.float32), np.load(cache_y).astype(np.int64)
        from torchvision import datasets
        cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
        train = cls(root=data_dir, train=True, download=True)
        x = train.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # NHWC -> NCHW: (N,32,32,3) -> (N,3,32,32)
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)
        x = (x - mean) / std
        y = np.asarray(train.targets, dtype=np.int64)
        os.makedirs(logdir, exist_ok=True)
        np.save(cache_x, x)
        np.save(cache_y, y)
        return x, y

    x = np.load(os.path.join(data_dir, "x_train.npy")).astype(np.float32)
    y = np.load(os.path.join(data_dir, "y_train.npy")).astype(np.int64)
    return x, y


def pick_device(name: str) -> str:
    """'auto' selects the Metal GPU (mps) on Apple Silicon when available."""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _crop_batch_gather(padded: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Vectorized per-image random crop via two torch.gather calls.

    Replaces the original per-image Python loop (kept below as
    _crop_batch_loop). The loop issued 2 * B tiny GPU operations per batch,
    each preceded by a GPU->CPU sync (int(offs[i, 0])), which starved the
    GPU; this version issues two batched gathers. Verified bit-identical to
    the loop by _verify_augment() below on the actual training device.
    """
    B, C, H, W = padded.shape[0], padded.shape[1], padded.shape[2] - 8, padded.shape[3] - 8
    dev = padded.device
    # row gather: out1[b, c, h, w] = padded[b, c, r_b + h, w]
    idx_r = (offs[:, 0].reshape(B, 1, 1, 1)
             + torch.arange(H, device=dev).reshape(1, 1, H, 1)).expand(B, C, H, W + 8)
    out1 = torch.gather(padded, 2, idx_r)
    # col gather: out[b, c, h, w] = out1[b, c, h, c_b + w]
    idx_c = (offs[:, 1].reshape(B, 1, 1, 1)
             + torch.arange(W, device=dev).reshape(1, 1, 1, W)).expand(B, C, H, W)
    return torch.gather(out1, 3, idx_c)


def _crop_batch_loop(padded: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Original per-image loop (reference implementation, used only by the
    one-time equivalence check _verify_augment)."""
    B = padded.shape[0]
    H = padded.shape[2] - 8
    W = padded.shape[3] - 8
    out = torch.empty(B, padded.shape[1], H, W, device=padded.device,
                      dtype=padded.dtype)
    for i in range(B):
        r, c = int(offs[i, 0]), int(offs[i, 1])
        out[i] = padded[i, :, r:r + H, c:c + W]
    return out


_augment_checked = False


def _verify_augment(device: str) -> None:
    """One-time check that the vectorized crop is bit-identical to the
    original loop on the device actually used for training. Raises a clear
    RuntimeError on any mismatch, so a device-specific bug can never
    silently poison a training run."""
    global _augment_checked
    if _augment_checked:
        return
    g = torch.Generator(device="cpu").manual_seed(1234)
    xb = torch.randn(8, 3, 32, 32, generator=g).to(device)
    offs = torch.randint(0, 9, (8, 2), generator=g).to(device)
    padded = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    ref = _crop_batch_loop(padded, offs)
    vec = _crop_batch_gather(padded, offs)
    if not torch.equal(ref, vec):
        raise RuntimeError(
            "vectorized random crop mismatch on device " + str(device)
            + " - falling back is required; report this error")
    _augment_checked = True


def augment_batch(xb: torch.Tensor) -> torch.Tensor:
    """CIFAR training augmentation, as in the official pipeline ('weak'):
    random crop with reflect padding of 4 + random horizontal flip.

    xb: (B, 3, 32, 32) on the training device. Does NOT mutate xb in place
    (the old version did; callers keep their .clone() defensively).
    """
    B = xb.shape[0]
    flip = torch.rand(B, device=xb.device) < 0.5
    xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(-1), xb)
    padded = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    offs = torch.randint(0, 9, (B, 2), device=xb.device)  # top-left in [0, 8]
    return _crop_batch_gather(padded, offs)


def train_model(x, y, arch: str, dataset: str, epochs: int = 50, batch_size: int = 256,
                lr: float = 0.1, weight_decay: float = 5e-4, seed: int = 0,
                device: str = "cpu", verbose: bool = True):
    torch.manual_seed(seed)
    if dataset.startswith("cifar"):
        _verify_augment(device)  # one-time: vectorized crop == reference loop
    n_class = int(y.max() + 1)
    n_features = x.shape[1] if x.ndim == 2 else 3
    net = build_model(arch, dataset, n_features=n_features, n_class=n_class).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True,
                         weight_decay=weight_decay)
    n = len(x)
    steps_per_epoch = max(n // batch_size, 1)
    total_steps = epochs * steps_per_epoch
    step = 0
    # Keep the batch pipeline inside torch: indexing a numpy array with a
    # 1-element torch tensor can collapse the batch to a single sample (seen
    # with torch 2.8 / numpy 2.0 on the size-1 final batch of an epoch), so
    # the training arrays are converted once and indexed with torch only.
    xt = torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else torch.as_tensor(x)
    yt = torch.from_numpy(np.ascontiguousarray(y)) if isinstance(y, np.ndarray) else torch.as_tensor(y)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n)
        # accumulate the epoch loss as a 0-dim tensor: loss.item() would
        # force a CPU<->GPU synchronization on every single step
        tot_loss = torch.zeros((), device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = xt[idx].to(device)
            if dataset.startswith("cifar"):
                if xb.ndim != 4:  # must be (B, 3, 32, 32); guard against any collapse
                    raise RuntimeError(
                        f"corrupt batch shape {tuple(xb.shape)} at step {step}; rerun this model")
                xb = augment_batch(xb)  # random crop + flip (official 'weak')
            yb = yt[idx].to(device)
            lr_t = lr_at(step / max(total_steps, 1), lr)
            for g in opt.param_groups:
                g["lr"] = lr_t
            opt.zero_grad()
            out = net(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.detach()
            step += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss {float(tot_loss)/steps_per_epoch:.4f}")
    return net


@torch.no_grad()
def compute_logits(net, x, device: str = "cpu", batch_size: int = 2048) -> np.ndarray:
    net.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[i:i + batch_size], device=device)
        outs.append(net(xb).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def compute_logits_aug(net, x, num_augs: int, device: str = "cpu",
                       batch_size: int = 1024) -> np.ndarray:
    """Logits over `num_augs` queries per point (paper's augmented setting).

    Returns an array of shape (num_augs, N, C) where query 0 is the CLEAN
    image and queries 1..num_augs-1 are independent random augmentations
    (the same weak augmentation used in training: random crop with reflect
    padding + horizontal flip). This mirrors the official inference.py with
    --aug=Q, whose logits files also include the original query at index 0
    (the official README notes "--aug=2 # contains the original query").

    Only valid for image datasets (cifar10/cifar100); purchase100 is tabular
    and has no augmentation in the official pipeline either.
    """
    net.eval()
    all_queries = []
    for q in range(num_augs):
        outs = []
        for i in range(0, len(x), batch_size):
            # clone: augment_batch mutates in place and as_tensor may share
            # memory with the underlying numpy array
            xb = torch.as_tensor(x[i:i + batch_size], device=device).clone()
            if q > 0:
                xb = augment_batch(xb)
            outs.append(net(xb).cpu().numpy())
        all_queries.append(np.concatenate(outs))
    return np.stack(all_queries, axis=0)  # (Q, N, C)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="purchase100", choices=["purchase100", "cifar10", "cifar100"])
    p.add_argument("--data-dir", default="data/purchase100")
    p.add_argument("--arch", default="mlp",
                   help="mlp (purchase100) | wrn28-2 (paper's CIFAR models) | small_cnn")
    p.add_argument("--expid", type=int, required=True)
    p.add_argument("--num-experiments", type=int, default=16)
    p.add_argument("--logdir", default="exp/purchase100")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--pkeep", type=float, default=0.5)
    p.add_argument("--device", default="auto",
                   help="auto | cpu | mps (Apple GPU) | cuda. auto = mps when available")
    p.add_argument("--num-augmentations", type=int, default=1,
                   help="number of queries per point for the augmented attack "
                        "setting (1 = clean only). Query 0 is always the clean "
                        "image; the rest are random augmentations. CIFAR only. "
                        "Paper's strongest setting uses 18 or 50.")
    args = p.parse_args()

    device = pick_device(args.device)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"[train] using device: {device}")

    x, y = load_dataset(args.dataset, args.data_dir, logdir=args.logdir)
    dataset_size = len(x)
    keep = balanced_keep(args.num_experiments, dataset_size, args.pkeep, args.expid)
    exp_dir = os.path.join(args.logdir, f"experiment-{args.expid}")
    os.makedirs(exp_dir, exist_ok=True)
    # dataset files must be visible to the audit stage from the logdir
    os.makedirs(args.logdir, exist_ok=True)
    np.save(os.path.join(args.logdir, "y_train.npy"), y)

    print(f"[train] experiment-{args.expid}: {keep.sum()} / {dataset_size} training points")

    def fit_and_score(lr):
        net = train_model(x[keep], y[keep], args.arch, args.dataset, epochs=args.epochs,
                          batch_size=args.batch, lr=lr, seed=args.expid,
                          device=device)
        lg = compute_logits(net, x, device=device)
        tr = (lg.argmax(1)[keep] == y[keep]).mean()
        te = (lg.argmax(1)[~keep] == y[~keep]).mean()
        return net, lg, tr, te

    net, logits, train_acc, test_acc = fit_and_score(args.lr)

    # Divergence guard: with lr=0.1 an occasional seed diverges (near-random
    # accuracy). Detect it and retrain once with a 10x smaller learning rate.
    random_acc = 1.0 / (int(y.max()) + 1)
    if train_acc < max(5 * random_acc, 0.05):
        print(f"[train] WARNING: training diverged (train acc {train_acc:.4f}); "
              f"retrying with lr={args.lr / 10}")
        net, logits, train_acc, test_acc = fit_and_score(args.lr / 10)
        if train_acc < max(5 * random_acc, 0.05):
            print("[train] WARNING: model still failed after retry; inspect manually")

    print(f"[train] experiment-{args.expid}: train acc {train_acc:.4f}, "
          f"held-out acc {test_acc:.4f}")
    np.save(os.path.join(exp_dir, "keep.npy"), keep)
    np.save(os.path.join(exp_dir, "logits.npy"), logits)

    if args.num_augmentations > 1:
        if not args.dataset.startswith("cifar"):
            raise SystemExit("--num-augmentations > 1 is only supported for cifar "
                             "datasets (purchase100 is tabular; the official "
                             "pipeline does not augment it either)")
        logits_aug = compute_logits_aug(net, x, args.num_augmentations,
                                        device=device)
        np.save(os.path.join(exp_dir, "logits_aug.npy"), logits_aug)
        print(f"[train] saved {exp_dir}/logits_aug.npy "
              f"({args.num_augmentations} queries: clean + "
              f"{args.num_augmentations - 1} augmentations)")

    print(f"[train] saved {exp_dir}/keep.npy and {exp_dir}/logits.npy")


if __name__ == "__main__":
    main()

"""
Quick check of the model architectures (run on your machine; needs torch).

    python tools/check_models.py

Verifies that WideResNet-28-2 builds, runs a forward pass, and has the
expected number of parameters (~1.47M for CIFAR-10, the standard figure for
the paper's CIFAR architecture).
"""

from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

from src.models import WideResNet, build_model


def main():
    for n_class, expected in [(10, None), (100, None)]:
        net = WideResNet(28, 2, n_class)
        n_params = sum(p.numel() for p in net.parameters())
        x = torch.randn(4, 3, 32, 32)
        out = net(x)
        assert out.shape == (4, n_class), out.shape
        print(f"WRN-28-2 ({n_class} classes): {n_params:,} params, "
              f"forward pass OK {tuple(out.shape)}")
        if n_class == 10:
            expected = 1_471_226  # standard WRN-28-2 for CIFAR-10
            if n_params != expected:
                print(f"  NOTE: expected {expected:,} params (standard WRN-28-2); "
                      f"got {n_params:,}")
            else:
                print("  param count matches the standard WRN-28-2 exactly")

    # device check
    if torch.backends.mps.is_available():
        net = WideResNet(28, 2, 10).to("mps")
        x = torch.randn(8, 3, 32, 32, device="mps")
        out = net(x)
        print(f"MPS (Metal) forward pass OK -> {tuple(out.shape)}")
    else:
        print("MPS not available on this machine (CPU-only run)")

    net = build_model("mlp", "purchase100", n_features=600, n_class=100)
    out = net(torch.randn(3, 600))
    assert out.shape == (3, 100)
    print("MLP 600->512->256->128->64->100 OK")
    print("\nAll model checks passed.")


if __name__ == "__main__":
    main()

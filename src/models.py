"""
Model architectures (PyTorch).

Purchase-100: MLP with hidden units [512, 256, 128, 64] and Glorot-uniform
initialization, exactly as in the official RMIA training scripts
(research/2024_rmia/scripts/mlp.py, layer_units=[512, 256, 128, 64]).

CIFAR-10/100: WideResNet-28-2, the architecture used by the paper (and by
Carlini et al.'s mi_lira_2021 training code that the authors reused):
depth 28, widen factor 2, 3 groups of (28-4)/6 = 4 basic blocks, widths
16/32/64 scaled by 2, BN-ReLU-Conv pre-activation blocks.

Apple Silicon: all models run on the Metal GPU via the MPS backend
(pass --device mps, or --device auto to use MPS when available).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(n_in: int, n_class: int, layer_units=(512, 256, 128, 64)) -> nn.Module:
    layers = []
    prev = n_in
    for u in layer_units:
        layers += [nn.Linear(prev, u), nn.ReLU()]
        prev = u
    layers.append(nn.Linear(prev, n_class))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Glorot uniform, as official
            nn.init.zeros_(m.bias)
    return net


def small_cnn(n_class: int = 10) -> nn.Module:
    """Small CIFAR-10 CNN (VGG-style), CPU-friendly quick-test model."""
    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2)]
    layers = block(3, 32) + block(32, 64) + block(64, 128)
    net = nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
        nn.Linear(256, n_class),
    )
    return net


class WideBasicBlock(nn.Module):
    """Pre-activation basic block of WideResNet (Zagoruyko & Komodakis, 2016)."""

    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1,
                               padding=1, bias=False)
        self.equal_in_out = in_planes == out_planes
        if not self.equal_in_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        if not self.equal_in_out:
            z = self.relu1(self.bn1(x))
            identity = self.shortcut(z)
        else:
            identity = x
            z = self.relu1(self.bn1(x))
        y = self.conv1(z)
        y = self.conv2(self.relu2(self.bn2(y)))
        return y + identity


class WideResNet(nn.Module):
    """WideResNet-{depth}-{widen}: depth 28, widen 2 for the paper's CIFAR models."""

    def __init__(self, depth: int = 28, widen: int = 2, n_class: int = 10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv0 = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
        layers = []
        for group, (cin, cout) in enumerate(zip(widths[:-1], widths[1:])):
            stride = 1 if group == 0 else 2
            for b in range(n):
                layers.append(WideBasicBlock(cin if b == 0 else cout, cout,
                                             stride if b == 0 else 1))
        self.body = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(widths[-1])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(widths[-1], n_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                         nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.conv0(x)
        y = self.body(y)
        y = self.relu(self.bn(y))
        y = torch.nn.functional.adaptive_avg_pool2d(y, 1).flatten(1)
        return self.fc(y)


def build_model(arch: str, dataset: str, n_features: int = 600, n_class: int = 100):
    if arch == "mlp":
        return mlp(n_features, n_class)
    elif arch == "small_cnn":
        return small_cnn(n_class)
    elif arch == "wrn28-2":
        return WideResNet(28, 2, n_class)
    elif arch.startswith("wrn28-"):
        return WideResNet(28, int(arch.split("-")[1]), n_class)
    raise ValueError(f"unknown arch {arch}")

"""
exp_shallowconvnet.py
=====================

Faz 5a: subject-specific ShallowConvNet (Schirrmeister et al. 2017).
FBCSP'nin DL karsiligi:

    Conv2D(40, kernel=(1, 25))     # temporal — kisa filtre bandi ogrenir
    Conv2D(40, kernel=(C, 1))      # spatial  — bilesik CSP-benzeri
    BatchNorm
    square                         # x**2
    AvgPool2D(kernel=(1, 75), stride=(1, 15))   # frequency power tahmini
    safe_log                       # log(clamp(x, min=eps))
    Dropout
    Flatten -> Linear(n_classes)

Girdi: (B, 1, C=22, T=500), bandpass 4-38 Hz, uV olcek.
Dis CV: 5-fold StratifiedKFold(random_state=42) — klasik fazlarla AYNI.
OOF -> results/oof/shallowconvnet/sub_NN.npz.

A0XE'ye DOKUNULMAZ.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _dl_common import (  # noqa: E402
    DEVICE,
    RANDOM_STATE,
    SubjectRunResult,
    TrainConfig,
    report_and_save,
    run_dl_subject,
)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #


def _safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


class ShallowConvNet(nn.Module):
    """ShallowConvNet (FBCSP DL karsiligi)."""

    def __init__(
        self,
        n_classes: int = 4,
        n_chans: int = 22,
        n_samples: int = 500,
        n_filters_time: int = 40,
        n_filters_spat: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.temporal = nn.Conv2d(
            1, n_filters_time, kernel_size=(1, filter_time_length), bias=True,
        )
        # Spatial conv — non-linearity YOK (paper'da kasitli)
        self.spatial = nn.Conv2d(
            n_filters_time, n_filters_spat, kernel_size=(n_chans, 1), bias=False,
        )
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time_length),
                                 stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.temporal(x)        # (B, 40, C, T-24)
        x = self.spatial(x)         # (B, 40, 1, T-24)
        x = self.bn(x)
        x = x * x                   # square
        x = self.pool(x)
        x = _safe_log(x)
        x = self.drop(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model() -> nn.Module:
    return ShallowConvNet()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="ShallowConvNet subject-specific CV.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--output", type=str, default="exp_shallowconvnet.csv")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        augment=not args.no_aug,
        verbose=False,
    )

    rows: List[SubjectRunResult] = []
    t_total = time.time()
    for sid in args.subjects:
        res = run_dl_subject(
            subject_id=sid,
            model_factory=build_model,
            cfg=cfg,
            oof_subdir="shallowconvnet",
            outer_splits=args.outer_splits,
            verbose=True,
        )
        rows.append(res)
    total_elapsed = time.time() - t_total

    report_and_save(
        rows=rows,
        output_name=args.output,
        total_elapsed=total_elapsed,
        title="ShallowConvNet (subject-specific, 5-fold CV)",
    )


if __name__ == "__main__":
    main()

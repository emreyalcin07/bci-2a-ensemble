"""
exp_eegnet.py
=============

Faz 5a: subject-specific EEGNet-8,2.

Mimari (Lawhence et al. 2018):
    Block 1
        Conv2D(F1=8, kernel=(1, 64)) [temporal]  -> BN
        DepthwiseConv2D(kernel=(C, 1), depth_multiplier=D=2) [spatial]
        BN -> ELU -> AvgPool(1,4) -> Dropout
    Block 2
        SeparableConv2D = DepthwiseConv2D(kernel=(1, 16)) + PointwiseConv2D(F2=16)
        BN -> ELU -> AvgPool(1,8) -> Dropout
    Flatten -> Linear(n_classes)

Girdi: (B, 1, C=22, T=500), bandpass 4-38 Hz, uV olcek.
4 sinif softmax. CrossEntropyLoss. AdamW lr=1e-3 wd=1e-2.

Dis CV: 5-fold StratifiedKFold(random_state=42) — klasik fazlarla AYNI.
Iç train/val (%80/20) early stopping. Augmentation: time shift + Gaussian noise.

OOF -> results/oof/eegnet/sub_NN.npz (ensemble icin).
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
import torch.nn.functional as F

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


class EEGNet(nn.Module):
    """EEGNet-8,2 (F1=8, D=2, F2=16). 4-sinif motor imagery icin standart."""

    def __init__(
        self,
        n_classes: int = 4,
        n_chans: int = 22,
        n_samples: int = 500,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(
            1, F1, kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(
            F1, F1 * D, kernel_size=(n_chans, 1),
            groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # Block 2 — Separable conv = depthwise + pointwise
        self.sep_depth = nn.Conv2d(
            F1 * D, F1 * D, kernel_size=(1, 16),
            padding=(0, 8), groups=F1 * D, bias=False,
        )
        self.sep_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)

        # Classifier — lazy linear: input shape conv-arithmetic'e bagli
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model() -> nn.Module:
    return EEGNet()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="EEGNet-8,2 nested-like CV.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no-aug", action="store_true", help="augmentation kapali")
    parser.add_argument("--output", type=str, default="exp_eegnet.csv")
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
            oof_subdir="eegnet",
            outer_splits=args.outer_splits,
            verbose=True,
        )
        rows.append(res)
    total_elapsed = time.time() - t_total

    report_and_save(
        rows=rows,
        output_name=args.output,
        total_elapsed=total_elapsed,
        title="EEGNet-8,2 (subject-specific, 5-fold CV)",
    )


if __name__ == "__main__":
    main()

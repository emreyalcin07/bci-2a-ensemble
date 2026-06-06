"""
exp_transfer_eegnet.py
======================

Faz 5b: cross-subject transfer (LOSO pretrain + finetune) — EEGNet-8,2.

Protokol (hedef denek S icin):
    1. Pretrain: S DISINDAKI 8 deneǧin TÜM A0XT verisi (~2304 trial)
       - leakage yok: S'in hiçbir verisi pretrain havuzunda degil
    2. Her dis 5-fold icin:
       - Pretrain'li modeli al, S'in dis-train fold'unda finetune et (düşük lr)
       - Early stop S'in iç-val fold'unda
       - Dis-test fold'unu tahmin et
    3. OOF kaydet -> results/oof/eegnet_transfer/sub_NN.npz

Dis CV: StratifiedKFold(5, shuffle=True, random_state=42) — klasik fazlarla AYNI,
ensemble hizali olacak.

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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _dl_common import (  # noqa: E402
    DEVICE,
    SubjectRunResult,
    TransferConfig,
    report_and_save,
    run_transfer_subject,
)
from exp_eegnet import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="EEGNet cross-subject transfer.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--pre-epochs", type=int, default=200)
    parser.add_argument("--ft-epochs", type=int, default=100)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--ft-patience", type=int, default=15)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--output", type=str, default="exp_transfer_eegnet.csv")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    cfg = TransferConfig(
        pre_epochs=args.pre_epochs,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        ft_patience=args.ft_patience,
        augment=not args.no_aug,
        verbose=False,
    )

    rows: List[SubjectRunResult] = []
    t_total = time.time()
    for sid in args.subjects:
        res = run_transfer_subject(
            subject_id=sid,
            model_factory=build_model,
            cfg=cfg,
            oof_subdir="eegnet_transfer",
            outer_splits=args.outer_splits,
            verbose=True,
        )
        rows.append(res)
    total_elapsed = time.time() - t_total

    report_and_save(
        rows=rows,
        output_name=args.output,
        total_elapsed=total_elapsed,
        title="EEGNet — LOSO pretrain + finetune (5-fold CV)",
    )


if __name__ == "__main__":
    main()

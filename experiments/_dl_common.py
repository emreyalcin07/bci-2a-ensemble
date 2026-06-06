"""
_dl_common.py
=============

Faz 5a deep learning baselineslari icin paylasilan altyapi:
    - bandpass 4-38 Hz (DL temporal filtreleri kendi ogrendigi icin genis bant)
    - EEGDataset: numpy -> tensor + augmentation (sadece train modunda)
    - train_one_fold: train/val split (early stopping) + son modelle test predict_proba
    - run_dl_subject: 5-fold outer CV (StratifiedKFold(5, shuffle=True, random_state=42)
                     — klasik fazlarla AYNI, ensemble hizalamasi icin) + OOF kaydet
    - report_and_save (yeniden ihrac, klasik fazlarla ayni format)

Leakage onlemleri:
    - Dis CV fold yapisi klasik fazlarla bit-bit ayni (random_state=42).
    - Dis-train icindeki iç train/val ayrimi yine stratified ve fold'a bagimli.
    - Augmentation YALNIZCA inner-train DataLoader'inda; val ve outer-test asla.
    - Per-fold z-score normalizasyon (dis-train uzerinden hesaplanir, dis-test'e
      uygulanir). Veya basitce uV olcege (1e6 carpan) — varsayilan: uV olcek.
    - A0XE bu modul tarafindan yuklenmez.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_loader import load_all_subjects, load_subject  # noqa: E402
from _fbcsp_common import SubjectRunResult, report_and_save  # noqa: E402

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OOF_ROOT = PROJECT_ROOT / "results" / "oof"


# --------------------------------------------------------------------------- #
# Reproducibility helpers                                                     #
# --------------------------------------------------------------------------- #


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Tum stochastic kanallari ayni tohumla baglar (tam determinizm garanti yok
    ama runlar arasinda buyuk varyans engellenir)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Bandpass (genis bant: DL temporal filtreleri kendi ogrenir)                 #
# --------------------------------------------------------------------------- #


def bandpass_wide(
    X: np.ndarray,
    sfreq: float,
    l_freq: float = 4.0,
    h_freq: float = 38.0,
    order: int = 4,
) -> np.ndarray:
    """4-38 Hz Butterworth (filtfilt, sifir faz)."""
    nyq = 0.5 * sfreq
    b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="band")
    Y = filtfilt(b, a, X, axis=-1)
    return Y.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Dataset (augmentation only during training)                                  #
# --------------------------------------------------------------------------- #


class EEGDataset(Dataset):
    """numpy (N, C, T) -> tensor (1, C, T) + label.

    Augmentation (sadece augment=True):
        - time_shift: rastgele -shift_max..+shift_max kayma (np.roll)
        - gaussian noise: x += N(0, sigma) where sigma = noise_std * channel_std

    Test/val icin augment=False kullan.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        shift_max: int = 12,        # ~50 ms at 250 Hz
        noise_std: float = 0.1,     # relative to per-channel std
    ):
        assert X.ndim == 3, f"X must be (N, C, T), got {X.shape}"
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment
        self.shift_max = shift_max
        self.noise_std = noise_std
        # Per-channel std (sample-bazli sabit) noise kalibrasyonu icin
        self._ch_std = X.std(axis=(0, 2), keepdims=True) + 1e-12  # (1, C, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()  # (C, T)
        if self.augment:
            # time shift
            if self.shift_max > 0:
                shift = np.random.randint(-self.shift_max, self.shift_max + 1)
                if shift != 0:
                    x = np.roll(x, shift, axis=-1)
            # gaussian noise (per-channel scaled)
            if self.noise_std > 0:
                std = self._ch_std[0]  # (C, 1)
                x = x + np.random.randn(*x.shape).astype(np.float32) * (std * self.noise_std)
        # add singleton "input channel" dim -> (1, C, T)
        x = x[np.newaxis, :, :]
        return torch.from_numpy(x), int(self.y[idx])


# --------------------------------------------------------------------------- #
# Train one fold (with early stopping)                                        #
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    early_stopping_patience: int = 30
    val_fraction: float = 0.2
    augment: bool = True
    shift_max: int = 12
    noise_std: float = 0.1
    verbose: bool = False


def _stratified_inner_split(
    y: np.ndarray, val_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Single stratified split icin StratifiedKFold(1/val_fraction)'in 1. fold'unu kullan."""
    n_splits = max(int(round(1.0 / val_fraction)), 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(skf.split(np.zeros(len(y)), y))
    return tr_idx, va_idx


def train_one_fold(
    model_factory: Callable[[], nn.Module],
    X_tr_outer: np.ndarray,
    y_tr_outer: np.ndarray,
    X_te_outer: np.ndarray,
    cfg: TrainConfig,
    inner_seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tek dis-fold icin egit + dis-test predict_proba dondur.

    Inner train/val split early stopping icin kullanilir; outer test ASLA gorulmez.

    Returns
    -------
    y_proba_test : ndarray (n_test, n_classes)
    info : dict (best_epoch, best_val_loss, n_epochs_trained)
    """
    set_seed(inner_seed)

    # Inner split (stratified)
    tr_idx, va_idx = _stratified_inner_split(y_tr_outer, cfg.val_fraction, inner_seed)
    X_tr, y_tr = X_tr_outer[tr_idx], y_tr_outer[tr_idx]
    X_va, y_va = X_tr_outer[va_idx], y_tr_outer[va_idx]

    # Datasets
    ds_tr = EEGDataset(X_tr, y_tr, augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(X_va, y_va, augment=False)
    ds_te = EEGDataset(X_te_outer, np.zeros(len(X_te_outer), dtype=np.int64), augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)

    # Model + optimizer
    model = model_factory().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    patience = 0

    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

        # ---- val ----
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_loss_sum / max(val_n, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                break

        if cfg.verbose and (epoch % 10 == 0 or epoch < 5):
            print(f"  epoch {epoch:3d}  val_loss={val_loss:.4f}  best@{best_epoch}={best_val_loss:.4f}")

    # ---- restore best + predict test ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probas = []
    with torch.no_grad():
        for xb, _ in dl_te:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probas.append(p)
    y_proba = np.concatenate(probas, axis=0)

    return y_proba, {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "n_epochs_trained": epoch + 1,
    }


# --------------------------------------------------------------------------- #
# Outer CV runner                                                             #
# --------------------------------------------------------------------------- #


def run_dl_subject(
    subject_id: int,
    model_factory: Callable[[], nn.Module],
    cfg: TrainConfig,
    oof_subdir: str,
    outer_splits: int = 5,
    verbose: bool = True,
) -> SubjectRunResult:
    """Tek denek icin dis 5-fold CV + OOF kaydet."""
    t0 = time.time()
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T icin etiket yok.")

    # Wide bandpass + uV olcek (DL icin makul amplitude)
    X = bandpass_wide(sd.X, sd.sfreq) * 1e6  # (N, C, T)
    y = sd.y

    if verbose:
        print(
            f"[A{subject_id:02d}T] X={X.shape}, y={y.shape}, device={DEVICE}",
            flush=True,
        )

    # AYNI fold yapisi (klasik fazlarla bit-bit ayni)
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    n = len(y)
    n_classes = int(np.max(y) + 1)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = np.zeros((n, n_classes), dtype=np.float32)
    fold_kappas: List[float] = []
    fold_info: List[Dict[str, Any]] = []

    for fi, (tr, te) in enumerate(skf.split(X, y)):
        # inner_seed'i fold'a goz: deterministik ama her fold'da farkli inner split
        inner_seed = RANDOM_STATE + fi
        y_proba_te, info = train_one_fold(
            model_factory=model_factory,
            X_tr_outer=X[tr], y_tr_outer=y[tr],
            X_te_outer=X[te],
            cfg=cfg,
            inner_seed=inner_seed,
        )
        oof_proba[te] = y_proba_te
        pred = y_proba_te.argmax(axis=1)
        oof_pred[te] = pred
        k = cohen_kappa_score(y[te], pred)
        fold_kappas.append(k)
        fold_info.append({"fold": fi, **info, "fold_kappa": k})
        if verbose:
            print(
                f"  fold {fi}: kappa={k:.4f}  best_epoch={info['best_epoch']}  "
                f"epochs_trained={info['n_epochs_trained']}",
                flush=True,
            )

    kappa = cohen_kappa_score(y, oof_pred)
    acc = accuracy_score(y, oof_pred)
    mf1 = f1_score(y, oof_pred, average="macro")

    # OOF kaydet (klasik base'lerle ayni format)
    oof_dir = OOF_ROOT / oof_subdir
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        oof_dir / f"sub_{subject_id:02d}.npz",
        y_true=y,
        y_pred=oof_pred,
        y_proba=oof_proba,
    )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"[A{subject_id:02d}T] kappa={kappa:.4f}  acc={acc:.4f}  "
            f"macro_f1={mf1:.4f}  fold_kappa_std={np.std(fold_kappas):.4f}  "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return SubjectRunResult(
        subject_id=subject_id,
        kappa=kappa,
        accuracy=acc,
        macro_f1=mf1,
        kappa_std=float(np.std(fold_kappas)),
        n_trials=n,
        elapsed_sec=elapsed,
        best_params_per_fold=fold_info,
    )


# --------------------------------------------------------------------------- #
# Cross-subject transfer (Faz 5b)                                             #
# --------------------------------------------------------------------------- #


@dataclass
class TransferConfig:
    """Pretrain + finetune ayri ayri tune edilebilen konfigurasyon."""
    # Pretrain (8 denek pool'unda, hedef denek HARIC)
    pre_epochs: int = 200
    pre_batch_size: int = 64
    pre_lr: float = 1e-3
    pre_weight_decay: float = 1e-2
    pre_patience: int = 30
    pre_val_fraction: float = 0.15  # 2304 trial'in %15'i ~345, anlamli val

    # Finetune (hedef denek dis-train'inde, sonra dis-test predict)
    ft_epochs: int = 100
    ft_batch_size: int = 64
    ft_lr: float = 1e-4               # düşük lr — pretrain'i çok bozma
    ft_weight_decay: float = 1e-2
    ft_patience: int = 15
    ft_val_fraction: float = 0.2

    # Augmentation (her iki asamada da train-only)
    augment: bool = True
    shift_max: int = 12
    noise_std: float = 0.1
    verbose: bool = False


def _load_pretrain_pool(
    target_subject_id: int,
    all_subjects: Sequence[int] = tuple(range(1, 10)),
) -> Tuple[np.ndarray, np.ndarray]:
    """LOSO havuzu: target_subject_id HARIC tüm deneklerin A0XT verisini birlestir.

    Returns
    -------
    X : (N_pool, C, T), float32, uV olcek
    y : (N_pool,), int64, 0..3
    """
    pool = [s for s in all_subjects if s != target_subject_id]
    per_subj = load_all_subjects(session="T", subjects=pool, concatenate=False)
    Xs, ys = [], []
    for sid, sd in per_subj.items():
        if sd.y is None:
            raise RuntimeError(f"A{sid:02d}T etiketi yok.")
        # Wide bandpass + uV (Faz 5a ile ayni on-isleme)
        Xs.append(bandpass_wide(sd.X, sd.sfreq) * 1e6)
        ys.append(sd.y)
    X = np.concatenate(Xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.int64, copy=False)
    return X, y


def _pretrain_model(
    model_factory: Callable[[], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    cfg: TransferConfig,
    seed: int = RANDOM_STATE,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """LOSO pool uzerinde pretrain. Best (val_loss) state dict dondurur.

    Pretrain val split: stratified, hedef denegi GORMEZ (zaten havuzda yok).
    Burada val sadece pretrain early-stop'u icin; hedef denegin hicbir verisi
    kullanilmadigi icin leakage yok.
    """
    set_seed(seed)
    tr_idx, va_idx = _stratified_inner_split(y, cfg.pre_val_fraction, seed)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    ds_tr = EEGDataset(X_tr, y_tr, augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(X_va, y_va, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.pre_batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)

    model = model_factory().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.pre_lr,
                              weight_decay=cfg.pre_weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    patience = 0

    for epoch in range(cfg.pre_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                loss = criterion(model(xb), yb)
                vs += loss.item() * xb.size(0)
                vn += xb.size(0)
        val_loss = vs / max(vn, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.pre_patience:
                break

    info = {
        "pretrain_best_epoch": best_epoch,
        "pretrain_best_val_loss": float(best_val_loss),
        "pretrain_epochs_trained": epoch + 1,
        "pretrain_n_samples": len(y),
    }
    return best_state, info


def _finetune_and_predict(
    pretrained_state: Dict[str, torch.Tensor],
    model_factory: Callable[[], nn.Module],
    X_tr_outer: np.ndarray,
    y_tr_outer: np.ndarray,
    X_te_outer: np.ndarray,
    cfg: TransferConfig,
    inner_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hedef denegin dis-train fold'unda fine-tune et, dis-test predict_proba dondur."""
    set_seed(inner_seed)

    tr_idx, va_idx = _stratified_inner_split(y_tr_outer, cfg.ft_val_fraction, inner_seed)
    X_tr, y_tr = X_tr_outer[tr_idx], y_tr_outer[tr_idx]
    X_va, y_va = X_tr_outer[va_idx], y_tr_outer[va_idx]

    ds_tr = EEGDataset(X_tr, y_tr, augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(X_va, y_va, augment=False)
    ds_te = EEGDataset(X_te_outer, np.zeros(len(X_te_outer), dtype=np.int64), augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.ft_batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)

    # Modeli yeniden olustur ve pretrain state'i yukle
    model = model_factory().to(DEVICE)
    # LazyLinear icin dummy forward (parametre olusumu)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, X_tr.shape[1], X_tr.shape[2], device=DEVICE)
        _ = model(dummy)
    model.load_state_dict(pretrained_state, strict=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.ft_lr,
                              weight_decay=cfg.ft_weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    patience = 0

    for epoch in range(cfg.ft_epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                loss = criterion(model(xb), yb)
                vs += loss.item() * xb.size(0)
                vn += xb.size(0)
        val_loss = vs / max(vn, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.ft_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probas = []
    with torch.no_grad():
        for xb, _ in dl_te:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
    y_proba = np.concatenate(probas, axis=0)

    info = {
        "ft_best_epoch": best_epoch,
        "ft_best_val_loss": float(best_val_loss),
        "ft_epochs_trained": epoch + 1,
    }
    return y_proba, info


def run_transfer_subject(
    subject_id: int,
    model_factory: Callable[[], nn.Module],
    cfg: TransferConfig,
    oof_subdir: str,
    outer_splits: int = 5,
    verbose: bool = True,
) -> SubjectRunResult:
    """LOSO pretrain (1 kere) + 5 dis-fold finetune/predict. OOF kaydet."""
    t0 = time.time()

    # 1) Hedef denegi yukle
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T etiketi yok.")
    X_target = bandpass_wide(sd.X, sd.sfreq) * 1e6
    y_target = sd.y

    if verbose:
        print(f"[A{subject_id:02d}T] target X={X_target.shape}, device={DEVICE}", flush=True)

    # 2) LOSO pool (hedef denek HARIC)
    t_pool = time.time()
    X_pool, y_pool = _load_pretrain_pool(subject_id)
    if verbose:
        print(
            f"[A{subject_id:02d}T] pretrain pool X={X_pool.shape}, y dist="
            f"{dict(zip(*np.unique(y_pool, return_counts=True)))}, load={time.time()-t_pool:.1f}s",
            flush=True,
        )

    # 3) Pretrain (1 kere, tum dis fold'lar bu state'i kullanir)
    t_pre = time.time()
    pretrained_state, pre_info = _pretrain_model(model_factory, X_pool, y_pool, cfg)
    if verbose:
        print(
            f"[A{subject_id:02d}T] pretrain best@{pre_info['pretrain_best_epoch']}  "
            f"val_loss={pre_info['pretrain_best_val_loss']:.4f}  "
            f"epochs={pre_info['pretrain_epochs_trained']}  elapsed={time.time()-t_pre:.1f}s",
            flush=True,
        )

    # 4) Dis 5-fold: her fold'da pretrain'den finetune
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    n = len(y_target)
    n_classes = int(np.max(y_target) + 1)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = np.zeros((n, n_classes), dtype=np.float32)
    fold_kappas: List[float] = []
    fold_info: List[Dict[str, Any]] = []

    for fi, (tr, te) in enumerate(skf.split(X_target, y_target)):
        inner_seed = RANDOM_STATE + fi
        y_proba_te, info = _finetune_and_predict(
            pretrained_state=pretrained_state,
            model_factory=model_factory,
            X_tr_outer=X_target[tr], y_tr_outer=y_target[tr],
            X_te_outer=X_target[te],
            cfg=cfg,
            inner_seed=inner_seed,
        )
        oof_proba[te] = y_proba_te
        pred = y_proba_te.argmax(axis=1)
        oof_pred[te] = pred
        k = cohen_kappa_score(y_target[te], pred)
        fold_kappas.append(k)
        fold_info.append({"fold": fi, **info, "fold_kappa": k})
        if verbose:
            print(
                f"  fold {fi}: kappa={k:.4f}  ft_best@{info['ft_best_epoch']}  "
                f"ft_epochs={info['ft_epochs_trained']}",
                flush=True,
            )

    kappa = cohen_kappa_score(y_target, oof_pred)
    acc = accuracy_score(y_target, oof_pred)
    mf1 = f1_score(y_target, oof_pred, average="macro")

    # OOF kaydet
    oof_dir = OOF_ROOT / oof_subdir
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        oof_dir / f"sub_{subject_id:02d}.npz",
        y_true=y_target,
        y_pred=oof_pred,
        y_proba=oof_proba,
    )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"[A{subject_id:02d}T] TRANSFER kappa={kappa:.4f}  acc={acc:.4f}  "
            f"macro_f1={mf1:.4f}  fold_std={np.std(fold_kappas):.4f}  "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    return SubjectRunResult(
        subject_id=subject_id,
        kappa=kappa,
        accuracy=acc,
        macro_f1=mf1,
        kappa_std=float(np.std(fold_kappas)),
        n_trials=n,
        elapsed_sec=elapsed,
        best_params_per_fold=[pre_info] + fold_info,
    )


__all__ = [
    "DEVICE",
    "RANDOM_STATE",
    "TrainConfig",
    "TransferConfig",
    "EEGDataset",
    "SubjectRunResult",
    "bandpass_wide",
    "report_and_save",
    "run_dl_subject",
    "run_transfer_subject",
    "set_seed",
    "train_one_fold",
]

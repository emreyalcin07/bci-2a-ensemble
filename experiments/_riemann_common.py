"""
_riemann_common.py
==================

Faz 3 Riemannian pipeline'ları için paylaşılan altyapı.

- bandpass_single: tek geniş bant filtre (8-30 Hz default).
                   Veri-bağımsız işlem; Pipeline DIŞINDA bir kez uygulanır.
- run_subject: SubjectData yükle, filtrele, nested CV çalıştır.
- report_and_save / SubjectRunResult: _fbcsp_common'dan yeniden kullanılır.

Leakage notu: pyriemann'ın Covariances, TangentSpace, MDM step'leri sklearn
uyumludur ve yalnızca fit'te train verisini görür. Outer/inner CV içindeki
hiçbir adım test fold'una bakmaz.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
# saga convergence uyarilari
warnings.filterwarnings(
    "ignore",
    message=".*ConvergenceWarning.*",
    module="sklearn",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_loader import load_subject  # noqa: E402
from evaluation import run_nested_cv  # noqa: E402
from _fbcsp_common import SubjectRunResult, report_and_save  # noqa: E402

import mne  # noqa: E402
mne.set_log_level("ERROR")

RANDOM_STATE = 42
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Tek geniş bant filtre                                                       #
# --------------------------------------------------------------------------- #


def bandpass_single(
    X: np.ndarray,
    sfreq: float,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """8-30 Hz Butterworth bant geçiren (filtfilt, sıfır faz).

    Riemannian yöntemler genellikle tek geniş bantta çalışır; bant başına
    kovaryans hesaplamak yerine geniş bant kovaryansı tercih edilir.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    sfreq : float
    l_freq, h_freq : float
    order : int

    Returns
    -------
    ndarray, X ile aynı şekil, float32.
    """
    nyq = 0.5 * sfreq
    b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="band")
    Y = filtfilt(b, a, X, axis=-1)
    return Y.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Multi-band (opsiyonel — exp_riemann_multiband_ts için)                       #
# --------------------------------------------------------------------------- #


def bandpass_multi(
    X: np.ndarray,
    sfreq: float,
    bands: Sequence[Tuple[float, float]],
    order: int = 4,
) -> np.ndarray:
    """Her bant için ayrı filtrelenmiş kopya: (n_trials, n_bands, n_ch, n_samp)."""
    out = np.empty((X.shape[0], len(bands), X.shape[1], X.shape[2]), dtype=np.float32)
    nyq = 0.5 * sfreq
    for bi, (l, h) in enumerate(bands):
        b, a = butter(order, [l / nyq, h / nyq], btype="band")
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32, copy=False)
    return out


# --------------------------------------------------------------------------- #
# Denek koşumu                                                                #
# --------------------------------------------------------------------------- #


def run_subject(
    subject_id: int,
    pipeline_factory: Callable,
    param_grid: Dict[str, Sequence[Any]],
    bandpass_fn: Callable[[np.ndarray, float], np.ndarray],
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_jobs: int = -1,
    verbose: bool = True,
    oof_dir: Any = None,
) -> SubjectRunResult:
    """Tek deneğe filtre + nested CV uygula. oof_dir verilirse OOF .npz kaydeder."""
    t0 = time.time()
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T için etiket yok.")
    X_filt = bandpass_fn(sd.X, sd.sfreq)
    y = sd.y

    if verbose:
        print(
            f"[A{subject_id:02d}T] X_filt={X_filt.shape}, y={y.shape}",
            flush=True,
        )

    result = run_nested_cv(
        estimator_factory=pipeline_factory,
        param_grid=param_grid,
        X=X_filt,
        y=y,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        return_proba=True,
    )

    elapsed = time.time() - t0
    if verbose:
        print(
            f"[A{subject_id:02d}T] {result.summary()}  elapsed={elapsed:.1f}s",
            flush=True,
        )

    if oof_dir is not None:
        oof_dir = Path(oof_dir)
        oof_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            oof_dir / f"sub_{subject_id:02d}.npz",
            y_true=result.y_true,
            y_pred=result.y_pred,
            y_proba=(
                result.y_proba if result.y_proba is not None else np.empty(0)
            ),
        )

    return SubjectRunResult(
        subject_id=subject_id,
        kappa=result.kappa,
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        kappa_std=float(np.std(result.fold_kappas)),
        n_trials=len(y),
        elapsed_sec=elapsed,
        best_params_per_fold=result.extras.get("best_params_per_fold", []),
    )


# --------------------------------------------------------------------------- #
# Multi-band Tangent Space transformer (Faz 3 Adim A için)                    #
# --------------------------------------------------------------------------- #


class MultiBandTangentSpace(BaseEstimator, TransformerMixin):
    """Bant başina Covariances + TangentSpace, sonra concatenate.

    Burada (modül seviyesinde) tanimlandi çünkü joblib.Memory pickle için
    sinifi modül yolundan bulmak zorunda — __main__'da tanimlanan siniflar
    cache hash'lemede başarisiz oluyor.

    Girdi:  (n_trials, n_bands, n_channels, n_samples)
    Çikti:  (n_trials, n_bands * n_features_ts)
    """

    def __init__(self, cov_estimator: str = "oas", ts_metric: str = "riemann"):
        self.cov_estimator = cov_estimator
        self.ts_metric = ts_metric

    def fit(self, X: np.ndarray, y=None):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        if X.ndim != 4:
            raise ValueError(
                f"MultiBandTangentSpace 4D bekler "
                f"(n_trials, n_bands, n_channels, n_samples), aldi: {X.shape}"
            )
        self.n_bands_ = X.shape[1]
        self.covs_ = []
        self.tss_ = []
        for bi in range(self.n_bands_):
            cov = Covariances(estimator=self.cov_estimator)
            X_b = X[:, bi].astype(np.float64, copy=False)
            cov.fit(X_b)
            cov_X = cov.transform(X_b)
            ts = TangentSpace(metric=self.ts_metric)
            ts.fit(cov_X)
            self.covs_.append(cov)
            self.tss_.append(ts)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for bi in range(self.n_bands_):
            X_b = X[:, bi].astype(np.float64, copy=False)
            cov_X = self.covs_[bi].transform(X_b)
            feats.append(self.tss_[bi].transform(cov_X))
        return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Multi-scale Tangent Space transformer (Faz 9 — zaman pencereleri)           #
# --------------------------------------------------------------------------- #


class MultiScaleTangentSpace(BaseEstimator, TransformerMixin):
    """Single-band, multi-scale Tangent Space.

    Input: (n_trials, n_channels, n_samples) — single bandpass uygulanmis.
    Her time window icin Cov + TS hesaplanir, feature'lar concat edilir.

    Modul seviyesinde tanimlandi (MultiBandTangentSpace gibi) — joblib.Memory
    pickle icin sinifi modul yolundan bulmak zorunda.

    Args:
        windows: list of (start, end) tuples — orn. [(0,500),(0,250),...]
        cov_estimator: 'oas' (default), 'lwf', 'cov'
        metric: 'riemann' (default)
    """

    def __init__(self, windows, cov_estimator="oas", metric="riemann"):
        self.windows = windows
        self.cov_estimator = cov_estimator
        self.metric = metric

    def fit(self, X, y=None):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace

        self.ts_per_window_ = []
        for (start, end) in self.windows:
            X_win = X[:, :, start:end].astype(np.float64, copy=False)
            cov = Covariances(estimator=self.cov_estimator).fit_transform(X_win)
            ts = TangentSpace(metric=self.metric).fit(cov)
            self.ts_per_window_.append(ts)
        return self

    def transform(self, X):
        from pyriemann.estimation import Covariances

        feats = []
        for (start, end), ts in zip(self.windows, self.ts_per_window_):
            X_win = X[:, :, start:end].astype(np.float64, copy=False)
            cov = Covariances(estimator=self.cov_estimator).fit_transform(X_win)
            feats.append(ts.transform(cov))
        return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


__all__ = [
    "RANDOM_STATE",
    "SubjectRunResult",
    "MultiBandTangentSpace",
    "MultiScaleTangentSpace",
    "bandpass_single",
    "bandpass_multi",
    "report_and_save",
    "run_subject",
]

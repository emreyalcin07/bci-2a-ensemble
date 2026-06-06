"""
_fbcsp_common.py
================

Faz 2 FBCSP varyantları için paylaşılan altyapı:
    - generate_bands / make_multiband_tensor (Faz 1 ile aynı tanım)
    - AllBandCSP: tüm bantlara CSP fit eder, log-variance feature döndürür.
                  Sabit parametreli ⇒ Pipeline(memory=...) ile cache'lenebilir.
    - TopKBandSelector: TRAIN-ONLY band selection. Her bandı MI ile skorlar,
                        en iyi top_k bandın feature'larını döndürür.
                        Bu fit yalnızca sklearn fold'unda train kısmına çağrılır
                        ⇒ leakage yok.
    - MRMRSelector: minimum-redundancy maximum-relevance feature seçimi.
    - run_subject / report_and_save: tekrar eden koşum + tablo kaydı.

Leakage notu:
    Pipeline içine konulan adımlar (AllBandCSP, TopKBandSelector, scaler,
    feature selector, classifier) yalnızca outer/inner CV'nin TRAIN parçasına
    fit edilir; test parçası fit'i tetiklemez. mne CSP fit, MI band skorlama,
    SelectKBest, mRMR — hepsi train-only.
"""

from __future__ import annotations

import sys
import time
import warnings

# SelectKBest k>n_features uyarıları (benign fallback) ve sklearn convergence
# uyarıları stdout'u boğmasın.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_random_state

# --- Proje köküne yol ekle ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_loader import load_subject  # noqa: E402
from evaluation import run_nested_cv  # noqa: E402

# MNE'nin CSP fit log'larını sustur (n_jobs alt-süreçlere de yansır
# çünkü mne her import'ta global handler set ediyor).
import mne  # noqa: E402
mne.set_log_level("ERROR")

RANDOM_STATE = 42
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Filter bank                                                                 #
# --------------------------------------------------------------------------- #


def generate_bands(
    f_low: float = 8.0,
    f_high: float = 30.0,
    widths_steps: Sequence[Tuple[float, float]] = ((4.0, 2.0), (6.0, 3.0)),
) -> List[Tuple[float, float]]:
    """Filter bank için (l_freq, h_freq) bant listesi üret.

    Parameters
    ----------
    f_low, f_high : float
        Filter bank aralığı (Hz).
    widths_steps : sequence of (width, step)
        (4Hz/step2 -> 10 bant) + (6Hz/step3 -> 6 bant) = 16 bant (varsayılan).

    Returns
    -------
    list of (l_freq, h_freq)
    """
    bands: List[Tuple[float, float]] = []
    for width, step in widths_steps:
        f = f_low
        while f + width <= f_high + 1e-9:
            bands.append((round(f, 4), round(f + width, 4)))
            f += step
    return bands


def _design_bandpass(l_freq: float, h_freq: float, sfreq: float, order: int = 4):
    nyq = 0.5 * sfreq
    return butter(order, [l_freq / nyq, h_freq / nyq], btype="band")


def make_multiband_tensor(
    X: np.ndarray,
    sfreq: float,
    bands: Sequence[Tuple[float, float]],
    order: int = 4,
) -> np.ndarray:
    """Veri-bağımsız filtreleme; (n_trials, n_bands, n_channels, n_samples)."""
    n_trials, n_channels, n_samples = X.shape
    out = np.empty((n_trials, len(bands), n_channels, n_samples), dtype=np.float32)
    for bi, (l, h) in enumerate(bands):
        b, a = _design_bandpass(l, h, sfreq, order=order)
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32, copy=False)
    return out


# --------------------------------------------------------------------------- #
# All-band CSP (Pipeline'da cache'lenebilir sabit step)                       #
# --------------------------------------------------------------------------- #


class AllBandCSP(BaseEstimator, TransformerMixin):
    """Her banda CSP fit eder, (n_trials, n_bands, n_components) tensörü döndürür.

    Üst seviyede 3D tutmasının nedeni: bir sonraki step (TopKBandSelector)
    bant başına feature gruplarını ayırt edebilsin. Son tüketiciler 2D bekler;
    o yüzden TopKBandSelector flatten eder.

    Sabit parametre tasarımı: bu transformer'ın output'u (n_components, reg,
    log, norm_trace) değişmediği sürece aynıdır; Pipeline(memory=...) cache'i
    inner CV'nin grid combo'ları arasında AllBandCSP fit'ini yeniden
    yapmadan tekrar kullanır.
    """

    def __init__(
        self,
        n_components: int = 4,
        reg: str = "ledoit_wolf",
        log: bool = True,
        norm_trace: bool = False,
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def fit(self, X: np.ndarray, y: np.ndarray):
        from mne.decoding import CSP

        if X.ndim != 4:
            raise ValueError(
                f"AllBandCSP 4D bekler (n_trials, n_bands, n_channels, n_samples), "
                f"aldı: {X.shape}"
            )
        self.n_bands_ = X.shape[1]
        self.csps_ = []
        for bi in range(self.n_bands_):
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=self.log,
                norm_trace=self.norm_trace,
                transform_into="average_power",
            )
            csp.fit(X[:, bi].astype(np.float64, copy=False), y)
            self.csps_.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = np.stack(
            [
                csp.transform(X[:, bi].astype(np.float64, copy=False))
                for bi, csp in enumerate(self.csps_)
            ],
            axis=1,
        )  # (n_trials, n_bands, n_components)
        return feats.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# TRAIN-ONLY Band Selection                                                   #
# --------------------------------------------------------------------------- #


class TopKBandSelector(BaseEstimator, TransformerMixin):
    """Her bandı MI ile skorla, en iyi top_k bandı seç.

    Girdi: (n_trials, n_bands, n_components)
    Çıktı: (n_trials, top_k * n_components) — flatten edilmiş.

    Skorlama: bant başına mutual_info_classif(features, y).sum().
    Sıralama train'de yapılır; transform sadece seçili bantları döndürür.
    """

    def __init__(self, top_k: int = 8, random_state: int = RANDOM_STATE):
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(
                f"TopKBandSelector 3D bekler (n_trials, n_bands, n_components), "
                f"aldı: {X.shape}"
            )
        n_trials, n_bands, n_comp = X.shape
        scores = np.empty(n_bands, dtype=np.float64)
        for bi in range(n_bands):
            mi = mutual_info_classif(
                X[:, bi, :], y,
                random_state=self.random_state,
                n_neighbors=3,
            )
            scores[bi] = float(mi.sum())
        self.band_scores_ = scores

        k = min(self.top_k, n_bands)
        # En yüksek skorlu k bandın indekslerini al (azalan sırada)
        self.selected_bands_ = np.argsort(-scores)[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        sel = X[:, self.selected_bands_, :]  # (n, k, n_comp)
        return sel.reshape(sel.shape[0], -1)  # (n, k*n_comp)


# --------------------------------------------------------------------------- #
# mRMR feature selection                                                       #
# --------------------------------------------------------------------------- #


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Minimum-redundancy Maximum-relevance feature selection.

    Relevance:   MI(feature, y)      (sklearn mutual_info_classif)
    Redundancy:  ortalama MI(feature, selected_feature)  (mutual_info_regression)
    Adım: argmax_f [ rel(f) - (1/|S|) * sum_{s in S} MI(f, s) ]

    Klasik MID (Mutual Information Difference) varyantı. Devamlı feature'lar
    varsayılır (CSP log-variance + StandardScaler sonrası geçerli).
    """

    def __init__(self, n_features: int = 20, random_state: int = RANDOM_STATE):
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = check_random_state(self.random_state)  # noqa: F841 — şu an deterministik
        n_total = X.shape[1]
        k = min(self.n_features, n_total)

        rel = mutual_info_classif(X, y, random_state=self.random_state, n_neighbors=3)
        red_sum = np.zeros(n_total, dtype=np.float64)
        not_sel = np.ones(n_total, dtype=bool)

        # 1. seçim: en yüksek relevance
        first = int(np.argmax(rel))
        selected = [first]
        not_sel[first] = False

        if k > 1 and not_sel.any():
            idx = np.where(not_sel)[0]
            red = mutual_info_regression(
                X[:, idx], X[:, first],
                random_state=self.random_state,
                n_neighbors=3,
            )
            red_sum[idx] += red

        while len(selected) < k:
            idx = np.where(not_sel)[0]
            scores = rel[idx] - red_sum[idx] / len(selected)
            best_local = int(np.argmax(scores))
            best = int(idx[best_local])
            selected.append(best)
            not_sel[best] = False

            if not_sel.any() and len(selected) < k:
                idx = np.where(not_sel)[0]
                red = mutual_info_regression(
                    X[:, idx], X[:, best],
                    random_state=self.random_state,
                    n_neighbors=3,
                )
                red_sum[idx] += red

        self.selected_ = np.array(selected, dtype=np.int64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_]


# --------------------------------------------------------------------------- #
# Koşum yardımcıları                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class SubjectRunResult:
    subject_id: int
    kappa: float
    accuracy: float
    macro_f1: float
    kappa_std: float
    n_trials: int
    elapsed_sec: float
    best_params_per_fold: List[Dict[str, Any]]


def run_subject(
    subject_id: int,
    pipeline_factory: Callable,
    param_grid: Dict[str, Sequence[Any]],
    bands: Sequence[Tuple[float, float]],
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_jobs: int = -1,
    verbose: bool = True,
    oof_dir: Any = None,
) -> SubjectRunResult:
    """Tek deneğe nested CV uygula. oof_dir verilirse OOF .npz kaydeder."""
    t0 = time.time()
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T için etiket yok.")
    X_mb = make_multiband_tensor(sd.X, sfreq=sd.sfreq, bands=bands, order=4)
    y = sd.y

    if verbose:
        print(
            f"[A{subject_id:02d}T] X_mb={X_mb.shape}, y={y.shape}, bands={len(bands)}",
            flush=True,
        )

    result = run_nested_cv(
        estimator_factory=pipeline_factory,
        param_grid=param_grid,
        X=X_mb,
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
        from pathlib import Path as _Path
        oof_dir = _Path(oof_dir)
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


def report_and_save(
    rows: List[SubjectRunResult],
    output_name: str,
    total_elapsed: float,
    title: str,
):
    """Tablo bas + CSV kaydet. Stdout utf-8 olduğu için ASCII-güvenli."""
    df = pd.DataFrame(
        [
            {
                "subject": r.subject_id,
                "cv_kappa": r.kappa,
                "cv_acc": r.accuracy,
                "macro_f1": r.macro_f1,
                "kappa_std": r.kappa_std,
                "n_trials": r.n_trials,
                "elapsed_sec": r.elapsed_sec,
            }
            for r in rows
        ]
    )
    mean_row = {
        "subject": "MEAN",
        "cv_kappa": df["cv_kappa"].mean(),
        "cv_acc": df["cv_acc"].mean(),
        "macro_f1": df["macro_f1"].mean(),
        "kappa_std": df["cv_kappa"].std(),  # denekler-arası std
        "n_trials": df["n_trials"].sum(),
        "elapsed_sec": total_elapsed,
    }
    df_out = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    out_path = RESULTS_DIR / output_name
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(df_out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(
        f"Mean kappa:    {df['cv_kappa'].mean():.4f}  "
        f"(std across subjects: {df['cv_kappa'].std():.4f})"
    )
    print(f"Mean accuracy: {df['cv_acc'].mean():.4f}")
    print(f"Total elapsed: {total_elapsed/60:.1f} min")
    print(f"Saved: {out_path}")

    threshold = 0.5
    low = df[df["cv_kappa"] < threshold]
    if len(low):
        print()
        print(f"[!] Dusuk denek (kappa < {threshold}):")
        for _, r in low.iterrows():
            print(f"   A{int(r['subject']):02d}: kappa={r['cv_kappa']:.4f}")
    return df_out

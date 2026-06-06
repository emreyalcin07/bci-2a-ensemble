"""
bci_lib.py  —  BCI Competition IV 2a yeniden-üretim kütüphanesi
================================================================

Bu modül, projedeki tüm yeniden kullanılabilir yapı taşlarını TEK bir
importlanabilir dosyada toplar. Notebook bu dosyayı `%%writefile` ile
çalışma zamanında diske yazar ve `import bci_lib` ile yükler.

Neden ayrı bir .py modülü?  GridSearchCV(n_jobs=-1) ve joblib.Memory, özel
transformer SINIFLARINI alt-süreçlere "pickle" eder. Notebook hücresinde
(__main__) tanımlanan sınıflar bu pickle işleminde sorun çıkarır; modül
yolundan import edilen sınıflar sorunsuz çalışır. Mantık birebir korunmuştur.

İçerik:
  - Veri yükleme (load_subject / load_all_subjects)        [data_loader.py]
  - Değerlendirme (run_nested_cv, soft_vote, metrikler)    [evaluation.py]
  - FBCSP yapı taşları (CSP, band-seçim, mRMR)             [_fbcsp_common.py]
  - Riemann yapı taşları (TS, multi-band TS)               [_riemann_common.py]
  - Derin öğrenme altyapısı (Dataset, train, transfer)     [_dl_common.py]
  - Modeller (EEGNet, ShallowConvNet)                      [exp_eegnet/scnet]
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple)

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================== #
# GENEL SABİTLER                                                              #
# =========================================================================== #

RANDOM_STATE = 42

#: Veri kök dizini. Notebook bunu bci_lib.DATA_DIR = ... ile değiştirir.
DATA_DIR = Path("/content/data")

SFREQ = 250.0
N_EEG_CHANNELS = 22
N_EOG_CHANNELS = 3

CUE_EVENT_IDS_TRAIN = {"769": 1, "770": 2, "771": 3, "772": 4}
CUE_EVENT_ID_TEST = {"783": 0}

TMIN_AFTER_CUE = 0.5
TMAX_AFTER_CUE = 2.5
EPOCH_SAMPLES = int(round((TMAX_AFTER_CUE - TMIN_AFTER_CUE) * SFREQ))  # 500
LABEL_OFFSET = 1


# =========================================================================== #
# VERİ YÜKLEME  (data_loader.py)                                              #
# =========================================================================== #


@dataclass
class SubjectData:
    """Tek bir oturumun epoch'lanmış verisi."""
    X: np.ndarray
    y: Optional[np.ndarray]
    X_eog: np.ndarray
    subject_id: int
    session: str
    sfreq: float
    ch_names: List[str]

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def class_distribution(self) -> Dict[int, int]:
        if self.y is None:
            return {}
        unique, counts = np.unique(self.y, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}


def _data_dir(data_dir: Optional[Path]) -> Path:
    return Path(data_dir) if data_dir is not None else DATA_DIR


def _build_gdf_path(subject_id: int, session: str, data_dir: Path) -> Path:
    if subject_id < 1 or subject_id > 9:
        raise ValueError(f"subject_id 1..9 olmalı, alınan: {subject_id}")
    session = session.upper()
    if session not in {"T", "E"}:
        raise ValueError(f"session 'T' veya 'E' olmalı, alınan: {session}")
    path = data_dir / f"A{subject_id:02d}{session}.gdf"
    if not path.exists():
        raise FileNotFoundError(f"GDF dosyası bulunamadı: {path}")
    return path


def _find_true_labels_file(subject_id: int, data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / f"A{subject_id:02d}E.mat",
        data_dir / "true_labels" / f"A{subject_id:02d}E.mat",
        data_dir / "labels" / f"A{subject_id:02d}E.mat",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_true_labels(mat_path: Path) -> np.ndarray:
    from scipy.io import loadmat
    mat = loadmat(str(mat_path))
    if "classlabel" not in mat:
        raise KeyError(f"{mat_path.name} içinde 'classlabel' yok.")
    labels = mat["classlabel"].ravel().astype(int)
    return labels - LABEL_OFFSET  # 1..4 -> 0..3


def load_subject(
    subject_id: int,
    session: str = "T",
    data_dir: Optional[Path] = None,
    tmin: float = TMIN_AFTER_CUE,
    tmax: float = TMAX_AFTER_CUE,
    verbose: bool = False,
) -> SubjectData:
    """Tek deneğin tek oturumunu yükle ve epoch'la (288, 22, 500)."""
    import mne
    data_dir = _data_dir(data_dir)
    gdf_path = _build_gdf_path(subject_id, session, data_dir)
    mne_verbose = "INFO" if verbose else "ERROR"

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=mne_verbose)

    ch_names_all = raw.ch_names
    if len(ch_names_all) < N_EEG_CHANNELS + N_EOG_CHANNELS:
        raise RuntimeError(
            f"Beklenen >= {N_EEG_CHANNELS + N_EOG_CHANNELS} kanal, "
            f"bulunan {len(ch_names_all)}"
        )
    eeg_names = ch_names_all[:N_EEG_CHANNELS]
    eog_names = ch_names_all[N_EEG_CHANNELS:N_EEG_CHANNELS + N_EOG_CHANNELS]

    ch_type_map = {name: "eeg" for name in eeg_names}
    ch_type_map.update({name: "eog" for name in eog_names})
    raw.set_channel_types(ch_type_map, verbose=mne_verbose)

    events, event_id_map = mne.events_from_annotations(raw, verbose=mne_verbose)
    wanted = CUE_EVENT_IDS_TRAIN if session.upper() == "T" else CUE_EVENT_ID_TEST

    selected_event_id: Dict[str, int] = {}
    for desc, label in wanted.items():
        if desc in event_id_map:
            selected_event_id[desc] = event_id_map[desc]
        elif session.upper() == "T":
            raise RuntimeError(f"Cue olayı '{desc}' GDF'de yok.")
    if not selected_event_id:
        raise RuntimeError("Hiçbir cue olayı bulunamadı.")

    epochs = mne.Epochs(
        raw, events=events, event_id=selected_event_id,
        tmin=tmin, tmax=tmax, baseline=None, preload=True, proj=False,
        picks=None, verbose=mne_verbose, on_missing="ignore",
    )

    eeg_data = epochs.copy().pick(eeg_names).get_data(copy=False)
    eog_data = epochs.copy().pick(eog_names).get_data(copy=False)
    eeg_data = eeg_data[:, :, :EPOCH_SAMPLES]
    eog_data = eog_data[:, :, :EPOCH_SAMPLES]

    if session.upper() == "T":
        inv_map = {v: CUE_EVENT_IDS_TRAIN[k] for k, v in selected_event_id.items()}
        raw_labels = np.array([inv_map[e] for e in epochs.events[:, 2]])
        y = raw_labels - LABEL_OFFSET
    else:
        mat_path = _find_true_labels_file(subject_id, data_dir)
        if mat_path is not None:
            y = _load_true_labels(mat_path)
            if len(y) != eeg_data.shape[0]:
                raise RuntimeError(
                    f"True label sayısı ({len(y)}) epoch sayısıyla "
                    f"({eeg_data.shape[0]}) eşleşmiyor."
                )
        else:
            y = None

    return SubjectData(
        X=eeg_data.astype(np.float32),
        y=y.astype(np.int64) if y is not None else None,
        X_eog=eog_data.astype(np.float32),
        subject_id=subject_id,
        session=session.upper(),
        sfreq=SFREQ,
        ch_names=eeg_names,
    )


def load_all_subjects(
    session: str = "T",
    data_dir: Optional[Path] = None,
    subjects: Optional[List[int]] = None,
    concatenate: bool = False,
    **kwargs,
):
    """Tüm denekleri yükle. concatenate=True ise (X, y, subject_ids) döner."""
    if subjects is None:
        subjects = list(range(1, 10))
    per_subject: Dict[int, SubjectData] = {}
    for sid in subjects:
        per_subject[sid] = load_subject(sid, session=session, data_dir=data_dir, **kwargs)
    if not concatenate:
        return per_subject
    if any(sd.y is None for sd in per_subject.values()):
        raise RuntimeError("concatenate=True session='E' ile kullanılamaz.")
    X_list, y_list, sid_list = [], [], []
    for sid, sd in per_subject.items():
        X_list.append(sd.X)
        y_list.append(sd.y)
        sid_list.append(np.full(sd.n_trials, sid, dtype=np.int64))
    return (np.concatenate(X_list, 0), np.concatenate(y_list, 0),
            np.concatenate(sid_list, 0))


def summarize(sd: SubjectData) -> str:
    lines = [
        f"Subject A{sd.subject_id:02d}{sd.session}",
        f"  X shape       : {sd.X.shape}  (trials, channels, samples)",
        f"  sfreq         : {sd.sfreq} Hz",
        f"  n_eeg_channels: {len(sd.ch_names)}",
    ]
    if sd.y is not None:
        lines.append(f"  y shape       : {sd.y.shape}")
        lines.append(f"  class dist    : {sd.class_distribution}")
    else:
        lines.append("  y             : None (true labels not loaded)")
    return "\n".join(lines)


# =========================================================================== #
# DEĞERLENDİRME  (evaluation.py)                                              #
# =========================================================================== #

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass
class CVResult:
    kappa: float
    accuracy: float
    macro_f1: float
    fold_kappas: List[float]
    confusion: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"kappa={self.kappa:.4f}  acc={self.accuracy:.4f}  "
                f"macro_f1={self.macro_f1:.4f}  "
                f"fold_kappa_std={np.std(self.fold_kappas):.4f}")


def compute_metrics(y_true, y_pred, labels=None) -> Dict[str, Any]:
    return {
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def _supports_proba(estimator) -> bool:
    return hasattr(estimator, "predict_proba")


def run_cv(estimator_factory, X, y, n_splits=5, shuffle=True,
           random_state=RANDOM_STATE, return_proba=True) -> CVResult:
    """Standart stratified k-fold CV; OOF tahminleri toplar."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    n = len(y)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = None
    fold_kappas: List[float] = []
    for tr, te in skf.split(X, y):
        est = estimator_factory()
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        oof_pred[te] = pred
        fold_kappas.append(cohen_kappa_score(y[te], pred))
        if return_proba and _supports_proba(est):
            proba = est.predict_proba(X[te])
            if oof_proba is None:
                oof_proba = np.zeros((n, proba.shape[1]), dtype=np.float32)
            oof_proba[te] = proba
    m = compute_metrics(y, oof_pred)
    return CVResult(m["kappa"], m["accuracy"], m["macro_f1"], fold_kappas,
                    m["confusion"], y.copy(), oof_pred, oof_proba)


def run_nested_cv(estimator_factory, param_grid, X, y, outer_splits=5,
                  inner_splits=5, scoring="accuracy", random_state=RANDOM_STATE,
                  n_jobs=-1, return_proba=True) -> CVResult:
    """Nested CV: dış fold performans, iç fold (GridSearchCV) hiperparametre."""
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    n = len(y)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = None
    fold_kappas: List[float] = []
    best_params: List[Dict[str, Any]] = []
    for tr, te in outer.split(X, y):
        gs = GridSearchCV(estimator=estimator_factory(), param_grid=param_grid,
                          cv=inner, scoring=scoring, n_jobs=n_jobs, refit=True)
        gs.fit(X[tr], y[tr])
        best_params.append(gs.best_params_)
        pred = gs.predict(X[te])
        oof_pred[te] = pred
        fold_kappas.append(cohen_kappa_score(y[te], pred))
        if return_proba and _supports_proba(gs.best_estimator_):
            proba = gs.predict_proba(X[te])
            if oof_proba is None:
                oof_proba = np.zeros((n, proba.shape[1]), dtype=np.float32)
            oof_proba[te] = proba
    m = compute_metrics(y, oof_pred)
    return CVResult(m["kappa"], m["accuracy"], m["macro_f1"], fold_kappas,
                    m["confusion"], y.copy(), oof_pred, oof_proba,
                    extras={"best_params_per_fold": best_params})


def soft_vote(proba_list: Sequence[np.ndarray],
              weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Ağırlıklı ortalama olasılık -> argmax."""
    if not proba_list:
        raise ValueError("proba_list boş olamaz.")
    stacked = np.stack(proba_list, axis=0)
    if weights is None:
        w = np.ones(stacked.shape[0]) / stacked.shape[0]
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    avg = np.tensordot(w, stacked, axes=([0], [0]))
    return avg.argmax(axis=1)


@dataclass
class SubjectRunResult:
    subject_id: int
    kappa: float
    accuracy: float
    macro_f1: float
    kappa_std: float
    n_trials: int
    elapsed_sec: float
    best_params_per_fold: List[Dict[str, Any]] = field(default_factory=list)


# =========================================================================== #
# FİLTRE BANKASI + FBCSP YAPI TAŞLARI  (_fbcsp_common.py)                     #
# =========================================================================== #

from scipy.signal import butter, filtfilt
from sklearn.feature_selection import (mutual_info_classif,
                                       mutual_info_regression)
from sklearn.utils.validation import check_random_state


def generate_bands(f_low=8.0, f_high=30.0,
                   widths_steps=((4.0, 2.0), (6.0, 3.0))) -> List[Tuple[float, float]]:
    """16 bant: (4Hz/adim2 -> 10 bant) + (6Hz/adim3 -> 6 bant)."""
    bands: List[Tuple[float, float]] = []
    for width, step in widths_steps:
        f = f_low
        while f + width <= f_high + 1e-9:
            bands.append((round(f, 4), round(f + width, 4)))
            f += step
    return bands


def _design_bandpass(l_freq, h_freq, sfreq, order=4):
    nyq = 0.5 * sfreq
    return butter(order, [l_freq / nyq, h_freq / nyq], btype="band")


def make_multiband_tensor(X, sfreq, bands, order=4) -> np.ndarray:
    """Veri-bağımsız filtreleme -> (n_trials, n_bands, n_channels, n_samples)."""
    n_trials, n_channels, n_samples = X.shape
    out = np.empty((n_trials, len(bands), n_channels, n_samples), dtype=np.float32)
    for bi, (l, h) in enumerate(bands):
        b, a = _design_bandpass(l, h, sfreq, order=order)
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32, copy=False)
    return out


class MultiBandCSP(BaseEstimator, TransformerMixin):
    """Her banda CSP (ledoit_wolf, log-var); feature'ları concat eder. 4D girdi."""

    def __init__(self, n_components=4, reg="ledoit_wolf", log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def fit(self, X, y):
        from mne.decoding import CSP
        if X.ndim != 4:
            raise ValueError(f"MultiBandCSP 4D bekler, aldı: {X.shape}")
        self.n_bands_ = X.shape[1]
        self.csps_ = []
        for bi in range(self.n_bands_):
            csp = CSP(n_components=self.n_components, reg=self.reg, log=self.log,
                      norm_trace=self.norm_trace, transform_into="average_power")
            csp.fit(X[:, bi].astype(np.float64, copy=False), y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        feats = [csp.transform(X[:, bi].astype(np.float64, copy=False))
                 for bi, csp in enumerate(self.csps_)]
        return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


class AllBandCSP(BaseEstimator, TransformerMixin):
    """Her banda CSP, (n_trials, n_bands, n_components) tensörü döndürür.
    Sabit parametre -> Pipeline(memory=...) ile cache'lenebilir."""

    def __init__(self, n_components=4, reg="ledoit_wolf", log=True, norm_trace=False):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace

    def fit(self, X, y):
        from mne.decoding import CSP
        if X.ndim != 4:
            raise ValueError(f"AllBandCSP 4D bekler, aldı: {X.shape}")
        self.n_bands_ = X.shape[1]
        self.csps_ = []
        for bi in range(self.n_bands_):
            csp = CSP(n_components=self.n_components, reg=self.reg, log=self.log,
                      norm_trace=self.norm_trace, transform_into="average_power")
            csp.fit(X[:, bi].astype(np.float64, copy=False), y)
            self.csps_.append(csp)
        return self

    def transform(self, X):
        feats = np.stack([csp.transform(X[:, bi].astype(np.float64, copy=False))
                          for bi, csp in enumerate(self.csps_)], axis=1)
        return feats.astype(np.float32, copy=False)


class TopKBandSelector(BaseEstimator, TransformerMixin):
    """Her bandı MI ile skorla, en iyi top_k bandı seç (TRAIN-ONLY).
    Girdi (n, n_bands, n_comp) -> Çıktı (n, top_k*n_comp)."""

    def __init__(self, top_k=8, random_state=RANDOM_STATE):
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X, y):
        if X.ndim != 3:
            raise ValueError(f"TopKBandSelector 3D bekler, aldı: {X.shape}")
        n_trials, n_bands, n_comp = X.shape
        scores = np.empty(n_bands, dtype=np.float64)
        for bi in range(n_bands):
            mi = mutual_info_classif(X[:, bi, :], y,
                                     random_state=self.random_state, n_neighbors=3)
            scores[bi] = float(mi.sum())
        self.band_scores_ = scores
        k = min(self.top_k, n_bands)
        self.selected_bands_ = np.argsort(-scores)[:k]
        return self

    def transform(self, X):
        sel = X[:, self.selected_bands_, :]
        return sel.reshape(sel.shape[0], -1)


class MRMRSelector(BaseEstimator, TransformerMixin):
    """mRMR (MID): relevance MI(f,y) - ortalama redundancy MI(f, secilen)."""

    def __init__(self, n_features=20, random_state=RANDOM_STATE):
        self.n_features = n_features
        self.random_state = random_state

    def fit(self, X, y):
        _ = check_random_state(self.random_state)
        n_total = X.shape[1]
        k = min(self.n_features, n_total)
        rel = mutual_info_classif(X, y, random_state=self.random_state, n_neighbors=3)
        red_sum = np.zeros(n_total, dtype=np.float64)
        not_sel = np.ones(n_total, dtype=bool)
        first = int(np.argmax(rel))
        selected = [first]
        not_sel[first] = False
        if k > 1 and not_sel.any():
            idx = np.where(not_sel)[0]
            red = mutual_info_regression(X[:, idx], X[:, first],
                                         random_state=self.random_state, n_neighbors=3)
            red_sum[idx] += red
        while len(selected) < k:
            idx = np.where(not_sel)[0]
            scores = rel[idx] - red_sum[idx] / len(selected)
            best = int(idx[int(np.argmax(scores))])
            selected.append(best)
            not_sel[best] = False
            if not_sel.any() and len(selected) < k:
                idx = np.where(not_sel)[0]
                red = mutual_info_regression(X[:, idx], X[:, best],
                                             random_state=self.random_state, n_neighbors=3)
                red_sum[idx] += red
        self.selected_ = np.array(selected, dtype=np.int64)
        return self

    def transform(self, X):
        return X[:, self.selected_]


# =========================================================================== #
# RIEMANN YAPI TAŞLARI  (_riemann_common.py)                                  #
# =========================================================================== #


def bandpass_single(X, sfreq, l_freq=8.0, h_freq=30.0, order=4) -> np.ndarray:
    """Tek geniş bant 8-30 Hz Butterworth (filtfilt)."""
    nyq = 0.5 * sfreq
    b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1).astype(np.float32, copy=False)


def bandpass_multi(X, sfreq, bands, order=4) -> np.ndarray:
    """Her bant ayrı kopya: (n_trials, n_bands, n_ch, n_samp)."""
    out = np.empty((X.shape[0], len(bands), X.shape[1], X.shape[2]), dtype=np.float32)
    nyq = 0.5 * sfreq
    for bi, (l, h) in enumerate(bands):
        b, a = butter(order, [l / nyq, h / nyq], btype="band")
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32, copy=False)
    return out


class MultiBandTangentSpace(BaseEstimator, TransformerMixin):
    """Bant başına Covariances(oas) + TangentSpace(riemann), sonra concat."""

    def __init__(self, cov_estimator="oas", ts_metric="riemann"):
        self.cov_estimator = cov_estimator
        self.ts_metric = ts_metric

    def fit(self, X, y=None):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        if X.ndim != 4:
            raise ValueError(f"MultiBandTangentSpace 4D bekler, aldı: {X.shape}")
        self.n_bands_ = X.shape[1]
        self.covs_, self.tss_ = [], []
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

    def transform(self, X):
        feats = []
        for bi in range(self.n_bands_):
            X_b = X[:, bi].astype(np.float64, copy=False)
            cov_X = self.covs_[bi].transform(X_b)
            feats.append(self.tss_[bi].transform(cov_X))
        return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


# =========================================================================== #
# DERİN ÖĞRENME ALTYAPISI  (_dl_common.py)                                    #
# =========================================================================== #

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Tüm stochastic kanalları aynı tohuma bağlar."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinizmi artır (T4 üzerinde runlar arası varyansı azaltır)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bandpass_wide(X, sfreq, l_freq=4.0, h_freq=38.0, order=4) -> np.ndarray:
    """4-38 Hz geniş bant (DL kendi temporal filtresini öğrenir)."""
    nyq = 0.5 * sfreq
    b, a = butter(order, [l_freq / nyq, h_freq / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1).astype(np.float32, copy=False)


class EEGDataset(Dataset):
    """numpy (N, C, T) -> tensor (1, C, T) + label. Augmentation sadece train'de."""

    def __init__(self, X, y, augment=False, shift_max=12, noise_std=0.1):
        assert X.ndim == 3, f"X (N, C, T) olmalı, alındı {X.shape}"
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.augment = augment
        self.shift_max = shift_max
        self.noise_std = noise_std
        self._ch_std = X.std(axis=(0, 2), keepdims=True) + 1e-12

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            if self.shift_max > 0:
                shift = np.random.randint(-self.shift_max, self.shift_max + 1)
                if shift != 0:
                    x = np.roll(x, shift, axis=-1)
            if self.noise_std > 0:
                std = self._ch_std[0]
                x = x + np.random.randn(*x.shape).astype(np.float32) * (std * self.noise_std)
        x = x[np.newaxis, :, :]
        return torch.from_numpy(x), int(self.y[idx])


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


def _stratified_inner_split(y, val_fraction, seed):
    n_splits = max(int(round(1.0 / val_fraction)), 2)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(skf.split(np.zeros(len(y)), y))
    return tr_idx, va_idx


def _train_loop(model, dl_tr, dl_va, dl_te, cfg_epochs, lr, weight_decay, patience):
    """Ortak eğitim döngüsü: early stopping + en iyi state ile test predict_proba."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    pat = 0
    epoch = 0
    for epoch in range(cfg_epochs):
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
                vs += criterion(model(xb), yb).item() * xb.size(0)
                vn += xb.size(0)
        val_loss = vs / max(vn, 1)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    probas = []
    with torch.no_grad():
        for xb, _ in dl_te:
            xb = xb.to(DEVICE, non_blocking=True)
            probas.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
    y_proba = np.concatenate(probas, axis=0)
    info = {"best_epoch": best_epoch, "best_val_loss": float(best_val_loss),
            "n_epochs_trained": epoch + 1, "best_state": best_state}
    return y_proba, info


def train_one_fold(model_factory, X_tr_outer, y_tr_outer, X_te_outer, cfg,
                   inner_seed=RANDOM_STATE):
    """Tek dış-fold: iç train/val (early stop) + dış-test predict_proba."""
    set_seed(inner_seed)
    tr_idx, va_idx = _stratified_inner_split(y_tr_outer, cfg.val_fraction, inner_seed)
    X_tr, y_tr = X_tr_outer[tr_idx], y_tr_outer[tr_idx]
    X_va, y_va = X_tr_outer[va_idx], y_tr_outer[va_idx]
    ds_tr = EEGDataset(X_tr, y_tr, augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(X_va, y_va, augment=False)
    ds_te = EEGDataset(X_te_outer, np.zeros(len(X_te_outer), dtype=np.int64), augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)
    model = model_factory().to(DEVICE)
    y_proba, info = _train_loop(model, dl_tr, dl_va, dl_te, cfg.epochs, cfg.lr,
                                cfg.weight_decay, cfg.early_stopping_patience)
    info.pop("best_state", None)
    return y_proba, info


def run_dl_subject(subject_id, model_factory, cfg, oof_store, oof_subdir,
                   outer_splits=5, verbose=True):
    """Tek denek 5-fold outer CV + OOF kaydet (oof_store: dict-of-dict)."""
    t0 = time.time()
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T için etiket yok.")
    X = bandpass_wide(sd.X, sd.sfreq) * 1e6
    y = sd.y
    if verbose:
        print(f"[A{subject_id:02d}T] X={X.shape}, device={DEVICE}", flush=True)
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    n = len(y)
    n_classes = int(np.max(y) + 1)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = np.zeros((n, n_classes), dtype=np.float32)
    fold_kappas: List[float] = []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        inner_seed = RANDOM_STATE + fi
        y_proba_te, info = train_one_fold(model_factory, X[tr], y[tr], X[te],
                                          cfg, inner_seed=inner_seed)
        oof_proba[te] = y_proba_te
        oof_pred[te] = y_proba_te.argmax(axis=1)
        k = cohen_kappa_score(y[te], oof_pred[te])
        fold_kappas.append(k)
        if verbose:
            print(f"  fold {fi}: kappa={k:.4f}  best_epoch={info['best_epoch']}",
                  flush=True)
    kappa = cohen_kappa_score(y, oof_pred)
    acc = accuracy_score(y, oof_pred)
    mf1 = f1_score(y, oof_pred, average="macro")
    oof_store.setdefault(oof_subdir, {})[subject_id] = {
        "y_true": y.copy(), "y_pred": oof_pred.copy(), "y_proba": oof_proba.copy()}
    elapsed = time.time() - t0
    if verbose:
        print(f"[A{subject_id:02d}T] kappa={kappa:.4f}  acc={acc:.4f}  "
              f"elapsed={elapsed:.1f}s", flush=True)
    return SubjectRunResult(subject_id, kappa, acc, mf1, float(np.std(fold_kappas)),
                            n, elapsed)


# ----- Cross-subject transfer (Faz 5b) ------------------------------------- #


@dataclass
class TransferConfig:
    pre_epochs: int = 200
    pre_batch_size: int = 64
    pre_lr: float = 1e-3
    pre_weight_decay: float = 1e-2
    pre_patience: int = 30
    pre_val_fraction: float = 0.15
    ft_epochs: int = 100
    ft_batch_size: int = 64
    ft_lr: float = 1e-4
    ft_weight_decay: float = 1e-2
    ft_patience: int = 15
    ft_val_fraction: float = 0.2
    augment: bool = True
    shift_max: int = 12
    noise_std: float = 0.1
    verbose: bool = False


def _load_pretrain_pool(target_subject_id, all_subjects=tuple(range(1, 10))):
    pool = [s for s in all_subjects if s != target_subject_id]
    per_subj = load_all_subjects(session="T", subjects=pool, concatenate=False)
    Xs, ys = [], []
    for sid, sd in per_subj.items():
        if sd.y is None:
            raise RuntimeError(f"A{sid:02d}T etiketi yok.")
        Xs.append(bandpass_wide(sd.X, sd.sfreq) * 1e6)
        ys.append(sd.y)
    return (np.concatenate(Xs, 0).astype(np.float32),
            np.concatenate(ys, 0).astype(np.int64))


def _pretrain_model(model_factory, X, y, cfg, seed=RANDOM_STATE):
    set_seed(seed)
    tr_idx, va_idx = _stratified_inner_split(y, cfg.pre_val_fraction, seed)
    ds_tr = EEGDataset(X[tr_idx], y[tr_idx], augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(X[va_idx], y[va_idx], augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.pre_batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)
    # Pretrain'de test gerekmez; küçük bir kukla loader kullan.
    ds_te = EEGDataset(X[va_idx][:2], y[va_idx][:2], augment=False)
    dl_te = DataLoader(ds_te, batch_size=2, shuffle=False)
    model = model_factory().to(DEVICE)
    _, info = _train_loop(model, dl_tr, dl_va, dl_te, cfg.pre_epochs, cfg.pre_lr,
                          cfg.pre_weight_decay, cfg.pre_patience)
    best_state = info.pop("best_state")
    return best_state, {"pretrain_best_epoch": info["best_epoch"],
                        "pretrain_n_samples": len(y)}


def _finetune_and_predict(pretrained_state, model_factory, X_tr_outer, y_tr_outer,
                          X_te_outer, cfg, inner_seed):
    set_seed(inner_seed)
    tr_idx, va_idx = _stratified_inner_split(y_tr_outer, cfg.ft_val_fraction, inner_seed)
    ds_tr = EEGDataset(X_tr_outer[tr_idx], y_tr_outer[tr_idx], augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(X_tr_outer[va_idx], y_tr_outer[va_idx], augment=False)
    ds_te = EEGDataset(X_te_outer, np.zeros(len(X_te_outer), dtype=np.int64), augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.ft_batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)
    model = model_factory().to(DEVICE)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, X_tr_outer.shape[1], X_tr_outer.shape[2], device=DEVICE)
        _ = model(dummy)  # LazyLinear parametrelerini oluştur
    model.load_state_dict(pretrained_state, strict=True)
    y_proba, info = _train_loop(model, dl_tr, dl_va, dl_te, cfg.ft_epochs, cfg.ft_lr,
                                cfg.ft_weight_decay, cfg.ft_patience)
    info.pop("best_state", None)
    return y_proba, info


def run_transfer_subject(subject_id, model_factory, cfg, oof_store, oof_subdir,
                         outer_splits=5, verbose=True):
    """LOSO pretrain (1 kere) + 5 dış-fold finetune/predict. OOF kaydet."""
    t0 = time.time()
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T etiketi yok.")
    X_target = bandpass_wide(sd.X, sd.sfreq) * 1e6
    y_target = sd.y
    X_pool, y_pool = _load_pretrain_pool(subject_id)
    if verbose:
        print(f"[A{subject_id:02d}T] target X={X_target.shape}  pool X={X_pool.shape}",
              flush=True)
    pretrained_state, pre_info = _pretrain_model(model_factory, X_pool, y_pool, cfg)
    if verbose:
        print(f"[A{subject_id:02d}T] pretrain best@{pre_info['pretrain_best_epoch']}",
              flush=True)
    skf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    n = len(y_target)
    n_classes = int(np.max(y_target) + 1)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba = np.zeros((n, n_classes), dtype=np.float32)
    fold_kappas: List[float] = []
    for fi, (tr, te) in enumerate(skf.split(X_target, y_target)):
        inner_seed = RANDOM_STATE + fi
        y_proba_te, info = _finetune_and_predict(pretrained_state, model_factory,
                                                 X_target[tr], y_target[tr],
                                                 X_target[te], cfg, inner_seed)
        oof_proba[te] = y_proba_te
        oof_pred[te] = y_proba_te.argmax(axis=1)
        k = cohen_kappa_score(y_target[te], oof_pred[te])
        fold_kappas.append(k)
        if verbose:
            print(f"  fold {fi}: kappa={k:.4f}", flush=True)
    kappa = cohen_kappa_score(y_target, oof_pred)
    acc = accuracy_score(y_target, oof_pred)
    mf1 = f1_score(y_target, oof_pred, average="macro")
    oof_store.setdefault(oof_subdir, {})[subject_id] = {
        "y_true": y_target.copy(), "y_pred": oof_pred.copy(), "y_proba": oof_proba.copy()}
    elapsed = time.time() - t0
    if verbose:
        print(f"[A{subject_id:02d}T] TRANSFER kappa={kappa:.4f}  elapsed={elapsed:.1f}s",
              flush=True)
    return SubjectRunResult(subject_id, kappa, acc, mf1, float(np.std(fold_kappas)),
                            n, elapsed)


# =========================================================================== #
# MODELLER  (exp_eegnet.py / exp_shallowconvnet.py)                           #
# =========================================================================== #


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018). 4-sınıf motor imagery."""

    def __init__(self, n_classes=4, n_chans=22, n_samples=500, F1=8, D=2, F2=16,
                 kernel_length=64, pool1=4, pool2=8, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                                   groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x)
        x = self.depthwise(x); x = self.bn2(x); x = F.elu(x)
        x = self.pool1(x); x = self.drop1(x)
        x = self.sep_depth(x); x = self.sep_point(x); x = self.bn3(x); x = F.elu(x)
        x = self.pool2(x); x = self.drop2(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_eegnet() -> nn.Module:
    return EEGNet()


def _safe_log(x, eps=1e-6):
    return torch.log(torch.clamp(x, min=eps))


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al. 2017). FBCSP'nin DL karşılığı."""

    def __init__(self, n_classes=4, n_chans=22, n_samples=500, n_filters_time=40,
                 n_filters_spat=40, filter_time_length=25, pool_time_length=75,
                 pool_time_stride=15, dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters_time, (1, filter_time_length), bias=True)
        self.spatial = nn.Conv2d(n_filters_time, n_filters_spat, (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x * x
        x = self.pool(x)
        x = _safe_log(x)
        x = self.drop(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_shallowconvnet() -> nn.Module:
    return ShallowConvNet()

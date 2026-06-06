"""
evaluation.py
=============

Sınıflandırma performansı için ortak metrikler ve cross-validation
runner'ları.

Hedef metrik: **Cohen's Kappa** (BCI Competition IV standart metriği).
Yan metrikler: doğruluk, sınıf-bazlı F1, confusion matrix.

Tasarım notları:
    - run_cv: standart stratified k-fold cross-validation runner.
    - run_nested_cv: dış CV ile performans tahmini, iç CV ile
      hiperparametre seçimi. Subject-specific kappa raporu için kullanılır.
    - Tüm runner'lar pipeline'ın predict_proba sağlayıp sağlamadığını
      kontrol eder; ileride soft-voting ensemble için olasılık matrisleri
      bu modül tarafından döndürülecek.
    - Sonuçlar pandas.DataFrame olarak da serileştirilebilir (results/'a
      kaydetmek için main.py kullanır).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# --------------------------------------------------------------------------- #
# Sonuç konteyneri                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class CVResult:
    """Bir CV koşusunun çıktısı.

    Attributes
    ----------
    kappa : float
        Tüm dış fold OOF tahminleri üzerinden Cohen's kappa.
    accuracy : float
        OOF doğruluk.
    macro_f1 : float
        Sınıf-dengeli F1.
    fold_kappas : list[float]
        Her dış fold için ayrı kappa (varyans / güven aralığı için).
    confusion : np.ndarray
        Toplam OOF confusion matrix.
    y_true : np.ndarray
        Ground truth (OOF sırasına göre).
    y_pred : np.ndarray
        Tahminler (OOF sırasına göre).
    y_proba : np.ndarray | None
        Sınıf olasılıkları (n, n_classes) — soft-voting ensemble için.
        Pipeline predict_proba sağlamıyorsa None.
    extras : dict
        Pipeline-özel metaveriler (örn. seçilen hiperparametreler).
    """

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
        return (
            f"kappa={self.kappa:.4f}  acc={self.accuracy:.4f}  "
            f"macro_f1={self.macro_f1:.4f}  "
            f"fold_kappa_std={np.std(self.fold_kappas):.4f}"
        )


# --------------------------------------------------------------------------- #
# Tek-set metrikleri                                                          #
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Cohen's kappa, accuracy, macro-F1 ve confusion matrix hesapla."""
    return {
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


# --------------------------------------------------------------------------- #
# Standart k-fold CV                                                          #
# --------------------------------------------------------------------------- #


def _supports_proba(estimator: BaseEstimator) -> bool:
    return hasattr(estimator, "predict_proba")


def run_cv(
    estimator_factory: Callable[[], BaseEstimator],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    return_proba: bool = True,
) -> CVResult:
    """Stratified k-fold CV ile sınıflandırıcıyı değerlendir.

    Parameters
    ----------
    estimator_factory : callable -> BaseEstimator
        Çağrıldığında taze bir sınıflandırıcı döner. Pipeline'ın her
        fold'da sıfırdan eğitildiğinden emin olmak için fabrika kullanılır.
    X : ndarray, shape (n_trials, ...)
    y : ndarray, shape (n_trials,)
    n_splits : int
    shuffle : bool
    random_state : int
    return_proba : bool
        True ve sınıflandırıcı predict_proba sağlıyorsa, OOF olasılık
        matrisi de döndürülür. Ensemble (soft voting) için gereklidir.

    Returns
    -------
    CVResult
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    n = len(y)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba: Optional[np.ndarray] = None
    fold_kappas: List[float] = []

    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
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

    metrics = compute_metrics(y, oof_pred)
    return CVResult(
        kappa=metrics["kappa"],
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        fold_kappas=fold_kappas,
        confusion=metrics["confusion"],
        y_true=y.copy(),
        y_pred=oof_pred,
        y_proba=oof_proba,
    )


# --------------------------------------------------------------------------- #
# Nested CV                                                                   #
# --------------------------------------------------------------------------- #


def run_nested_cv(
    estimator_factory: Callable[[], BaseEstimator],
    param_grid: Dict[str, Sequence[Any]],
    X: np.ndarray,
    y: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "accuracy",
    random_state: int = 42,
    n_jobs: int = 1,
    return_proba: bool = True,
) -> CVResult:
    """Nested CV: dış fold performans, iç fold hiperparametre seçimi.

    Subject-specific raporlama için tercih edilen yöntem. İç döngüde
    GridSearchCV ile en iyi hiperparametre seçilir; dış döngüde bu
    sınıflandırıcının OOF tahmini alınır.

    Parameters
    ----------
    estimator_factory : callable -> BaseEstimator
    param_grid : dict
        sklearn GridSearchCV uyumlu parametre ızgarası. Pipeline kullanırken
        'step__param' notasyonu geçerlidir.
    scoring : str
        İç CV skoru. Varsayılan accuracy; kappa için 'cohen_kappa' özel
        scorer gerekir (bkz. sklearn make_scorer).

    Returns
    -------
    CVResult
        extras['best_params'] her dış fold için seçilen hiperparametre.
    """
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    n = len(y)
    oof_pred = np.full(n, -1, dtype=np.int64)
    oof_proba: Optional[np.ndarray] = None
    fold_kappas: List[float] = []
    best_params: List[Dict[str, Any]] = []

    for tr, te in outer.split(X, y):
        gs = GridSearchCV(
            estimator=estimator_factory(),
            param_grid=param_grid,
            cv=inner,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
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

    metrics = compute_metrics(y, oof_pred)
    return CVResult(
        kappa=metrics["kappa"],
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        fold_kappas=fold_kappas,
        confusion=metrics["confusion"],
        y_true=y.copy(),
        y_pred=oof_pred,
        y_proba=oof_proba,
        extras={"best_params_per_fold": best_params},
    )


# --------------------------------------------------------------------------- #
# Ensemble yardımcısı (ileride main.py kullanır)                              #
# --------------------------------------------------------------------------- #


def soft_vote(proba_list: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Birden fazla pipeline'ın OOF olasılık matrislerini soft-vote ile birleştir.

    Parameters
    ----------
    proba_list : list of ndarray, her biri (n, n_classes)
    weights : list of float, optional
        Pipeline ağırlıkları. None ise eşit.

    Returns
    -------
    ndarray, shape (n,), birleşik tahminler (argmax).
    """
    if not proba_list:
        raise ValueError("proba_list boş olamaz.")
    stacked = np.stack(proba_list, axis=0)  # (n_models, n, n_classes)
    if weights is None:
        w = np.ones(stacked.shape[0]) / stacked.shape[0]
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    avg = np.tensordot(w, stacked, axes=([0], [0]))  # (n, n_classes)
    return avg.argmax(axis=1)


# --------------------------------------------------------------------------- #
# Tablo çıktısı                                                               #
# --------------------------------------------------------------------------- #


def result_to_row(name: str, subject_id: int, result: CVResult) -> Dict[str, Any]:
    """Sonucu CSV satırı olarak serileştir (results/tables için)."""
    return {
        "pipeline": name,
        "subject": subject_id,
        "kappa": result.kappa,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "kappa_std": float(np.std(result.fold_kappas)),
        "n_folds": len(result.fold_kappas),
    }

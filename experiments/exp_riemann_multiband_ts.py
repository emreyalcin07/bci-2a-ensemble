"""
exp_riemann_multiband_ts.py
===========================

Faz 3 — Adım A: Multi-band Riemannian Tangent Space + LR.

Filter bank mantığını Riemannian'a tasıyoruz:
    Bantlar: [(8,14), (12,20), (18,30)]  (3 geniş bant)
    Her banta:  Covariances('oas') -> TangentSpace('riemann')
    Concatenate -> StandardScaler -> [PCA | L1-LR] -> classifier

Boyut/örnek: 3 bant × 22*23/2 = 759 TS feature, 288 trial.
Bu oran tehlikeli; bu yüzden boyut indirgeme iki varyantta denenir:

  - PCA variant: PCA(n_components) -> LogisticRegression(L2)
        n_components tune: [30, 50, 80]
        C tune:            [0.01, 0.1, 1, 10]
  - L1 variant:  LogisticRegression(penalty='l1', solver='saga')  # implicit sparse
        C tune:            [0.01, 0.1, 1, 10]

Pipeline(memory=...) MultiBandTangentSpace step'ini grid combo'ları arasinda
yeniden kullanir -> büyük hızlanma.

OOF y_true ve y_proba diske kaydedilir (ensemble adimi için hazirlik).

A0XE'ye DOKUNULMAZ.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joblib import Memory  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from _riemann_common import (  # noqa: E402
    RANDOM_STATE,
    MultiBandTangentSpace,
    bandpass_multi,
    report_and_save,
)
from _fbcsp_common import SubjectRunResult  # noqa: E402
from data_loader import load_subject  # noqa: E402
from evaluation import run_nested_cv  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
OOF_DIR = PROJECT_ROOT / "results" / "oof"
OOF_DIR.mkdir(parents=True, exist_ok=True)

BANDS: List[Tuple[float, float]] = [(8.0, 14.0), (12.0, 20.0), (18.0, 30.0)]


# --------------------------------------------------------------------------- #
# Pipeline fabrikalari                                                        #
# --------------------------------------------------------------------------- #

_MEMORY = Memory(location=tempfile.mkdtemp(prefix="mbts_cache_"), verbose=0)


def build_pipeline_pca() -> Pipeline:
    """Multi-band TS + StandardScaler + PCA + LogReg(L2)."""
    return Pipeline(
        steps=[
            ("mbts", MultiBandTangentSpace(cov_estimator="oas", ts_metric="riemann")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=50, random_state=RANDOM_STATE)),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        memory=_MEMORY,
    )


PARAM_GRID_PCA = {
    "pca__n_components": [30, 50, 80],
    "lr__C": [0.01, 0.1, 1.0, 10.0],
}


def build_pipeline_l1() -> Pipeline:
    """Multi-band TS + StandardScaler + LogReg(L1, saga)."""
    return Pipeline(
        steps=[
            ("mbts", MultiBandTangentSpace(cov_estimator="oas", ts_metric="riemann")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=4000,
                    C=1.0,
                    penalty="l1",
                    solver="saga",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        memory=_MEMORY,
    )


PARAM_GRID_L1 = {
    "lr__C": [0.01, 0.1, 1.0, 10.0],
}


# --------------------------------------------------------------------------- #
# Denek koşumu (run_nested_cv'yi dogrudan çagiriyoruz çünkü OOF kaydedecegiz) #
# --------------------------------------------------------------------------- #


def run_subject_save_oof(
    subject_id: int,
    pipeline_factory: Callable,
    param_grid: Dict[str, Sequence[Any]],
    variant_name: str,
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_jobs: int = -1,
    verbose: bool = True,
) -> SubjectRunResult:
    """Tek deneğe nested CV uygula, OOF'i diske kaydet."""
    t0 = time.time()
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T için etiket yok.")

    X_mb = bandpass_multi(sd.X, sfreq=sd.sfreq, bands=BANDS, order=4)
    y = sd.y

    if verbose:
        print(
            f"[A{subject_id:02d}T] X_mb={X_mb.shape}, y={y.shape}, bands={len(BANDS)}",
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

    # OOF kaydet (ensemble için)
    oof_subdir = OOF_DIR / variant_name
    oof_subdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        oof_subdir / f"sub_{subject_id:02d}.npz",
        y_true=result.y_true,
        y_pred=result.y_pred,
        y_proba=result.y_proba if result.y_proba is not None else np.empty(0),
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
# Main                                                                        #
# --------------------------------------------------------------------------- #


def run_variant(
    variant_name: str,
    factory: Callable,
    param_grid: Dict[str, Sequence[Any]],
    subjects: Sequence[int],
    outer_splits: int,
    inner_splits: int,
    n_jobs: int,
    output_csv: str,
    title: str,
) -> None:
    print()
    print(f"### Variant: {variant_name}")
    print(f"Grid: {param_grid}")
    print()
    rows: List[SubjectRunResult] = []
    t0 = time.time()
    for sid in subjects:
        rows.append(
            run_subject_save_oof(
                subject_id=sid,
                pipeline_factory=factory,
                param_grid=param_grid,
                variant_name=variant_name,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                n_jobs=n_jobs,
                verbose=True,
            )
        )
    total = time.time() - t0
    report_and_save(
        rows=rows,
        output_name=output_csv,
        total_elapsed=total,
        title=title,
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-band Riemann TS nested CV.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=["pca", "l1"],
        choices=["pca", "l1"],
    )
    args = parser.parse_args()

    print(f"Bands: {BANDS}")
    print()

    try:
        if "pca" in args.variants:
            run_variant(
                variant_name="riemann_multiband_ts_pca",
                factory=build_pipeline_pca,
                param_grid=PARAM_GRID_PCA,
                subjects=args.subjects,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
                n_jobs=args.n_jobs,
                output_csv="exp_riemann_multiband_ts_pca.csv",
                title="Multi-band Riemann TS + PCA + LR(L2) - nested CV (5x5)",
            )
        if "l1" in args.variants:
            run_variant(
                variant_name="riemann_multiband_ts_l1",
                factory=build_pipeline_l1,
                param_grid=PARAM_GRID_L1,
                subjects=args.subjects,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
                n_jobs=args.n_jobs,
                output_csv="exp_riemann_multiband_ts_l1.csv",
                title="Multi-band Riemann TS + LR(L1, saga) - nested CV (5x5)",
            )
    finally:
        try:
            shutil.rmtree(_MEMORY.location, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()

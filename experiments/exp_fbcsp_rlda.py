"""
exp_fbcsp_rlda.py
=================

Filter Bank CSP + Shrinkage-LDA baseline, leakage-free nested CV ile.

Pipeline:
    [her bant için] band-pass (4. dereceden Butterworth, filtfilt)
    -> Regularized CSP (mne.decoding.CSP, reg='ledoit_wolf', log=True)
    -> tüm bantların CSP log-variance feature'ları concatenate
    -> SelectKBest(mutual_info_classif, k=K)
    -> LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

Leakage önleme:
    - Band-pass filtre veriden öğrenmez; trial-bağımsız bir DSP işlemidir.
      Bu yüzden filtrelenmiş çoklu-bant tensörünü Pipeline DIŞINDA, bir
      kereye mahsus hesaplayıp X olarak veririz.
    - CSP ve SelectKBest sklearn Pipeline içindedir → her iç/dış CV
      fold'unda yalnızca TRAIN verisine fit edilir.
    - Test fold'u ne CSP'ye ne de SelectKBest'e dokunmaz.
    - A0XE bu betikte HİÇ kullanılmaz.

İç CV (GridSearchCV) tune ettiği hiperparametreler:
    csp__n_components ∈ {2, 4, 6}
    select__k         ∈ {10, 20, 40, 'all'}
    (LDA shrinkage='auto' Ledoit-Wolf ile, ayrıca tune edilmez)

Dış CV:
    5-fold StratifiedKFold üzerinden OOF tahminleri toplanır;
    denek başına tek bir Cohen's kappa raporlanır (fold-bazlı kappa
    ortalamasından değil, OOF birleştirmeden).

Çalıştırma:
    python experiments/exp_fbcsp_rlda.py            # 9 deneğin hepsi
    python experiments/exp_fbcsp_rlda.py --subjects 1 3 7
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

# Modülü doğrudan çalıştırınca proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import load_subject  # noqa: E402
from evaluation import run_nested_cv  # noqa: E402

# MNE'nin CSP fit sırasındaki "Estimating class=... covariance" gibi
# çıktılarını sustur — denek başına yüzlerce satır basıyor.
import mne  # noqa: E402
mne.set_log_level("ERROR")

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OOF_DIR = PROJECT_ROOT / "results" / "oof" / "fbcsp_rlda"
OOF_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Filter bank tanımı                                                          #
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
        Filter bank'in alt ve üst kenarı (Hz).
    widths_steps : sequence of (width, step)
        Her eleman bir bant genişliği ve adım büyüklüğü tanımlar.
        Varsayılan: 4 Hz / step 2 (10 bant) + 6 Hz / step 3 (6 bant) = 16 bant.

    Returns
    -------
    list of (l_freq, h_freq) tuple.
    """
    bands: List[Tuple[float, float]] = []
    eps = 1e-9
    for width, step in widths_steps:
        f = f_low
        while f + width <= f_high + eps:
            bands.append((round(f, 4), round(f + width, 4)))
            f += step
    return bands


# --------------------------------------------------------------------------- #
# Çoklu-bant filtre tensörü hesaplama                                         #
# --------------------------------------------------------------------------- #


def _design_bandpass(l_freq: float, h_freq: float, sfreq: float, order: int = 4):
    """Butterworth bant geçiren filtre katsayıları."""
    nyq = 0.5 * sfreq
    return butter(order, [l_freq / nyq, h_freq / nyq], btype="band")


def make_multiband_tensor(
    X: np.ndarray,
    sfreq: float,
    bands: Sequence[Tuple[float, float]],
    order: int = 4,
) -> np.ndarray:
    """X'in her bant için filtrelenmiş halini bir tensörde topla.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    sfreq : float
    bands : list of (l_freq, h_freq)
    order : int
        Butterworth derecesi. filtfilt ile efektif derece 2*order.

    Returns
    -------
    ndarray, shape (n_trials, n_bands, n_channels, n_samples), float32.

    Notes
    -----
    Bu adım veri-bağımsız (filtre katsayıları sadece sfreq ve frekans
    aralığına bağlı), dolayısıyla CV dışında bir kez hesaplanmasında
    leakage yoktur.
    """
    n_trials, n_channels, n_samples = X.shape
    out = np.empty((n_trials, len(bands), n_channels, n_samples), dtype=np.float32)
    for bi, (l, h) in enumerate(bands):
        b, a = _design_bandpass(l, h, sfreq, order=order)
        # filtfilt son eksen boyunca çalışır → zaman ekseni doğru.
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32, copy=False)
    return out


# --------------------------------------------------------------------------- #
# Multi-band CSP transformer (sklearn-uyumlu)                                 #
# --------------------------------------------------------------------------- #


class MultiBandCSP(BaseEstimator, TransformerMixin):
    """Her bant için ayrı bir CSP fit et, log-variance feature'ları concat et.

    Parameters
    ----------
    n_components : int
        Her bantta tutulacak CSP bileşeni sayısı (mne CSP n_components).
    reg : str | float | None
        mne.decoding.CSP'ye geçirilen düzenlileştirme. 'ledoit_wolf' önerilir.
    log : bool
        CSP feature'ları log-variance olarak ver.
    norm_trace : bool
        CSP filtrelerinin iz-normalize edilip edilmeyeceği.

    Notes
    -----
    - mne.decoding.CSP multiclass'ı joint approximate diagonalization ile
      destekler — OVR wrapper gerekmez.
    - sklearn Pipeline tarafından çağrılır; fit yalnızca train indeksleri
      üzerinde tetiklenir, böylece test fold'una leakage olmaz.
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
                f"MultiBandCSP X'in 4D olmasını bekler "
                f"(n_trials, n_bands, n_channels, n_samples), aldı: {X.shape}"
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
            # mne CSP epoch verisini float64 olarak ister, ama numpy
            # array kabul eder. Channels önce, sonra samples.
            csp.fit(X[:, bi].astype(np.float64, copy=False), y)
            self.csps_.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError("MultiBandCSP.transform 4D X bekler.")
        feats = [
            csp.transform(X[:, bi].astype(np.float64, copy=False))
            for bi, csp in enumerate(self.csps_)
        ]
        return np.concatenate(feats, axis=1).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Pipeline fabrikası                                                          #
# --------------------------------------------------------------------------- #


def build_pipeline() -> Pipeline:
    """Eğitime hazır taze pipeline döndür (her fold'da çağrılır)."""
    mi_score = partial(mutual_info_classif, random_state=RANDOM_STATE)
    return Pipeline(
        steps=[
            ("csp", MultiBandCSP(n_components=4, reg="ledoit_wolf")),
            ("select", SelectKBest(score_func=mi_score, k=20)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


#: GridSearchCV için iç-CV parametre ızgarası
PARAM_GRID = {
    "csp__n_components": [2, 4, 6],
    "select__k": [10, 20, 40, "all"],
}


# --------------------------------------------------------------------------- #
# Denek başına koşum                                                          #
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


def run_subject(
    subject_id: int,
    bands: Sequence[Tuple[float, float]],
    outer_splits: int = 5,
    inner_splits: int = 5,
    n_jobs: int = -1,
    verbose: bool = True,
) -> SubjectRunResult:
    """Tek denek için multi-band tensörü hazırla + nested CV çalıştır."""
    t0 = time.time()
    sd = load_subject(subject_id=subject_id, session="T", verbose=False)
    if sd.y is None:
        raise RuntimeError(f"A{subject_id:02d}T için etiket yok.")

    X_mb = make_multiband_tensor(sd.X, sfreq=sd.sfreq, bands=bands, order=4)
    y = sd.y

    if verbose:
        print(
            f"[A{subject_id:02d}T] data ready: X_mb={X_mb.shape}, y={y.shape}, "
            f"bands={len(bands)}",
            flush=True,
        )

    result = run_nested_cv(
        estimator_factory=build_pipeline,
        param_grid=PARAM_GRID,
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
            f"[A{subject_id:02d}T] {result.summary()}  "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    # OOF kaydet — ensemble icin (Adim B)
    np.savez(
        OOF_DIR / f"sub_{subject_id:02d}.npz",
        y_true=result.y_true,
        y_pred=result.y_pred,
        y_proba=(result.y_proba if result.y_proba is not None else np.empty(0)),
    )

    return SubjectRunResult(
        subject_id=subject_id,
        kappa=result.kappa,
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        kappa_std=float(np.std(result.fold_kappas)),
        n_trials=len(y),
        elapsed_sec=elapsed,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="FBCSP + sLDA nested CV.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", type=str, default="exp_fbcsp_rlda.csv")
    args = parser.parse_args()

    bands = generate_bands()
    print(f"Filter bank: {len(bands)} bands")
    for l, h in bands:
        print(f"  {l:>5.1f} – {h:>5.1f} Hz")
    print()

    rows: List[SubjectRunResult] = []
    t_total = time.time()
    for sid in args.subjects:
        res = run_subject(
            subject_id=sid,
            bands=bands,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            n_jobs=args.n_jobs,
            verbose=True,
        )
        rows.append(res)

    total_elapsed = time.time() - t_total

    # ---- Tablo ----
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

    # Ortalama satırı
    mean_row = {
        "subject": "MEAN",
        "cv_kappa": df["cv_kappa"].mean(),
        "cv_acc": df["cv_acc"].mean(),
        "macro_f1": df["macro_f1"].mean(),
        "kappa_std": df["cv_kappa"].std(),  # denekler arası std
        "n_trials": df["n_trials"].sum(),
        "elapsed_sec": total_elapsed,
    }
    df_out = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    out_path = RESULTS_DIR / args.output
    df_out.to_csv(out_path, index=False)

    print()
    print("=" * 70)
    print(f"FBCSP + sLDA — nested CV ({args.outer_splits}x{args.inner_splits})")
    print("=" * 70)
    print(df_out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"Mean kappa:    {df['cv_kappa'].mean():.4f}  (std across subjects: {df['cv_kappa'].std():.4f})")
    print(f"Mean accuracy: {df['cv_acc'].mean():.4f}")
    print(f"Total elapsed: {total_elapsed/60:.1f} min")
    print(f"Saved: {out_path}")

    # Düşük kappa flagleme
    threshold = 0.5
    low = df[df["cv_kappa"] < threshold]
    if len(low):
        print()
        print(f"[!] Dusuk denek (kappa < {threshold}):")
        for _, r in low.iterrows():
            print(f"   A{int(r['subject']):02d}: kappa={r['cv_kappa']:.4f}")


if __name__ == "__main__":
    main()

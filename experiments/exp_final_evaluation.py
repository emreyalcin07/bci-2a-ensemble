"""
exp_final_evaluation.py
=======================

Faz 6: FINAL DEGERLENDIRME — A0XE saklı test seti üzerinde tek-shot tahmin.

Protokol (her denek için, sirasiyla):
    1. A0XT (288 trial) yükle → tüm veri TRAIN, CV yok
    2. Final ensemble base'leri fit:
        - fbcsp_rlda                 (Faz 1)
        - riemann_multiband_ts_l1    (Faz 3c-L1)
        - eegnet                     (Faz 5a)
        - shallowconvnet             (Faz 5a)
       Klasik base'ler: tek GridSearchCV 5-fold A0XT'de → best params → refit
       (nested CV'deki best_params_per_fold'un mod'una denk; aynı param grid).
       DL: A0XT %80/%20 stratified train/val + early stop + augmentation train-only.
    3. A0XE (288 trial) yükle — ILK ve TEK kez. Sinyal: bandpass + ölçek; her
       base predict_proba.
    4. Soft vote (1.0/1.0/0.5/0.5) → final tahmin.
    5. A0XE true label ile κ, accuracy, confusion. + Faz 1 (sLDA tek) ve
       Faz 4 (FBCSP+sLDA + MB-TS-L1 eşit) referans tahminler.
    6. Modeller results/models/sub_NN/ altına kaydet.

LEAKAGE KONTROLU:
    - A0XE'nin sinyali HER deneğin tahmin adimında bir kez okunur, etiketi YALNIZCA
      en sondaki skorlamada kullanılır.
    - GridSearchCV, DL early-stopping, augmentation: hepsi A0XT'de.
    - A0XE'ye hiçbir noktada veri-bağımlı fit (scaler, PCA vs.) uygulanmaz —
      tüm transformer'lar A0XT'de fit edilir, A0XE'ye yalnızca transform.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump as joblib_dump
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(PROJECT_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Klasik base'ler
from data_loader import load_subject  # noqa: E402
from _fbcsp_common import generate_bands as fbcsp_generate_bands  # noqa: E402
from _fbcsp_common import make_multiband_tensor as fbcsp_multiband  # noqa: E402
from exp_fbcsp_rlda import PARAM_GRID as FBCSP_PARAM_GRID  # noqa: E402
from exp_fbcsp_rlda import build_pipeline as fbcsp_build_pipeline  # noqa: E402

from _riemann_common import bandpass_multi  # noqa: E402
from exp_riemann_multiband_ts import (  # noqa: E402
    BANDS as RIEMANN_BANDS,
    PARAM_GRID_L1 as RIEMANN_L1_PARAM_GRID,
    build_pipeline_l1 as riemann_l1_build_pipeline,
)

# DL base'leri
from _dl_common import (  # noqa: E402
    DEVICE,
    RANDOM_STATE,
    TrainConfig,
    bandpass_wide,
    train_one_fold,
)
from exp_eegnet import build_model as eegnet_build_model  # noqa: E402
from exp_shallowconvnet import build_model as scnet_build_model  # noqa: E402

import mne  # noqa: E402
mne.set_log_level("ERROR")

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WEAK_SUBJECTS = {2, 4, 5, 6}


# --------------------------------------------------------------------------- #
# Tek-base "fit on A0XT, predict on A0XE" helper'lari                         #
# --------------------------------------------------------------------------- #


def _fit_classical(
    pipeline_factory,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """A0XT'de GridSearchCV 5-fold → best_params → refit → A0XE predict_proba."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=pipeline_factory(),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        refit=True,
    )
    gs.fit(X_train, y_train)
    proba = gs.predict_proba(X_test)
    return proba, gs.best_estimator_, {
        "best_params": gs.best_params_,
        "best_inner_cv_score": float(gs.best_score_),
    }


def _fit_dl_with_state(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[np.ndarray, Dict[str, torch.Tensor], Dict[str, Any]]:
    """train_one_fold'un yaptıgini yapar AMA en iyi state_dict'i de döndürür.

    Mevcut train_one_fold helper'i state'i geri vermiyor; final değerlendirmede
    modeli kaydedebilmek için burada paralel implementasyon. Kod tekrarı,
    ama net.
    """
    from torch.utils.data import DataLoader
    from _dl_common import EEGDataset, _stratified_inner_split, set_seed

    set_seed(RANDOM_STATE)
    tr_idx, va_idx = _stratified_inner_split(y_train, cfg.val_fraction, RANDOM_STATE)
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    ds_tr = EEGDataset(X_tr, y_tr, augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(X_va, y_va, augment=False)
    ds_te = EEGDataset(X_test, np.zeros(len(X_test), dtype=np.int64), augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)

    model = model_factory().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience = 0

    for epoch in range(cfg.epochs):
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
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
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
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "n_epochs_trained": epoch + 1,
    }
    return y_proba, best_state, info


# --------------------------------------------------------------------------- #
# Per-subject runner                                                          #
# --------------------------------------------------------------------------- #


def evaluate_subject(subject_id: int, n_jobs: int = -1, save_models: bool = True) -> Dict[str, Any]:
    """Tek denek icin tüm boru hattını koştur, sonuç dict döndür."""
    t0 = time.time()
    print(f"\n{'='*78}\nA{subject_id:02d}  FINAL EVALUATION\n{'='*78}", flush=True)

    # ---- 1. A0XT load ----
    sd_t = load_subject(subject_id=subject_id, session="T", verbose=False)
    X_t = sd_t.X            # (288, 22, 500), float32 (Volt)
    y_t = sd_t.y
    print(f"  A0{subject_id}T  X={X_t.shape}  y={y_t.shape}  classes={dict(zip(*np.unique(y_t, return_counts=True)))}", flush=True)

    # ---- 2. A0XE load (FIRST AND ONLY) ----
    sd_e = load_subject(subject_id=subject_id, session="E", verbose=False)
    X_e = sd_e.X
    y_e = sd_e.y
    if y_e is None:
        raise RuntimeError(f"A0{subject_id}E etiket yüklenemedi (.mat eksik).")
    print(f"  A0{subject_id}E  X={X_e.shape}  y={y_e.shape}  classes={dict(zip(*np.unique(y_e, return_counts=True)))}", flush=True)

    sub_model_dir = MODELS_DIR / f"sub_{subject_id:02d}"
    if save_models:
        sub_model_dir.mkdir(parents=True, exist_ok=True)

    info_all: Dict[str, Any] = {"subject": subject_id}

    # ============ Base 1: FBCSP + sLDA ============
    print("  [1/4] fbcsp_rlda ...", flush=True)
    t1 = time.time()
    fbcsp_bands = fbcsp_generate_bands()
    X_t_fb = fbcsp_multiband(X_t, sfreq=sd_t.sfreq, bands=fbcsp_bands, order=4)
    X_e_fb = fbcsp_multiband(X_e, sfreq=sd_e.sfreq, bands=fbcsp_bands, order=4)
    proba_fbcsp, model_fbcsp, info_fbcsp = _fit_classical(
        pipeline_factory=fbcsp_build_pipeline,
        param_grid=FBCSP_PARAM_GRID,
        X_train=X_t_fb, y_train=y_t, X_test=X_e_fb,
        n_jobs=n_jobs,
    )
    info_all["fbcsp_best_params"] = info_fbcsp["best_params"]
    info_all["fbcsp_inner_cv_score"] = info_fbcsp["best_inner_cv_score"]
    if save_models:
        joblib_dump(model_fbcsp, sub_model_dir / "fbcsp_rlda.joblib")
    print(f"        best_params={info_fbcsp['best_params']}  inner_cv={info_fbcsp['best_inner_cv_score']:.4f}  {time.time()-t1:.1f}s", flush=True)

    # ============ Base 2: Riemann Multi-band TS + L1-LR ============
    print("  [2/4] riemann_multiband_ts_l1 ...", flush=True)
    t1 = time.time()
    X_t_rm = bandpass_multi(X_t, sfreq=sd_t.sfreq, bands=RIEMANN_BANDS, order=4)
    X_e_rm = bandpass_multi(X_e, sfreq=sd_e.sfreq, bands=RIEMANN_BANDS, order=4)
    proba_riemann, model_riemann, info_riemann = _fit_classical(
        pipeline_factory=riemann_l1_build_pipeline,
        param_grid=RIEMANN_L1_PARAM_GRID,
        X_train=X_t_rm, y_train=y_t, X_test=X_e_rm,
        n_jobs=n_jobs,
    )
    info_all["riemann_best_params"] = info_riemann["best_params"]
    info_all["riemann_inner_cv_score"] = info_riemann["best_inner_cv_score"]
    if save_models:
        joblib_dump(model_riemann, sub_model_dir / "riemann_multiband_ts_l1.joblib")
    print(f"        best_params={info_riemann['best_params']}  inner_cv={info_riemann['best_inner_cv_score']:.4f}  {time.time()-t1:.1f}s", flush=True)

    # DL ön-isleme: 4-38 Hz + uV
    X_t_dl = bandpass_wide(X_t, sd_t.sfreq) * 1e6
    X_e_dl = bandpass_wide(X_e, sd_e.sfreq) * 1e6

    dl_cfg = TrainConfig(
        epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-2,
        early_stopping_patience=30, augment=True, verbose=False,
    )

    # ============ Base 3: EEGNet ============
    print("  [3/4] eegnet ...", flush=True)
    t1 = time.time()
    proba_eeg, state_eeg, info_eeg = _fit_dl_with_state(
        model_factory=eegnet_build_model,
        X_train=X_t_dl, y_train=y_t, X_test=X_e_dl,
        cfg=dl_cfg,
    )
    info_all["eegnet_best_epoch"] = info_eeg["best_epoch"]
    info_all["eegnet_best_val_loss"] = info_eeg["best_val_loss"]
    info_all["eegnet_epochs_trained"] = info_eeg["n_epochs_trained"]
    if save_models:
        torch.save(state_eeg, sub_model_dir / "eegnet.pt")
    print(f"        best@{info_eeg['best_epoch']}  val_loss={info_eeg['best_val_loss']:.4f}  epochs={info_eeg['n_epochs_trained']}  {time.time()-t1:.1f}s", flush=True)

    # ============ Base 4: ShallowConvNet ============
    print("  [4/4] shallowconvnet ...", flush=True)
    t1 = time.time()
    proba_scn, state_scn, info_scn = _fit_dl_with_state(
        model_factory=scnet_build_model,
        X_train=X_t_dl, y_train=y_t, X_test=X_e_dl,
        cfg=dl_cfg,
    )
    info_all["scnet_best_epoch"] = info_scn["best_epoch"]
    info_all["scnet_best_val_loss"] = info_scn["best_val_loss"]
    info_all["scnet_epochs_trained"] = info_scn["n_epochs_trained"]
    if save_models:
        torch.save(state_scn, sub_model_dir / "shallowconvnet.pt")
    print(f"        best@{info_scn['best_epoch']}  val_loss={info_scn['best_val_loss']:.4f}  epochs={info_scn['n_epochs_trained']}  {time.time()-t1:.1f}s", flush=True)

    # ============ Per-base κ (sınıflandırma sağlık kontrolü) ============
    base_probas = {
        "fbcsp_rlda": proba_fbcsp,
        "riemann_multiband_ts_l1": proba_riemann,
        "eegnet": proba_eeg,
        "shallowconvnet": proba_scn,
    }
    base_kappas = {}
    for name, p in base_probas.items():
        pred = p.argmax(axis=1)
        base_kappas[name] = float(cohen_kappa_score(y_e, pred))
        info_all[f"{name}_test_kappa"] = base_kappas[name]
    print(f"  per-base test κ: {base_kappas}", flush=True)

    # ============ Soft-vote ensembles ============
    def soft_vote(probas: List[np.ndarray], weights: List[float]) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        stacked = np.stack(probas, axis=0)
        avg = np.tensordot(w, stacked, axes=([0], [0]))
        return avg.argmax(axis=1)

    # 1. Faz 1 referans (fbcsp_rlda tek)
    pred_faz1 = proba_fbcsp.argmax(axis=1)
    # 2. Faz 4 ensemble (fbcsp_rlda + riemann_multiband_ts_l1 esit)
    pred_faz4 = soft_vote([proba_fbcsp, proba_riemann], [1.0, 1.0])
    # 3. Faz 5a/6 final lider: fbcsp + riemann + eegnet + scnet, 1/1/0.5/0.5
    pred_final = soft_vote(
        [proba_fbcsp, proba_riemann, proba_eeg, proba_scn],
        [1.0, 1.0, 0.5, 0.5],
    )

    def score(pred):
        return {
            "kappa": float(cohen_kappa_score(y_e, pred)),
            "acc": float(accuracy_score(y_e, pred)),
            "macro_f1": float(f1_score(y_e, pred, average="macro")),
        }

    s_faz1 = score(pred_faz1)
    s_faz4 = score(pred_faz4)
    s_final = score(pred_final)
    cm_final = confusion_matrix(y_e, pred_final).tolist()

    info_all.update({
        "faz1_test_kappa":  s_faz1["kappa"],
        "faz1_test_acc":    s_faz1["acc"],
        "faz4_test_kappa":  s_faz4["kappa"],
        "faz4_test_acc":    s_faz4["acc"],
        "final_test_kappa": s_final["kappa"],
        "final_test_acc":   s_final["acc"],
        "final_macro_f1":   s_final["macro_f1"],
        "final_confusion":  cm_final,
        "elapsed_sec":      time.time() - t0,
    })

    print(
        f"  TEST κ: Faz1={s_faz1['kappa']:.4f}  Faz4={s_faz4['kappa']:.4f}  "
        f"FINAL={s_final['kappa']:.4f}  acc={s_final['acc']:.4f}",
        flush=True,
    )
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)
    return info_all


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def _load_cv_kappas() -> Dict[str, Dict[int, float]]:
    """results/oof/<base>/sub_NN.npz dosyalarindan denek-bazli OOF κ'lari topla.

    Bu CV (A0XT) κ ile Test (A0XE) κ arasinda karsilastirma icin kullanilir.
    """
    OOF = PROJECT_ROOT / "results" / "oof"
    out: Dict[str, Dict[int, float]] = {}
    for base in ["fbcsp_rlda", "riemann_multiband_ts_l1", "eegnet", "shallowconvnet"]:
        d = {}
        for sid in range(1, 10):
            p = OOF / base / f"sub_{sid:02d}.npz"
            if not p.exists():
                continue
            z = np.load(p)
            d[sid] = float(cohen_kappa_score(z["y_true"], z["y_pred"]))
        out[base] = d
    # Final ensemble (Faz 5a lider): yeniden hesapla
    final_kappas = {}
    for sid in range(1, 10):
        try:
            ps = []
            yt = None
            for base, w in zip(
                ["fbcsp_rlda", "riemann_multiband_ts_l1", "eegnet", "shallowconvnet"],
                [1.0, 1.0, 0.5, 0.5],
            ):
                z = np.load(OOF / base / f"sub_{sid:02d}.npz")
                ps.append(z["y_proba"] * w)
                yt = z["y_true"]
            avg = np.sum(ps, axis=0)
            pred = avg.argmax(axis=1)
            final_kappas[sid] = float(cohen_kappa_score(yt, pred))
        except Exception:
            pass
    out["final_ensemble_cv"] = final_kappas
    return out


def main():
    parser = argparse.ArgumentParser(description="Final A0XE evaluation.")
    parser.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 10)))
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # CV κ'lari (referans)
    cv_kappas = _load_cv_kappas()

    rows: List[Dict[str, Any]] = []
    t_total = time.time()
    for sid in args.subjects:
        info = evaluate_subject(sid, n_jobs=args.n_jobs, save_models=not args.no_save_models)
        # CV κ'lari ekle
        info["cv_kappa_fbcsp"] = cv_kappas.get("fbcsp_rlda", {}).get(sid)
        info["cv_kappa_riemann"] = cv_kappas.get("riemann_multiband_ts_l1", {}).get(sid)
        info["cv_kappa_eegnet"] = cv_kappas.get("eegnet", {}).get(sid)
        info["cv_kappa_scnet"] = cv_kappas.get("shallowconvnet", {}).get(sid)
        info["cv_kappa_final_ensemble"] = cv_kappas.get("final_ensemble_cv", {}).get(sid)
        rows.append(info)
    total_elapsed = time.time() - t_total

    # ===== Tablo ve özet =====
    df = pd.DataFrame(rows)

    # Ana özet tablosu (per denek)
    summary_cols = [
        "subject",
        "cv_kappa_fbcsp", "faz1_test_kappa",
        "cv_kappa_final_ensemble", "final_test_kappa", "final_test_acc",
        "faz4_test_kappa",
        "fbcsp_best_params", "riemann_best_params",
        "elapsed_sec",
    ]
    df_summary = df[summary_cols].copy()
    df_summary["delta_final"] = (
        df_summary["final_test_kappa"] - df_summary["cv_kappa_final_ensemble"]
    )
    df_summary["delta_faz1"] = (
        df_summary["faz1_test_kappa"] - df_summary["cv_kappa_fbcsp"]
    )

    # Toplam ortalama satırı
    mean_row = {c: np.nan for c in df_summary.columns}
    mean_row["subject"] = "MEAN"
    for c in [
        "cv_kappa_fbcsp", "faz1_test_kappa",
        "cv_kappa_final_ensemble", "final_test_kappa", "final_test_acc",
        "faz4_test_kappa", "delta_final", "delta_faz1",
    ]:
        mean_row[c] = float(df_summary[c].mean())
    mean_row["elapsed_sec"] = total_elapsed
    df_out = pd.concat([df_summary, pd.DataFrame([mean_row])], ignore_index=True)

    out_csv = TABLES_DIR / "final_evaluation.csv"
    df_out.to_csv(out_csv, index=False, encoding="utf-8")

    # Per-denek confusion'i ayri JSON kaydet (cok büyük CSV olmasin)
    cm_records = {int(r["subject"]): r["final_confusion"] for r in rows}
    (TABLES_DIR / "final_confusion_matrices.json").write_text(
        json.dumps(cm_records, indent=2), encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("FINAL DEGERLENDIRME — A0XE saklı test seti (ILK ve TEK kullanım)")
    print("=" * 78)
    cols_show = [
        "subject",
        "cv_kappa_fbcsp", "faz1_test_kappa",
        "cv_kappa_final_ensemble", "final_test_kappa",
        "delta_final", "final_test_acc",
    ]
    print(df_out[cols_show].to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v),
    ))
    print()
    print(f"FINAL mean κ = {mean_row['final_test_kappa']:.4f}  acc = {mean_row['final_test_acc']:.4f}")
    print(f"Faz1 (FBCSP+sLDA tek) mean κ = {mean_row['faz1_test_kappa']:.4f}")
    print(f"Faz4 (klasik ensemble) mean κ = {mean_row['faz4_test_kappa']:.4f}")
    print(f"BCI Comp IV 2a kazananı (Kai Keng Ang, FBCSP) yarışma mean κ = 0.57 (referans)")
    print(f"Total elapsed: {total_elapsed/60:.1f} min")
    print(f"\nSaved:")
    print(f"  {out_csv}")
    print(f"  {TABLES_DIR / 'final_confusion_matrices.json'}")
    print(f"  {MODELS_DIR}/sub_NN/  (her denek icin .joblib + .pt)")


if __name__ == "__main__":
    main()

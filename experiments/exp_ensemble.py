"""
exp_ensemble.py
===============

Leakage-free soft-voting ensemble. Tüm base modeller AYNI fold yapisi ile
(StratifiedKFold(5, shuffle=True, random_state=42)) nested CV koşmuş olmali;
out-of-fold (OOF) tahminleri trial-by-trial hizali bekleniyor.

Base'ler (results/oof/<base>/sub_NN.npz olarak okunur):
    - fbcsp_rlda                (Faz 1: FBCSP + sLDA)
    - riemann_ts_1band          (Faz 3a: TS + LR, tek bant)
    - riemann_multiband_ts_pca  (Faz 3c PCA varyanti)
    - riemann_multiband_ts_l1   (Faz 3c L1 varyanti)

Kombinasyonlar 1-5: kullaniciin tanimladigi sabit aǧirlikli soft-vote'lar.
Kombinasyon 6 (per-subject adaptif) SKIPLENDI: her base icin ayri inner-CV
skoru saklamadik, OOF κ'siyla seçim leakage olur.

Cikti: results/tables/exp_ensemble.csv — her kombinasyon bir satir.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = PROJECT_ROOT / "results" / "oof"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

# Sabit base label haritasi -> dizin adi.
# DL base'leri (eegnet, shallowconvnet) opsiyonel — OOF dosyasi yoksa atlanir.
BASE_DIRS = {
    "fbcsp_rlda":               "Faz1: FBCSP+sLDA",
    "riemann_ts_1band":         "Faz3a: TS 1-band",
    "riemann_multiband_ts_pca": "Faz3c: MB-TS PCA",
    "riemann_multiband_ts_l1":  "Faz3c: MB-TS L1",
    "eegnet":                   "Faz5a: EEGNet",
    "shallowconvnet":           "Faz5a: ShallowConvNet",
    "eegnet_transfer":          "Faz5b: EEGNet LOSO transfer",
    "shallowconvnet_transfer":  "Faz5b: ShallowConvNet LOSO transfer",
    "atcnet":                   "Faz6: ATCNet (pre-norm)",
    "eegnet_10seed":            "Faz11: EEGNet 10-seed",
    "shallowconvnet_10seed":    "Faz11: ShallowConvNet 10-seed",
    "atcnet_10seed":            "Faz11: ATCNet 10-seed",
}

WEAK_SUBJECTS = {2, 4, 5, 6}
ALL_SUBJECTS = list(range(1, 10))


# --------------------------------------------------------------------------- #
# OOF I/O                                                                     #
# --------------------------------------------------------------------------- #


def load_oof(base: str, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_proba) for given base + subject."""
    path = OOF_DIR / base / f"sub_{subject_id:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"OOF dosyasi yok: {path}")
    npz = np.load(path)
    y_true = npz["y_true"]
    y_proba = npz["y_proba"]
    if y_proba.size == 0:
        raise RuntimeError(f"y_proba bos: {path} (base predict_proba kaydetmemis)")
    return y_true, y_proba


def verify_alignment(bases: Sequence[str]) -> bool:
    """Tüm base'lerin tüm denek y_true vektörlerinin esit oldugunu dogrula."""
    ref_y: Dict[int, np.ndarray] = {}
    for base in bases:
        for sid in ALL_SUBJECTS:
            y, _ = load_oof(base, sid)
            if sid not in ref_y:
                ref_y[sid] = y
            elif not np.array_equal(ref_y[sid], y):
                print(
                    f"[FAIL] alignment: base={base} sub={sid} y_true farkli "
                    f"(fold yapisi bozulmus olabilir!)"
                )
                return False
    return True


# --------------------------------------------------------------------------- #
# Soft voting                                                                 #
# --------------------------------------------------------------------------- #


def soft_vote(
    proba_list: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Soft-vote: aǧirlikli ortalama olasilik -> argmax."""
    if weights is None:
        w = np.ones(len(proba_list)) / len(proba_list)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    stacked = np.stack(proba_list, axis=0)  # (n_models, n_trials, n_classes)
    avg = np.tensordot(w, stacked, axes=([0], [0]))  # (n_trials, n_classes)
    return avg.argmax(axis=1)


def evaluate_combo(
    bases: Sequence[str],
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Tüm 9 denek icin soft-vote sonucu döndür; denek-satirli DataFrame."""
    rows = []
    for sid in ALL_SUBJECTS:
        y_ref = None
        probas = []
        for base in bases:
            y_true, y_proba = load_oof(base, sid)
            if y_ref is None:
                y_ref = y_true
            probas.append(y_proba)
        y_pred = soft_vote(probas, weights)
        rows.append({
            "subject": sid,
            "kappa": cohen_kappa_score(y_ref, y_pred),
            "acc": accuracy_score(y_ref, y_pred),
            "macro_f1": f1_score(y_ref, y_pred, average="macro"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Faz 9: Dynamic (confidence-weighted) soft voting                            #
# --------------------------------------------------------------------------- #


def dynamic_soft_vote(
    proba_list: Sequence[np.ndarray],
    global_weights: Optional[Sequence[float]] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Hybrid dynamic combination: global × per-trial confidence.

    Her trial için her base'in confidence skoru = negatif entropy.
    Per-trial weight = softmax(confidence / temperature) — temperature
    keskinligi kontrol eder.

    Final weight = global_weight × per_trial_weight (normalized).

    Args:
        proba_list: List of (n_trials, n_classes) probability matrices.
        global_weights: Per-base global weights (len = n_bases). None ise esit.
        temperature: T=0 limit en guvenli base hard-selection; T=inf esit
                     soft voting. Default T=1.0.

    Returns:
        Predicted labels (n_trials,).
    """
    n_bases = len(proba_list)
    if global_weights is None:
        gw = np.ones(n_bases) / n_bases
    else:
        gw = np.asarray(global_weights, dtype=np.float64)
        gw = gw / gw.sum()

    stacked = np.stack(proba_list, axis=0)  # (n_bases, n_trials, n_classes)

    # Per-trial confidence: negatif entropy (yuksek = guvenli)
    eps = 1e-12
    log_probs = np.log(stacked + eps)
    entropy = -np.sum(stacked * log_probs, axis=2)  # (n_bases, n_trials)
    confidence = -entropy  # daha yuksek = daha guvenli

    # Per-trial weight: softmax(confidence / T) across bases
    if temperature <= 0:
        # T=0 limit: hard selection of most confident base
        per_trial_w = np.zeros_like(confidence)
        best = confidence.argmax(axis=0)
        per_trial_w[best, np.arange(confidence.shape[1])] = 1.0
    else:
        scaled = confidence / temperature
        # Numerical stability: subtract max
        scaled = scaled - scaled.max(axis=0, keepdims=True)
        exp_scaled = np.exp(scaled)
        per_trial_w = exp_scaled / exp_scaled.sum(axis=0, keepdims=True)

    # Combine: global * per_trial, normalize per trial
    combined_w = gw[:, np.newaxis] * per_trial_w  # (n_bases, n_trials)
    combined_w = combined_w / combined_w.sum(axis=0, keepdims=True)

    # Weighted average of probabilities
    avg = np.einsum("bt,btc->tc", combined_w, stacked)
    return avg.argmax(axis=1)


def evaluate_combo_dynamic(
    bases: Sequence[str],
    global_weights: Optional[Sequence[float]] = None,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Dynamic combination ile her denek icin sonuc DataFrame'i."""
    rows = []
    for sid in ALL_SUBJECTS:
        y_ref = None
        probas = []
        for base in bases:
            y_true, y_proba = load_oof(base, sid)
            if y_ref is None:
                y_ref = y_true
            probas.append(y_proba)
        y_pred = dynamic_soft_vote(probas, global_weights, temperature)
        rows.append({
            "subject": sid,
            "kappa": cohen_kappa_score(y_ref, y_pred),
            "acc": accuracy_score(y_ref, y_pred),
            "macro_f1": f1_score(y_ref, y_pred, average="macro"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Per-subject base κ (referans tablosu, leakage-safe sadece raporlama)        #
# --------------------------------------------------------------------------- #


def per_subject_base_kappa() -> pd.DataFrame:
    rows = []
    for base in BASE_DIRS:
        for sid in ALL_SUBJECTS:
            y_true, y_proba = load_oof(base, sid)
            y_pred = y_proba.argmax(axis=1)
            rows.append({
                "base": base,
                "subject": sid,
                "kappa": cohen_kappa_score(y_true, y_pred),
            })
    return pd.DataFrame(rows).pivot(index="subject", columns="base", values="kappa")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main():
    # Sadece OOF dosyasi MEVCUT olan base'leri kullan (DL henuz koşmamissa atla)
    bases_all = []
    for b in BASE_DIRS:
        path = OOF_DIR / b / "sub_01.npz"
        if path.exists():
            bases_all.append(b)
        else:
            print(f"[skip] {b} OOF yok ({path})")
    if len(bases_all) < 2:
        print("[ABORT] en az 2 base gerekli.")
        sys.exit(1)
    print(f"Aktif base'ler: {bases_all}\n")

    # 1. Hizalama dogrulamasi
    print("Fold alignment check ...")
    if not verify_alignment(bases_all):
        print("[ABORT] alignment failure")
        sys.exit(1)
    print("  OK: tüm base'lerin y_true vektörleri 9 denek icin esit.\n")

    # 2. Referans: her base'in tek başina OOF κ'si (denek bazinda)
    base_kappa = per_subject_base_kappa()
    print("Per-subject base κ (OOF):")
    print(base_kappa.round(4).to_string())
    print()

    # 3. CV κ orantili aǧirlik (genel ortalamaya göre)
    base_mean = {b: base_kappa[b].mean() for b in bases_all}
    print("Base mean OOF κ:")
    for b, k in base_mean.items():
        print(f"  {b:30s} {k:.4f}")
    print()

    # 4. Kombinasyonlar — mevcut base'lere gore dinamik kuruluyor
    has_eegnet = "eegnet" in bases_all
    has_scnet = "shallowconvnet" in bases_all
    has_dl = has_eegnet or has_scnet

    BEST_CLASSIC = ["fbcsp_rlda", "riemann_multiband_ts_l1"]  # Faz 4 lider

    combos: List[Tuple[str, List[str], Optional[List[float]]]] = []

    # ===== Klasik referans set (Faz 4'ten) =====
    combos.extend([
        ("FBCSP+sLDA + MB-TS-PCA (esit)",
         ["fbcsp_rlda", "riemann_multiband_ts_pca"], None),
        ("FBCSP+sLDA + MB-TS-L1 (esit) [Faz4 lider]",
         ["fbcsp_rlda", "riemann_multiband_ts_l1"], None),
        ("FBCSP+sLDA + MB-TS-PCA + MB-TS-L1 (esit)",
         ["fbcsp_rlda", "riemann_multiband_ts_pca", "riemann_multiband_ts_l1"], None),
        ("4 klasik base (esit)",
         ["fbcsp_rlda", "riemann_ts_1band",
          "riemann_multiband_ts_pca", "riemann_multiband_ts_l1"], None),
    ])

    # Transfer (Faz 5b) durumu
    has_eegnet_tr = "eegnet_transfer" in bases_all
    has_scnet_tr = "shallowconvnet_transfer" in bases_all
    has_tr = has_eegnet_tr or has_scnet_tr

    # ===== Faz 5a DL kombinasyonlari =====
    if has_eegnet:
        combos.append((
            "Faz4-lider + EEGNet (esit, 3-lu)",
            BEST_CLASSIC + ["eegnet"], None,
        ))
        combos.append((
            "FBCSP+sLDA + EEGNet (esit)",
            ["fbcsp_rlda", "eegnet"], None,
        ))
    if has_scnet:
        combos.append((
            "Faz4-lider + ShallowConvNet (esit, 3-lu)",
            BEST_CLASSIC + ["shallowconvnet"], None,
        ))
        combos.append((
            "FBCSP+sLDA + ShallowConvNet (esit)",
            ["fbcsp_rlda", "shallowconvnet"], None,
        ))
    if has_eegnet and has_scnet:
        combos.append((
            "Faz4-lider + EEGNet + ShallowConvNet (esit, 4-lu)",
            BEST_CLASSIC + ["eegnet", "shallowconvnet"], None,
        ))
        combos.append((
            "Tum 6 base (esit)",
            bases_all, None,
        ))
        # Agirlikli: klasik gucluler 1.0, DL 0.5 (orta agirlik)
        w = [1.0, 1.0, 0.5, 0.5]
        combos.append((
            "Faz4-lider + EEGNet + ShallowConvNet (1/1/0.5/0.5)",
            BEST_CLASSIC + ["eegnet", "shallowconvnet"], w,
        ))
        # Agresif klasik agirlik
        w2 = [1.5, 1.0, 0.5, 0.5]
        combos.append((
            "Faz4-lider + EEGNet + ShallowConvNet (1.5/1/0.5/0.5)",
            BEST_CLASSIC + ["eegnet", "shallowconvnet"], w2,
        ))
        # DL'e dusuk agirlik (klasige guven)
        w3 = [1.0, 1.0, 0.3, 0.3]
        combos.append((
            "Faz4-lider + EEGNet + ShallowConvNet (1/1/0.3/0.3)",
            BEST_CLASSIC + ["eegnet", "shallowconvnet"], w3,
        ))

    # ===== Faz 5b transfer kombinasyonlari =====
    if has_eegnet_tr:
        combos.append((
            "Faz4-lider + EEGNet-transfer (esit, 3-lu)",
            BEST_CLASSIC + ["eegnet_transfer"], None,
        ))
    if has_scnet_tr:
        combos.append((
            "Faz4-lider + ShallowConvNet-transfer (esit, 3-lu)",
            BEST_CLASSIC + ["shallowconvnet_transfer"], None,
        ))
    if has_eegnet_tr and has_scnet_tr:
        # Sadece transfer DL
        combos.append((
            "Faz4-lider + EEGNet-tr + ShallowConvNet-tr (1/1/0.5/0.5)",
            BEST_CLASSIC + ["eegnet_transfer", "shallowconvnet_transfer"],
            [1.0, 1.0, 0.5, 0.5],
        ))
        combos.append((
            "Faz4-lider + EEGNet-tr + ShallowConvNet-tr (esit, 4-lu)",
            BEST_CLASSIC + ["eegnet_transfer", "shallowconvnet_transfer"], None,
        ))
        # Transfer + subject-specific DL birlikte
        if has_eegnet and has_scnet:
            combos.append((
                "Faz4-lider + 2 subj-DL + 2 transfer-DL (1/1/0.5/0.5/0.5/0.5)",
                BEST_CLASSIC + ["eegnet", "shallowconvnet",
                                "eegnet_transfer", "shallowconvnet_transfer"],
                [1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
            ))
            combos.append((
                "Tüm 8 base (esit)",
                bases_all, None,
            ))
            # Klasik 1.0, DL 0.3 (DL ağırlıği daha düşük — toplam DL etkisi 4×0.3=1.2)
            w_lower = [1.0, 1.0, 0.3, 0.3, 0.3, 0.3]
            combos.append((
                "Faz4-lider + 4 DL (1/1/0.3/0.3/0.3/0.3)",
                BEST_CLASSIC + ["eegnet", "shallowconvnet",
                                "eegnet_transfer", "shallowconvnet_transfer"],
                w_lower,
            ))

    # ===== Faz 6: ATCNet kombinasyonlari =====
    if "atcnet" in bases_all:
        combos.append((
            "Faz4-lider + ATCNet (esit, 3-lu)",
            BEST_CLASSIC + ["atcnet"], None,
        ))
        combos.append((
            "Faz4-lider + ATCNet (1/1/0.5)",
            BEST_CLASSIC + ["atcnet"], [1.0, 1.0, 0.5],
        ))
        # Onceki lider (Faz4-lider + EEGNet + ShallowConvNet 1/1/0.5/0.5) ile
        # tam karsilastirma: ATCNet'i 5. base olarak ekle
        if "eegnet" in bases_all and "shallowconvnet" in bases_all:
            combos.append((
                "Faz4-lider + EEGNet + ShallowConvNet + ATCNet (1/1/0.5/0.5/0.5)",
                BEST_CLASSIC + ["eegnet", "shallowconvnet", "atcnet"],
                [1.0, 1.0, 0.5, 0.5, 0.5],
            ))
            combos.append((
                "Faz4-lider + EEGNet + ShallowConvNet + ATCNet (1/1/0.5/0.5/0.3)",
                BEST_CLASSIC + ["eegnet", "shallowconvnet", "atcnet"],
                [1.0, 1.0, 0.5, 0.5, 0.3],
            ))

    # ===== Faz 11: 10-seed DL kombinasyonlari =====
    if all(b in bases_all for b in ["eegnet_10seed", "shallowconvnet_10seed"]):
        # 2 DL_10s — agirlik ablation (eski lider 1/1/0.5/0.5 idi, simdi yukseltiyoruz)
        for w in [0.5, 0.7, 1.0]:
            combos.append((
                f"Faz4 + EEGNet_10s + Shallow_10s (1/1/{w}/{w})",
                BEST_CLASSIC + ["eegnet_10seed", "shallowconvnet_10seed"],
                [1.0, 1.0, w, w],
            ))
        # Asimetrik: EEGNet daha yuksek (+0.080 > Shallow +0.038)
        combos.append((
            "Faz4 + EEGNet_10s + Shallow_10s (1/1/1.0/0.7) asymmetric",
            BEST_CLASSIC + ["eegnet_10seed", "shallowconvnet_10seed"],
            [1.0, 1.0, 1.0, 0.7],
        ))

        # 3 DL_10s ekli varyantlar
        if "atcnet_10seed" in bases_all:
            for w in [0.5, 0.7, 1.0]:
                combos.append((
                    f"Faz4 + EEGNet_10s + Shallow_10s + ATCNet_10s (1/1/{w}/{w}/{w})",
                    BEST_CLASSIC + ["eegnet_10seed", "shallowconvnet_10seed", "atcnet_10seed"],
                    [1.0, 1.0, w, w, w],
                ))
            # Asimetrik: EEGNet ve ATCNet esit yuksek (+0.080 ikisi de), Shallow daha dusuk (+0.038)
            combos.append((
                "Faz4 + EEGNet_10s + Shallow_10s + ATCNet_10s (1/1/1.0/0.7/1.0)",
                BEST_CLASSIC + ["eegnet_10seed", "shallowconvnet_10seed", "atcnet_10seed"],
                [1.0, 1.0, 1.0, 0.7, 1.0],
            ))
            # Tek basina FBCSP + 3 DL_10s (Riemann olmadan)
            combos.append((
                "FBCSP + 3 DL_10s (1/1/1/1)",
                ["fbcsp_rlda", "eegnet_10seed", "shallowconvnet_10seed", "atcnet_10seed"],
                [1.0, 1.0, 1.0, 1.0],
            ))

    # 5. Cozumle ve tabloyu olustur
    summary_rows = []
    per_subject_table = pd.DataFrame({"subject": ALL_SUBJECTS}).set_index("subject")

    for name, bases, weights in combos:
        df = evaluate_combo(bases, weights)
        mean_k = df["kappa"].mean()
        mean_a = df["acc"].mean()
        weak_k = df[df["subject"].isin(WEAK_SUBJECTS)]["kappa"].mean()
        strong_k = df[~df["subject"].isin(WEAK_SUBJECTS)]["kappa"].mean()
        a05 = float(df[df["subject"] == 5]["kappa"].iloc[0])
        summary_rows.append({
            "combo": name,
            "mean_kappa": round(mean_k, 4),
            "mean_acc":   round(mean_a, 4),
            "weak_kappa": round(weak_k, 4),
            "strong_kappa": round(strong_k, 4),
            "a05_kappa":  round(a05, 4),
            "n_bases": len(bases),
            "weights": str(weights) if weights else "equal",
        })
        per_subject_table[name] = df.set_index("subject")["kappa"].values.round(4)

    # ===== Faz 9: Dynamic combination =====
    DYNAMIC_BASES_5 = BEST_CLASSIC + ["eegnet", "shallowconvnet", "atcnet"]
    DYNAMIC_WEIGHTS_5 = [1.0, 1.0, 0.5, 0.5, 0.5]
    DYNAMIC_BASES_4 = BEST_CLASSIC + ["eegnet", "shallowconvnet"]
    DYNAMIC_WEIGHTS_4 = [1.0, 1.0, 0.5, 0.5]

    dynamic_combos = []
    if all(b in bases_all for b in DYNAMIC_BASES_5):
        for T in [0.5, 1.0, 2.0, 5.0]:
            dynamic_combos.append((
                f"DYNAMIC 5-base (1/1/0.5/0.5/0.5) T={T}",
                DYNAMIC_BASES_5, DYNAMIC_WEIGHTS_5, T,
            ))
    if all(b in bases_all for b in DYNAMIC_BASES_4):
        for T in [0.5, 1.0, 2.0, 5.0]:
            dynamic_combos.append((
                f"DYNAMIC 4-base (1/1/0.5/0.5) T={T}",
                DYNAMIC_BASES_4, DYNAMIC_WEIGHTS_4, T,
            ))

    for name, bases, weights, T in dynamic_combos:
        df = evaluate_combo_dynamic(bases, weights, T)
        mean_k = df["kappa"].mean()
        mean_a = df["acc"].mean()
        weak_k = df[df["subject"].isin(WEAK_SUBJECTS)]["kappa"].mean()
        strong_k = df[~df["subject"].isin(WEAK_SUBJECTS)]["kappa"].mean()
        a05 = float(df[df["subject"] == 5]["kappa"].iloc[0])
        summary_rows.append({
            "combo": name,
            "mean_kappa": round(mean_k, 4),
            "mean_acc": round(mean_a, 4),
            "weak_kappa": round(weak_k, 4),
            "strong_kappa": round(strong_k, 4),
            "a05_kappa": round(a05, 4),
            "n_bases": len(bases),
            "weights": f"{weights} | T={T}",
        })
        per_subject_table[name] = df.set_index("subject")["kappa"].values.round(4)

    # Faz 1 referans satiri
    faz1_df = pd.DataFrame({"subject": ALL_SUBJECTS}).set_index("subject")
    faz1_kappa = base_kappa["fbcsp_rlda"]
    per_subject_table.insert(0, "Faz1 sLDA (referans)", faz1_kappa.round(4).values)
    summary_rows.insert(0, {
        "combo": "Faz1 sLDA tek başina (referans)",
        "mean_kappa": round(faz1_kappa.mean(), 4),
        "mean_acc": float("nan"),
        "weak_kappa": round(faz1_kappa[list(WEAK_SUBJECTS)].mean(), 4),
        "strong_kappa": round(faz1_kappa[[s for s in ALL_SUBJECTS if s not in WEAK_SUBJECTS]].mean(), 4),
        "a05_kappa": round(faz1_kappa[5], 4),
        "n_bases": 1,
        "weights": "—",
    })

    summary = pd.DataFrame(summary_rows)
    out_path = RESULTS_DIR / "exp_ensemble.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8")

    print("=" * 95)
    print("Ensemble karsilastirma (sirali: mean_kappa azalan)")
    print("=" * 95)
    sorted_summary = summary.sort_values("mean_kappa", ascending=False)
    print(sorted_summary.to_string(index=False))
    print()
    print(f"Saved: {out_path}")

    # Per-subject tablo
    print()
    print("=" * 95)
    print("Per-subject κ (her satir bir denek, her kolon bir kombinasyon)")
    print("=" * 95)
    print(per_subject_table.to_string())
    per_subj_path = RESULTS_DIR / "exp_ensemble_per_subject.csv"
    per_subject_table.to_csv(per_subj_path, encoding="utf-8")
    print()
    print(f"Saved: {per_subj_path}")


if __name__ == "__main__":
    main()

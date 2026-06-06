"""
_make_report_figures.py
=======================

Rapor figurleri (3 adet, double-column geniliginde, TR etiketli):
    1. fig_cv_vs_test.png        — Denek bazli CV (A0XT) vs Test (A0XE) kappa
    2. fig_ablation.png          — Faz 1..6 mean kappa ilerleme cubugu
    3. fig_confusion_matrix.png  — A0XE final ensemble normalize CM (9 denek ort.)
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# Double-column genislik: tek sutun ~3.4 inch, double sutun ~7 inch
COL_W = 3.4
DBL_W = 7.0


# =========================================================================== #
# Figur 1: CV vs Test kappa
# =========================================================================== #
def fig_cv_vs_test():
    df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "final_evaluation.csv")
    df = df[df["subject"] != "MEAN"].copy()
    df["subject"] = df["subject"].astype(int)

    subs = df["subject"].tolist()
    cv = df["cv_kappa_final_ensemble"].astype(float).tolist()
    te = df["final_test_kappa"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(DBL_W, 2.5))
    x = np.arange(len(subs))
    w = 0.38
    bars_cv = ax.bar(x - w/2, cv, w, label="CV (A0XT)", color="#3a6ea5")
    bars_te = ax.bar(x + w/2, te, w, label="Test (A0XE)", color="#c0504d")

    # Yarisma kazanani referans cizgisi (mean)
    ax.axhline(0.569, color="gray", linestyle="--", linewidth=0.8,
               label="Yarisma kazanani (Ang FBCSP, 0.569)")
    # Bizim Test ortalama
    ax.axhline(np.mean(te), color="#c0504d", linestyle=":", linewidth=0.8,
               label=f"Test ort. ({np.mean(te):.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels([f"A0{s}" for s in subs])
    ax.set_ylabel("Cohen's $\\kappa$")
    ax.set_ylim(0, 1.0)
    ax.set_title("Denek bazli CV (A0XT, nested 5x5) vs Test (A0XE saklı set) kappa")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "fig_cv_vs_test.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


# =========================================================================== #
# Figur 2: Ablation ilerleme
# =========================================================================== #
def fig_ablation():
    # (label, mean kappa, kategori)
    data = [
        ("Faz 1\nFBCSP+sLDA",         0.6250, "klasik"),
        ("Faz 2a\nBandSel+SVM(MI)",   0.5787, "negatif"),
        ("Faz 2b\nBandSel+SVM(mRMR)", 0.5849, "negatif"),
        ("Faz 3a\nRiemann TS 1-band", 0.5303, "negatif"),
        ("Faz 3b\nRiemann MDM",       0.4635, "negatif"),
        ("Faz 3c-PCA\nMB-TS+PCA",     0.5417, "negatif"),
        ("Faz 3c-L1\nMB-TS+L1",       0.5499, "negatif"),
        ("Faz 4\nKlasik ens.",        0.6425, "ensemble"),
        ("Faz 5a\nEEGNet (CV)",       0.3976, "negatif"),
        ("Faz 5a\nSCN (CV)",          0.3971, "negatif"),
        ("Faz 5a ens.\n+ 2 DL (CV)",  0.6553, "ensemble"),
        ("Faz 5b\nEEGNet-tr (CV)",    0.3863, "negatif"),
        ("Faz 5b\nSCN-tr (CV)",       0.4213, "negatif"),
        ("Faz 6 Test\nA0XE final",    0.6137, "test"),
    ]
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    cats = [d[2] for d in data]
    color_map = {
        "klasik":   "#3a6ea5",
        "negatif":  "#a0a0a0",
        "ensemble": "#2e7d32",
        "test":     "#c0504d",
    }
    colors = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(DBL_W, 2.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors)

    # Referans cizgiler
    ax.axhline(0.569, color="black", linestyle="--", linewidth=0.7,
               label="Yarisma kazanani (0.569)")

    # Bar uzerine deger
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=6.5)
    ax.set_ylabel("Mean Cohen's $\\kappa$ (9 denek)")
    ax.set_ylim(0, 0.78)
    ax.set_title("Sistematik ablasyon — Faz 1'den final A0XE testine kadar")

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(color=color_map["klasik"],   label="Klasik baseline"),
        Patch(color=color_map["ensemble"], label="Ensemble (CV)"),
        Patch(color=color_map["negatif"],  label="Negatif sonuc (ablasyon)"),
        Patch(color=color_map["test"],     label="Final A0XE test"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    out = FIG_DIR / "fig_ablation.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


# =========================================================================== #
# Figur 3: A0XE final confusion matrix (9 denek, normalize satir)
# =========================================================================== #
def fig_confusion():
    cm_path = PROJECT_ROOT / "results" / "tables" / "final_confusion_matrices.json"
    if not cm_path.exists():
        print(f"[skip] {cm_path} yok")
        return
    cms = json.loads(cm_path.read_text(encoding="utf-8"))
    # Topla, sonra normalize (satir bazli)
    total = np.zeros((4, 4), dtype=float)
    for sid, cm in cms.items():
        total += np.array(cm, dtype=float)
    row_sum = total.sum(axis=1, keepdims=True)
    norm = total / np.maximum(row_sum, 1)

    class_names = ["Sol El", "Sag El", "Iki Ayak", "Dil"]

    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek sinif")
    ax.set_title("A0XE final ensemble — normalize CM (9 denek toplam)")

    for i in range(4):
        for j in range(4):
            c = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    color=c, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = FIG_DIR / "fig_confusion_matrix.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def main():
    fig_cv_vs_test()
    fig_ablation()
    fig_confusion()


if __name__ == "__main__":
    main()

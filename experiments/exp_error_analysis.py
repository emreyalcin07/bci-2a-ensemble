"""
exp_error_analysis.py
=====================

Faz 5a teshisi: DL ve klasik base'lerin yanlis sinifladigi trial'lar ne kadar
ortusuyor? Dusuk ortusme -> farkli hata profili -> ensemble'da deger var.

Metrikler (her denek icin, base ciftleri uzerinden):
    - n_err_A: A'nin yanlis sayisi
    - n_err_B: B'nin yanlis sayisi
    - both_wrong: ikisinin DE yanlisladigi trial sayisi
    - jaccard_err = both_wrong / (n_err_A + n_err_B - both_wrong)
                    [0 -> tamamen ayri hatalar; 1 -> ozdes hatalar]
    - disagreement_rate = (A yanlis xor B yanlis) / n_trials
                          [iki modelin farkli yerlerde yanlis yaptigi oran]

Ozellikle A05 icin DL <-> fbcsp_rlda kiyasi: DL gercekten klasiklerin
kaciirdigi farkli trial'lari yakaliyor mu?

Cikti:
    results/tables/error_overlap.csv     (her base cifti × her denek)
    results/tables/error_overlap_a05.csv (A05 odakli, kompakt)
    Stdout: kompakt ozet tablosu
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOF_DIR = PROJECT_ROOT / "results" / "oof"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

# Tum base'ler (mevcut olanlar otomatik tespit edilir)
BASES = [
    "fbcsp_rlda",
    "riemann_ts_1band",
    "riemann_multiband_ts_pca",
    "riemann_multiband_ts_l1",
    "eegnet",
    "shallowconvnet",
    "eegnet_transfer",
    "shallowconvnet_transfer",
]

ALL_SUBJECTS = list(range(1, 10))
WEAK_SUBJECTS = {2, 4, 5, 6}


def load_oof_pred(base: str, sid: int) -> Tuple[np.ndarray, np.ndarray]:
    path = OOF_DIR / base / f"sub_{sid:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"OOF yok: {path}")
    npz = np.load(path)
    return npz["y_true"], npz["y_pred"]


def overlap_metrics(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict:
    """A ve B'nin hata ortusme metrikleri."""
    wrong_a = (y_pred_a != y_true)
    wrong_b = (y_pred_b != y_true)
    both_wrong = wrong_a & wrong_b
    either_wrong = wrong_a | wrong_b
    xor_wrong = wrong_a ^ wrong_b

    n_a = int(wrong_a.sum())
    n_b = int(wrong_b.sum())
    n_both = int(both_wrong.sum())
    n_either = int(either_wrong.sum())

    jaccard = n_both / n_either if n_either > 0 else 1.0
    disagree = float(xor_wrong.mean())  # iki modelin tahmin uyusmazligi degil; *hatada* ayrildigi orani
    a_acc = float((~wrong_a).mean())
    b_acc = float((~wrong_b).mean())
    return {
        "n_err_A": n_a,
        "n_err_B": n_b,
        "both_wrong": n_both,
        "jaccard_err": jaccard,
        "disagreement_rate": disagree,
        "acc_A": a_acc,
        "acc_B": b_acc,
        "n_trials": int(len(y_true)),
    }


def main():
    # Tum bazlarin yuklenebilirligini kontrol et
    available = []
    for b in BASES:
        try:
            _ = load_oof_pred(b, 1)
            available.append(b)
        except FileNotFoundError as e:
            print(f"[uyari] {b} eksik: {e}")
    if len(available) < 2:
        print("[ABORT] en az 2 base gerekli.")
        sys.exit(1)
    print(f"Kullanilabilir base'ler: {available}\n")

    rows: List[Dict] = []
    for sid in ALL_SUBJECTS:
        # y_true ortak — ilk base'den al, sonra dogrula
        y_true_ref = None
        preds = {}
        for b in available:
            y_t, y_p = load_oof_pred(b, sid)
            if y_true_ref is None:
                y_true_ref = y_t
            else:
                if not np.array_equal(y_true_ref, y_t):
                    raise RuntimeError(f"y_true mismatch: {b} sub_{sid:02d}")
            preds[b] = y_p

        for i, a in enumerate(available):
            for b in available[i + 1:]:
                m = overlap_metrics(y_true_ref, preds[a], preds[b])
                rows.append({
                    "subject": sid,
                    "base_A": a,
                    "base_B": b,
                    **m,
                })

    df = pd.DataFrame(rows)
    out_full = RESULTS_DIR / "error_overlap.csv"
    df.to_csv(out_full, index=False, encoding="utf-8")
    print(f"Saved: {out_full}\n")

    # Kompakt ozet: 9 denek ortalamasi pairwise jaccard_err matrisi
    pivot = df.pivot_table(
        index="base_A", columns="base_B",
        values="jaccard_err", aggfunc="mean",
    )
    print("=" * 80)
    print("Mean Jaccard error overlap (9 denek ortalamasi)")
    print("0 = tamamen ayri hatalar (ensemble cok degerli)")
    print("1 = ozdes hatalar (ensemble degersiz)")
    print("=" * 80)
    print(pivot.round(3).to_string())
    print()

    # Klasik en iyi (fbcsp_rlda) <-> DL ciftleri, denek bazinda
    print("=" * 80)
    print("fbcsp_rlda <-> DL hata ortusmesi (denek bazinda)")
    print("=" * 80)
    dl_bases = [b for b in ["eegnet", "shallowconvnet"] if b in available]
    if dl_bases and "fbcsp_rlda" in available:
        for dl in dl_bases:
            sub_df = df[
                ((df["base_A"] == "fbcsp_rlda") & (df["base_B"] == dl))
                | ((df["base_B"] == "fbcsp_rlda") & (df["base_A"] == dl))
            ].copy()
            sub_df = sub_df[["subject", "acc_A", "acc_B",
                             "n_err_A", "n_err_B", "both_wrong",
                             "jaccard_err", "disagreement_rate"]]
            sub_df.columns = ["subject", "acc_fbcsp", f"acc_{dl}",
                              "err_fbcsp", f"err_{dl}", "both_wrong",
                              "jaccard_err", "disagree_rate"]
            print(f"\n--- fbcsp_rlda vs {dl} ---")
            print(sub_df.to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))
            # A05 vurgu
            a05 = sub_df[sub_df["subject"] == 5]
            if len(a05):
                r = a05.iloc[0]
                weak_flag = ("DUSUK ortusme: DL farkli hatalar"
                             if r["jaccard_err"] < 0.5 else
                             "YUKSEK ortusme: DL ayni hatalari yapiyor")
                print(f"  A05 yorumu: {weak_flag}  (jaccard={r['jaccard_err']:.3f})")
    print()

    # A05'e ozel kompakt cikti
    a05_df = df[df["subject"] == 5][["base_A", "base_B", "acc_A", "acc_B",
                                      "n_err_A", "n_err_B", "both_wrong",
                                      "jaccard_err", "disagreement_rate"]]
    out_a05 = RESULTS_DIR / "error_overlap_a05.csv"
    a05_df.to_csv(out_a05, index=False, encoding="utf-8")
    print(f"Saved (A05 detayli): {out_a05}")


if __name__ == "__main__":
    main()

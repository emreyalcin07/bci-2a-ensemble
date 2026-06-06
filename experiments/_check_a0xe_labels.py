"""
_check_a0xe_labels.py
=====================

A0XE etiket dosyalarinin formatini dogrula. A0XE SINYALINE DOKUNMA.
Sadece .mat etiketlerini incele + A0XT kodlamasiyla cross-check.

Yapilanlar:
1. A01E.mat ve A05E.mat'i scipy.io.loadmat ile ac
2. Anahtarlari, ham etiket degerlerini (ilk 10 + unique), count'u goster
3. data_loader.LABEL_OFFSET (=1) ile kaydirma sonrasi dagilim
4. A01T (training) etiketleri ile encoding kiyaslamasi:
   - A0XT: GDF annotations '769'/'770'/'771'/'772' -> 1/2/3/4 -> 0/1/2/3
   - A0XE: .mat classlabel 1/2/3/4 -> 0/1/2/3 (eger ayni semadaysa)
   - Iki dagilim da 72/72/72/72 dengeli olmali
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Proje koku
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import LABEL_OFFSET, load_subject  # noqa: E402

TRUE_LABELS_DIR = ROOT / "data" / "true_labels"


def inspect_mat(subject_id: int):
    path = TRUE_LABELS_DIR / f"A{subject_id:02d}E.mat"
    print(f"\n{'='*70}")
    print(f"A{subject_id:02d}E.mat  ({path})")
    print(f"{'='*70}")
    if not path.exists():
        print("  [HATA] dosya yok")
        return None

    mat = loadmat(str(path))
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"  Anahtarlar       : {keys}")

    if "classlabel" not in mat:
        print(f"  [HATA] 'classlabel' anahtari yok. Mevcut: {keys}")
        return None

    raw = mat["classlabel"]
    print(f"  classlabel shape : {raw.shape}, dtype: {raw.dtype}")
    raw_flat = raw.ravel().astype(int)
    print(f"  count            : {len(raw_flat)}  (288 bekleniyor)")
    print(f"  ilk 10 (HAM)     : {raw_flat[:10].tolist()}")
    print(f"  unique (HAM)     : {sorted(np.unique(raw_flat).tolist())}")
    uniq, cnt = np.unique(raw_flat, return_counts=True)
    print(f"  HAM dagilim      : {dict(zip(uniq.tolist(), cnt.tolist()))}")

    shifted = raw_flat - LABEL_OFFSET  # data_loader -1 kaydirma
    print(f"  ilk 10 (kaydir)  : {shifted[:10].tolist()}")
    print(f"  unique (kaydir)  : {sorted(np.unique(shifted).tolist())}")
    uniq_s, cnt_s = np.unique(shifted, return_counts=True)
    print(f"  KAYDIR dagilim   : {dict(zip(uniq_s.tolist(), cnt_s.tolist()))}")
    return raw_flat, shifted


def compare_with_training(subject_id: int, shifted: np.ndarray):
    """A0XT (training) etiket dagilimini yukle ve A0XE ile kiyasla.
    NOT: A0XT yuklemesi sinyal de yukler ama biz yalnizca .y'ye bakacagiz.
    A0XE sinyaline dokunulmaz."""
    print(f"\n  --- A0{subject_id}T ile encoding cross-check ---")
    sd_t = load_subject(subject_id=subject_id, session="T", verbose=False)
    train_y = sd_t.y
    uniq, cnt = np.unique(train_y, return_counts=True)
    print(f"  A0{subject_id}T classes        : {sorted(uniq.tolist())}")
    print(f"  A0{subject_id}T dagilim        : {dict(zip(uniq.tolist(), cnt.tolist()))}")

    # KIYASLAMA
    a0xe_classes = sorted(np.unique(shifted).tolist())
    a0xt_classes = sorted(uniq.tolist())
    match = a0xe_classes == a0xt_classes
    print(f"  Sinif setleri esit mi? {match} "
          f"(A0XE={a0xe_classes}, A0XT={a0xt_classes})")
    if not match:
        print("  [UYARI] sinif setleri esit degil — yanlis kodlama riski")


def main():
    print("A0XE etiket dosyasi dogrulama")
    print(f"true_labels dizini: {TRUE_LABELS_DIR}")
    print(f"data_loader LABEL_OFFSET (kaydirma): {LABEL_OFFSET}")

    for sid in (1, 5):
        result = inspect_mat(sid)
        if result is None:
            continue
        _, shifted = result
        compare_with_training(sid, shifted)

    print("\nBitti — A0XE sinyaline DOKUNULMADI, sadece .mat etiketleri ve A0XT y vektoru okundu.")


if __name__ == "__main__":
    main()

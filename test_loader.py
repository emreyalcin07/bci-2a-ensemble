"""
test_loader.py
==============

data_loader.py için duman testi (smoke test):
    1. A01T'yi yükler.
    2. Beklenen şekilleri ve değer aralıklarını doğrular.
    3. Sınıf dağılımını ve özet bilgileri ekrana basar.

Bu betik, herhangi bir model eğitmeden ÖNCE çalıştırılır. Burada bir
şey patlarsa, sonraki tüm deneyler güvenilmez olur.

Çalıştırma:
    python test_loader.py
"""

from __future__ import annotations

import sys

import numpy as np

from data_loader import (
    EPOCH_SAMPLES,
    N_EEG_CHANNELS,
    N_EOG_CHANNELS,
    SFREQ,
    load_subject,
    summarize,
)


def assert_eq(label, actual, expected):
    ok = actual == expected
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {label}: {actual}  (expected {expected})")
    if not ok:
        sys.exit(1)


def assert_true(label, condition, detail=""):
    mark = "OK " if condition else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        sys.exit(1)


def main():
    print(">>> Loading A01T ...")
    sd = load_subject(subject_id=1, session="T", verbose=False)

    print("\n>>> Summary")
    print(summarize(sd))

    print("\n>>> Sanity checks")
    # Şekiller
    assert_eq("X.ndim", sd.X.ndim, 3)
    assert_eq("X.shape[1] (n_eeg_channels)", sd.X.shape[1], N_EEG_CHANNELS)
    assert_eq("X.shape[2] (n_samples)", sd.X.shape[2], EPOCH_SAMPLES)
    assert_eq("X_eog.shape[1]", sd.X_eog.shape[1], N_EOG_CHANNELS)
    assert_eq("X_eog.shape[2]", sd.X_eog.shape[2], EPOCH_SAMPLES)
    assert_eq("sfreq", sd.sfreq, SFREQ)

    # Etiketler
    assert_true("y is not None", sd.y is not None)
    assert_eq("y.shape[0] == X.shape[0]", sd.y.shape[0], sd.X.shape[0])
    classes = sorted(set(int(c) for c in sd.y.tolist()))
    assert_eq("classes", classes, [0, 1, 2, 3])

    # Sınıf dağılımı dengeli mi? (BCI Comp IV 2a: 72 trial/sınıf, 288 toplam)
    dist = sd.class_distribution
    for c, n in dist.items():
        assert_true(f"class {c} non-empty", n > 0)
    total = sum(dist.values())
    print(f"  [INFO] toplam trial: {total}")
    print(f"  [INFO] sınıf dağılımı: {dist}")

    # Veri sayısal sağlık kontrolü
    assert_true("X is finite", np.all(np.isfinite(sd.X)))
    assert_true("X has variance", float(sd.X.std()) > 0)
    print(f"  [INFO] X dtype={sd.X.dtype}, range=[{sd.X.min():.3e}, {sd.X.max():.3e}], std={sd.X.std():.3e}")

    print("\n>>> A01E (evaluation) yüklenebilir mi? (etiketler kullanılmaz)")
    sd_e = load_subject(subject_id=1, session="E", verbose=False)
    print(f"  X.shape (E): {sd_e.X.shape}")
    print(f"  y (E):      {'present' if sd_e.y is not None else 'None (true labels file missing — beklenen)'}")
    assert_eq("X.shape[1] (E)", sd_e.X.shape[1], N_EEG_CHANNELS)
    assert_eq("X.shape[2] (E)", sd_e.X.shape[2], EPOCH_SAMPLES)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()

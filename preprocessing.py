"""
preprocessing.py
================

Epoch'lanmış EEG/EOG verisi üzerinde uygulanacak ön işleme adımları.

Bu modül, tipik motor imagery işleme zincirinin yapı taşlarını sağlar:
    - Bant geçiren filtreleme (mu/beta veya geniş bant)
    - Kanal başına z-score / mean-removal
    - Basit EOG regresyonu (regression-based EOG artifact removal)

İleride eklenecek ileri yöntemler (ICA, RASR, ASR, riemannian whitening)
için yer hazır bırakılmıştır.

Önemli: Bu aşamada hiçbir fonksiyon main akışta çağrılmaz. Pipeline'lar
geliştirildiğinde kullanılacaktır.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt


# --------------------------------------------------------------------------- #
# Filtreleme                                                                  #
# --------------------------------------------------------------------------- #


def bandpass_filter(
    X: np.ndarray,
    sfreq: float,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    order: int = 5,
) -> np.ndarray:
    """Sıfır-fazlı Butterworth bant geçiren filtre uygula.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    sfreq : float
        Örnekleme frekansı (Hz).
    l_freq, h_freq : float
        Alt ve üst kesim frekansları (Hz). Varsayılan 8–30 Hz mu+beta.
    order : int
        Butterworth filtre derecesi. filtfilt ile efektif derece 2*order olur.

    Returns
    -------
    ndarray, aynı şekilde filtrelenmiş veri.

    Notes
    -----
    filtfilt sıfır-faz filtreleme yapar; nedensel değildir, ama epoch
    tabanlı analizde standarttır.
    """
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    b, a = butter(order, [low, high], btype="band")
    # filtfilt son eksen boyunca çalışır — kanallarımızın son ekseni zaman.
    return filtfilt(b, a, X, axis=-1).astype(X.dtype, copy=False)


# --------------------------------------------------------------------------- #
# Normalizasyon                                                               #
# --------------------------------------------------------------------------- #


def channel_zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Her trial içinde, her kanal için z-score normalizasyonu.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)

    Returns
    -------
    ndarray, aynı şekil, kanal başına sıfır ortalama / birim std.
    """
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return (X - mean) / (std + eps)


def remove_channel_mean(X: np.ndarray) -> np.ndarray:
    """Her trial × kanal için ortalamayı çıkar (DC kaldırma)."""
    return X - X.mean(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# EOG handling (iskelet)                                                      #
# --------------------------------------------------------------------------- #


def regress_out_eog(X_eeg: np.ndarray, X_eog: np.ndarray) -> np.ndarray:
    """En küçük kareler ile EOG bileşenini EEG'den çıkar.

    Trial-bazlı basit lineer regresyon:
        EEG_clean[c, t] = EEG[c, t] - sum_k beta[c, k] * EOG[k, t]

    Parameters
    ----------
    X_eeg : ndarray, shape (n_trials, n_eeg, n_samples)
    X_eog : ndarray, shape (n_trials, n_eog, n_samples)

    Returns
    -------
    ndarray, EEG ile aynı şekilde, EOG bileşeni çıkarılmış.

    Notes
    -----
    - Trial başına ayrı regresyon yapılır; küçük n_samples'ta gürültülü olabilir.
    - Daha sağlam alternatifler: oturum bazlı tek regresyon, ICA tabanlı
      EOG kaldırma. İleride buraya eklenir.
    - Şu an pipeline'larda otomatik çağrılmaz — opsiyonel.
    """
    if X_eeg.shape[0] != X_eog.shape[0] or X_eeg.shape[-1] != X_eog.shape[-1]:
        raise ValueError("EEG ve EOG trial / sample sayıları uyuşmuyor.")

    cleaned = np.empty_like(X_eeg)
    for i in range(X_eeg.shape[0]):
        eeg = X_eeg[i]            # (n_eeg, n_samples)
        eog = X_eog[i]            # (n_eog, n_samples)
        # beta: (n_eeg, n_eog). EEG ≈ beta @ EOG  ->  beta = EEG @ EOG.T @ inv(EOG @ EOG.T)
        eog_cov = eog @ eog.T
        beta = eeg @ eog.T @ np.linalg.pinv(eog_cov)
        cleaned[i] = eeg - beta @ eog
    return cleaned


# --------------------------------------------------------------------------- #
# Pipeline yardımcısı                                                         #
# --------------------------------------------------------------------------- #


def default_preprocess(
    X_eeg: np.ndarray,
    sfreq: float,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> np.ndarray:
    """Varsayılan ön işleme: bant geçiren filtre + DC kaldırma.

    Pipeline'lar bunu opsiyonel olarak çağırabilir. Bu modül, tek bir
    "doğru" yol dayatmaz — her pipeline kendi ön işleme tercihini seçer.
    """
    X = bandpass_filter(X_eeg, sfreq, l_freq=l_freq, h_freq=h_freq)
    X = remove_channel_mean(X)
    return X

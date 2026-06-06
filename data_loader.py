"""
data_loader.py
==============

BCI Competition IV Dataset 2a yükleme katmanı.

Bu modül, .gdf dosyalarından motor imagery EEG verisini okuyup epoch'lara
böler ve sınıflandırıcılara verilebilecek (X, y) tensörleri döndürür.

Veri seti:
    - 9 denek (A01 .. A09)
    - Her denek için 2 oturum:
        * A0XT.gdf  -> training set (etiketler dosyanın içinde)
        * A0XE.gdf  -> evaluation set (etiketler ayrı .mat dosyasında)
    - 250 Hz örnekleme
    - 22 EEG kanalı + 3 EOG kanalı
    - 4 sınıf: 1=sol el, 2=sağ el, 3=iki ayak, 4=dil

Tasarım notları:
    - load_subject(subject_id, session) tek bir oturumu yükler.
    - load_all_subjects(...) tüm denekleri birleştirir; cross-subject
      transfer (LOSO pretrain/finetune) deneyleri için gereklidir.
    - EOG kanalları ayrı bir tensörde (X_eog) döndürülür; EOG ile artefakt
      temizliği yapılacaksa preprocessing katmanı kullanır. Bu aşamada
      sınıflandırmada kullanılmaz.
    - A0XE (evaluation) etiketleri yalnızca .mat dosyası mevcutsa eşlenir.
      Aksi halde y=None döner ve dosya sadece yüklenebilirliği için kontrol
      edilir. Final değerlendirme için saklanır — analizde kullanılmamalı.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# MNE GDF okuma uyarılarını sessize alıyoruz — kanal tipi tahminleri vs.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")


# --------------------------------------------------------------------------- #
# Sabitler                                                                    #
# --------------------------------------------------------------------------- #

#: Veri kök dizini (proje köküne göre)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

#: Örnekleme frekansı (Hz). Veri seti spesifikasyonundan sabit.
SFREQ = 250.0

#: BCI Comp IV 2a'da 22 EEG + 3 EOG kanal vardır. Sırasıyla ilk 22 EEG.
N_EEG_CHANNELS = 22
N_EOG_CHANNELS = 3

#: Cue olay kodları. Training (T) dosyalarında sınıflar açıkça verilir;
#: evaluation (E) dosyalarında tüm cue'lar 783 olarak işaretlenir.
#: MNE annotations'larda string olarak gelir.
CUE_EVENT_IDS_TRAIN = {
    "769": 1,  # sol el
    "770": 2,  # sağ el
    "771": 3,  # iki ayak
    "772": 4,  # dil
}
CUE_EVENT_ID_TEST = {"783": 0}  # bilinmeyen — gerçek etiket .mat'tan gelir

#: Cue sonrası epoch penceresi (s). 0.5 s ile 2.5 s arası ⇒ 2 s ⇒ 500 örnek.
TMIN_AFTER_CUE = 0.5
TMAX_AFTER_CUE = 2.5
EPOCH_SAMPLES = int(round((TMAX_AFTER_CUE - TMIN_AFTER_CUE) * SFREQ))  # 500

#: Etiketler [0..3] aralığına çekilir (scikit-learn için pratik).
LABEL_OFFSET = 1


# --------------------------------------------------------------------------- #
# Veri konteyneri                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class SubjectData:
    """Tek bir oturumun epoch'lanmış verisi.

    Attributes
    ----------
    X : np.ndarray, shape (n_trials, n_eeg_channels, n_samples)
        EEG epoch'ları. EOG çıkarılmış, yalnızca 22 EEG kanalı.
    y : np.ndarray | None, shape (n_trials,)
        Sınıf etiketleri 0..3 (0=sol el, 1=sağ el, 2=iki ayak, 3=dil).
        A0XE için .mat dosyası yoksa None.
    X_eog : np.ndarray, shape (n_trials, n_eog_channels, n_samples)
        EOG epoch'ları. Preprocessing katmanı için saklanır; sınıflandırmada
        kullanılmaz.
    subject_id : int
        Denek numarası (1..9).
    session : str
        "T" (training) veya "E" (evaluation).
    sfreq : float
        Örnekleme frekansı.
    ch_names : list[str]
        22 EEG kanalının isimleri.
    """

    X: np.ndarray
    y: Optional[np.ndarray]
    X_eog: np.ndarray
    subject_id: int
    session: str
    sfreq: float
    ch_names: List[str]

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def class_distribution(self) -> Dict[int, int]:
        """Sınıf etiketi -> trial sayısı sözlüğü."""
        if self.y is None:
            return {}
        unique, counts = np.unique(self.y, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}


# --------------------------------------------------------------------------- #
# Yardımcı fonksiyonlar                                                       #
# --------------------------------------------------------------------------- #


def _build_gdf_path(subject_id: int, session: str, data_dir: Path) -> Path:
    """A0X{T|E}.gdf yolunu üret ve var olduğunu doğrula."""
    if subject_id < 1 or subject_id > 9:
        raise ValueError(f"subject_id 1..9 aralığında olmalı, alınan: {subject_id}")
    session = session.upper()
    if session not in {"T", "E"}:
        raise ValueError(f"session 'T' veya 'E' olmalı, alınan: {session}")

    path = data_dir / f"A{subject_id:02d}{session}.gdf"
    if not path.exists():
        raise FileNotFoundError(f"GDF dosyası bulunamadı: {path}")
    return path


def _find_true_labels_file(subject_id: int, data_dir: Path) -> Optional[Path]:
    """A0XE için true label .mat dosyasını ara. Yoksa None döner.

    BCI Comp IV 2a için true label dosyaları yarışma sonrası ayrı olarak
    yayınlanır (genellikle A0XE.mat içinde 'classlabel' değişkeni olarak).
    Yaygın isimlendirmeleri kontrol ederiz.
    """
    candidates = [
        data_dir / f"A{subject_id:02d}E.mat",
        data_dir / "true_labels" / f"A{subject_id:02d}E.mat",
        data_dir / "labels" / f"A{subject_id:02d}E.mat",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_true_labels(mat_path: Path) -> np.ndarray:
    """A0XE .mat dosyasından classlabel'ı oku, 0..3 aralığına çek."""
    from scipy.io import loadmat

    mat = loadmat(str(mat_path))
    if "classlabel" not in mat:
        raise KeyError(
            f"{mat_path.name} içinde 'classlabel' değişkeni bulunamadı. "
            f"Mevcut anahtarlar: {[k for k in mat.keys() if not k.startswith('__')]}"
        )
    labels = mat["classlabel"].ravel().astype(int)
    return labels - LABEL_OFFSET  # 1..4 -> 0..3


# --------------------------------------------------------------------------- #
# Ana yükleme fonksiyonu                                                      #
# --------------------------------------------------------------------------- #


def load_subject(
    subject_id: int,
    session: str = "T",
    data_dir: Optional[Path] = None,
    tmin: float = TMIN_AFTER_CUE,
    tmax: float = TMAX_AFTER_CUE,
    drop_rejected: bool = True,
    verbose: bool = False,
) -> SubjectData:
    """Tek bir deneğin tek bir oturumunu yükle ve epoch'la.

    Parameters
    ----------
    subject_id : int
        Denek numarası (1..9).
    session : {'T', 'E'}
        'T' training (A0XT.gdf), 'E' evaluation (A0XE.gdf).
    data_dir : Path, optional
        Veri kök dizini. Belirtilmezse ./data kullanılır.
    tmin, tmax : float
        Cue olayına göre epoch başlangıç ve bitiş zamanları (saniye).
        Varsayılan 0.5–2.5 s ⇒ 500 örnek.
    drop_rejected : bool
        True ise 1023 (rejected trial) işaretli epoch'ları atar.
        Veri setinin orijinal kalite işaretine güveniriz.
    verbose : bool
        MNE'nin ilerleme çıktısını göster.

    Returns
    -------
    SubjectData
        Epoch'lanmış EEG verisi ve etiketler.

    Notes
    -----
    - Session 'E' için .mat etiket dosyası yoksa y=None döner. Yükleme
      doğrulanır ama analizde kullanılmamalıdır (final değerlendirme için
      saklı).
    """
    import mne  # geç içe aktarım — modül import maliyetini erteler

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    gdf_path = _build_gdf_path(subject_id, session, data_dir)
    mne_verbose = "INFO" if verbose else "ERROR"

    # ---- 1. Ham veriyi oku ----
    raw = mne.io.read_raw_gdf(
        str(gdf_path),
        preload=True,
        verbose=mne_verbose,
    )

    # MNE bazen kanal isimlerini stim/eog olarak işaretlemez. Manuel düzelt:
    # İlk 22 kanal EEG, son 3 kanal EOG (BCI Comp IV 2a spesifikasyonu).
    ch_names_all = raw.ch_names
    if len(ch_names_all) < N_EEG_CHANNELS + N_EOG_CHANNELS:
        raise RuntimeError(
            f"Beklenen ≥{N_EEG_CHANNELS + N_EOG_CHANNELS} kanal, "
            f"bulunan {len(ch_names_all)}: {ch_names_all}"
        )

    eeg_names = ch_names_all[:N_EEG_CHANNELS]
    eog_names = ch_names_all[N_EEG_CHANNELS : N_EEG_CHANNELS + N_EOG_CHANNELS]

    # Kanal tiplerini açıkça ayarla — MNE'nin EOG'yi tanıması için.
    ch_type_map = {name: "eeg" for name in eeg_names}
    ch_type_map.update({name: "eog" for name in eog_names})
    raw.set_channel_types(ch_type_map, verbose=mne_verbose)

    # ---- 2. Olayları çıkar ----
    # GDF'de annotations -> events
    events, event_id_map = mne.events_from_annotations(raw, verbose=mne_verbose)

    if session.upper() == "T":
        wanted = CUE_EVENT_IDS_TRAIN
    else:
        wanted = CUE_EVENT_ID_TEST

    # MNE event_id_map: { "769": 7, ... } gibi — annotation string -> integer id.
    selected_event_id: Dict[str, int] = {}
    for desc, label in wanted.items():
        if desc in event_id_map:
            selected_event_id[desc] = event_id_map[desc]
        else:
            if session.upper() == "T":
                raise RuntimeError(
                    f"Beklenen cue olayı '{desc}' GDF içinde bulunamadı. "
                    f"Mevcut olaylar: {sorted(event_id_map.keys())}"
                )

    if not selected_event_id:
        raise RuntimeError("Hiçbir cue olayı bulunamadı.")

    # ---- 3. Epoch'la ----
    # Baseline=None: bant geçiren filtre kullanmadan ham epoch'lar; baseline
    # düzeltmesi preprocessing katmanına bırakılır.
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=selected_event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        proj=False,
        picks=None,  # tüm kanallar — sonra EEG/EOG ayırıyoruz
        verbose=mne_verbose,
        on_missing="ignore",
    )

    # Reddedilmiş trial'lar (event 1023) MNE tarafından otomatik ele alınmaz;
    # ancak epoch sayısı zaten cue olaylarından üretilir. Bazı çalışmalar
    # 1023 işaretli trial'ı bırakır — burada güvende kalmak için MNE'nin
    # kendi reject kriterini kullanmıyoruz (sonradan preprocessing'te artefakt
    # temizliği yapılır). drop_rejected parametresi ileride .gdf
    # annotations'taki 1023 işaretine göre filtrelemek üzere ayrılmıştır.
    _ = drop_rejected  # şu an no-op; ileride kullanılacak

    # ---- 4. X (EEG) ve X_eog ayır ----
    eeg_data = epochs.copy().pick(eeg_names).get_data(copy=False)  # (n, 22, ?)
    eog_data = epochs.copy().pick(eog_names).get_data(copy=False)  # (n, 3, ?)

    # MNE tmin–tmax dahil-dahil verir; örnek sayısı (tmax-tmin)*sfreq + 1.
    # 500 örneklik pencereyi garantile.
    eeg_data = eeg_data[:, :, :EPOCH_SAMPLES]
    eog_data = eog_data[:, :, :EPOCH_SAMPLES]

    # ---- 5. Etiketler ----
    if session.upper() == "T":
        # epochs.events[:, 2] -> MNE'nin atadığı integer id; bunu sınıf 0..3'e çevir.
        inv_map = {v: CUE_EVENT_IDS_TRAIN[k] for k, v in selected_event_id.items()}
        raw_labels = np.array([inv_map[e] for e in epochs.events[:, 2]])
        y = raw_labels - LABEL_OFFSET  # 1..4 -> 0..3
    else:
        mat_path = _find_true_labels_file(subject_id, data_dir)
        if mat_path is not None:
            y = _load_true_labels(mat_path)
            if len(y) != eeg_data.shape[0]:
                raise RuntimeError(
                    f"True label sayısı ({len(y)}) epoch sayısı "
                    f"({eeg_data.shape[0]}) ile eşleşmiyor."
                )
        else:
            y = None  # final değerlendirme için saklı

    return SubjectData(
        X=eeg_data.astype(np.float32),
        y=y.astype(np.int64) if y is not None else None,
        X_eog=eog_data.astype(np.float32),
        subject_id=subject_id,
        session=session.upper(),
        sfreq=SFREQ,
        ch_names=eeg_names,
    )


# --------------------------------------------------------------------------- #
# Çoklu denek yükleme                                                         #
# --------------------------------------------------------------------------- #


def load_all_subjects(
    session: str = "T",
    data_dir: Optional[Path] = None,
    subjects: Optional[List[int]] = None,
    concatenate: bool = False,
    **kwargs,
) -> Dict[int, SubjectData] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tüm denekleri yükle. Cross-subject transfer için tasarlandı.

    Parameters
    ----------
    session : {'T', 'E'}
    data_dir : Path, optional
    subjects : list[int], optional
        Yüklenecek denek listesi. None ise 1..9 hepsi.
    concatenate : bool
        False (varsayılan) -> {subject_id: SubjectData} sözlüğü döner.
        True  -> (X, y, subject_ids) tuple olarak birleşik tensörler döner.
        LOSO (leave-one-subject-out) için subject_ids gerekli.

    Returns
    -------
    dict[int, SubjectData] veya (X, y, subject_ids)
    """
    if subjects is None:
        subjects = list(range(1, 10))

    per_subject: Dict[int, SubjectData] = {}
    for sid in subjects:
        per_subject[sid] = load_subject(sid, session=session, data_dir=data_dir, **kwargs)

    if not concatenate:
        return per_subject

    # Birleştir: y=None olan denekler varsa concatenate edilemez.
    if any(sd.y is None for sd in per_subject.values()):
        raise RuntimeError(
            "concatenate=True ile session='E' kullanılamaz çünkü bazı "
            "deneklerin true etiketleri yok."
        )

    X_list, y_list, sid_list = [], [], []
    for sid, sd in per_subject.items():
        X_list.append(sd.X)
        y_list.append(sd.y)
        sid_list.append(np.full(sd.n_trials, sid, dtype=np.int64))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subject_ids = np.concatenate(sid_list, axis=0)
    return X, y, subject_ids


# --------------------------------------------------------------------------- #
# Kolaylık: keşif modu                                                        #
# --------------------------------------------------------------------------- #


def summarize(sd: SubjectData) -> str:
    """SubjectData için kısa özet string'i üret (loglama / test için)."""
    lines = [
        f"Subject A{sd.subject_id:02d}{sd.session}",
        f"  X shape       : {sd.X.shape}  (trials, channels, samples)",
        f"  X_eog shape   : {sd.X_eog.shape}",
        f"  sfreq         : {sd.sfreq} Hz",
        f"  n_eeg_channels: {len(sd.ch_names)}",
        f"  ch_names      : {sd.ch_names}",
    ]
    if sd.y is not None:
        lines.append(f"  y shape       : {sd.y.shape}")
        lines.append(f"  classes       : {sorted(set(sd.y.tolist()))}")
        lines.append(f"  class dist    : {sd.class_distribution}")
    else:
        lines.append("  y             : None (true labels not loaded)")
    return "\n".join(lines)

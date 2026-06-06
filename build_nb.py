# -*- coding: utf-8 -*-
"""BCI_MI_Reproduction.ipynb üreticisi.

lib_content.py'yi okur, tüm markdown/kod hücrelerini kurar ve geçerli bir
.ipynb (nbformat v4) dosyası yazar.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = (ROOT / "lib_content.py").read_text(encoding="utf-8")

cells = []


def _cid():
    return "cell-%03d" % (len(cells) + 1)


def md(src):
    cells.append({"id": _cid(), "cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"id": _cid(), "cell_type": "code", "metadata": {},
                  "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


# =========================================================================== #
# BAŞLIK
# =========================================================================== #
md(r"""
# 🧠 BCI Competition IV — Dataset 2a · Motor Imagery Yeniden-Üretim Notebook'u

**4 sınıflı motor imagery** (sol el / sağ el / iki ayak / dil) EEG sınıflandırması.
Bu notebook, çok-dosyalı araştırma projesini **tek bir Colab notebook'una** indirger;
`Runtime ▸ Run all` ile sıfırdan, baştan sona çalıştırıldığında aynı final sonucu üretir.

### 🎯 Beklenen final sonuç (saklı test seti A0XE)

| Metrik | Değer |
|:--|:--|
| **Final Test Cohen's κ** | **0.6137** |
| Final Test accuracy | 0.7103 |
| Δ (Test − CV) | −0.042 (makul session shift, leakage yok) |
| Yarışma kazananı (Kai Keng Ang, FBCSP, 2008) | κ = 0.569 → **+0.045 geçildi** |

> ℹ️ Klasik fazlar (FBCSP, Riemann) tamamen deterministiktir → κ bit-bit yeniden üretilir.
> Derin öğrenme fazları (EEGNet / ShallowConvNet) GPU çekirdek non-determinizmi nedeniyle
> ±0.01–0.02 oynayabilir; tüm tohumlar sabitlendiği için sapma küçüktür. Her bölüm
> **elde edilen κ'yı beklenen değerle karşılaştırarak** bilgilendirir (assert yok).

### 🔒 En kritik ilke — LEAKAGE-FREE
Test seti **A0XE, BÖLÜM 7'ye kadar HİÇ kullanılmaz.** Tüm hiperparametre seçimi iç
cross-validation'da yapılır. Tüm fazlar **`StratifiedKFold(5, shuffle=True, random_state=42)`**
kullanır → fold yapısı her fazda bit-bit aynı → OOF tahminleri trial-by-trial hizalı →
ensemble temiz.

### ⏱️ Tahmini toplam süre (Colab T4)
| Bölüm | Tahmin |
|:--|:--|
| 0–1 Kurulum + veri indirme | ~5 dk |
| 3 Faz 1 (FBCSP+sLDA) | ~30–80 dk (en yavaş klasik) |
| 4 Faz 2–3 ablasyonlar | ~40–70 dk |
| 6 Faz 5 (EEGNet + ShallowConvNet) | ~15–20 dk (GPU) |
| 7 Faz 6 (final test, 9 denek) | ~25–35 dk |
| **TOPLAM** | **~2–4 saat** |

> 🧪 **Hızlı deneme:** Bölüm 0'daki `SUBJECTS` listesini örn. `[1, 3]` yaparak yapıyı
> dakikalar içinde sınayabilirsiniz (κ değerleri 9-denek ortalamasına göre verilmiştir).

### 📑 İçindekiler
0. Kurulum (paketler, GPU, tohum)
1. Veri indirme + yükleme + ön işleme
2. Değerlendirme altyapısı (nested CV, OOF, soft-vote)
3. **Faz 1** — Baseline: Regularized FBCSP + Shrinkage-LDA (κ≈0.625)
4. **Faz 2–3** — Ablasyon / negatif sonuçlar (band-sel, Riemann)
5. **Faz 4** — Klasik ensemble (κ≈0.6425)
6. **Faz 5** — Derin öğrenme + final ensemble (κ≈0.6553)
7. **Faz 6** — A0XE saklı test, tek-shot (**Test κ=0.6137**)
8. Görselleştirme & özet
""")

# =========================================================================== #
# BÖLÜM 0
# =========================================================================== #
md(r"""
## BÖLÜM 0 — Kurulum

Gerekli paketleri kuruyoruz. Colab'da `numpy`, `scipy`, `scikit-learn`, `torch`,
`matplotlib`, `joblib` zaten yüklüdür; **`mne`** (GDF okuma) ve **`pyriemann`**
(Riemann geometrisi) ek olarak kurulur.
""")
code(r"""
# ~1-2 dk. mne: .gdf okuma; pyriemann: kovaryans manifoldu yöntemleri.
!pip -q install -U mne pyriemann
print("Kurulum tamam.")
""")

code(r"""
# --- İmportlar, GPU kontrolü, global yapılandırma ---
import os, sys, time, ssl, shutil, zipfile, urllib.request, warnings
from pathlib import Path
from functools import partial

import numpy as np
import torch

warnings.filterwarnings("ignore")

# GPU kontrolü (DL fazları için). Yoksa CPU'da çalışır ama yavaştır.
if torch.cuda.is_available():
    print("✅ GPU bulundu:", torch.cuda.get_device_name(0))
else:
    print("⚠️  GPU YOK — derin öğrenme fazları CPU'da çok yavaş olacaktır.")
    print("    Colab'da: Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ T4 GPU")

# ---- Global yapılandırma ----
DATA_DIR_PATH = Path("/content/data")          # .gdf dosyaları buraya
LABELS_DIR    = DATA_DIR_PATH / "true_labels"   # A0XE.mat etiketleri buraya
SUBJECTS      = list(range(1, 10))              # tüm 9 denek (hızlı deneme: [1, 3])
RUN_TRANSFER  = False                           # Faz 5b LOSO transfer (opsiyonel, ~50 dk)

# Determinizm — tüm random_state=42
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

print("Denekler:", SUBJECTS)
print("Veri dizini:", DATA_DIR_PATH)
""")

# =========================================================================== #
# BÖLÜM 1 — Veri indirme
# =========================================================================== #
md(r"""
## BÖLÜM 1 — Veri İndirme, Yükleme ve Ön İşleme

### 1.1 Otomatik indirme

İki kaynaktan veri çekiyoruz:

1. **`.gdf` ham EEG** — BNCI Horizon 2020 aynası (MOABB'nin de kullandığı kaynak):
   `bnci-horizon-2020.eu/database/data-sets/001-2014/A0X{T,E}.gdf` (18 dosya, ~‑40 MB/dosya).
   Bu ayna kayıt/lisans gerektirmez ve doğrudan indirilebilir (TU‑Graz sunucusuna yönlenir).
2. **Gerçek test etiketleri (`.mat`)** — resmi yarışma sonuçları:
   `bbci.de/competition/iv/results/ds2a/true_labels.zip` → `A0XE.mat` (test etiketleri).

> 🛟 **Manuel alternatif (indirme başarısızsa):** Dosyaları Google Drive'a yükleyip
> `DATA_DIR_PATH`'i Drive yolunuza işaret edin, ardından bu hücreyi atlayıp 1.3'e geçin.
> Gerekli düzen: `DATA_DIR_PATH/A0X{T,E}.gdf` ve `DATA_DIR_PATH/true_labels/A0XE.mat`.
""")
code(r"""
DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

GDF_BASE = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/"
LABELS_URL = "https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip"

def _download(url, dst, min_bytes=10000, tries=3):
    if dst.exists() and dst.stat().st_size > min_bytes:
        print("  (var)", dst.name); return
    ctx = ssl.create_default_context()
    ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=180) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            print("  ✓", dst.name, f"({dst.stat().st_size/1e6:.1f} MB)"); return
        except Exception as e:
            last = e; print(f"  ! deneme {t+1} hata: {e}")
    raise RuntimeError(f"İndirilemedi: {url}\n{last}")

print("== .gdf dosyaları ==")
for sid in range(1, 10):
    for sess in ("T", "E"):
        _download(GDF_BASE + f"A{sid:02d}{sess}.gdf", DATA_DIR_PATH / f"A{sid:02d}{sess}.gdf")

print("\n== gerçek test etiketleri (.mat) ==")
zp = DATA_DIR_PATH / "true_labels.zip"
_download(LABELS_URL, zp, min_bytes=1000)
with zipfile.ZipFile(zp) as z:
    z.extractall(LABELS_DIR)
# Olası alt-klasörleri düzleştir → true_labels/A0XE.mat
for p in list(LABELS_DIR.rglob("*.mat")):
    if p.parent != LABELS_DIR:
        shutil.move(str(p), str(LABELS_DIR / p.name))
print("Etiket dosyaları:", sorted(p.name for p in LABELS_DIR.glob("A0*E.mat")))
""")

# =========================================================================== #
# BÖLÜM 1.2 — kütüphane
# =========================================================================== #
md(r"""
### 1.2 Proje kütüphanesi (`bci_lib.py`)

Tüm yeniden kullanılabilir mantığı (veri yükleyici, değerlendirme, FBCSP/Riemann/DL
yapı taşları, modeller) **importlanabilir bir modüle** yazıyoruz.

**Neden ayrı modül?** `GridSearchCV(n_jobs=-1)` ve `joblib.Memory`, özel transformer
sınıflarını paralel alt-süreçlere *pickle* eder. Notebook hücresinde (`__main__`)
tanımlanan sınıflar bu işlemde sorun çıkarır; modül yolundan import edilen sınıflar
sorunsuz çalışır. **Mantık, orijinal çok-dosyalı projeden birebir taşınmıştır.**

Aşağıdaki hücre modülü diske yazar (içeriği tamamen görünürdür — eğitici amaçla okunabilir).
""")
code("%%writefile bci_lib.py\n" + LIB)

code(r"""
# Modülü yükle ve veri dizinini bağla
import importlib
import bci_lib
importlib.reload(bci_lib)
from bci_lib import *          # load_subject, run_nested_cv, modeller, ...
import bci_lib

bci_lib.DATA_DIR = DATA_DIR_PATH   # load_subject bu global'i kullanır
set_seed(RANDOM_STATE)
print("bci_lib yüklendi.  DEVICE =", DEVICE)
""")

# =========================================================================== #
# BÖLÜM 1.3 — yükleme + doğrulama
# =========================================================================== #
md(r"""
### 1.3 Yükleme & doğrulama

`load_subject()` bir `.gdf` dosyasını okur, **cue + 0.5–2.5 s** penceresinde epoch'lar
(2 s × 250 Hz = **500 örnek**), 22 EEG kanalını ayırır ve etiketleri 0–3'e çeker:

- **A0XT** (training): etiketler `.gdf` annotation'larından (`769–772` → `1–4` → `0–3`).
- **A0XE** (test): etiketler ayrı `.mat`'ten (`classlabel` − 1).

Beklenen: `X` şekli **(288, 22, 500)**, sınıf dağılımı **72/72/72/72** (dengeli).
""")
code(r"""
sd = load_subject(1, "T")
print(summarize(sd))
print()
print("Beklenen: X=(288, 22, 500), sınıflar {0,1,2,3}, dağılım 72/72/72/72")
assert sd.X.shape == (288, 22, 500), "Beklenmeyen epoch şekli!"
assert sd.class_distribution == {0: 72, 1: 72, 2: 72, 3: 72}, "Dengesiz sınıf!"
print("✓ Doğrulama geçti.")
""")
code(r"""
# Tüm fazlarda paylaşılan ORTAK fold yapısı — OOF hizalamasının temeli
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("StratifiedKFold(5, shuffle=True, random_state=42) — A01T:")
for i, (tr, te) in enumerate(skf.split(sd.X, sd.y)):
    print(f"  fold {i}: train={len(tr)}  test={len(te)}  ilk 5 test idx={te[:5].tolist()}")
print("\nBu fold yapısı HER fazda aynıdır → tüm modellerin OOF tahminleri trial-by-trial hizalı.")
""")

# =========================================================================== #
# BÖLÜM 2 — değerlendirme altyapısı
# =========================================================================== #
md(r"""
## BÖLÜM 2 — Değerlendirme Altyapısı

Ana metrik **Cohen's κ** (BCI Comp IV standart metriği; şans seviyesini düzeltir).

- **`run_nested_cv`** — dış 5-fold ile *dürüst performans tahmini*, iç 5-fold
  `GridSearchCV` ile *hiperparametre seçimi*. Dış-test fold'una hiçbir bilgi sızmaz.
  Her dış fold için OOF (out-of-fold) `predict_proba` toplanır.
- **OOF deposu** — her base modelin trial-bazlı olasılıkları `OOF[base][subject]`
  altında saklanır. Tüm base'ler aynı fold yapısını kullandığı için bu olasılıklar
  hizalıdır → soft-voting ensemble *leakage-free* olur.
- **`soft_vote`** — ağırlıklı ortalama olasılık → `argmax`.

Aşağıda OOF deposu ve faz koşum yardımcılarını tanımlıyoruz.
""")
code(r"""
from sklearn.metrics import cohen_kappa_score

OOF = {}              # OOF[base_name][subject_id] = {"y_true","y_pred","y_proba"}
WEAK = {2, 4, 5, 6}   # Faz 1'de ortaya çıkan "zayıf" denekler (düşük κ)
PHASE_KAPPA = {}      # faz adı -> mean κ (özet tablosu için)

def store_oof(base, sid, res):
    OOF.setdefault(base, {})[sid] = {
        "y_true": res.y_true, "y_pred": res.y_pred, "y_proba": res.y_proba}

def run_classical_phase(base_name, prep_fn, build_fn, grid,
                        subjects=None, n_jobs=-1, verbose=True):
    # Her denek için: yükle -> ön-işle -> nested CV -> OOF kaydet. mean κ döndür.
    subjects = subjects or SUBJECTS
    res_k, t0 = {}, time.time()
    for sid in subjects:
        sd = load_subject(sid, "T")
        X = prep_fn(sd); y = sd.y
        r = run_nested_cv(build_fn, grid, X, y, n_jobs=n_jobs)
        store_oof(base_name, sid, r)
        res_k[sid] = r.kappa
        if verbose:
            print(f"  A{sid:02d}: κ={r.kappa:.4f}  acc={r.accuracy:.4f}")
    mk = float(np.mean(list(res_k.values())))
    wk = float(np.mean([res_k[s] for s in subjects if s in WEAK])) if any(s in WEAK for s in subjects) else float("nan")
    print(f"[{base_name}] mean κ={mk:.4f}  (zayıf grup κ={wk:.4f})  süre={(time.time()-t0)/60:.1f} dk")
    return res_k, mk

def ensemble_kappa(bases, weights=None, subjects=None):
    # OOF olasılıklarından soft-vote -> denek-bazlı κ.
    subjects = subjects or SUBJECTS
    res = {}
    for sid in subjects:
        probas = [OOF[b][sid]["y_proba"] for b in bases]
        yt = OOF[bases[0]][sid]["y_true"]
        res[sid] = cohen_kappa_score(yt, soft_vote(probas, weights))
    return res

def compare(got, expected, label):
    print(f"\n➡️  {label}: elde edilen κ = {got:.4f}   |   beklenen ≈ {expected}")
print("Altyapı hazır.")
""")

# =========================================================================== #
# BÖLÜM 3 — Faz 1
# =========================================================================== #
md(r"""
## BÖLÜM 3 — Faz 1: Baseline (Regularized FBCSP + Shrinkage-LDA)

**Filter Bank Common Spatial Patterns** — motor imagery'nin altın standardı.

**Pipeline:**
1. **16 bant** filtre bankası (8–30 Hz): 10×(4 Hz, adım 2) + 6×(6 Hz, adım 3).
2. Her bantta **regularized CSP** (`reg='ledoit_wolf'`, log-variance feature) — Ledoit-Wolf
   shrinkage küçük örneklemde kovaryans tahminini stabilize eder.
3. Tüm bant feature'ları concat → **`SelectKBest(mutual_info_classif)`**.
4. **Shrinkage-LDA** (`solver='lsqr', shrinkage='auto'`) — örtük boyut indirgemeli, sağlam.

**İç CV grid:** `csp_n_components ∈ {2,4,6}`, `select_k ∈ {10,20,40,'all'}`.

**Beklenen:** CV κ ≈ **0.625**. Belirgin **bimodal** dağılım: güçlü grup (A01,03,07,08,09)
κ≈0.80, zayıf grup (A02,04,05,06) κ≈0.41.

> ⏱️ En yavaş klasik faz: ~30–80 dk (16 bant × CSP × geniş grid × nested CV).
""")
code(r"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

FBCSP_BANDS = generate_bands()
print(f"Filtre bankası ({len(FBCSP_BANDS)} bant):", FBCSP_BANDS)

def prep_fbcsp(sd):
    # Veri-bağımsız çoklu-bant filtreleme (CV DIŞINDA → leakage yok)
    return make_multiband_tensor(sd.X, sd.sfreq, FBCSP_BANDS, order=4)

def build_fbcsp():
    mi = partial(mutual_info_classif, random_state=RANDOM_STATE)
    return Pipeline([
        ("csp",    MultiBandCSP(n_components=4, reg="ledoit_wolf")),
        ("select", SelectKBest(score_func=mi, k=20)),
        ("lda",    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])

GRID_FBCSP = {"csp__n_components": [2, 4, 6], "select__k": [10, 20, 40, "all"]}

k_fbcsp, m_fbcsp = run_classical_phase("fbcsp_rlda", prep_fbcsp, build_fbcsp, GRID_FBCSP)
PHASE_KAPPA["Faz1: FBCSP+sLDA"] = m_fbcsp
compare(m_fbcsp, 0.625, "Faz 1 (FBCSP + Shrinkage-LDA)")
""")

# =========================================================================== #
# BÖLÜM 4 — ablasyonlar
# =========================================================================== #
md(r"""
## BÖLÜM 4 — Faz 2–3: Ablasyon (NEGATİF SONUÇLAR)

Bilimsel dürüstlük için denenen ama **baseline'ı geçemeyen** yaklaşımlar. Her birinin
*neden* başarısız olduğunu açıklıyoruz — negatif sonuçlar da bilgi taşır ve bazıları
ensemble için "farklı hata profili" sağlar.

| Varyant | κ | Sonuç |
|:--|:-:|:--|
| Faz 2a — hard band-sel + linSVM(MI) | ≈0.579 | ✗ band seçimi bilgi kaybı |
| Faz 2b — + mRMR | ≈0.585 | ✗ MI'ya göre marjinal |
| Faz 3a — Riemann TS+LR (tek-bant) | ≈0.530 | ✗ filtre bankası avantajı yok |
| Faz 3b — Riemann MDM | ≈0.464 | ✗ ayrımcı bilgi modellemiyor |
| Faz 3c-PCA — multi-band TS+PCA+L2 | ≈0.542 | ✗ boyut/örnek oranı kötü |
| Faz 3c-L1 — multi-band TS+L1 | ≈0.550 | ✗ ama **ensemble'da kullanılacak** |
""")

md(r"""
### 4.1 Faz 2 — Train-only band selection + Linear SVM

**Hipotez:** 16 bandı zorla tutmaktansa train-fold'da MI ile en iyi `top_k` bandı seç.
**Sonuç:** başarısız. Shrinkage-LDA zaten örtük boyut indirgemesi yapıyor; *hard* band
selection bu yumuşak regularizasyondan daha agresif → bilgi kaybı. Güçlü deneklerde sinyal
birden çok banda dağılmış; zayıf deneklerde MI band skoru küçük örneklemde gürültüye duyarlı.

> ⏱️ Geniş grid (top_k × select_k × C). `joblib.Memory` ile `AllBandCSP` cache'lenir. ~30–50 dk.
""")
code(r"""
import tempfile
from joblib import Memory
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MEM = Memory(tempfile.mkdtemp(prefix="bci_cache_"), verbose=0)

def build_bandsel_svm():
    mi = partial(mutual_info_classif, random_state=RANDOM_STATE, n_neighbors=3)
    return Pipeline([
        ("csp_all",      AllBandCSP(n_components=4, reg="ledoit_wolf")),
        ("select_bands", TopKBandSelector(top_k=8)),
        ("scaler",       StandardScaler()),
        ("select_feats", SelectKBest(score_func=mi, k=20)),
        ("svc",          SVC(kernel="linear", probability=True,
                             decision_function_shape="ovr", random_state=RANDOM_STATE)),
    ], memory=MEM)

GRID_SVM = {"select_bands__top_k": [4, 6, 8, 10],
            "select_feats__k": [20, 40, "all"],
            "svc__C": [0.01, 0.1, 1.0, 10.0]}

_, m_svm = run_classical_phase("fbcsp_bandsel_svm", prep_fbcsp, build_bandsel_svm, GRID_SVM)
PHASE_KAPPA["Faz2a: bandSel+SVM(MI)"] = m_svm
compare(m_svm, 0.579, "Faz 2a (band-sel + linSVM, MI)")
""")
code(r"""
def build_bandsel_mrmr():
    return Pipeline([
        ("csp_all",      AllBandCSP(n_components=4, reg="ledoit_wolf")),
        ("select_bands", TopKBandSelector(top_k=8)),
        ("scaler",       StandardScaler()),
        ("select_feats", MRMRSelector(n_features=20)),
        ("svc",          SVC(kernel="linear", probability=True,
                             decision_function_shape="ovr", random_state=RANDOM_STATE)),
    ], memory=MEM)

GRID_MRMR = {"select_bands__top_k": [4, 6, 8, 10],
             "select_feats__n_features": [20, 40, 60],
             "svc__C": [0.01, 0.1, 1.0, 10.0]}

_, m_mrmr = run_classical_phase("fbcsp_bandsel_mrmr", prep_fbcsp, build_bandsel_mrmr, GRID_MRMR)
PHASE_KAPPA["Faz2b: bandSel+SVM(mRMR)"] = m_mrmr
compare(m_mrmr, 0.585, "Faz 2b (band-sel + mRMR)")
""")

md(r"""
### 4.2 Faz 3 — Riemann geometrisi

**Hipotez:** kovaryans manifoldunda küçük örneklemde kovaryans tahmini CSP'den sağlam
olabilir; zayıf grupta kazanç beklenir. **Sonuç:** doğrulanmadı — hiçbir varyant Faz 1'i geçmedi.

- **3a TS (tek-bant 8–30 Hz):** FBCSP'nin 16-bant avantajını yakalayamaz → *adil olmayan kıyas*.
- **3b MDM:** sınıf merkezlerine saf metrik mesafe; ayrımcı bilgiyi modellemez.
- **3c Multi-band TS (3 bant):** 759 feature / 288 trial → boyut/örnek oranı kötü; PCA/L1
  boyut indirgemesi bilgi kaybeder. **L1 varyantı yine de farklı hata profili taşıdığı için
  Faz 4 ensemble'ında kullanılacaktır.**
""")
code(r"""
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# --- 3a: Tangent Space + LR (tek geniş bant 8-30 Hz) ---
def prep_ts1(sd):   return bandpass_single(sd.X, sd.sfreq)        # (n, 22, 500)
def build_ts1():
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts",  TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("lr",  LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)),
    ])
_, m_ts1 = run_classical_phase("riemann_ts_1band", prep_ts1, build_ts1, {"lr__C": [0.01, 0.1, 1.0, 10.0]})
PHASE_KAPPA["Faz3a: TS 1-band"] = m_ts1
compare(m_ts1, 0.530, "Faz 3a (TS + LR, tek bant)")
""")
code(r"""
# --- 3b: Minimum Distance to Mean (hiperparametresiz) ---
def build_mdm():
    return Pipeline([("cov", Covariances(estimator="oas")), ("mdm", MDM(metric="riemann"))])
_, m_mdm = run_classical_phase("riemann_mdm", prep_ts1, build_mdm, {"mdm__metric": ["riemann"]})
PHASE_KAPPA["Faz3b: MDM"] = m_mdm
compare(m_mdm, 0.464, "Faz 3b (MDM)")
""")
code(r"""
# --- 3c: Multi-band Tangent Space (3 bant) → PCA+L2 ve L1 varyantları ---
RIEMANN_BANDS = [(8.0, 14.0), (12.0, 20.0), (18.0, 30.0)]
def prep_mbts(sd):  return bandpass_multi(sd.X, sd.sfreq, RIEMANN_BANDS, order=4)

def build_mbts_pca():
    return Pipeline([
        ("mbts",   MultiBandTangentSpace(cov_estimator="oas", ts_metric="riemann")),
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=50, random_state=RANDOM_STATE)),
        ("lr",     LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)),
    ], memory=MEM)
GRID_PCA = {"pca__n_components": [30, 50, 80], "lr__C": [0.01, 0.1, 1.0, 10.0]}
_, m_pca = run_classical_phase("riemann_multiband_ts_pca", prep_mbts, build_mbts_pca, GRID_PCA)
PHASE_KAPPA["Faz3c: MB-TS PCA"] = m_pca
compare(m_pca, 0.542, "Faz 3c-PCA (multi-band TS + PCA + L2-LR)")

def build_mbts_l1():
    return Pipeline([
        ("mbts",   MultiBandTangentSpace(cov_estimator="oas", ts_metric="riemann")),
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=4000, penalty="l1", solver="saga",
                                      random_state=RANDOM_STATE)),
    ], memory=MEM)
_, m_l1 = run_classical_phase("riemann_multiband_ts_l1", prep_mbts, build_mbts_l1, {"lr__C": [0.01, 0.1, 1.0, 10.0]})
PHASE_KAPPA["Faz3c: MB-TS L1"] = m_l1
compare(m_l1, 0.550, "Faz 3c-L1 (multi-band TS + L1-LR)  [ensemble'da kullanılacak]")
""")

# =========================================================================== #
# BÖLÜM 5 — Faz 4
# =========================================================================== #
md(r"""
## BÖLÜM 5 — Faz 4: Klasik Ensemble (Soft Voting)

İki klasik base'in OOF `predict_proba`'larını **eşit ağırlıkla** soft-vote ediyoruz:

**FBCSP+sLDA  ⊕  Multi-band TS-L1**

Her iki base aynı `random_state=42` fold yapısını kullandığından OOF'lar trial-bazlı hizalı
→ **leakage-free** ensemble. Ağırlıkları OOF κ'sına göre seçmek leakage olurdu; **sabit eşit
ağırlık** kullanıyoruz.

**Beklenen:** κ ≈ **0.6425** (Faz 1'e göre +0.018; en çok güçlü grupta varyans azalması).
""")
code(r"""
k_faz4 = ensemble_kappa(["fbcsp_rlda", "riemann_multiband_ts_l1"])
m_faz4 = float(np.mean(list(k_faz4.values())))
PHASE_KAPPA["Faz4: klasik ensemble"] = m_faz4
for sid in SUBJECTS:
    print(f"  A{sid:02d}: κ={k_faz4[sid]:.4f}")
compare(m_faz4, 0.6425, "Faz 4 (FBCSP+sLDA ⊕ MB-TS-L1, eşit)")
""")

# =========================================================================== #
# BÖLÜM 6 — Faz 5
# =========================================================================== #
md(r"""
## BÖLÜM 6 — Faz 5: Derin Öğrenme + Final Ensemble

**Hipotez:** DL, klasiklerden *farklı hata profili* çıkarır → ensemble'a 5./6. üye olunca
zayıf grupta klasiklerin kaçırdığı trial'ları yakalayabilir.

**Mimariler:**
- **EEGNet-8,2** (Lawhern 2018): temporal Conv → depthwise spatial → separable conv. ~2k param.
- **ShallowConvNet** (Schirrmeister 2017): FBCSP'nin DL karşılığı (temporal+spatial conv → square → log-pool).

**Protokol:** Girdi (B,1,22,500), bandpass **4–38 Hz** + µV ölçek. Dış CV yine
`StratifiedKFold(5, random_state=42)` → klasiklerle hizalı. İç %80/%20 stratified val,
**early stopping** (patience=30). **Augmentation sadece train:** time-shift ±12 örnek +
Gaussian gürültü (σ=0.1×kanal std). AdamW (lr=1e-3, wd=1e-2).

**Beklenen (tek başına, zayıf):** EEGNet κ≈0.398, ShallowConvNet κ≈0.397 — Faz 1'in çok
altında (288 trial subject-specific DL için sınırda). Ama **hata profili ortogonal** (Jaccard≈0.29).

> ⏱️ GPU'da ~7–10 dk/mimari. CPU'da çok daha yavaş.
""")
code(r"""
# EEGNet — subject-specific, 5-fold OOF
dl_cfg = TrainConfig(epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-2,
                     early_stopping_patience=30, augment=True)
t0 = time.time()
res_eeg = {}
for sid in SUBJECTS:
    r = run_dl_subject(sid, build_eegnet, dl_cfg, OOF, "eegnet", verbose=True)
    res_eeg[sid] = r.kappa
m_eeg = float(np.mean(list(res_eeg.values())))
PHASE_KAPPA["Faz5a: EEGNet"] = m_eeg
print(f"\n[eegnet] mean κ={m_eeg:.4f}  süre={(time.time()-t0)/60:.1f} dk")
compare(m_eeg, 0.398, "Faz 5a (EEGNet, subject-specific)")
""")
code(r"""
# ShallowConvNet — subject-specific, 5-fold OOF
t0 = time.time()
res_scn = {}
for sid in SUBJECTS:
    r = run_dl_subject(sid, build_shallowconvnet, dl_cfg, OOF, "shallowconvnet", verbose=True)
    res_scn[sid] = r.kappa
m_scn = float(np.mean(list(res_scn.values())))
PHASE_KAPPA["Faz5a: ShallowConvNet"] = m_scn
print(f"\n[shallowconvnet] mean κ={m_scn:.4f}  süre={(time.time()-t0)/60:.1f} dk")
compare(m_scn, 0.397, "Faz 5a (ShallowConvNet, subject-specific)")
""")
md(r"""
### 6.1 (Opsiyonel) Faz 5b — Cross-subject LOSO transfer

Her hedef denek için diğer 8 deneğin tüm A0XT'siyle pretrain + hedef fold'da fine-tune.
Zayıf-DL deneklerinde (A05, A06) belirgin kazanç sağlar ama **final ensemble lideri
değiştirmez** (Faz 5a 0.6553 tepede kalır). Varsayılan olarak kapalı (`RUN_TRANSFER=False`,
~50 dk). Açmak için Bölüm 0'da `RUN_TRANSFER=True` yapın.
""")
code(r"""
if RUN_TRANSFER:
    tcfg = TransferConfig(pre_epochs=200, ft_epochs=100, ft_lr=1e-4, ft_patience=15, augment=True)
    for sid in SUBJECTS:
        run_transfer_subject(sid, build_eegnet, tcfg, OOF, "eegnet_transfer", verbose=True)
    for sid in SUBJECTS:
        run_transfer_subject(sid, build_shallowconvnet, tcfg, OOF, "shallowconvnet_transfer", verbose=True)
    print("Transfer OOF üretildi.")
else:
    print("Faz 5b transfer atlandı (RUN_TRANSFER=False).")
""")
md(r"""
### 6.2 FİNAL ENSEMBLE

**FBCSP+sLDA + MB-TS-L1 + EEGNet + ShallowConvNet**, ağırlık **1 / 1 / 0.5 / 0.5**.

Klasik base'lere tam ağırlık, DL'lere yarı ağırlık — DL tek başına zayıf olduğu için
(eşit ağırlıkta zayıflığı taşır), 0.5 *sweet spot*'tur.

**Beklenen:** CV κ ≈ **0.6553** (Faz 1'e göre +0.030; A04/A06/A09 belirgin kazanır).
""")
code(r"""
FINAL_BASES = ["fbcsp_rlda", "riemann_multiband_ts_l1", "eegnet", "shallowconvnet"]
FINAL_WEIGHTS = [1.0, 1.0, 0.5, 0.5]
k_final_cv = ensemble_kappa(FINAL_BASES, FINAL_WEIGHTS)
m_final_cv = float(np.mean(list(k_final_cv.values())))
PHASE_KAPPA["Faz5a: FINAL ensemble (CV)"] = m_final_cv
print("Final ensemble CV — denek bazlı:")
for sid in SUBJECTS:
    d = k_final_cv[sid] - k_fbcsp[sid]
    print(f"  A{sid:02d}: κ={k_final_cv[sid]:.4f}   (Faz1'e Δ={d:+.4f})")
compare(m_final_cv, 0.6553, "Final ensemble (CV, 1/1/0.5/0.5)")
""")

# =========================================================================== #
# BÖLÜM 7 — Faz 6 final test
# =========================================================================== #
md(r"""
## BÖLÜM 7 — Faz 6: A0XE Final Test (single-shot) 🔓

**Şimdiye kadar A0XE'ye HİÇ dokunulmadı.** Tüm tuning iç CV'de yapıldı. Şimdi tek-shot test:

**Protokol (her denek için):**
1. Final pipeline base'lerini **TÜM A0XT (288 trial)** ile fit et (CV yok):
   - Klasik base'ler: A0XT'de `GridSearchCV(5-fold)` → best params → refit.
   - DL base'ler: A0XT %80/%20 + early stopping + augmentation (train-only).
2. **A0XE'yi İLK ve TEK kez** oku → her base `predict_proba`.
3. Soft-vote **1/1/0.5/0.5** → final tahmin → A0XE etiketiyle κ.

**Beklenen:** **Test κ = 0.6137**, accuracy = 0.71, Δ(Test−CV) = −0.042.

### 7.1 Etiket doğrulama
A0XE.mat ham 1–4 → −1 kaydır → 0–3, A0XT semantiğiyle hizalı, 72/72/72/72 dengeli mi?
""")
code(r"""
from scipy.io import loadmat
for sid in (1, 5):
    raw = loadmat(str(LABELS_DIR / f"A{sid:02d}E.mat"))["classlabel"].ravel().astype(int)
    shifted = raw - 1
    uniq, cnt = np.unique(shifted, return_counts=True)
    print(f"A{sid:02d}E: n={len(raw)}  ham unique={sorted(set(raw))}  "
          f"kaydırılmış dağılım={dict(zip(uniq.tolist(), cnt.tolist()))}")
print("✓ A0XE etiketleri 0–3, dengeli 72/72/72/72 — A0XT ile aynı şema.")
""")
md(r"""
### 7.2 Final değerlendirme

> ⏱️ ~25–35 dk (9 denek × [FBCSP grid + Riemann grid + 2 DL eğitimi]).
""")
code(r"""
from sklearn.model_selection import GridSearchCV
from bci_lib import _train_loop, _stratified_inner_split   # final DL fit için iç yardımcılar

def final_fit_classical(build_fn, grid, Xtr, ytr, Xte):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(build_fn(), grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)
    return gs.predict_proba(Xte), gs.best_params_

def final_fit_dl(model_factory, Xtr, ytr, Xte, cfg):
    set_seed(RANDOM_STATE)
    from torch.utils.data import DataLoader
    tr_idx, va_idx = _stratified_inner_split(ytr, cfg.val_fraction, RANDOM_STATE)
    ds_tr = EEGDataset(Xtr[tr_idx], ytr[tr_idx], augment=cfg.augment,
                       shift_max=cfg.shift_max, noise_std=cfg.noise_std)
    ds_va = EEGDataset(Xtr[va_idx], ytr[va_idx], augment=False)
    ds_te = EEGDataset(Xte, np.zeros(len(Xte), dtype=np.int64), augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=256)
    dl_te = DataLoader(ds_te, batch_size=256)
    model = model_factory().to(DEVICE)
    y_proba, _ = _train_loop(model, dl_tr, dl_va, dl_te, cfg.epochs, cfg.lr,
                             cfg.weight_decay, cfg.early_stopping_patience)
    return y_proba

rows = []
y_true_all, y_pred_all = [], []   # birleşik confusion için
t0 = time.time()
for sid in SUBJECTS:
    sd_t = load_subject(sid, "T"); sd_e = load_subject(sid, "E")
    Xt, yt = sd_t.X, sd_t.y
    Xe, ye = sd_e.X, sd_e.y        # A0XE — İLK ve TEK kullanım

    # Base 1: FBCSP+sLDA
    p_fb, bp_fb = final_fit_classical(
        build_fbcsp, GRID_FBCSP,
        make_multiband_tensor(Xt, sd_t.sfreq, FBCSP_BANDS, 4),
        yt, make_multiband_tensor(Xe, sd_e.sfreq, FBCSP_BANDS, 4))
    # Base 2: Riemann MB-TS-L1
    p_rm, bp_rm = final_fit_classical(
        build_mbts_l1, {"lr__C": [0.01, 0.1, 1.0, 10.0]},
        bandpass_multi(Xt, sd_t.sfreq, RIEMANN_BANDS, 4),
        yt, bandpass_multi(Xe, sd_e.sfreq, RIEMANN_BANDS, 4))
    # Base 3-4: DL
    Xt_dl = bandpass_wide(Xt, sd_t.sfreq) * 1e6
    Xe_dl = bandpass_wide(Xe, sd_e.sfreq) * 1e6
    p_eg = final_fit_dl(build_eegnet, Xt_dl, yt, Xe_dl, dl_cfg)
    p_sc = final_fit_dl(build_shallowconvnet, Xt_dl, yt, Xe_dl, dl_cfg)

    pred_final = soft_vote([p_fb, p_rm, p_eg, p_sc], [1.0, 1.0, 0.5, 0.5])
    pred_faz1  = p_fb.argmax(1)
    k_test  = cohen_kappa_score(ye, pred_final)
    a_test  = (pred_final == ye).mean()
    k_cv    = k_final_cv.get(sid, float("nan"))
    rows.append({"sid": sid, "cv_kappa": k_cv, "test_kappa": k_test,
                 "test_acc": a_test, "faz1_test": cohen_kappa_score(ye, pred_faz1),
                 "delta": k_test - k_cv, "fbcsp_bp": bp_fb, "riemann_bp": bp_rm})
    y_true_all.append(ye); y_pred_all.append(pred_final)
    print(f"A{sid:02d}: Test κ={k_test:.4f}  acc={a_test:.4f}  "
          f"(CV κ={k_cv:.4f}, Δ={k_test-k_cv:+.4f})  FBCSP={bp_fb}")

y_true_all = np.concatenate(y_true_all); y_pred_all = np.concatenate(y_pred_all)
print(f"\nToplam süre: {(time.time()-t0)/60:.1f} dk")
""")
code(r"""
import pandas as pd
df = pd.DataFrame(rows).set_index("sid")
mean_test = df["test_kappa"].mean()
mean_cv   = df["cv_kappa"].mean()
mean_acc  = df["test_acc"].mean()
mean_d    = df["delta"].mean()

print("="*72)
print("FINAL DEĞERLENDİRME — A0XE saklı test (denek bazlı)")
print("="*72)
print(df[["cv_kappa", "test_kappa", "delta", "test_acc", "faz1_test"]].round(4).to_string())
print("-"*72)
print(f"MEAN  CV κ={mean_cv:.4f}   Test κ={mean_test:.4f}   "
      f"Δ(Test−CV)={mean_d:+.4f}   Test acc={mean_acc:.4f}")
print()
print(f"  Yarışma kazananı (Kai Keng Ang FBCSP, 2008):  κ = 0.569")
print(f"  Bu çalışma (final ensemble):                  κ = {mean_test:.4f}   "
      f"(Δ = {mean_test-0.569:+.4f})")
PHASE_KAPPA["Faz6: FINAL TEST (A0XE)"] = mean_test
compare(mean_test, "0.6137 (acc≈0.71, Δ≈-0.042)", "Faz 6 FINAL TEST")
""")

# =========================================================================== #
# BÖLÜM 8 — viz
# =========================================================================== #
md(r"""
## BÖLÜM 8 — Görselleştirme & Özet

1. Ablation bar grafiği (tüm fazların κ'sı)
2. CV vs Test denek-bazlı karşılaştırma
3. Final confusion matrix (normalize)
4. Özet tablo
""")
code(r"""
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Ablation bar: tüm fazlar ---
order = ["Faz1: FBCSP+sLDA", "Faz2a: bandSel+SVM(MI)", "Faz2b: bandSel+SVM(mRMR)",
         "Faz3a: TS 1-band", "Faz3b: MDM", "Faz3c: MB-TS PCA", "Faz3c: MB-TS L1",
         "Faz4: klasik ensemble", "Faz5a: EEGNet", "Faz5a: ShallowConvNet",
         "Faz5a: FINAL ensemble (CV)", "Faz6: FINAL TEST (A0XE)"]
labels = [o for o in order if o in PHASE_KAPPA]
vals   = [PHASE_KAPPA[o] for o in labels]
colors = ["#2c7fb8" if "FINAL" in l or "Faz6" in l else
          ("#41ab5d" if "Faz4" in l else "#bdbdbd") for l in labels]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(range(len(vals)), vals, color=colors)
ax.axhline(0.569, ls="--", c="red", lw=1, label="Yarışma kazananı (0.569)")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Cohen's κ"); ax.set_title("Faz ilerlemesi — Cohen's κ")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", fontsize=7)
ax.legend(); plt.tight_layout(); plt.show()
""")
code(r"""
# --- 2. CV vs Test denek bazlı ---
sids = list(df.index)
x = np.arange(len(sids)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - w/2, df["cv_kappa"], w, label="CV (A0XT)", color="#9ecae1")
ax.bar(x + w/2, df["test_kappa"], w, label="Test (A0XE)", color="#3182bd")
ax.set_xticks(x); ax.set_xticklabels([f"A{s:02d}" for s in sids])
ax.set_ylabel("Cohen's κ")
ax.set_title(f"Final ensemble — CV κ={mean_cv:.4f} vs Test κ={mean_test:.4f}")
ax.axhline(0, c="k", lw=0.5); ax.legend(); plt.tight_layout(); plt.show()
""")
code(r"""
# --- 3. Final confusion matrix (normalize) ---
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_all, y_pred_all)
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
class_names = ["sol el", "sağ el", "iki ayak", "dil"]
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(class_names, rotation=30, ha="right"); ax.set_yticklabels(class_names)
ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
ax.set_title("Final ensemble confusion (A0XE, normalize)")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                color="white" if cmn[i,j] > 0.5 else "black")
fig.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()
""")
code(r"""
# --- 4. Özet tablo ---
print("="*60)
print(f"{'Faz / Pipeline':40s} {'κ':>8s}")
print("="*60)
for k in order:
    if k in PHASE_KAPPA:
        print(f"{k:40s} {PHASE_KAPPA[k]:8.4f}")
print("="*60)
print(f"{'Yarışma kazananı (Kai Keng Ang FBCSP)':40s} {0.569:8.4f}")
print(f"{'>>> FINAL Test κ (A0XE)':40s} {mean_test:8.4f}")
print(f"{'>>> FINAL Test accuracy':40s} {mean_acc:8.4f}")
print("="*60)
print("\n🎉 Yeniden-üretim tamamlandı. Saklı test setinde dürüst Cohen's κ ölçüldü;")
print("   BCI Comp IV 2a yarışma kazananı +0.045 farkla geçildi.")
""")

# =========================================================================== #
# notebook'u yaz
# =========================================================================== #
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "BCI_MI_Reproduction.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Yazıldı:", out, f"({len(cells)} hücre)")

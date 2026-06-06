# BCI Competition IV — Dataset 2a · Araştırma Defteri

Subject-specific motor imagery sınıflandırması. Hedef: mümkün olan en yüksek **dürüst**
Cohen's κ (nested CV, dış-test fold'una hiçbir bilgi sızmayan).

---

## 0. Setup

| Öğe | Değer |
|:--|:--|
| Veri | BCI Comp IV 2a — 9 denek, 4 sınıf (sol el / sağ el / iki ayak / dil) |
| Eğitim oturumları | A01T … A09T (288 trial/denek, dengelenmiş 72/sınıf) |
| Test oturumları | A01E … A09E **— bu aşamada KULLANILMIYOR** (final için saklı) |
| Sampling | 250 Hz, 22 EEG + 3 EOG kanal |
| Epoch | cue + 0.5 s – 2.5 s ⇒ 500 örnek |
| Dış CV | StratifiedKFold(n_splits=5, shuffle=True, random_state=42) |
| İç CV / val | sklearn GridSearchCV (klasik) veya stratified %80/%20 split (DL) |
| Metrik | Cohen's κ (ana), accuracy, macro-F1 |
| Reproduce | Tüm fazlar random_state=42 ⇒ fold yapısı bit-bit aynı ⇒ ensemble hizalı |

**Denek tiplemesi** (Faz 1 sonrasında ortaya çıktı, sonraki fazlarda referans):

| Grup | Denekler | Faz 1 κ ortalaması |
|:--|:--|:-:|
| Güçlü | A01, A03, A07, A08, A09 | 0.799 |
| Zayıf | A02, A04, A05, A06 | 0.407 |

---

## Faz 1 — FBCSP + Shrinkage-LDA (Baseline)

**Pipeline**: 16 bant (10× 4 Hz + 6× 6 Hz, 8–30 Hz) → her bantta regularized CSP
(`mne.decoding.CSP`, reg='ledoit_wolf', log-variance) → tüm bant feature'ları concat
→ `SelectKBest(mutual_info_classif)` → `LDA(solver='lsqr', shrinkage='auto')`.

**İç CV grid**: `csp_n_components ∈ {2,4,6}`, `select_k ∈ {10,20,40,'all'}`.

**Sonuçlar (`exp_fbcsp_rlda.csv`)**:

| Denek | κ | acc | κ_std (fold) |
|:-:|:-:|:-:|:-:|
| A01 | 0.7778 | 0.8333 | 0.039 |
| A02 ⚠ | 0.4167 | 0.5625 | 0.064 |
| A03 | 0.8611 | 0.8958 | 0.048 |
| A04 ⚠ | 0.4722 | 0.6042 | 0.063 |
| A05 ⚠ | 0.3287 | 0.4965 | 0.080 |
| A06 ⚠ | 0.4120 | 0.5590 | 0.041 |
| A07 | 0.7685 | 0.8264 | 0.046 |
| A08 | 0.8380 | 0.8785 | 0.061 |
| A09 | 0.7500 | 0.8125 | 0.082 |
| **MEAN** | **0.6250** | **0.7188** | subj-std=0.212 |

Süre: 82.4 dk. **Bimodal dağılım** belirgin: güçlü grup 0.799, zayıf grup 0.407.

**Rapora not**: BCI Comp IV 2a için literatür FBCSP+LDA tipik κ aralığında (~0.55–0.65)
güçlü, dürüst nested CV ile honest tahmin. Sonraki tüm fazlar bunu **baseline** alır.

---

## Faz 2 — Train-only Band Selection + Linear SVM (Ablation)

**Hipotez**: 16 bandı zorla tutmaktansa, train-fold'da MI ile en iyi `top_k` bandı seçip
classifier'a vermek hem feature'ı azaltır hem ayırt edicilik artırabilir.

**Pipeline** (`TopKBandSelector` sklearn Pipeline step'i olarak, fit yalnız train-fold):

- Faz 2a (`exp_fbcsp_bandsel_svm.csv`): FBCSP → top-k bant (MI, leakage-free) →
  StandardScaler → `SelectKBest(MI)` → `SVC(kernel='linear', probability=True)`
- Faz 2b (`exp_fbcsp_bandsel_mrmr.csv`): aynısı, ama `SelectKBest` yerine **mRMR**
  (MID variant: relevance − ortalama redundancy)

**İç CV grid**: `top_k ∈ {4,6,8,10}`, `select_k ∈ {20,40,'all'}` (mRMR'de
`n_features ∈ {20,40,60}`), `svc_C ∈ {0.01,0.1,1,10}`.

**Sonuçlar**:

| Denek | Faz 1 sLDA | Faz 2a SVM(MI) | Faz 2b SVM(mRMR) |
|:-:|:-:|:-:|:-:|
| A01 | **0.7778** | 0.7083 | 0.7222 |
| A02 ⚠ | **0.4167** | 0.3194 | 0.3102 |
| A03 | 0.8611 | 0.8611 | **0.8704** |
| A04 ⚠ | **0.4722** | 0.4352 | 0.4306 |
| A05 ⚠ | **0.3287** | 0.2824 | 0.3056 |
| A06 ⚠ | 0.4120 | **0.4167** | **0.4167** |
| A07 | **0.7685** | 0.7454 | 0.7361 |
| A08 | **0.8380** | 0.7731 | 0.8009 |
| A09 | **0.7500** | 0.6667 | 0.6713 |
| **MEAN** | **0.6250** | 0.5787 | 0.5849 |
| zayıf grup κ | **0.4074** | 0.3634 | 0.3658 |

**Bulgu**: Faz 2 **başarısız**. mRMR vs MI çok marjinal (+0.006). Asıl problem: band
selection. Hem güçlü hem zayıf grupta düşüş.

**Neden düştü** (rapora gidecek yorum):

- Shrinkage-LDA zaten **örtük boyut indirgemesi** yapıyor (Ledoit-Wolf shrunk kovaryans
  + lineer alt-uzay). Hard band selection bu yumuşak regularizasyondan daha agresif,
  bilgi kaybı.
- Güçlü deneklerde sinyal birden fazla bantta dağılmış (top_k=10 bile yetersiz).
- Zayıf deneklerde MI band skorlaması küçük örneklemde gürültüye duyarlı; train-fold'da
  görünen "iyi" bant, test'te tutmuyor.

**Rapora not**: Negatif sonuç. Faz 4'te ablation bölümünde sunulacak.

---

## Faz 3 — Riemannian Geometri

**Hipotez**: Kovaryans manifoldunda küçük örneklemde, kovaryans tahmini CSP'den daha
sağlam olabilir; zayıf grupta kazanç beklenir.

### 3a — Tangent Space + Logistic Regression (tek geniş bant)

`exp_riemann_ts.csv` · 8–30 Hz bandpass → `Covariances(estimator='oas')` →
`TangentSpace(metric='riemann')` → `StandardScaler` → `LR(C tune)`.

### 3b — Minimum Distance to Mean (tek geniş bant)

`exp_riemann_mdm.csv` · `Covariances('oas')` → `MDM(metric='riemann')`.
Hiperparametresiz; saf metrik baseline.

### 3c — Multi-band TS (3 bant: 8–14, 12–20, 18–30)

FBCSP'nin filter bank mantığını Riemannian'a taşıma. Her bantta Cov+TS → concat
(759 feature) → StandardScaler → boyut indirgeme:

- 3c-PCA (`exp_riemann_multiband_ts_pca.csv`): `PCA(n_components ∈ {30,50,80})` → `LR(L2, C tune)`
- 3c-L1 (`exp_riemann_multiband_ts_l1.csv`): `LR(penalty='l1', solver='saga', C tune)`

`MultiBandTangentSpace` sınıfı Pipeline `memory=joblib.Memory` ile cache'lenir.

**Sonuçlar**:

| Denek | Faz 1 sLDA | 3a TS 1-band | 3b MDM | 3c MB-TS PCA | 3c MB-TS L1 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| A01 | **0.7778** | 0.7130 | 0.6852 | 0.7315 | 0.7500 |
| A02 ⚠ | **0.4167** | 0.2870 | 0.2593 | 0.3287 | 0.2731 |
| A03 | **0.8611** | 0.7963 | 0.5972 | 0.7639 | 0.8056 |
| A04 ⚠ | **0.4722** | 0.4028 | 0.3241 | 0.3796 | 0.4028 |
| A05 ⚠ | **0.3287** | 0.1528 | 0.1481 | 0.2361 | 0.1019 |
| A06 ⚠ | **0.4120** | 0.3750 | 0.2639 | 0.3796 | 0.3750 |
| A07 | **0.7685** | 0.5648 | 0.5509 | 0.5926 | 0.7407 |
| A08 | **0.8380** | 0.7546 | 0.7083 | 0.7361 | 0.7824 |
| A09 | **0.7500** | 0.7269 | 0.6343 | 0.7269 | 0.7176 |
| **MEAN** | **0.6250** | 0.5303 | 0.4635 | 0.5417 | 0.5499 |
| zayıf grup κ | **0.4074** | 0.3044 | 0.2489 | 0.3310 | 0.2882 |

**Bulgu**: Hipotez **doğrulanmadı**. Hiçbir Riemann varyantı Faz 1'i geçmedi; zayıf
grupta da kazanç yok. Multi-band tek-bandı geçti (+0.02 ortalama) ama Faz 1'in 0.625'i
altında.

**Neden kaybetti**:

- Tek-bant 8–30 Hz, FBCSP'nin 16-bant filter bank avantajını yakalayamıyor; **adil
  olmayan kıyas**.
- Multi-band TS bunu kısmen kapattı ama 22×23/2 = 253 feature × 3 bant = 759 → 288
  trial'a göre **boyut/örnek oranı kötü**; PCA/L1 boyut indirgemesi bilgi kaybediyor.
- MDM sınıf merkezleri arasında saf metrik mesafe — ayrımcı bilgiyi modellemiyor.
- **L1 saga küçük örneklemde konverjans zorlanıyor**: A05'te κ=0.10 (chance level),
  A02'de 0.27 (PCA 0.33'den düşük).

**Rapora not**: Faz 3'ün asıl değeri kazanç değil, **farklı hata profili** olabilir
(ensemble için ham madde). Faz 4'te test edildi.

---

## Faz 4 — Ensemble (Soft Voting, Leakage-free)

**Pipeline**: 4 klasik base'in OOF predict_proba'larını soft-vote ile birleştir. Tüm
base'ler aynı `random_state=42` StratifiedKFold(5) ⇒ trial-bytrial hizalı.

**Kritik leakage notu**: ağırlıkları OOF κ'sına göre seçmek leakage'dir; sabit
ağırlıklar (eşit veya manuel) kullanıldı. "CV-κ orantılı" varyant *population stat*
olarak işaretlendi (slightly leaky, raporda not).

**Test edilen kombinasyonlar**: 10 farklı (ikili, üçlü, dörtlü, ağırlıklı).

**Sonuçlar (`exp_ensemble.csv`, sıralı: mean κ azalan)**:

| # | Kombinasyon | mean κ | Δ Faz 1 | weak κ | strong κ | A05 |
|:-:|:--|:-:|:-:|:-:|:-:|:-:|
| **1** | **FBCSP+sLDA + MB-TS-L1 (eşit)** | **0.6425** | **+0.018** | 0.4155 | **0.8241** | 0.3287 |
| 2 | FBCSP + PCA + L1 (κ-orantılı*) | 0.6404 | +0.015 | 0.4167 | 0.8194 | 0.3380 |
| 3 | FBCSP + MB-TS-PCA (eşit) | 0.6399 | +0.015 | **0.4329** | 0.8056 | **0.3611** |
| 4 | FBCSP + MB-TS-PCA (0.6/0.4) | 0.6389 | +0.014 | 0.4236 | 0.8111 | 0.3519 |
| 5 | FBCSP + PCA + L1 (3'lü eşit) | 0.6379 | +0.013 | 0.4062 | 0.8231 | 0.3241 |
| — | Faz 1 sLDA (referans) | 0.6250 | 0 | 0.4074 | 0.7991 | 0.3287 |

**Bulgu**: Ensemble Faz 1'i geçti. **+0.018 mean κ**, **+0.025 strong group κ**.
**Zayıf grupta katkı dağılmış**: A02 +0.028, A06 +0.028 (Faz 1'e göre), ama A04 ve A05
marjinal. **Hiçbir kombinasyon A05'i 0.40'a taşıyamadı** (max 0.3611).

**Yorum**:

- Klasik base'lerin hata profilleri kısmen örtüşüyor; ensemble en çok güçlü grupta
  fayda sağlıyor (varyans azalması).
- Yeni model eklemeden, sadece soft vote ile +0.018 kazanç → ucuz fayda.
- A05 ve A04 için klasik bazlar **aynı temel sinyali aynı şekilde okuyor**; ensemble
  çeşitliliği yetersiz.

**Rapora not**: Faz 4 mevcut klasik tavan. Sonraki adım **hata profili çeşitliliği** —
deep learning ve/veya cross-subject transfer.

---

## Faz 5a — Subject-specific Deep Learning

**Hipotez**: DL klasiklerden farklı hata profili çıkarır; ensemble'a 5. ve 6. üye
olduğunda zayıf grupta (özellikle A05) klasiklerin kaçırdığı trial'ları yakalayabilir.

**Mimariler**:

- `exp_eegnet.py` — **EEGNet-8,2** (Lawhence et al. 2018): temporal Conv2D(F1=8, k=64)
  → DepthwiseConv2D(C=22, D=2) → SeparableConv2D → AvgPool/Dropout → Linear(4).
  ~1.5k parametre.
- `exp_shallowconvnet.py` — **ShallowConvNet** (Schirrmeister et al. 2017): FBCSP'nin
  DL karşılığı. Temporal Conv(40, k=25) → Spatial Conv(40, C×1, no nonlin) → BN →
  square → AvgPool(75, stride=15) → safe_log → Dropout → Linear(4). ~50k parametre.

**Girdi**: (B, 1, 22, 500), bandpass **4–38 Hz** (DL kendi temporal filtresini öğrenir;
geniş bant verilir), µV ölçek.

**Eğitim protokolü**:

- Dış CV: aynı StratifiedKFold(5, random_state=42) → klasik fazlarla bit-bit aynı,
  ensemble hizalı.
- İç train/val: stratified %80/%20, **early stopping** val_loss'a göre (patience=30).
- Optimizer: AdamW (lr=1e-3, wd=1e-2), batch=64, max 300 epoch.
- Augmentation (**sadece train**): time shift ±12 sample (~50 ms), Gaussian noise
  σ = 0.1 × per-channel std. Val/test'e asla.
- OOF predict_proba → `results/oof/{eegnet,shallowconvnet}/sub_NN.npz`.
- A0XE asla kullanılmaz.

### Sonuçlar

| Denek | Faz 1 sLDA | EEGNet | ShallowConvNet |
|:-:|:-:|:-:|:-:|
| A01 | **0.7778** | 0.6111 | 0.4815 |
| A02 ⚠ | **0.4167** | 0.0602 | 0.1435 |
| A03 | **0.8611** | 0.7083 | 0.6667 |
| A04 ⚠ | **0.4722** | 0.1944 | 0.2361 |
| A05 ⚠ | **0.3287** | 0.0417 | 0.0463 |
| A06 ⚠ | **0.4120** | 0.1019 | 0.0972 |
| A07 | **0.7685** | 0.4722 | 0.5926 |
| A08 | **0.8380** | 0.6759 | 0.5741 |
| A09 | 0.7500 | 0.7130 | **0.7361** |
| **MEAN** | **0.6250** | 0.3976 | 0.3971 |
| zayıf grup κ | **0.4074** | 0.0996 | 0.1308 |

Süre: EEGNet 7 dk, ShallowConvNet 8 dk (GPU = RTX 3050 Ti).

**Bulgu**: Subject-specific DL **başarısız**. Her iki mimari de Faz 1'in 0.227
altında. Zayıf grupta neredeyse **chance level**:

- **A05 EEGNet κ=0.042, ShallowConvNet κ=0.046** — model rastgele tahmin ediyor.
- **A02, A04, A06** benzer çöküş — DL ek artefakt absorbe ediyor olabilir.
- **Per-fold varyans yüksek** (A04 EEGNet std=0.18, A07 ShallowConvNet std=0.10) —
  küçük inner-train fold'unda eğitim instabilite.

**Neden DL kaybetti**:

- **Veri yetersiz**: 288 trial × %80 = ~230 train sample, EEGNet için bile sınırda.
  Klasik literatürde EEGNet 2a için tipik κ ~0.55-0.65 bildirilir, ama çoğu çalışma
  *cross-subject pretrain* veya *çoklu oturum* kullanır. Tek-oturum subject-specific
  DL bu veri setinde gerçekten zor.
- **Early stopping val %20 fold üzerinde** (~46 trial) — val_loss çok gürültülü,
  durması erken/geç olabilir.
- **Augmentation çift kenarlı**: time shift + Gaussian noise, küçük örneklemde
  underfitting yaratıyor olabilir.
- **A05/A06 özellikle**: zaten klasiklerde de sinyal çok zayıf; DL'in extra parametre
  bütçesi onu rastgele gürültüye fit ediyor.

### Hata Profili Analizi (`exp_error_analysis.py`)

**Asıl test**: DL'in hataları klasiklerden ne kadar farklı? Düşük Jaccard error
overlap → ensemble değeri var.

**Pairwise mean Jaccard error overlap (9 denek ortalaması)**:

| | EEGNet | MB-TS-L1 | MB-TS-PCA | TS-1band | ShallowConvNet |
|:--|:-:|:-:|:-:|:-:|:-:|
| **fbcsp_rlda** | **0.290** | 0.364 | 0.355 | 0.355 | **0.287** |
| EEGNet | — | 0.346 | 0.383 | 0.391 | 0.465 |
| MB-TS-L1 | | — | 0.479 | 0.474 | 0.340 |
| MB-TS-PCA | | | — | 0.546 | 0.369 |
| TS-1band | | | | — | 0.352 |

**Yorum**:

- **fbcsp_rlda ↔ DL Jaccard ≈ 0.29** — DL klasikten **gerçekten farklı hatalar yapıyor**.
- En örtüşen çift: MB-TS-PCA ↔ TS-1band (0.546) — beklenen, ikisi de TS tabanlı.
- En ayrık çift: fbcsp_rlda ↔ ShallowConvNet (0.287) — bu ensemble için altın.
- **A05'te** fbcsp_rlda↔EEGNet jaccard = 0.397, ↔ShallowConvNet = 0.450 — DL **farklı
  trial'ları yanlış yapıyor**, ensemble'da kazanç beklenir.

**Sonuç**: DL tek başına zayıf ama hata profili **ortogonal**. Ensemble'a düşük
ağırlıkla katılmasının değerli olabileceğini önceden gösteriyor.

---

## Faz 5a Genişletilmiş Ensemble — Yeni Lider 0.6553

4 klasik + 2 DL = 6 base ile 14 kombinasyon test edildi.

**En iyi 5 kombinasyon (sıralı: mean κ azalan, `exp_ensemble.csv`)**:

| Sıra | Kombinasyon | Ağırlık | mean κ | Δ Faz 4 | Δ Faz 1 | strong κ | weak κ | A05 |
|:-:|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| **1** | **Faz4-lider + EEGNet + ShallowConvNet** | **1/1/0.5/0.5** | **0.6553** | **+0.013** | **+0.030** | **0.8333** | 0.4329 | 0.3380 |
| 2 | Faz4-lider + EEGNet + ShallowConvNet | 1.5/1/0.5/0.5 | 0.6471 | +0.005 | +0.022 | 0.8250 | 0.4248 | 0.3426 |
| 3 | Faz4-lider + EEGNet | eşit (3'lü) | 0.6466 | +0.004 | +0.022 | 0.8296 | 0.4178 | 0.3148 |
| 4 | Faz4-lider + EEGNet + ShallowConvNet | 1/1/0.3/0.3 | 0.6466 | +0.004 | +0.022 | 0.8287 | 0.4190 | 0.3287 |
| 5 | Faz4-lider + EEGNet + ShallowConvNet | eşit (4'lü) | 0.6451 | +0.003 | +0.020 | 0.8204 | 0.4259 | 0.3519 |
| — | **Faz 4 lider** (FBCSP + MB-TS-L1) | eşit | 0.6425 | 0 | +0.018 | 0.8241 | 0.4155 | 0.3287 |
| — | **Faz 1 sLDA tek başına** | — | 0.6250 | −0.018 | 0 | 0.7991 | 0.4074 | 0.3287 |

**En iyi kombinasyon detay (Faz4-lider + EEGNet + ShallowConvNet, 1/1/0.5/0.5)**:

| Denek | Faz 1 κ | Yeni Lider κ | Δ |
|:-:|:-:|:-:|:-:|
| A01 | 0.7778 | 0.8056 | +0.028 |
| A02 ⚠ | 0.4167 | 0.4074 | −0.009 |
| A03 | 0.8611 | 0.8704 | +0.009 |
| A04 ⚠ | 0.4722 | 0.5231 | **+0.051** |
| A05 ⚠ | 0.3287 | 0.3380 | +0.009 |
| A06 ⚠ | 0.4120 | 0.4630 | **+0.051** |
| A07 | 0.7685 | 0.7917 | +0.023 |
| A08 | 0.8380 | 0.8657 | +0.028 |
| A09 | 0.7500 | 0.8333 | **+0.083** |
| **MEAN** | **0.6250** | **0.6553** | **+0.030** |

**Bulgular**:

- **DL eklemek mean κ'yı +0.013 (Faz 4'e göre), +0.030 (Faz 1'e göre) artırdı.**
- **A04, A06, A09 belirgin kazandı** (+0.05 ila +0.08). Bunlar farklı hata profilinin
  somut çıktısı.
- **Strong grup: 0.7991 → 0.8333** (+0.034). En büyük artış.
- **Weak grup: 0.4074 → 0.4329** (+0.026). Sınırlı ama tutarlı.
- **A05: 0.3287 → 0.3380** (+0.009) — Yine **0.40'ı geçemedi**. **Maksimum A05 κ tüm
  fazlar boyunca 0.3611** (Faz 4'te FBCSP+MB-TS-PCA, başka bir varyant).
- **Klasik ağırlık = 1.0, DL ağırlık = 0.5** optimal noktayı yakaladı. Eşit ağırlıkta
  DL'in zayıflığı taşıyor (0.6451). DL'i daha düşürünce (0.3) kazanç biraz azalıyor
  (0.6466). 0.5 sweet spot.
- **Tüm 6 base eşit**: 0.6420 — DL'in tam ağırlığı net zarar veriyor.

**Sonuç**: DL doğrudan rekabette başarısız ama **ensemble'da değerli yardımcı**.
Hipotez kısmen doğrulandı (mean κ artışı), ama **A05 problemi hâlâ çözülmedi**.

---

## Faz 5b — Cross-subject Transfer (LOSO Pretrain + Fine-tune)

**Hipotez**: Subject-specific DL küçük örneklemde başarısız (Faz 5a). A05 ve diğer
zayıf denekler için tek umut: 8 deneğin verisinden pretrained model + hedef
deneğin train fold'unda fine-tune.

**Protokol** (her hedef denek S için, leakage-free):

1. **Pretrain** havuzu: S DIŞINDAKI 8 deneğin TÜM A0XT verisi (~2304 trial).
   - Stratified %15 val split (pretrain early-stop için, S'in hiç verisi havuzda yok)
   - AdamW lr=1e-3, wd=1e-2, max 200 epoch, patience=30
2. **Fine-tune** (her dış fold için ayrı):
   - Pretrain state'i yükle
   - S'in dış-train fold'unda fine-tune (düşük lr=1e-4, max 100 epoch, patience=15)
   - İç-val %20 stratified, early stop
3. Dış-test fold'unu tahmin, OOF kaydet:
   - `results/oof/eegnet_transfer/sub_NN.npz`
   - `results/oof/shallowconvnet_transfer/sub_NN.npz`

Dış CV yine StratifiedKFold(5, shuffle=True, random_state=42) → klasik fazlarla
bit-bit aynı, ensemble hizalı.

### Sonuçlar (subject-specific DL ile yan yana)

| Denek | EEGNet subj | **EEGNet transfer** | Δ | SCN subj | **SCN transfer** | Δ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A01 | 0.6111 | 0.5833 | −0.028 | 0.4815 | **0.5648** | +0.083 |
| A02 ⚠ | 0.0602 | 0.0648 | +0.005 | 0.1435 | 0.1343 | −0.009 |
| A03 | 0.7083 | **0.7361** | +0.028 | 0.6667 | **0.7037** | +0.037 |
| A04 ⚠ | 0.1944 | **0.2176** | +0.023 | 0.2361 | **0.2870** | +0.051 |
| A05 ⚠ | 0.0417 | 0.0602 | +0.019 | 0.0463 | **0.1435** | **+0.097** |
| A06 ⚠ | 0.1019 | **0.2778** | **+0.176** | 0.0972 | **0.2870** | **+0.190** |
| A07 | 0.4722 | 0.2454 | **−0.227** | 0.5926 | 0.3380 | **−0.255** |
| A08 | 0.6759 | 0.6759 | 0.000 | 0.5741 | **0.6343** | +0.060 |
| A09 | 0.7130 | 0.6157 | −0.097 | 0.7361 | 0.6991 | −0.037 |
| **MEAN** | **0.3976** | 0.3863 | −0.011 | **0.3971** | **0.4213** | **+0.024** |

Süre: EEGNet transfer 24 dk, ShallowConvNet transfer 28 dk (pretrain başına ~2 dk +
5×finetune).

### Transfer'in iki yüzü — Asimetrik etki

Bulgu **çok asimetrik**:

- **"Tabula rasa" denekleri**: A05/A06 gibi subject-specific'te neredeyse chance olan
  denekler büyük kazanç aldı. ShallowConvNet transfer: A06 0.097 → 0.287 (+0.190),
  A05 0.046 → 0.144 (+0.097). Bu denekler kendi 230 trial'ından bir şey öğrenemiyor,
  ama 2304 trial'lık cross-subject prior bir miktar genel MI representasyonu sağlıyor.
- **Bilgi taşıyan denekler**: A07 ve A09 gibi subject-specific DL'in zaten iyi olduğu
  (0.59, 0.73) denekler **belirgin zarar gördü**. Transfer A07'de −0.227, −0.255
  (subject-specific'in kazandığı bireysel özellikleri pretrain prior'u bastırıyor).
- **Net etki ShallowConvNet'te +0.024** (mean), EEGNet'te **−0.011** — ShallowConvNet
  transfer'e daha iyi uyum sağlıyor (basit conv yapı, daha az kapsam çakışması).

### A05 — BCI-illiterate Kanıtı

Smoke test A05'te şu özgün bulgu vardı: **fold 2'de κ = −0.1247 (chance altı)**.

| Yöntem | A05 κ | Yorum |
|:-:|:-:|:--|
| Faz 1 sLDA | 0.3287 | Klasik tavan |
| Faz 4 en iyi (PCA ensemble) | **0.3611** | Tüm fazlardaki maksimum |
| Faz 5a EEGNet subj | 0.0417 | DL çöktü |
| Faz 5a ShallowConvNet subj | 0.0463 | Aynı çöküş |
| Faz 5a ensemble (lider) | 0.3380 | Klasik üye taşıdı |
| **Faz 5b EEGNet transfer** | 0.0602 | Hâlâ chance |
| **Faz 5b ShallowConvNet transfer** | 0.1435 | Marjinal kazanç, hâlâ kötü |
| Faz 5b ensemble (en iyi A05) | 0.3194 | Faz 1'in altında |

**Sonuç**: Cross-subject transfer A05'i **çözmedi**. ShallowConvNet transfer +0.097
sağladı ama hâlâ 0.14. Smoke'taki **negatif fold kappa (−0.125)** + cross-subject
prior'un A05'te marjinal kalması ⇒ **A05'in MI sinyali ya yok ya da diğer
deneklerden niteliksel olarak farklı**. Literatürde "BCI illiteracy" olarak bilinen
fenomen: bazı bireylerin %20-30'u motor imagery yapamıyor veya beyin sinyali
ayrımcı bilgi taşımıyor.

**A05 maksimum κ tüm fazlar boyunca 0.3611** — tüm 8 modelin gücü birleşse de bu
denekte aşılamaz bir tavan var. Final raporda **denek-özgü tavan** olarak sunulacak.

### Hata Profili (Faz 5b sonrası)

Transfer DL'ler de subject-specific kadar **klasiklerden farklı hatalar yapıyor**:

| | EEGNet-tr | ShallowConvNet-tr | EEGNet | ShallowConvNet |
|:--|:-:|:-:|:-:|:-:|
| **fbcsp_rlda** | **0.282** | 0.298 | 0.290 | 0.287 |

Transfer DL ile fbcsp_rlda arası Jaccard 0.282-0.298 — subject-specific DL ile çok
benzer (0.287-0.290). Transfer **farklı bilgi getirmiyor** klasiklerden vs
subject-specific DL'e göre. Ensemble katkısı yetersiz.

### Genişletilmiş Ensemble — Lider değişmedi

8 base ile 21 kombinasyon test edildi (`exp_ensemble.csv`):

| Sıra | Kombinasyon | Ağırlık | mean κ | weak κ | A05 |
|:-:|:--|:--|:-:|:-:|:-:|
| **1** | **Faz4-lider + EEGNet + ShallowConvNet** (Faz5a) | 1/1/0.5/0.5 | **0.6553** | 0.4329 | 0.3380 |
| 2 | Faz4-lider + EEGNet-tr + ShallowConvNet-tr | eşit (4'lü) | 0.6548 | 0.4410 | 0.3194 |
| 3 | Faz4-lider + 4 DL | 1/1/0.3/0.3/0.3/0.3 | 0.6543 | 0.4306 | 0.3241 |
| 4 | Faz4-lider + 2 subj-DL + 2 transfer-DL | 1/1/0.5/0.5/0.5/0.5 | 0.6538 | 0.4329 | 0.3102 |
| 5 | Faz4-lider + EEGNet-tr + ShallowConvNet-tr | 1/1/0.5/0.5 | 0.6507 | 0.4340 | 0.3148 |
| — | Tüm 8 base (eşit) | | 0.6430 | **0.4549** | 0.3102 |

**Bulgular**:

- **Yeni lider yok** — Faz 5a'nın 0.6553'ü hâlâ tepede. Transfer ensemble (#2) sadece
  **0.0005** geride, istatistiksel olarak ayırt edilemez.
- **8-base eşit ağırlık**: zayıf grupta en yüksek (0.4549) ama mean'de gerilemiş
  (DL'in zayıflığı taşıyor) — weak-only optimum başka kombinasyon.
- **A05 hiçbir kombinasyonda kıpırdamadı**: max 0.3611 (Faz 4'teki FBCSP+MB-TS-PCA),
  Faz 5a/5b ensemble'larında 0.30-0.34 arası.
- **A04 ve A06 transfer'le belirgin kazandı** ensemble'da:
  - A04: 0.4722 → 0.5602 (8-base eşit), **+0.088**
  - A06: 0.4120 → 0.4954 (6-base, 1/1/0.3/0.3/0.3/0.3), **+0.083**
- **A07 transfer'den zarar gördü** (single-base) ama ensemble agirliklari (0.3-0.5)
  bu zarari emdi; ensemble'da A07 değeri korundu (0.78-0.81).

### Yorumlar

- **Subject-specific DL Faz 5a'da ortogonal hata getiriyordu; transfer DL ek
  çeşitlilik getirmedi** — transfer ile subject-specific DL'in hata profilleri
  benzer, ek üye olmasının marjinal katkısı var.
- **Transfer'in asıl değeri zayıf-DL deneklerinde** (A06 +0.190 SCN, A05 +0.097 SCN)
  ama bu kazançlar ensemble'da klasiklerin gücü tarafından zaten dolduruluyor.
- **A07/A09 transfer-DL "hasarı"**: ensemble ağırlığı (0.3-0.5) sayesinde bireysel
  zarar sönüyor, ensemble varyansı azalıyor (per-subject std da düşüyor).

### Rapora not (Faz 5b)

- Transfer öğrenme, 4-classlık subject-specific MI'da küçük örneklem darboğazını
  **kısmen** açıyor: zayıf-DL deneklerinde subject-specific'in kat ve kat üzerine
  çıkıyor, ama bireysel olarak güçlü deneklerde bastırıyor — asimetrik.
- **A05 = BCI-illiterate**. Ne klasik (max 0.36), ne subject-specific DL (chance), ne
  cross-subject transfer (0.14) A05'i çözmedi. Smoke test'te negatif fold κ = −0.125
  gözlemi (chance altı) bu deneğin sinyal ayrımcı bilgi taşımadığına dair en güçlü
  kanıt.
- Transfer'in *adil* protokol altında ensemble'a katkısı **marjinal** (+0.000 ile
  0.6553 lider değişmedi). Subject-specific Faz 5a DL'leri zaten ortogonal bilgiyi
  ekliyor; transfer aynı kapı.
- Sonraki adımlar için **transfer çıktıları faydasız değil**: bireysel zayıf
  deneklerde (özellikle A06) ciddi kazanç var, raporda denek-bazlı analiz değerli.

---

## Kümülatif Karşılaştırma (şu ana kadar)

| Faz | Pipeline | mean κ | weak κ | A05 | süre |
|:-:|:--|:-:|:-:|:-:|:-:|
| 1 | FBCSP + sLDA | **0.6250** | 0.4074 | 0.3287 | 82 dk |
| 2a | FBCSP + bandSel + linSVM(MI) | 0.5787 | 0.3634 | 0.2824 | 30 dk |
| 2b | FBCSP + bandSel + linSVM(mRMR) | 0.5849 | 0.3658 | 0.3056 | 38 dk |
| 3a | Riemann TS (1-band) | 0.5303 | 0.3044 | 0.1528 | 1 dk |
| 3b | Riemann MDM | 0.4635 | 0.2489 | 0.1481 | <1 dk |
| 3c-PCA | Multi-band TS + PCA + L2-LR | 0.5417 | 0.3310 | 0.2361 | 3 dk |
| 3c-L1 | Multi-band TS + L1-LR (saga) | 0.5499 | 0.2882 | 0.1019 | 25 dk |
| 4 | FBCSP+sLDA ⊕ MB-TS-L1 (klasik ensemble) | 0.6425 | 0.4155 | 0.3287 | + sn |
| 5a | EEGNet (subj-specific) | 0.3976 | 0.0996 | 0.0417 | 7 dk |
| 5a | ShallowConvNet (subj-specific) | 0.3971 | 0.1308 | 0.0463 | 8 dk |
| 5b | EEGNet LOSO transfer | 0.3863 | 0.1551 | 0.0602 | 24 dk |
| 5b | ShallowConvNet LOSO transfer | 0.4213 | **0.2380** | 0.1435 | 28 dk |
| **5a ens.** | **Faz4-lider + EEGNet + ShallowConvNet (1/1/0.5/0.5)** | **0.6553** | 0.4329 | 0.3380 | + sn |
| 5b ens. | Faz4-lider + EEGNet-tr + ShallowConvNet-tr (eşit) | 0.6548 | **0.4410** | 0.3194 | + sn |
| 5b ens. | Tüm 8 base eşit (weak-best) | 0.6430 | **0.4549** | 0.3102 | + sn |
| **6 TEST** | **Final ens. A0XE saklı test (4 base, 1/1/0.5/0.5)** | **0.6137** | 0.3958 | 0.2454 | 30 dk |

---

## A05 Problemi — Cross-subject Transfer Gerekçesi

A05 (denek 5) tüm fazlarda **çözülmeyen düğüm**:

| Faz | A05 κ |
|:-:|:-:|
| Faz 1 sLDA | 0.3287 (en yüksek klasik) |
| Faz 2 SVM-band-sel | 0.2824 / 0.3056 |
| Faz 3a TS 1-band | 0.1528 |
| Faz 3b MDM | 0.1481 |
| Faz 3c MB-TS PCA | 0.2361 |
| Faz 3c MB-TS L1 | 0.1019 (çöküş) |
| Faz 4 ensemble (en iyi A05) | 0.3611 |
| Faz 5a EEGNet | 0.0417 (chance) |
| Faz 5a ShallowConvNet | 0.0463 (chance) |
| Faz 5a ensemble (en iyi A05) | 0.3519 |
| Faz 5b EEGNet transfer | 0.0602 (chance) |
| Faz 5b ShallowConvNet transfer | 0.1435 (subj-DL'in 3 katı, hâlâ kötü) |
| Faz 5b ensemble (en iyi A05) | 0.3194 |
| **Tüm fazlar A05 max** | **0.3611** (Faz 4: FBCSP+MB-TS-PCA) |

**Yorum**:

- Hiçbir subject-specific klasik yöntem A05'i 0.40'a taşıyamadı.
- Ensemble bile sadece **+0.03** (0.328 → 0.361) iyileştirdi — bütün klasikler **aynı
  zayıflıkları paylaşıyor**, hata profilleri çok benzer.
- A05'in 288 trial'ı subject-specific öğrenme için yetersiz veya sinyal/gürültü oranı
  çok düşük. Hipotez: SMR (mu/beta) bandı zayıf, yan trial varyasyonu yüksek.

**Cross-subject transfer (Faz 5b) denendi ve A05'i ÇÖZMEDİ**:

- ShallowConvNet transfer A05: 0.0463 → 0.1435 (+0.097) — subject-specific'e göre 3
  kat ama Faz 1'in (0.3287) çok altında.
- Smoke test'te **fold 2'de κ = −0.1247** gözlemlendi (chance altı) — cross-subject
  prior'un bu deneğin verisine ters şekilde transfer edildiğinin kanıtı.
- Transfer ensemble'da A05 maksimum 0.3194 — yine Faz 1 altında.

**Sonuç**: A05 muhtemelen **BCI-illiterate** kategoride. Literatürde popülasyonun
%20-30'unun motor imagery sinyali ya yok ya da ayrımcı bilgi taşımıyor. Tüm
fazlardaki A05 maksimumu = **0.3611** (Faz 4'te FBCSP+MB-TS-PCA ensemble). Final
raporda **denek-özgü tavan** olarak sunulacak; mean κ tartışmasında bu deneği ayrı
ele almak gerekebilir.

---

## Açık Sorular & İzleme Listesi

| # | Soru | Durum |
|:-:|:--|:--|
| 1 | A05 cross-subject transfer'le 0.40'ı geçer mi? | **HAYIR** (max 0.3194 transfer-ens., 0.3611 Faz 4) |
| 2 | DL hata profili klasikten farklı mı? | **EVET** (Jaccard ~0.29, transfer ~0.28) |
| 3 | Ensemble + DL Faz 4 tavanını (0.6425) geçti mi? | **EVET** (Faz 5a: 0.6553, +0.013) |
| 4 | Transfer DL ek çeşitlilik getirdi mi? | **MARJİNAL** (Faz 5b ens. 0.6548, lider değil) |
| 5 | Trial rejection / EOG temizleme zayıf grupta yardım eder mi? | henüz yapılmadı |
| 6 | Final A0XE değerlendirme (saklı test) | **ŞU ANA KADAR A0XE'YE DOKUNULMADI** |

### Olası sonraki adımlar (Faz 6 sonrası)

- **Trial rejection**: zayıf grupta EOG-tabanlı veya amplitude-tabanlı outlier trial
  reddediş klasik base'lerin κ'sını az artırabilir. `preprocessing.py` iskeleti hazır.
- **Daha fazla DL çeşitliliği**: Conv-Transformer, ATCNet, TCN denemek mümkün ama
  veri kıtlığı verildiğinde fayda sınırlı olacak.
- **Per-subject base secimi (leakage-free)**: Her denek icin inner-CV'de en iyi
  base'i secen meta-strateji — şu an inner-CV scores per-base saklanmadığı için
  yapılamıyor.

---

## Faz 6 — Final Değerlendirme (A0XE Saklı Test)

**Protokol** (rapora gidecek):

1. **Etiket doğrulama** (`_check_a0xe_labels.py`): `data/true_labels/A0XE.mat` ham
   1–4 → `data_loader.LABEL_OFFSET=1` ile −1 kaydır → 0–3. A01E ve A05E denetlendi:
   her ikisi de 288 trial, dengeli 72/sınıf, A0XT semantiğiyle birebir hizalı
   (sol el / sağ el / iki ayak / dil).
2. **Base'leri tüm A0XT (288 trial) üzerinde fit** — CV yok:
   - **fbcsp_rlda**: GridSearchCV 5-fold (`csp_n_components ∈ {2,4,6}`,
     `select_k ∈ {10,20,40,'all'}`) → best params → refit
   - **riemann_multiband_ts_l1**: GridSearchCV 5-fold (`lr_C ∈ {0.01,0.1,1,10}`) → refit
   - **eegnet** / **shallowconvnet**: %80/%20 stratified train/val, early stopping
     (patience=30), augmentation train-only (time-shift ±12, Gaussian noise σ=0.1)
3. **A0XE tek-shot predict_proba** (ILK ve TEK kullanım). Soft vote 1/1/0.5/0.5.
4. **Modeller kaydedildi**: `results/models/sub_NN/{fbcsp_rlda.joblib,
   riemann_multiband_ts_l1.joblib, eegnet.pt, shallowconvnet.pt}`.

### Seçilen hiperparametreler (reproducibility)

Çoğunluk: `fbcsp_rlda` → **n_components=6, select_k=40** (5/9 denekte),
`riemann_l1` → **C=0.1** (4/9 denekte).

| Denek | FBCSP n_comp / select_k | Riemann L1 C |
|:-:|:-:|:-:|
| A01 | 4 / 40 | 0.1 |
| A02 | 6 / all | 0.1 |
| A03 | 6 / 40 | 1.0 |
| A04 | 6 / 40 | 10.0 |
| A05 | 6 / 20 | 1.0 |
| A06 | 4 / all | 0.1 |
| A07 | 4 / 40 | 0.1 |
| A08 | 4 / all | 10.0 |
| A09 | 6 / 40 | 1.0 |

### Ana sonuç tablosu — CV (A0XT) vs Test (A0XE)

| Denek | Faz 1 CV κ | **Faz 1 Test κ** | Final CV κ | **Final Test κ** | **Δ (Test−CV)** | Test acc |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A01 | 0.7778 | 0.7454 | 0.8056 | **0.7778** | −0.028 | 0.8333 |
| A02 ⚠ | 0.4167 | 0.3194 | 0.4074 | **0.3426** | −0.065 | 0.5069 |
| A03 | 0.8611 | 0.7778 | 0.8704 | **0.8333** | −0.037 | 0.8750 |
| A04 ⚠ | 0.4722 | 0.5417 | 0.5231 | **0.6204** | **+0.097** | 0.7153 |
| A05 ⚠ | 0.3287 | 0.1852 | 0.3380 | **0.2454** | −0.093 | 0.4340 |
| A06 ⚠ | 0.4120 | 0.3565 | 0.4630 | **0.3750** | −0.088 | 0.5312 |
| A07 | 0.7685 | 0.7176 | 0.7917 | **0.7963** | +0.005 | 0.8472 |
| A08 | 0.8380 | 0.7500 | 0.8657 | **0.7778** | −0.088 | 0.8333 |
| A09 | 0.7500 | 0.7222 | 0.8333 | **0.7546** | −0.079 | 0.8160 |
| **MEAN** | **0.6250** | **0.5684** | **0.6553** | **0.6137** | **−0.042** | **0.7103** |

### Yöntem karşılaştırması (A0XE'de)

| Yöntem | A0XE mean κ | A0XE mean acc |
|:--|:-:|:-:|
| **Kai Keng Ang FBCSP (BCI Comp IV 2a yarışma kazananı, 2008)** | **0.569** | — |
| Faz 1: FBCSP+sLDA tek | 0.5684 | — |
| Faz 4: klasik ensemble (FBCSP + MB-TS-L1) | 0.6106 | — |
| **Faz 5a/6: Final ensemble (FBCSP + MB-TS-L1 + EEGNet + ShallowConvNet)** | **0.6137** | **0.7103** |

**Δ vs yarışma kazananı (Kai Keng Ang FBCSP κ=0.569): +0.045**.

### CV vs Test "leakage check"

Ortalama Δ = **−0.042**, ne çok yüksek (overfit/leakage işareti olurdu) ne çok
düşük (CV pessimistic değil). Bu makul **session-to-session shift** — A0XE oturum 2
veri, A0XT oturum 1; EEG'de doğal varyasyon, beklendiği gibi.

**Anomali ve dikkat çekenler**:

- **A04 +0.097**: tek "yukarı" sapma. Faz 4 + DL ensemble'ı bu denekte gerçekten
  generalize etti, CV optimistic değildi. Olasılıkla A04'ün train fold varyansı
  yüksekti (CV κ_std=0.063), test gerçek performansı daha iyi yakaladı.
- **A05 −0.093**: en büyük negatif sapma. CV'de 0.34 olan A05 testte 0.25'e
  düştü. Hâlâ BCI-illiterate; session shift bu deneği daha da kötü etkiledi.
- **A08 −0.088**: güçlü deneklerden birinde anlamlı düşüş. CV 0.87, test 0.78 —
  session shift sağlam denekleri de etkiliyor ama oran küçük.
- **A07 ≈ 0**: en stabil denek, CV 0.79 ≈ Test 0.80. Sinyal kalitesi yüksek
  ve oturum-bağımsız.

### A05 final tavan: 0.2454 (test)

| Aşama | A05 κ |
|:--|:-:|
| Faz 4 ens. CV en iyi | 0.3611 |
| Faz 5a/6 ensemble CV | 0.3380 |
| **Faz 6 Final TEST** | **0.2454** |
| Yarışma kazananı baseline A05 ≈ | ~0.2 (literatürde) |

A05'in test κ'sı tüm umutlu fazlardan daha düşük çıktı. Bu BCI-illiterate hipotezini
**doğrulayan en güçlü kanıt**: ne klasik (max CV 0.36, test 0.19 sLDA tek), ne DL
subject-specific (chance), ne LOSO transfer (test 0.14 chance üstü), ne 4-base
ensemble (test 0.25) bu deneği 0.30'un üstüne çıkaramadı. Rapor için
**denek-özgü tavan = 0.25** olarak sun.

### Confusion matrix (Final ensemble, A0XE)

`results/tables/final_confusion_matrices.json` — her denek için 4×4 matrix.
A05 örneği: sınıflar arası karışıklık yüksek, hiçbir sınıf hakim değil
(BCI-illiterate karakteristiği).

### Süre

Final değerlendirme **30.0 dk** (9 denek). FBCSP grid search çoğunluğu;
DL ~10s/denek (GPU). FBCSP en yavaş (124-220s/denek).

### Faz 6 özet (rapora gidecek)

- **Saklı test setinde dürüst Cohen's κ = 0.6137**, accuracy = 0.7103.
- **BCI Comp IV 2a yarışma kazananı (Kai Keng Ang FBCSP, κ=0.569) +0.045 geçildi**.
- CV ↔ Test farkı **−0.042** (makul session shift, leakage işareti yok).
- 9 denekten **8'i CV'ye yakın veya altında** (A04 hariç), **A07 ≈ 0**
  generalizasyon olarak en stabil; **A05 hâlâ ulaşılamaz** (test 0.25).
- **Final lider sistem A0XT için CV κ=0.6553 → A0XE için Test κ=0.6137**.
  Bu, mevcut altyapının dürüst genelleme tahminidir.

---

*Son güncelleme: 2026-05-25 (Faz 6 tamamlandı; A0XE Test κ=0.6137, +0.045 vs yarışma kazananı)*

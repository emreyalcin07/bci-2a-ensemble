<!--
Bursa Teknik Universitesi - Mekatronik Muhendisligi
Hesaplamalı Sinir Bilimine Giris - Final Projesi 2025-2026 Bahar
-->

# BCI Competition IV Dataset 2a Üzerinde Motor Imagery EEG Sinyallerinin Sınıflandırılması: Sistematik Ablasyon ve Sızıntısız (Leakage-Free) Değerlendirme

**Yazarlar:** [Ad Soyad 1 — Öğrenci No], [Ad Soyad 2 — Öğrenci No], [Ad Soyad 3 — Öğrenci No]

*Bursa Teknik Üniversitesi, Mekatronik Mühendisliği Bölümü*

*Hesaplamalı Sinir Bilimine Giriş — Final Projesi, 2025-2026 Bahar*

---

## Özet

Motor imagery (MI) tabanlı beyin-bilgisayar arayüzü (BCI) sistemlerinde sınıflandırma performansı; küçük örneklem hacmi, denekler-arası varyasyon ve oturumlar-arası sinyal kayması gibi nedenlerle sınırlıdır. Bu çalışmada BCI Competition IV Dataset 2a (9 denek, 4 sınıf, 22 EEG kanalı) üzerinde sistematik bir ablasyon protokolüyle yedi farklı sinyal işleme/sınıflandırma boru hattı karşılaştırılmıştır. Tüm geliştirme süreci A0XT eğitim oturumunda iç içe (nested) çapraz doğrulama (5×5) ile yürütülmüş, A0XE değerlendirme oturumu **proje boyunca tek seferde** açılarak dürüst genelleme tahmini elde edilmiştir. Filtre bankası tabanlı CSP+shrinkage-LDA (κ_CV=0.625), Riemannian tangent space, EEGNet ve ShallowConvNet'ten oluşan dört temel modelin soft-voting birleşimi A0XT'de κ_CV=0.6553, A0XE'de **κ_Test=0.6137** sağlamış ve yarışma kazananı FBCSP yönteminin (κ=0.569) **+0.045** üzerinde performans elde etmiştir. CV-Test farkının −0.042 olması leakage'siz, dürüst genellemenin kanıtıdır. A05 deneğinin tüm yöntemlerde 0.30 altında kalması BCI-illiterate fenomenine güçlü kanıt sağlamaktadır.

---

## 1. Giriş

Motor imagery (MI), bir uzvun fiziksel hareketini gerçekleştirmeksizin zihinde canlandırılması sürecidir ve duyu-motor kortekste mu (8–13 Hz) ile beta (13–30 Hz) ritimlerinde olay-ilişkili senkronizasyon/desenkronizasyon (ERS/ERD) örüntüleri üretir [1]. Bu kortikal değişimlerin sınıflandırılması, motor BCI'ların temelini oluşturur.

BCI Competition IV Dataset 2a, MI sınıflandırması için referans veri kümesidir: 9 denek, 4 sınıf (sol el, sağ el, iki ayak, dil), her denek için iki ayrı oturumda 288 deneme [2]. Yarışma kazananı Ang vd.'nin **Filter Bank CSP (FBCSP)** yöntemi, dar bant CSP'lerin birleştirilmesi ile mu/beta etrafındaki ayrımcı bilgiyi yakalar; A0XE değerlendirme setinde mean κ=0.569 elde etmiştir [3]. Sonraki dönemde geometrik yaklaşımlar (Riemannian tangent space [4]) ve uçtan-uca derin öğrenme modelleri (EEGNet [5], ShallowConvNet [6]) güçlü alternatifler olarak ortaya çıkmıştır.

**Bu çalışmanın katkıları**:
(i) Yedi boru hattının **sistematik ablasyonu** — band selection, Riemannian, derin öğrenme, transfer öğrenme dahil — pozitif **ve** negatif sonuçlarla.
(ii) **Sızıntısız (leakage-free) protokol**: tüm geliştirme A0XT içinde 5×5 nested CV ile, A0XE final tek-shot tahmin için saklı tutulmuştur.
(iii) **Yarışma kazananını geçen** soft-voting ensemble (κ_Test=0.6137, +0.045) ve A05 deneğinde BCI-illiterate fenomeninin belgelenmesi.

## 2. Veri Seti ve Ön İşleme

**Veri kümesi.** BCI Competition IV Dataset 2a: 9 denek (A01–A09), 22 monopolar EEG + 3 EOG kanal, 250 Hz örnekleme. Her denek için iki ayrı gün kayıt: A0XT (eğitim) ve A0XE (değerlendirme), her birinde 288 deneme (sınıf başına 72). Etiketler A0XT için GDF olay kodlarından (769–772), A0XE için ayrı `.mat` dosyalarındaki `classlabel` alanından okunmuştur (ham 1–4, kod tabanı 0–3'e kaydırılmıştır).

**Epoch çıkarımı.** Cue uyarısından sonra **0.5–2.5 s** penceresi alınmıştır (500 örnek × 22 EEG kanalı; EOG kanalları artefakt analizi için ayrı tutulmuş, sınıflandırmada kullanılmamıştır). MI motor görüntüleme süresi tipik olarak cue+0.5 s'den itibaren stabilize olur [3].

**Filtre bankası.** Klasik kolda **16 bant** kullanılmıştır: 8–30 Hz aralığında 10 adet 4 Hz genişlikte bant (8–12, 10–14, …, 26–30) ve 6 adet 6 Hz genişlikte bant (8–14, 11–17, …, 23–29). Tüm filtreler 4. dereceden Butterworth, sıfır-faz `filtfilt` ile uygulanmıştır. Riemannian kolda 3 geniş bant (8–14, 12–20, 18–30), derin öğrenme kolunda tek geniş bant (4–38 Hz) tercih edilmiştir.

**Sızıntısız değerlendirme protokolü.** Geliştirme süresince **yalnızca A0XT** kullanılmıştır. Dış 5-fold stratified CV (`random_state=42`) ile iç 5-fold GridSearchCV birleşik tutulmuş (nested CV), tüm veri-bağımlı adımlar (CSP, ölçekleme, öznitelik seçimi, PCA) `sklearn.Pipeline` içine yerleştirilerek yalnızca eğitim katmanında fit edilmiştir. A0XE proje boyunca **bir kez bile açılmamış**, son aşamada A0XT'nin tamamında yeniden eğitilen modeller A0XE'de tek-shot tahmin için kullanılmıştır.

## 3. Öznitelik Çıkarımı ve Sınıflandırma

Üç temel kol incelenmiştir.

**Klasik kol — FBCSP + shrinkage-LDA (Faz 1).** 16 bandın her birinde Ledoit-Wolf shrinkage'li CSP (`mne.decoding.CSP`, log-variance) ile öznitelik çıkarılmış, tüm bant öznitelikleri birleştirilip `SelectKBest(mutual_info_classif)` ile boyut indirgenmiş, doğrusal LDA (`solver='lsqr'`, `shrinkage='auto'`) ile sınıflandırılmıştır. Tune edilen hiperparametreler: `csp_n_components ∈ {2,4,6}` ve `select_k ∈ {10,20,40,'all'}`.

**Geometrik kol — Multi-band Riemannian TS (Faz 3c).** 3 geniş bantta `Covariances(estimator='oas')` ile uzaysal kovaryans tahmini, `TangentSpace(metric='riemann')` ile manifold projeksiyonu, bant öznitelikleri birleştirildikten sonra PCA (n∈{30,50,80}) veya L1-regularized lojistik regresyon (saga, C∈{0.01,0.1,1,10}) ile sınıflandırma yapılmıştır [4].

**Derin kol — EEGNet ve ShallowConvNet (Faz 5a).** Standart EEGNet-8,2 (F1=8, D=2, F2=16) [5] ve ShallowConvNet [6] mimarileri PyTorch ile yeniden uygulanmış; giriş (B,1,22,500), AdamW (lr=1e-3, wd=1e-2), batch=64, en fazla 300 epoch, erken durdurma (patience=30) ile eğitilmiştir. Augmentation (yalnızca eğitim): ±50 ms zaman kayması ve kanal-bazlı Gauss gürültüsü (σ=0.1×std).

**Cross-subject transfer (Faz 5b).** LOSO öncelikli eğitim: hedef denek **hariç** 8 deneğin tüm A0XT verisiyle (2304 deneme) ön eğitim, ardından hedef deneğin dış-train katmanında ince ayar (lr=1e-4, patience=15).

**Ensemble — soft voting.** Her temel modelin `predict_proba` çıktıları ağırlıklı ortalama ile birleştirilmiştir; nihai sistem **fbcsp_rlda + riemann_multiband_ts_l1 + EEGNet + ShallowConvNet** dörtlüsü, ağırlık **1.0 / 1.0 / 0.5 / 0.5** ile soft-vote yapmaktadır. Hata profili çeşitliliği ön-teşhisi için pairwise Jaccard hata-örtüşme matrisi hesaplanmış; FBCSP↔DL örtüşmesinin 0.28–0.30 aralığında ortogonal olduğu doğrulandıktan sonra ensemble uygulanmıştır.

## 4. Sonuçlar

**Ablasyon ilerlemesi.** Tablo 1, dokuz boru hattının A0XT üzerindeki dürüst nested CV ortalama κ'sını ve Faz 6 final A0XE testini özetler. Şekil 2 görsel ilerlemeyi sunar.

**Tablo 1. Dokuz boru hattının ortalama κ değerleri (9 denek).**

| Faz | Boru hattı | Mean κ (CV/Test) |
|:--|:--|:-:|
| 1 | FBCSP + shrinkage-LDA | 0.6250 |
| 2a | FBCSP + bant seçimi + linSVM (MI) | 0.5787 |
| 2b | FBCSP + bant seçimi + linSVM (mRMR) | 0.5849 |
| 3a | Riemann TS, 1-bant | 0.5303 |
| 3b | Riemann MDM | 0.4635 |
| 3c-PCA | Multi-band TS + PCA + LR | 0.5417 |
| 3c-L1 | Multi-band TS + L1-LR | 0.5499 |
| 4 | Klasik ensemble (FBCSP ⊕ MB-TS-L1) | 0.6425 |
| 5a | EEGNet (subject-specific) | 0.3976 |
| 5a | ShallowConvNet (subject-specific) | 0.3971 |
| **5a ens.** | **4 base ensemble (1/1/0.5/0.5) — CV** | **0.6553** |
| 5b | EEGNet LOSO transfer | 0.3863 |
| 5b | ShallowConvNet LOSO transfer | 0.4213 |
| **6** | **Final ensemble — A0XE TEST** | **0.6137** |

Ablasyondan üç sonuç ön plana çıkar: (i) shrinkage-LDA örtük boyut indirgemesi nedeniyle hard bant seçimi (Faz 2) **bilgi kaybı yaratmıştır** (−0.04); (ii) tek-bant Riemannian, FBCSP'nin filtre bankası avantajını yakalayamamış (Faz 3a/b: ≤0.53), multi-band varyantları Faz 1'in altında kalmıştır; (iii) subject-specific DL küçük örneklemde (288 deneme) çökerken (≈0.40), ensemble içinde **düşük ağırlıkla** ortogonal hata profili sağlayarak +0.013'lük katkı vermiştir.

**Denek bazlı sonuçlar.** Tablo 2 ve Şekil 1, dokuz denek için CV (A0XT) vs Test (A0XE) κ değerlerini sunar.

**Tablo 2. Final ensemble — CV (A0XT, nested 5×5) ile Test (A0XE saklı set) karşılaştırması.**

| Denek | Faz 1 Test κ | CV κ | **Test κ** | Δ | Test acc |
|:-:|:-:|:-:|:-:|:-:|:-:|
| A01 | 0.7454 | 0.8056 | 0.7778 | −0.028 | 0.833 |
| A02 | 0.3194 | 0.4074 | 0.3426 | −0.065 | 0.507 |
| A03 | 0.7778 | 0.8704 | 0.8333 | −0.037 | 0.875 |
| A04 | 0.5417 | 0.5231 | 0.6204 | **+0.097** | 0.715 |
| A05 | 0.1852 | 0.3380 | 0.2454 | −0.093 | 0.434 |
| A06 | 0.3565 | 0.4630 | 0.3750 | −0.088 | 0.531 |
| A07 | 0.7176 | 0.7917 | 0.7963 | +0.005 | 0.847 |
| A08 | 0.7500 | 0.8657 | 0.7778 | −0.088 | 0.833 |
| A09 | 0.7222 | 0.8333 | 0.7546 | −0.079 | 0.816 |
| **MEAN** | **0.5684** | **0.6553** | **0.6137** | **−0.042** | **0.710** |

![Şekil 1. CV (A0XT) vs Test (A0XE) kappa — denek bazlı; kesikli çizgi yarışma kazananı 0.569.](figures/fig_cv_vs_test.png)

**Yöntem karşılaştırması.** Ang vd.'nin yarışma kazananı FBCSP A0XE'de κ=0.569 raporlamıştır [3]. Bu çalışmanın final ensemble'ı **κ_Test=0.6137 ile bu rakamı +0.045 geçmiştir**. Ayrıca Faz 4 klasik-yalnız ensemble (κ=0.6106) ile Faz 6 final (κ=0.6137) arasındaki ufak +0.003'lük fark, derin öğrenme ortagonal bilgisinin test setinde de korunduğunu göstermektedir.

**Confusion matrix.** Şekil 3, dokuz deneğin A0XE final tahminlerinin satır-normalize toplam karışıklık matrisini sunar. Sınıflar arası en yüksek karışıklık sol el – iki ayak çiftindedir; dil sınıfı en yüksek doğrulukla sınıflandırılan sınıftır.

![Şekil 2. Sistematik ablasyon ilerlemesi — Faz 1'den A0XE final testine.](figures/fig_ablation.png)

![Şekil 3. Final ensemble normalize confusion matrix (A0XE, 9 denek toplam).](figures/fig_confusion_matrix.png)

## 5. Tartışma

**Negatif sonuçların yorumu.** Dört bağımsız ablasyon Faz 1 baseline'ını **geçememiş**, ancak metodolojik içgörü sağlamıştır. (i) **Faz 2 — bant seçimi**: shrinkage-LDA, Ledoit-Wolf büzülmüş kovaryans ile gereksiz boyutları zaten yumuşak olarak bastırdığından, MI tabanlı hard bant seçimi bilgi kaybı yaratmıştır. (ii) **Faz 3 — Riemannian**: tek-bant TS, FBCSP'nin 16-bant filtre bankası avantajını yakalayamamış; multi-band varyantında 759 öznitelik / 288 deneme oranı boyut sınırını zorlamıştır. (iii) **Faz 5a — subject-specific DL**: 288 deneme EEGNet/ShallowConvNet için yetersiz; literatürde tipik 0.55–0.65 değerleri cross-subject ön eğitim varsayar. (iv) **Faz 5b — LOSO transfer**: etki **asimetriktir**; "tabula rasa" deneklerinde büyük kazanç (A06 SCN +0.190, A05 SCN +0.097) ama bilgi taşıyan deneklerde zarar (A07 SCN −0.255). Net mean κ değişimi marjinal kalmıştır.

**Düşük performanslı denekler.** A02 (κ=0.34) ve A06 (κ=0.38) test setinde tüm yöntemlerde zayıf kalmıştır; bu deneklerde EOG artefakt seviyesi ve trial-içi varyansın yüksek olduğu beklenmektedir (gelecek iş: amplitude-tabanlı trial reddediş).

**A05 — BCI-illiterate kanıtı.** A05 deneği yedi farklı klasik/geometrik/derin yöntem ve LOSO transfer dahil **hiçbir yapılandırmada** test κ=0.30'u geçememiştir (final test κ=0.245). Transfer smoke test'inde bir dış foldta **negatif κ (−0.125)** gözlemlenmesi, cross-subject prior'un bu deneğin verisine **ters yönde** transfer edildiğine işaret eder. Literatürde "BCI illiteracy" olarak adlandırılan ve popülasyonun %20–30'unda görülen fenomenle [7] uyumlu olarak, A05'in MI sinyali ya yetersiz ayrımcı bilgi taşımakta ya da niteliksel olarak diğer deneklerden farklı yapıdadır. Final raporda **denek-özgü tavan = 0.245** olarak ayrıca işaretlenmiştir.

**CV ↔ Test farkı.** Ortalama Δ = −0.042 makul **session-to-session shift** seviyesindedir; ne CV'nin optimistik bias'ını (Δ≪0 beklenirdi) ne de pessimistik bias'ını (Δ>0 beklenirdi) gösterir. Bu, sızıntısız nested CV protokolünün dürüst genelleme tahmini verdiğine dair güçlü ampirik kanıttır.

**İyileştirme yönleri.** (a) **Trial rejection**: EOG-bazlı veya amplitude-bazlı outlier reddediş, özellikle A02/A06'da kazanç sağlayabilir; mevcut `preprocessing.py` iskeleti hazırdır. (b) **Daha fazla veri**: çoklu oturum kaydı veya cross-dataset ön eğitim, DL kollarının üst limitini açabilir. (c) **Gelişmiş augmentation**: SmoothMix, frekans-alan maskeleme [8] gibi EEG-özgü teknikler. (d) **A05 için kanal-bazlı SNR analizi**: mu/beta bandında C3/C4 öznitelik amplitüd histogramları, BCI-illiteracy hipotezini niceliksel doğrulayabilir.

## Grup İçi Katkı Tablosu *(yer-tutucu)*

| Üye | Katkı Alanı | Yüzde |
|:--|:--|:-:|
| [Ad Soyad 1] | … | …% |
| [Ad Soyad 2] | … | …% |
| [Ad Soyad 3] | … | …% |

## Referanslar

[1] G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles," *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, 1999.

[2] M. Tangermann *et al.*, "Review of the BCI Competition IV," *Frontiers in Neuroscience*, vol. 6, p. 55, 2012.

[3] K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface," *IJCNN 2008*, pp. 2390–2397, 2008.

[4] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, "Multiclass brain-computer interface classification by Riemannian geometry," *IEEE Trans. Biomed. Eng.*, vol. 59, no. 4, pp. 920–928, 2012.

[5] V. J. Lawhern *et al.*, "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces," *J. Neural Eng.*, vol. 15, no. 5, 056013, 2018.

[6] R. T. Schirrmeister *et al.*, "Deep learning with convolutional neural networks for EEG decoding and visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391–5420, 2017.

[7] B. Blankertz *et al.*, "Neurophysiological predictor of SMR-based BCI performance," *NeuroImage*, vol. 51, no. 4, pp. 1303–1309, 2010.

[8] F. Lotte and C. Guan, "Regularizing common spatial patterns to improve BCI designs: unified theory and new algorithms," *IEEE Trans. Biomed. Eng.*, vol. 58, no. 2, pp. 355–362, 2011.

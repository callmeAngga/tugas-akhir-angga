# Techno-Economic Assessment (TEA) Sistem PV Atap Berbasis Prediksi *Machine Learning* Iradiansi Matahari

Repositori Tugas Akhir ini berisi keseluruhan artefak penelitian — mulai dari
pengolahan data, pemodelan *machine learning* untuk prediksi iradiansi matahari
(*Global Horizontal Irradiance*, GHI), hingga aplikasi web interaktif untuk
melakukan **penilaian kelayakan tekno-ekonomi (Techno-Economic Assessment)**
pemasangan panel surya atap (*rooftop PV*).

Aplikasi memungkinkan pengguna menggambar area atap pada peta, mengonfigurasi
asumsi sistem dan ekonomi, lalu memperoleh estimasi produksi energi tahunan dan
kelayakan finansial (**Net Present Value / NPV**) secara *real-time*.

---

## Daftar Isi

- [Ringkasan Metodologi](#ringkasan-metodologi)
- [Struktur Repositori](#struktur-repositori)
- [Alur Kerja Penelitian](#alur-kerja-penelitian)
- [Dataset](#dataset)
- [Pemodelan Machine Learning](#pemodelan-machine-learning)
- [Aplikasi Web (Techno-Economic Assessment)](#aplikasi-web-techno-economic-assessment)
- [Cara Menjalankan](#cara-menjalankan)
- [Hasil Utama](#hasil-utama)
- [Referensi Ilmiah](#referensi-ilmiah)

---

## Ringkasan Metodologi

Penelitian ini menggabungkan **prediksi berbasis data** (*machine learning*) dan
**pemodelan fisika radiasi surya** untuk menghasilkan penilaian tekno-ekonomi yang
kredibel:

1. **Prediksi GHI** menggunakan model *gradient boosting* (**LightGBM** dan
   **XGBoost**) yang dilatih pada data meteorologi dan geospasial.
2. **Forecast 2026** dibentuk dari *future covariates* berbasis **klimatologi
   harian** (rata-rata *day-of-year* 2021–2025).
3. **Dekomposisi** GHI menjadi komponen langsung (DNI) dan difus (DHI) dengan
   model **Erbs et al. (1982)**.
4. **Transposisi** ke bidang panel (*Plane of Array*, POA) dengan model
   **Perez et al. (1990)** melalui pustaka `pvlib`.
5. **Estimasi produksi energi tahunan (AEP)** dan analisis finansial **NPV** dengan
   memperhitungkan degradasi panel, eskalasi tarif, *self-consumption*, dan OPEX.

---

## Struktur Repositori

```text
tugas-akhir-angga/
├── README.md                     # Dokumen ini
├── app/                          # Aplikasi web Flask (TEA interaktif)
│   ├── app.py                    # Routing & endpoint API
│   ├── service.py                # Pipeline: prediksi GHI → POA → energi → NPV
│   ├── requirements.txt          # Dependensi aplikasi
│   └── templates/                # Antarmuka (landing, peta, panel konfigurasi)
├── ml/                           # Pipeline machine learning
│   ├── notebooks/                # Notebook tahapan penelitian (01–06)
│   ├── data/
│   │   ├── raw/                  # Data meteorologi harian mentah 2021–2025
│   │   ├── processed/            # Dataset fitur hasil rekayasa
│   │   └── results/              # Metrik model, prediksi, ringkasan TEA
│   └── models/                   # Model terlatih (LightGBM & XGBoost)
└── docs/                         # Bahan, referensi, dan dokumen pendukung
```

---

## Alur Kerja Penelitian

Notebook pada folder `ml/notebooks/` disusun secara berurutan agar mudah ditelusuri:

| Notebook | Tahapan |
|---|---|
| `01_data_preparation.ipynb` | Pembersihan & penggabungan data meteorologi mentah |
| `02_exploratory_analysis.ipynb` | Analisis data eksploratif (EDA) |
| `03_feature_engineering.ipynb` | Rekayasa fitur (siklikal, geospasial, meteorologi) |
| `04_lightgbm_model.ipynb` | Pelatihan & tuning model LightGBM |
| `05_xgboost_model.ipynb` | Pelatihan & tuning model XGBoost |
| `06_model_comparison_and_tea.ipynb` | Perbandingan model & penilaian tekno-ekonomi |

---

## Dataset

- **Sumber variabel** (harian, 2021–2025): GHI, DNI, DHI, Suhu (*Temperature*),
  Kelembapan (*Humidity*), dan Presipitasi (*Precipitation*).
- **Cakupan spasial**: beberapa grid koordinat (lat/lon) dengan target utama
  wilayah **Jakarta** (≈ −6,2; 106,8).
- **Dataset fitur akhir**: `ml/data/processed/features_dataset.csv`.

**Fitur skenario METEO** (fitur yang tersedia secara operasional untuk prediksi):

```text
lat, lon, month, day_of_year,
month_sin, month_cos, doy_sin, doy_cos,   # encoding siklikal musiman
TEMP, RH, PRECIP                          # variabel meteorologi
```

---

## Pemodelan Machine Learning

Dua skenario fitur dievaluasi untuk setiap algoritma:

- **METEO** — hanya menggunakan variabel meteorologi & geospasial (dipakai di
  aplikasi karena dapat diprediksi ke masa depan melalui klimatologi).
- **FULL** — mencakup fitur tambahan untuk pembanding performa.

Model produksi dipilih **otomatis** berdasarkan **Test RMSE terendah** pada
skenario METEO (lihat `_select_best_model()` di `app/service.py`).

**Contoh metrik (LightGBM · METEO · data uji):**

| Metrik | Nilai |
|---|---|
| RMSE | ~1,057 kWh/m²/hari |
| MAE | ~0,793 kWh/m²/hari |
| R² | ~0,27 |

> Metrik lengkap kedua algoritma tersimpan pada
> `ml/data/results/*_results.json` dan `model_comparison.csv`.

---

## Aplikasi Web (Techno-Economic Assessment)

Aplikasi **Flask** (`app/`) menyediakan antarmuka interaktif:

- **Peta interaktif** (Leaflet + Leaflet.Draw) untuk menggambar area atap dan
  menghitung luas secara geodesik.
- **Rekomendasi orientasi otomatis** — *tilt* optimal dicari dengan memaksimalkan
  POA tahunan (pemindaian Perez), dan *azimuth* diarahkan menghadap ekuator.
- **Panel konfigurasi** untuk modul PV serta parameter ekonomi (CAPEX, tarif,
  diskonto, degradasi, *self-consumption*, OPEX, eskalasi tarif).
- **Hasil analisis**: kapasitas terpasang (kWp), estimasi GHI tahunan, produksi
  energi tahunan (AEP), *specific yield*, grafik profil GHI 2026, dan **NPV**.

### Endpoint API

| Metode | Rute | Fungsi |
|---|---|---|
| `GET` | `/` | Halaman *landing* |
| `GET` | `/app` | Halaman analisis (peta + konfigurasi) |
| `POST` | `/api/recommend` | Rekomendasi *tilt* & *azimuth* untuk lokasi |
| `POST` | `/api/analyze` | Menjalankan pipeline TEA dan mengembalikan hasil |

### Parameter Ekonomi Default

| Parameter | Nilai default |
|---|---|
| CAPEX | Rp 15.000.000 / kWp |
| Tarif listrik | Rp 1.444,70 / kWh |
| Umur proyek | 25 tahun |
| Tingkat diskonto | 6% |
| Degradasi tahunan | 0,5% |
| *Performance Ratio* | 0,80 |
| *Self-consumption* | 0,65 |
| OPEX | 1% CAPEX / tahun |
| Eskalasi tarif | 3% / tahun |

---

## Cara Menjalankan

### 1. Prasyarat

- Python 3.10+ (disarankan)
- `pip` dan `venv`

### 2. Menyiapkan lingkungan virtual

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Memasang dependensi

Untuk menjalankan **aplikasi web**:

```powershell
pip install -r app\requirements.txt
```

Untuk menjalankan **notebook** (pemodelan ML):

```powershell
pip install pandas numpy matplotlib jupyter scikit-learn xgboost lightgbm pvlib joblib
```

### 4. Menjalankan aplikasi web

```powershell
python app\app.py
```

Kemudian buka `http://127.0.0.1:5000` pada peramban.

### 5. Menjalankan notebook

```powershell
jupyter notebook
```

Jalankan notebook `ml/notebooks/` secara berurutan (01 → 06).

---

## Hasil Utama

Ringkasan TEA (lokasi target Jakarta, tersimpan pada
`ml/data/results/tea_summary.json`):

- **Sumber daya multi-tahun (2021–2025)**: GHI rata-rata ≈ **1.721 kWh/m²/tahun**
  (koefisien variasi ≈ 3,8%).
- **Forecast 2026**: GHI ≈ **1.651 kWh/m²/tahun**, dengan rentang
  ketidakpastian (P10–P90) hasil simulasi **Monte Carlo (N = 300)** pada variabel
  TEMP/RH/PRECIP.
- **Produksi energi (AEP)** dan **NPV** dihitung dinamis di aplikasi sesuai
  konfigurasi pengguna.

> Artefak hasil lengkap: `ml/data/results/` (prediksi model, perbandingan model,
> profil energi TEA, forecast GHI 2026).

---

## Referensi Ilmiah

- **Erbs, D. G., Klein, S. A., & Duffie, J. A. (1982).** Estimation of the diffuse
  radiation fraction for hourly, daily and monthly-average global radiation.
  *Solar Energy.* — dasar dekomposisi GHI → DHI/DNI.
- **Perez, R., et al. (1990).** Modeling daylight availability and irradiance
  components from direct and global irradiance. *Solar Energy.* — dasar transposisi
  ke bidang panel (POA).
- **`pvlib-python`** — pustaka pemodelan sistem fotovoltaik yang digunakan untuk
  posisi matahari, dekomposisi, dan transposisi radiasi.

---

*Repositori ini disusun sebagai bagian dari Tugas Akhir dan ditujukan untuk
keperluan akademik.*

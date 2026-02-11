# 📊 Submission Akhir BMLP — Ridho Bintang Aulia

> **Belajar Machine Learning untuk Pemula (BMLP)**  
> Dicoding Indonesia

---

## 👤 Informasi Penulis

| Item         | Detail                                |
| ------------ | ------------------------------------- |
| **Nama**     | Ridho Bintang Aulia                   |
| **Program**  | Belajar Machine Learning untuk Pemula |
| **Platform** | Dicoding Indonesia                    |

---

## 📖 Deskripsi Proyek

Proyek ini merupakan **Submission Akhir** kelas _Belajar Machine Learning untuk Pemula (BMLP)_ di Dicoding Indonesia. Proyek terdiri dari **dua bagian utama** yang saling terhubung:

1. **Clustering** — Mengelompokkan data transaksi bank menggunakan algoritma **K-Means**.
2. **Klasifikasi** — Mengklasifikasikan data hasil clustering menggunakan algoritma **Decision Tree** dan **Random Forest**.

---

## 📁 Dataset

**Bank Transaction Dataset for Fraud Detection** (versi modifikasi)

| Properti     | Nilai                                     |
| ------------ | ----------------------------------------- |
| Jumlah Baris | 2.537                                     |
| Jumlah Kolom | 16                                        |
| Format       | CSV (`bank_transactions_data_edited.csv`) |

### Deskripsi Kolom

| Kolom                     | Deskripsi                           |
| ------------------------- | ----------------------------------- |
| `TransactionID`           | ID unik transaksi                   |
| `AccountID`               | ID akun pelanggan                   |
| `TransactionAmount`       | Jumlah transaksi                    |
| `TransactionDate`         | Tanggal dan waktu transaksi         |
| `TransactionType`         | Jenis transaksi (Debit/Credit)      |
| `Location`                | Lokasi transaksi                    |
| `DeviceID`                | ID perangkat                        |
| `IP Address`              | Alamat IP                           |
| `MerchantID`              | ID merchant                         |
| `Channel`                 | Kanal transaksi (ATM/Online/Branch) |
| `CustomerAge`             | Usia pelanggan                      |
| `CustomerOccupation`      | Pekerjaan pelanggan                 |
| `TransactionDuration`     | Durasi transaksi (detik)            |
| `LoginAttempts`           | Jumlah percobaan login              |
| `AccountBalance`          | Saldo akun                          |
| `PreviousTransactionDate` | Tanggal transaksi sebelumnya        |

---

## 🔬 Metodologi

### Bagian 1 — Clustering (`[Clustering]_Submission_Akhir_BMLP_Ridho_Bintang_Aulia.ipynb`)

Pipeline clustering mencakup tahapan berikut:

| No  | Tahapan                         | Metode / Detail                                                                                  |
| --- | ------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1   | Import Library                  | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `yellowbrick`, `joblib`              |
| 2   | Memuat Dataset                  | `pd.read_csv()` — 2.537 baris × 16 kolom                                                         |
| 3   | Exploratory Data Analysis (EDA) | Matriks korelasi, histogram fitur numerik & kategorikal                                          |
| 4   | Pembersihan Data                | Menangani _missing values_ — strategi: **drop** baris dengan data kosong (→ 1.501 baris tersisa) |
| 5   | Pemilihan Fitur                 | Memilih fitur yang relevan untuk clustering                                                      |
| 6   | Encoding & Scaling              | `LabelEncoder` untuk fitur kategorikal, `StandardScaler` untuk normalisasi                       |
| 7   | Reduksi Dimensi                 | `PCA` untuk mengurangi dimensi fitur                                                             |
| 8   | Pemilihan k Optimal             | **Elbow Method** (`KElbowVisualizer`) dan **Silhouette Score**                                   |
| 9   | Model Clustering                | `KMeans` dari scikit-learn                                                                       |
| 10  | Interpretasi Hasil              | Analisis karakteristik setiap cluster                                                            |
| 11  | Ekspor Data                     | Menyimpan hasil clustering ke CSV untuk digunakan pada tahap klasifikasi                         |

### Bagian 2 — Klasifikasi (`[Klasifikasi]_Submission_Akhir_BMLP_Ridho_Bintang_Aulia.ipynb`)

Pipeline klasifikasi mencakup tahapan berikut:

| No  | Tahapan                          | Metode / Detail                                                                  |
| --- | -------------------------------- | -------------------------------------------------------------------------------- |
| 1   | Import Library                   | `pandas`, `scikit-learn`, `joblib`                                               |
| 2   | Memuat Dataset                   | Memuat `data_clustering_inverse.csv` — hasil clustering (1.501 baris × 12 kolom) |
| 3   | Feature Encoding                 | `LabelEncoder` untuk fitur kategorikal                                           |
| 4   | Data Splitting                   | `train_test_split` — rasio **80:20**, `stratify=y`, `random_state=42`            |
| 5   | Model Utama                      | **Decision Tree Classifier** — sebagai baseline                                  |
| 6   | Model Tambahan (Skilled)         | **Random Forest Classifier** — `n_estimators=100`                                |
| 7   | Perbandingan Model               | Membandingkan Accuracy, Precision, Recall, F1-Score                              |
| 8   | Hyperparameter Tuning (Advanced) | Tuning model terbaik menggunakan `GridSearchCV` / `RandomizedSearchCV`           |
| 9   | Simpan Model                     | Menyimpan model terbaik ke file `.h5` dengan `joblib`                            |

---

## 📈 Hasil Model

### Performa Klasifikasi

| Model             | Accuracy | Precision | Recall | F1-Score |
| ----------------- | -------- | --------- | ------ | -------- |
| **Decision Tree** | 0.9900   | 0.9901    | 0.9900 | 0.9901   |
| **Random Forest** | 0.9967   | 0.9967    | 0.9967 | 0.9967   |

> ✅ **Model terbaik:** Random Forest (F1-Score: **0.9967**)

---

## 📂 Struktur File

```
BMLP_Ridho-Bintang-Aulia/
│
├── README.md                                                     # Dokumentasi proyek
├── .gitignore                                                    # File gitignore
│
├── [Clustering]_Submission_Akhir_BMLP_Ridho_Bintang_Aulia.ipynb  # Notebook Clustering
├── [Klasifikasi]_Submission_Akhir_BMLP_Ridho_Bintang_Aulia.ipynb # Notebook Klasifikasi
│
├── bank_transactions_data_edited.csv                             # Dataset utama
├── data_clustering.csv                                           # Output clustering (encoded)
├── data_clustering_inverse.csv                                   # Output clustering (inverse)
│
├── model_clustering.h5                                           # Model clustering (K-Means)
├── PCA_model_clustering.h5                                       # Model PCA
├── decision_tree_model.h5                                        # Model Decision Tree
├── best_model_classification.h5                                  # Model terbaik (Random Forest)
├── explore_RandomForest_classification.h5                        # Model Random Forest (eksplorasi)
└── tuning_classification.h5                                      # Model setelah tuning
```

---

## 🛠️ Tech Stack & Dependencies

| Library        | Kegunaan                                                         |
| -------------- | ---------------------------------------------------------------- |
| `pandas`       | Manipulasi dan analisis data                                     |
| `numpy`        | Komputasi numerik                                                |
| `matplotlib`   | Visualisasi data                                                 |
| `seaborn`      | Visualisasi statistik                                            |
| `scikit-learn` | Machine learning (KMeans, PCA, DecisionTree, RandomForest, dll.) |
| `yellowbrick`  | Visualisasi ML (Elbow Method)                                    |
| `joblib`       | Menyimpan dan memuat model                                       |

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/Rbin01yuh/Submission_Dicoding_Machine_Learning.git
cd Submission_Dicoding_Machine_Learning/BMLP_Ridho-Bintang-Aulia
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib
```

### 3. Jalankan Notebook

Buka dan jalankan notebook secara berurutan:

1. **Clustering** terlebih dahulu:

   ```
   [Clustering]_Submission_Akhir_BMLP_Ridho_Bintang_Aulia.ipynb
   ```

2. **Klasifikasi** setelah clustering selesai:
   ```
   [Klasifikasi]_Submission_Akhir_BMLP_Ridho_Bintang_Aulia.ipynb
   ```

> ⚠️ **Penting:** Notebook Klasifikasi bergantung pada output dari notebook Clustering (`data_clustering_inverse.csv`). Pastikan notebook Clustering dijalankan terlebih dahulu.

---

## 📝 Kriteria Penilaian yang Dipenuhi

| Kriteria                                   | Status | Keterangan                                     |
| ------------------------------------------ | ------ | ---------------------------------------------- |
| Clustering menggunakan K-Means             | ✅     | Implementasi lengkap dengan Elbow & Silhouette |
| Klasifikasi menggunakan Decision Tree      | ✅     | Baseline model dengan akurasi 99%              |
| **(Skilled)** Matriks Korelasi & Histogram | ✅     | EDA lengkap pada notebook Clustering           |
| **(Skilled)** Algoritma Klasifikasi Lain   | ✅     | Random Forest sebagai model tambahan           |
| **(Advanced)** Hyperparameter Tuning       | ✅     | Tuning pada model terbaik                      |

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan submission Dicoding Indonesia.

---

_Dibuat dengan ❤️ oleh Ridho Bintang Aulia_

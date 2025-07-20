# 📊 PREDIKSI SOFTWARE DEFECT PADA PROYEK OPEN-SOURCE APACHE MENGGUNAKAN METODE MACHINE LEARNING

Repositori ini berisi kerangka kerja (skeleton) proyek data mining yang terstruktur dan modular.

## Anggota Kelompok & NIM

- Waskitho Cito Adiwiguno (714220019)
- Muhammad Rifky (714220022)
- Ukasyah Abdulloh Azzam (714220016)

## Deskripsi Kasus

Proyek ini bertujuan untuk memprediksi apakah suatu modul perangkat lunak pada proyek open-source Apache mengandung cacat (defect) atau tidak. Dengan memanfaatkan data historis dari kode sumber seperti kompleksitas kode, ukuran, dan riwayat perubahan, model machine learning dilatih untuk mengidentifikasi modul yang berisiko tinggi mengalami cacat. Prediksi ini dapat membantu pengembang dalam proses pengujian dan pemeliharaan perangkat lunak agar lebih efisien dan efektif.

## Sumber Dataset

Dataset yang digunakan pada proyek adalah ApacheJIT, dataset ini diambil dari Zenodo.org (https://zenodo.org/records/5907847). ApacheJIT adalah dataset primer yang dikembangkan untuk penelitian Just-In-Time (JIT) defect prediction. Dataset ini dikumpulkan dari lebih dari 100.000 commit historis dari proyek-proyek open-source di bawah naungan Apache Software Foundation.

## Langkah Preprocessing

- Penanganan Missing Values
- Scalling Data
- Resampling
- Split data menjadi training dan testing

## Algoritma yang Digunakan

- Random Forest
- Logistic Regression
- K-Nearest Neighbors
- LightGBM
- XGBoost

## Evaluasi & Hasil

Tabel di bawah menunjukkan hasil evaluasi dari lima model machine learning yang digunakan, berdasarkan metrik Accuracy, Precision, Recall, F1-Score, dan ROC-AUC:

| Model                 | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | 0.76     | 0.78      | 0.76   | 0.78     | 0.8028  |
| K-Nearest Neighbors   | 0.79     | 0.80      | 0.70   | 0.75     | 0.7743  |
| LightGBM              | 0.78     | 0.81      | 0.78   | 0.78     | 0.8528  |
| Random Forest         | 0.82     | 0.82      | 0.78   | 0.83     | 0.8356  |
| XGBoost               | 0.79     | 0.75      | 0.78   | 0.75     | 0.8493  |

Model **Random Forest** memberikan performa terbaik secara keseluruhan berdasarkan nilai **accuracy** dan **F1-score**, sementara **LightGBM** memiliki nilai **ROC-AUC** tertinggi.


## 🗂 Struktur Direktori

```
tube_data_mining/
│
├── data/                      # Folder untuk menyimpan dataset
│   ├── raw/                   # Data mentah (belum diproses)
│   └── processed/             # Data setelah preprocessing
│
├── notebook/                 # Jupyter Notebook interaktif
│   ├── eda_template.ipynb     # Template untuk eksplorasi data
│   ├── preprocessing_template.ipynb  # Template untuk preprocessing
│   └── modeling_template.ipynb       # Template untuk pelatihan model
│
├── report/                   # Template laporan akhir
│   ├── laporan-akhir_template.pdf
│   ├── lampiran_template.docx
│   └── struktur-lampiran.md
│
├── src/                      # Source code modular
│   ├── data_loader.py         # Fungsi load dan simpan data
│   ├── model.py               # Fungsi training model
│   ├── utils.py               # Evaluasi model dan fungsi bantu
│   ├── main.py                # Main pipeline untuk dijalankan via terminal
│   └── main_notebook.ipynb    # Versi notebook dari main.py
│
├── run.sh                    # Script bash untuk menjalankan pipeline
├── requirements.txt          # Daftar dependensi Python
└── README.md                 # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan

### ✅ 1. Persiapkan Environment

Install dependensi:
```bash
pip install -r requirements.txt
```

### ✅ 2. Jalankan Pipeline

#### 💻 Via Terminal:
```bash
bash run.sh
```

#### 📒 Via Jupyter Notebook:
Buka dan jalankan:
```text
src/main_notebook.ipynb
```

---

## 📦 Struktur Modular

- **`data_loader.py`**: fungsi `load_raw_data()` dan `save_processed_data()`
- **`model.py`**: fungsi `train_model()`, split data, dan prediksi
- **`utils.py`**: evaluasi model (akurasi, classification report, dll.)

---

## 📓 Catatan

- Semua path diasumsikan relatif dari root project
- Tambahkan file data kamu ke dalam `data/raw/`
- Hasil preprocessing disimpan di `data/processed/` 
- Pastikan target label diberi nama kolom `target` (atau sesuaikan di script)

---

## 👩‍💻 Kontributor

- Seluruh Anggota Kelompok

---

## 📄 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk edukasi dan pengembangan pribadi.


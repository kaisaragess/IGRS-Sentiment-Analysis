# IGRS-Sentiment-Analysis
🎮 Analisis opini publik di X (Twitter) terkait kebijakan Indonesia Game Rating System (IGRS) menggunakan NLP. Dilengkapi pipeline lengkap dari web scraping hingga fine-tuning model IndoBERT untuk klasifikasi teks bahasa Indonesia (gaul/campur kode). 

# Analisis Sentimen Kebijakan IGRS di X (Twitter) Menggunakan IndoBERT

Proyek ini berisi kode dan dokumentasi untuk proyek analisis sentimen masyarakat Indonesia di platform X (sebelumnya Twitter) terhadap kebijakan **Indonesia Game Rating System (IGRS)** yang diinisiasi oleh **Kementerian Komunikasi dan Digital (Komdigi)**. Proyek ini menggunakan model bahasa berbasis Transformer, yaitu **IndoBERT**, untuk mengklasifikasikan opini publik ke dalam tiga kategori: **Positif**, **Netral**, dan **Negatif**.

---

## 📌 Deskripsi Proyek

Kebijakan IGRS bertujuan untuk memberikan panduan klasifikasi konten game agar sesuai dengan usia pengguna di Indonesia. Implementasi kebijakan ini menuai beragam reaksi di media sosial. 

Proyek ini bertujuan untuk memetakan distribusi sentimen masyarakat secara otomatis menggunakan pendekatan *Natural Language Processing* (NLP). Kami melakukan *fine-tuning* pada model `indolem/indobert-base-uncased` untuk memahami nuansa bahasa Indonesia sehari-hari, termasuk bahasa gaul (*slang*) dan campur kode (*code-switching*).

---

## 🛠️ Teknologi dan Library

Proyek ini dibangun dan dijalankan di atas lingkungan **Google Colab** dengan memanfaatkan GPU untuk akselerasi pelatihan. Library utama yang digunakan meliputi:

* **Transformers (Hugging Face)**: Untuk memuat model IndoBERT dan tokenizer.
* **PyTorch**: Sebagai framework deep learning utama.
* **Scikit-learn**: Untuk pembagian dataset (*train-test split*) dan perhitungan metrik evaluasi.
* **Imbalanced-learn (imblearn)**: Untuk menangani ketidakseimbangan kelas menggunakan `RandomOverSampler`.
* **Pandas**: Untuk manipulasi dan analisis data tabular.

---

## 📊 Dataset

Data dikumpulkan melalui proses *web scraping* kustom dari platform X pada periode **Januari - April 2026**.

* **Kata Kunci Pencarian**: `"IGRS"`, `"Game Rating System"`, `"Rating Game Indonesia"`.
* **Distribusi Awal**: Dataset bersifat *imbalanced* (tidak seimbang), dengan dominasi sentimen:
    * 🔴 **Negatif**: 60,6%
    * 🟡 **Netral**: 30,0%
    * 🟢 **Positif**: 9,4%

---

## ⚙️ Pipeline Pemrosesan

1.  **Data Preprocessing**: Meliputi penghapusan URL, mention, hashtag, tanda baca, serta penerapan *case folding* (mengubah teks menjadi huruf kecil).
2.  **Data Splitting & Oversampling**: Pemisahan rasio 80% data latih dan 20% data uji. Mengingat distribusi yang tidak seimbang, `RandomOverSampler` diterapkan **hanya pada data latih** untuk mencegah terjadinya *data leakage* (kebocoran data).
3.  **Tokenisasi**: Menggunakan tokenizer IndoBERT dengan panjang maksimal (`max_length`) 128 token.
4.  **Fine-Tuning**: Pelatihan model IndoBERT dengan konfigurasi:
    * *Learning Rate*: `2e-5`
    * *Batch Size*: `8`
    * *Epoch*: `12`
    * *Optimizer*: `AdamW`
    * Ditambah mekanisme *Early Stopping* untuk mencegah *overfitting*.

---

## 📈 Hasil Evaluasi

Model dievaluasi menggunakan metrik akurasi dan *Macro F1-Score* untuk memastikan keadilan penilaian di setiap kelas:

| Metrik Evaluasi | Nilai |
| :--- | :--- |
| **Accuracy** | 76.47% |
| **F1-Score (Macro)** | 0.7609 |
| **Precision (Macro)** | 0.7942 |
| **Recall (Macro)** | 0.7647 |

### Analisis Per Kelas
* Model memiliki performa sangat baik pada **kelas negatif** dengan **F1-Score 0.829**.
* Model mengalami kesulitan pada **kelas positif** dengan **F1-Score 0.500** yang disebabkan oleh minimnya sampel (*class imbalance*) dalam dataset asli.

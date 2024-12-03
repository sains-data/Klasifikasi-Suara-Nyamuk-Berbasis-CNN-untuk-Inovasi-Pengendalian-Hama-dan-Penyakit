# Team [1] _Deep Learning_ - Klasifikasi Suara Nyamuk Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit
Proyek ini dikembangkan oleh **Team-1** dari kelas **_Deep Learning_** tahun **2024**. Tujuan utama proyek ini adalah mengklasifikasikan suara nyamuk berdasarkan spesies menggunakan **_Convolutional Neural Network_ (CNN)** untuk mendukung inovasi dalam pengendalian hama dan penyakit.

![Gambar Nyamuk](https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo0.jpg)

## 📌 Anggota Kelompok
1. Ignatius Krisna Issaputra (121140037)
2. Ardoni Yeriko Rifana Gultom (121140141)
3. Rika Ajeng Finatih (121450036)
4. M. Gilang Martiansyah (121450056)
5. Sasa Rahma Lia (121450119)
6. Nazwa Nabila (121450122)

## 🚀 Tujuan Proyek
Mengembangkan sistem klasifikasi suara nyamuk secara otomatis untuk mendeteksi spesies seperti:
* 🦟 _Aedes aegypti_ (vektor demam berdarah)
* 🦟 _Anopheles stephensi_ (vektor malaria)
* 🦟 _Culex pipiens_ (vektor filariasis)

## 📂Dataset 
Dataset yang digunakan untuk proyek ini dapat diakses melalui sumber berikut:
```
[Dataset Suara Nyamuk](https://drive.google.com/drive/folders/109Spn_kf2DCFK1Xqb1f9K2w70kUPVaAj?usp=sharing )
```
Dataset mencakup rekaman audio nyamuk dalam format **.wav** serta label spesies dalam file **.csv**.

## 🛠️ Teknologi yang Digunakan
* **Pyhton**
* **TensorFlow/Keras** untuk model CNN
* **Librosa** untuk ekstraksi fitur audio
* **Matplotlib** dan **Seaborn** untuk visualisasi data

## 🧠 Model CNN _(Convolutional Neural Network)_
Arsitektur CNN dirancang untuk memproses spektrum audio dari suara nyamuk, memungkinkan klasifikasi spesies dengan akurasi tinggi. Model dilatih menggunakan teknik augmentasi data dan regularisasi untuk meningkatkan performa dan mencegah overfitting.

## 📊 Metodologi
1. **Preprocessing Audio**: Menggunakan fitur MFCC dan Mel Spectrogram.
2. **Pelatihan Model**: Model CNN dilatih untuk mengklasifikasikan suara nyamuk.
3. **Evaluasi Model**: Menggunakan metrik seperti akurasi, precision, recall, dan F1-score.
   
## 📈 Hasil yang diharapkan
* Akurasi klasifikasi ≥ 75%.
* Sistem real-time dengan waktu prediksi ≤ 1 detik.
* Sensitivitas deteksi spesies nyamuk ≥ 80%.

## 🧑‍💻 Cara Menjalankan Proyek
1. **Clone repositori**:
   ```
   git clone https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit.git
   ```
2. **Masuk ke direktori proyek** yang baru di-clone:
   ```
   cd Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit
   ```
4. **Instal dependensi**:
   ```
   pip install -r requirements.txt
   ```
5. **Jalankan notebook atau script**:
   ```
   python main.py
   ```
   
## 👥 Kontibutor
* Ignatius Krisna Issaputra - [Github](https://github.com/inExcelsis1710)
* Ardoni Yeriko Rifana Gultom - [Github](https://github.com/gultom20)
* Rika Ajeng Finatih - [Github](https://github.com/rika623)
* M. Gilang Martiansyah - [Github](https://github.com/mgilang56)
* Sasa Rahma Lia - [Github](https://github.com/sasarahmalia)
* Nazwa Nabila - [Github](https://github.com/nazwanabila)

## 📫 Kontak
Jika ada pertanyaan, silakan hubungi:

✉️Email: rika.121450036@student.itera.ac.id

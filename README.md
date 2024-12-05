# Team [1] _Deep Learning_ - Klasifikasi Suara Nyamuk Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit
Proyek ini dikembangkan oleh Team-1 dari kelas Deep Learning tahun 2024. Tujuan utamanya adalah mengklasifikasikan suara kepakan sayap nyamuk berdasarkan spesies menggunakan model Convolutional Neural Network (CNN) untuk mendukung inovasi dalam pengendalian hama dan penyakit di wilayah tropis, khususnya di Indonesia. Proyek ini berfokus pada tiga spesies nyamuk utama: Aedes aegypti, Anopheles stephensi, dan Culex pipiens. Melalui proyek ini, diharapkan dapat mendukung upaya pemerintah dalam mencapai target eliminasi malaria dan filariasis pada tahun 2030 serta mengurangi insiden demam berdarah dengue (DBD) hingga di bawah 49 kasus per 100.000 jiwa.

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
  
Dengan identifikasi spesies nyamuk secara akurat, diharapkan dapat mengurangi dampak dari penyakit-penyakit tersebut dan membantu pemerintah dalam pengendalian hama.


# 📂 Dataset
Dataset yang digunakan untuk proyek ini berisi rekaman audio nyamuk dalam format **.wav** dan label spesies dalam file **.csv**. Dataset ini mencakup suara dari tiga spesies nyamuk yang disebutkan di atas dan telah melalui tahap preprocessing untuk ekstraksi fitur.
[Dataset Suara Nyamuk](https://drive.google.com/drive/folders/109Spn_kf2DCFK1Xqb1f9K2w70kUPVaAj?usp=sharing)

## 🛠️ Teknologi yang Digunakan
### 1. **Python** 🐍
Bahasa pemrograman utama yang digunakan untuk mengembangkan model, memproses data, dan membangun aplikasi prediksi suara nyamuk.

<img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="150"/>

### 2. **TensorFlow/Keras** 🧠
TensorFlow adalah framework open-source yang digunakan untuk membangun dan melatih model deep learning, sementara Keras adalah API tingkat tinggi yang menyediakan antarmuka yang lebih sederhana untuk pengembangan model.

<img src="https://media.wired.com/photos/5927105acfe0d93c474323d7/master/pass/google-tensor-flow-logo-black-S.jpg" width="150"/>

### 3. **Librosa** 🎶
Librosa adalah pustaka Python yang digunakan untuk analisis dan ekstraksi fitur audio, seperti Mel-Frequency Cepstral Coefficients (MFCC) yang digunakan untuk mengolah data suara nyamuk dalam proyek ini.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcReiDgm71NRUOVsiA_rTGi8lsIZmO1rlYt4cw&s" width="150"/>

### 4. **Adobe Audition** 🎧
Adobe Audition adalah perangkat lunak pengolahan audio yang digunakan dalam proyek ini untuk membersihkan noise dan melakukan preprocessing audio sebelum ekstraksi fitur. Ini membantu memastikan data audio berkualitas tinggi.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Adobe_Audition_CC_icon_%282020%29.svg/800px-Adobe_Audition_CC_icon_%282020%29.svg.png" width="150"/>

### 5. **Streamlit** 🌐
Digunakan untuk membuat aplikasi web interaktif, yang memungkinkan pengguna untuk mengunggah suara dan mendapatkan prediksi spesies nyamuk secara langsung.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1IS9rNAuZkFawNTS7W3dgsNcuOjNfh9imKQ&s" width="150"/>

### 6. **Matplotlib & Seaborn** 📊
Digunakan untuk visualisasi data dan evaluasi kinerja model dengan grafik dan plot yang mudah dipahami.

<img src="https://matplotlib.org/stable/_static/logo2.svg" width="150"/>
  
## 🧠 Model CNN _(Convolutional Neural Network)_
Arsitektur CNN dirancang untuk memproses spektrum audio dari suara nyamuk, memungkinkan klasifikasi spesies dengan akurasi tinggi. Model dilatih menggunakan teknik augmentasi data dan regularisasi untuk meningkatkan performa dan mencegah overfitting.

## 📊 Metodologi
1. **Preprocessing Audio**: Menggunakan fitur MFCC dan Mel Spectrogram.
2. **Pelatihan Model**: Model CNN dilatih untuk mengklasifikasikan suara nyamuk.
3. **Evaluasi Model**: Menggunakan metrik seperti akurasi, precision, recall, dan F1-score.
   
## 🏆 **Hasil yang Diharapkan**
Kami berharap dapat mencapai:
- **Akurasi klasifikasi ≥ 75%** untuk semua spesies nyamuk.
- Waktu prediksi sistem **< 1 detik** untuk identifikasi suara nyamuk.
- **Sensitivitas ≥ 80%** untuk deteksi spesies nyamuk yang relevan.

## 📡 Flowchart Proses
![Deskripsi Gambar](https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/flowchart%20.png)


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

- ✉️ **Email:**
  - ignatius.121140037@student.itera.ac.id
  - ardoni.121140141@student.itera.ac.id
  - rika.121450036@student.itera.ac.id
  - mgilang.121450056@student.itera.ac.id
  - sasa.121450119@student.itera.ac.id
  - nazwa.121450122@student.itera.ac.id


## 🙏 Ucapan Terima Kasih
Kami ingin mengucapkan terima kasih yang sebesar-besarnya kepada:

1. **Dosen Pembimbing**:
   - Bapak Ardika Satria, S.Si. M.Si yang telah memberikan bimbingan, arahan, dan dukungan yang sangat berharga selama pengerjaan proyek ini. Terima kasih atas saran-saran yang membantu kami dalam mengembangkan ide dan implementasi sistem ini
  
  **Dosen Matakuliah**:
   - Bapak Christyan Tamaro Nadeak, M.Si selaku dosen pengampu mata kuliah Deep Learning.
   - ibu Ade Lailani, M.Si selaku dosen pengampu mata kuliah   Deep Learning.

2. **Anggota Kelompok**:
   - **Ardoni Yeriko Rifana Gultom**:  Terima kasih atas kerja keras dalam mengembangkan model CNN dan kontribusinya dalam preprocessing data audio dan terimakasih atas  optimasi model sangat membantu kami mencapai hasil yang lebih baik.
   - **M. Gilang Martiansyah**: Terima kasih atas kontribusinya dalam pembuatan aplikasi prediksi menggunakan Streamlit, implementasi pipeline data, serta analisis dan evaluasi model dan juga berperan penting dalam proses debugging.
   - **Rika Ajeng Finatih**: Terima kasih atas dedikasinya dalam memimpin proyek, serta perannya dalam pembuatan laporan dan dokumentasi.
   - **Ignatius Krisna Issaputra**: Terima kasih atas kerja kerasnya dalam pemrosesan audio, terutama dalam ekstraksi fitur audio menggunakan MFCC dan Mel Spectrogram. Krisna juga banyak berkontribusi dalam pengembangan model CNN untuk klasifikasi suara nyamuk.
   - **Sasa Rahma Lia**: Terima kasih atas kontribusinya dalam pembuatan laporan, serta bantuan dalam menyusun dan merapikan dokumentasi teknis proyek terimakasih juga atas   merapihkan organisir notion.
   - **Nazwa Nabila**: Terima kasih atas kontribusinya dalam pembuatan laporan serta dukungan dalam dokumentasi dan pengujian sistem.

Kami sangat mengapresiasi setiap kontribusi yang diberikan oleh setiap anggota tim. Tanpa kerjasama yang solid, proyek ini tidak akan tercapai dengan baik.

---

## 📑 Referensi
Daftar referensi yang digunakan dalam pengerjaan proyek ini:
## 🔗 Tautan Kelompok 1
Kunjungi Notion Kami: [Notion](https://aquamarine-dove-b45.notion.site/Team-1-Proyek-Tugas-Besar-Deep-Learning-133607a60e95805294dada205aea761d)
## 🎥 Demo Video
Tonton video berikut untuk melihat cara aplikasi ini bekerja: [in progress(https://www.youtube.com/watch?v=XXXXXXX)

1. [Referensi 1](https://...)
2. [Referensi 2](https://...)
3. [Referensi 3](https://...)

# **#DeepLearning #CNN #Classification #Malaria #Dengue #DiseasesControl**

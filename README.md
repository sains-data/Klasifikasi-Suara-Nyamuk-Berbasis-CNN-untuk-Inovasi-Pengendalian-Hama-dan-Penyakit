# 🌟 **Team 1: Deep Learning** - Klasifikasi Suara Nyamuk Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit

Proyek ini dikembangkan oleh **Team-1** dari kelas **Deep Learning 2024**. Tujuan utamanya adalah mengklasifikasikan suara kepakan sayap nyamuk berdasarkan spesies menggunakan model **Convolutional Neural Network (CNN)** untuk mendukung inovasi dalam pengendalian hama dan penyakit di wilayah tropis, khususnya di Indonesia.

## 🎯 **Fokus Proyek**
- **Spesies yang Diklasifikasi:**
  - 🦟 **_Aedes aegypti_** (vektor demam berdarah dengue)
  - 🦟 **_Anopheles stephensi_** (vektor malaria)
  - 🦟 **_Culex pipiens_** (vektor filariasis)

Proyek ini diharapkan mendukung **upaya pemerintah** dalam mencapai target:
- **Eliminasi malaria dan filariasis pada tahun 2030**
- **Pengurangan insiden demam berdarah dengue (DBD)** hingga di bawah **49 kasus per 100.000 jiwa**.

  
## Logo Produk
![🦟 Gambar Nyamuk](https://raw.githubusercontent.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/main/Deploy/Logo%20MosquID.png)

---

## 👥 **Anggota Kelompok**
1. **Ignatius Krisna Issaputra** (121140037)  
2. **Ardoni Yeriko Rifana Gultom** (121140141)  
3. **Rika Ajeng Finatih** (121450036)  
4. **M. Gilang Martiansyah** (121450056)  
5. **Sasa Rahma Lia** (121450119)  
6. **Nazwa Nabilla** (121450122)  

---

## 🚀 **Tujuan Proyek**
- Mengembangkan sistem klasifikasi suara nyamuk otomatis untuk mendeteksi spesies nyamuk.
- Mendukung inovasi pengendalian hama dan penyakit tropis dengan teknologi berbasis AI.


---

## 📂 **Dataset**
Dataset yang digunakan mencakup:
- **Rekaman audio nyamuk** dalam format **.wav**
- **Label spesies** dalam file **.csv**.

**🔗 [Download Dataset](https://drive.google.com/drive/folders/109Spn_kf2DCFK1Xqb1f9K2w70kUPVaAj?usp=sharing)**

### **Pengolahan Data:**
1. Audio difilter untuk menghilangkan noise.
2. Fitur diekstraksi menggunakan **MFCC** dan **Mel Spectrogram**.
3. Data di-augmentasi untuk meningkatkan generalisasi model.

---

## 🛠️ **Teknologi yang Digunakan**
| Teknologi          | Deskripsi                                                                 |
|--------------------|---------------------------------------------------------------------------|
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="80"> | **Python**: Bahasa pemrograman utama untuk pemrosesan data dan pengembangan model. |
| <img src="https://media.wired.com/photos/5927105acfe0d93c474323d7/master/pass/google-tensor-flow-logo-black-S.jpg" width="80"> | **TensorFlow/Keras**: Framework untuk membangun dan melatih model deep learning. |
| <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcReiDgm71NRUOVsiA_rTGi8lsIZmO1rlYt4cw&s" width="80"> | **Librosa**: Pustaka untuk analisis dan ekstraksi fitur audio (MFCC, Mel Spectrogram). |
| <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Adobe_Audition_CC_icon_%282020%29.svg/800px-Adobe_Audition_CC_icon_%282020%29.svg.png" width="80"> | **Adobe Audition**: Untuk preprocessing audio dan pembersihan noise. |
| <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT1IS9rNAuZkFawNTS7W3dgsNcuOjNfh9imKQ&s" width="80"> | **Streamlit**: Framework untuk membuat aplikasi web interaktif. |
| <img src="https://matplotlib.org/stable/_static/logo2.svg" width="80"> | **Matplotlib & Seaborn**: Untuk visualisasi data dan evaluasi kinerja model. |

---
  
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
   
---

## 👥 **Kontributor**
| Nama                           | Github                                                 |
|--------------------------------|--------------------------------------------------------|
| Ignatius Krisna Issaputra      | [Github](https://github.com/inExcelsis1710)            |
| Ardoni Yeriko Rifana Gultom    | [Github](https://github.com/gultom20)                  |
| Rika Ajeng Finatih             | [Github](https://github.com/rika623)                   |
| M. Gilang Martiansyah          | [Github](https://github.com/mgilang56)                 |
| Sasa Rahma Lia                 | [Github](https://github.com/sasarahmalia)              |
| Nazwa Nabilla                  | [Github](https://github.com/nazwanabila)               |

---
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

### **1. Dosen Pembimbing**
- **Bapak Ardika Satria, S.Si., M.Si.** yang telah memberikan bimbingan, arahan, dan dukungan yang sangat berharga selama pengerjaan proyek ini. Terima kasih atas saran-saran yang membantu kami dalam mengembangkan ide dan implementasi sistem ini
  
### **2. Dosen Mata Kuliah**
- **Bapak Christyan Tamaro Nadeak, M.Si**: Atas ilmu yang diberikan dalam mata kuliah Deep Learning.
- **Ibu Ade Lailani, M.Si**: Atas kontribusi dalam pengajaran konsep-konsep dasar yang mendukung proyek ini.

### **3. Anggota Kelompok**
   - **Ardoni Yeriko Rifana Gultom**:  Terima kasih atas kerja keras dalam mengembangkan model CNN dan kontribusinya dalam preprocessing data audio dan terimakasih atas  optimasi model sangat membantu kami mencapai hasil yang lebih baik.
   - **M. Gilang Martiansyah**: Terima kasih atas kontribusinya dalam pembuatan aplikasi prediksi menggunakan Streamlit, implementasi pipeline data, serta analisis dan evaluasi model dan juga berperan penting dalam proses debugging.
   - **Rika Ajeng Finatih**: Terima kasih atas dedikasinya dalam memimpin proyek, serta perannya dalam pembuatan laporan dan dokumentasi.
   - **Ignatius Krisna Issaputra**: Terima kasih atas kerja kerasnya dalam pemrosesan audio, terutama dalam ekstraksi fitur audio menggunakan MFCC dan Mel Spectrogram. Krisna juga banyak berkontribusi dalam pengembangan model CNN untuk klasifikasi suara nyamuk.
   - **Sasa Rahma Lia**: Terima kasih atas kontribusinya dalam pembuatan laporan, serta bantuan dalam menyusun dan merapikan dokumentasi teknis proyek terimakasih juga atas   merapihkan organisir notion.
   - **Nazwa Nabila**: Terima kasih atas kontribusinya dalam pembuatan laporan serta dukungan dalam dokumentasi dan pengujian sistem.

Kami sangat mengapresiasi setiap kontribusi yang diberikan oleh setiap anggota tim. Tanpa kerjasama yang solid, proyek ini tidak akan tercapai dengan baik.

---

## 🔗 **Tautan Penting**
- **Notion Kelompok 1**: [Notion](https://aquamarine-dove-b45.notion.site/Team-1-Proyek-Tugas-Besar-Deep-Learning-133607a60e95805294dada205aea761d)  
- **Demo Aplikasi Streamlit**: [Coba Aplikasi](https://mosquitoclassify1.streamlit.app/) _(disarankan memakai mode gelap untuk pengalaman terbaik)_  
- **Train Model CNN**: [Download Model](https://drive.google.com/file/d/1rbfhPOQLBKxyRvrSUS5jpHjjVBGgCKqx/view?usp=drive_link)  
- **Train History JSON**: [Download History](https://drive.google.com/file/d/1tl_NtfvabLha3-hrwYIaQmPu3hrxYgYv/view?usp=drive_link)  

---
---

## 🎥 **Video Dokumentasi**
**Iklan Produk**: [Tonton Video](https://drive.google.com/file/d/1o89d0438NAPzlXVg36KmjCbgButCD0k-/view?usp=drive_link)

---

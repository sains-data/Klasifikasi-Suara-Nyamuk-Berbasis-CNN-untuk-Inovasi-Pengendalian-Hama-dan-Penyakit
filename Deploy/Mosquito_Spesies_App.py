import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import io

# Fungsi untuk memuat model
def load_model():
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Fungsi untuk memproses file audio
def load_and_preprocess_file(file, target_shape=(180, 180)):
    data = []
    audio_data, sample_rate = librosa.load(file, sr=None)

    chunk_duration = 2
    overlap_duration = 1

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)

        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=13)
        mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)

        combined_features = np.concatenate([mel_spectrogram, mfcc_resized], axis=-1)
        data.append(combined_features)

    return np.array(data)

# Fungsi untuk prediksi model
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# Menambahkan background gambar dari URL
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('https://asset.kompas.com/crops/Uoby6be9TIeMzC18327oT1MCjlI=/13x0:500x325/1200x800/data/photo/2020/03/12/5e69cae0eb1d1.jpg');
            background-size: cover;
            background-position: top center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# HTML untuk bagian konten dan formulir upload
st.markdown(
    """
    <style>
        body {
            background-size: cover;
            color: #FFFFFF;  /* Warna putih untuk teks */
            font-family: 'Roboto', sans-serif;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            text-align: center;
        }

        h1 {
            font-size: 2.5em;
            font-weight: bold;
            color: #FFFFFF;  /* Warna putih */
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.6); /* Menambahkan bayangan pada teks */
        }

        h3 {
            font-size: 1.5em;
            color: #FFFFFF;  /* Warna putih */
            text-shadow: 1px 1px 6px rgba(0, 0, 0, 0.6); /* Bayangan pada teks */
        }

        .upload-form {
            background-color: rgba(255, 255, 255, 0.2); /* Background transparan putih */
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 20px;
        }

        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background-color: #333; /* Background hitam */
            color: white;
            font-size: 16px;
            cursor: pointer;
        }

        input[type="submit"] {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background-color: #3b3d6b;
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        input[type="submit"]:hover {
            background-color: #5f5f91;
        }

        img {
            margin-top: 20px;
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }

        .footer {
            margin-top: 20px;
            font-size: 1rem;
            color: #FFFFFF;  /* Warna putih untuk footer */
            font-weight: bold;
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Menampilkan header dan logo
st.markdown("""
    <div>
        <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" alt="Logo Nyamuk 1" width="65" height="65">
        <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" alt="Logo Nyamuk 2" width="65" height="65">
        <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" alt="Logo Nyamuk 3" width="65" height="65">
    </div>
    <h1>Klasifikasi Suara Nyamuk Berdasarkan Spesiesnya Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit</h1>
    <h3>Upload file suara nyamuk untuk memprediksi spesiesnya</h3>
""", unsafe_allow_html=True)

# Formulir upload file audio
st.title("Upload Audio Suara Nyamuk")

test_wav = st.file_uploader("Pilih file audio...", type=["wav"])

if test_wav is not None:
    # Proses file langsung tanpa menyimpannya
    audio_bytes = test_wav.read()

    # Pre-process file audio dan prediksi
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            # Menggunakan BytesIO untuk membaca file dari memori
            X_test = load_and_preprocess_file(io.BytesIO(audio_bytes))
            result_index = model_prediction(X_test)
            label = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            st.markdown(f"Predicted Species: {label[result_index]}")

# Footer
st.markdown("""
    <div class="footer">
        <h4>© Developer: Kelompok 1 Deep Learning</h4>
    </div>
""", unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import io
import os
import gdown
import json
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import pytz
import csv
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive URLs
MODEL_URL = "https://drive.google.com/uc?id=1rbfhPOQLBKxyRvrSUS5jpHjjVBGgCKqx"
HISTORY_URL = "https://drive.google.com/uc?id=1tl_NtfvabLha3-hrwYIaQmPu3hrxYgYv"

# File Names
MODEL_FILE = "trained_model.keras"
HISTORY_FILE = "training_history.json"

# Google Drive API Authentication
DRIVE_FOLDER_ID = "your_drive_folder_id_here"

# Utility Functions
def download_file(url, output):
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False)

def upload_to_drive(file_path, file_name):
    credentials = None  # Load your Google API credentials here
    service = build("drive", "v3", credentials=credentials)
    file_metadata = {"name": file_name, "parents": [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(file_path, mimetype="text/csv")
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()

def load_model():
    download_file(MODEL_URL, MODEL_FILE)
    return tf.keras.models.load_model(MODEL_FILE)

def load_training_history(file_path=HISTORY_FILE):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None

def load_and_preprocess_file(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        target_shape = (180, 180)
        mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)
        X_test = np.concatenate((mel_spectrogram_resized, mfcc_resized), axis=-1)
        return np.expand_dims(X_test, axis=0)
    except Exception as e:
        st.error(f"Error during audio preprocessing: {e}")
        return None

def model_prediction(X_test, model):
    try:
        prediction = model.predict(X_test)
        return np.argmax(prediction, axis=1)[0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

def show_prediction_result(audio_file, model):
    X_test = load_and_preprocess_file(audio_file)
    if X_test is not None:
        result_index = model_prediction(X_test, model)
        if result_index is not None:
            labels = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            st.markdown(f"**Predicted Species:** {labels[result_index]}")
        else:
            st.error("Model failed to provide a prediction.")
    else:
        st.error("Failed to process the audio file.")

def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://jenis.net/wp-content/uploads/2020/06/jenis-nyamuk-e1591437296119-768x456.jpg');
            background-size: cover;
            background-position: top center;
            color: white;
        }
        h1, h2, h3, h4, h5 {
            color: white;
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            text-align: center;
        }
        .social-icons {
            position: fixed;
            bottom: 20px;
            left: 20px;
            display: flex;
            gap: 15px;
        }
        .social-icons img {
            width: 40px;
            height: 40px;
            cursor: pointer;
        }
        .custom-box {
            margin-top: 30px;
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_header_logo():
    st.markdown(
        """
        <div class="center-content">
            <h1>Klasifikasi Suara Nyamuk Berdasarkan Spesies Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

def add_user_guide():
    st.markdown("""
    <div class="custom-box">
        <h2>Panduan Penggunaan</h2>
        <ol>
            <li>Upload file audio nyamuk dalam format <strong>.wav</strong> atau <strong>.mp3</strong>.</li>
            <li>Tekan tombol "Prediksi" untuk mengetahui spesies nyamuk.</li>
            <li>Tekan tombol "Show Training History" untuk melihat riwayat pelatihan model.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def add_feedback_section():
    st.markdown("""
    <div class="custom-box">
        <h2>Bagaimana Pendapat Anda tentang Aplikasi Ini?</h2>
        <p>Kami sangat menghargai masukan Anda untuk meningkatkan kualitas aplikasi ini.</p>
    </div>
    """, unsafe_allow_html=True)

    feedback = st.text_area("Tulis pendapat atau saran Anda di sini:")
    if st.button("Kirim Feedback"):
        if feedback.strip():
            feedback_file = "feedback.csv"
            with open(feedback_file, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([feedback, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            upload_to_drive(feedback_file, "feedback.csv")
            st.success("Terima kasih atas masukan Anda! Saran telah disimpan.")
        else:
            st.error("Silakan isi kotak saran sebelum mengirim.")

def add_dynamic_footer():
    wib = pytz.timezone('Asia/Jakarta')
    current_time = datetime.now(wib).strftime('%H:%M:%S')
    st.markdown(f"""
    <div class="footer">
        <span>© Developer: Kelompok 1 Deep Learning | Jam: {current_time} WIB</span>
    </div>
    """, unsafe_allow_html=True)

def main():
    add_bg_from_url()
    add_header_logo()
    add_user_guide()

    model = load_model()

    audio_file = st.file_uploader("Pilih file audio untuk diprediksi", type=["wav", "mp3"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        show_prediction_result(audio_file, model)

    if st.button("Show Training History"):
        history = load_training_history()
        if history:
            epochs = range(1, len(history['accuracy']) + 1)

            plt.figure()
            plt.plot(epochs, history['accuracy'], label="Training Accuracy")
            plt.plot(epochs, history['val_accuracy'], label="Validation Accuracy")
            plt.title("Accuracy Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            st.pyplot(plt)

            plt.figure()
            plt.plot(epochs, history['loss'], label="Training Loss")
            plt.plot(epochs, history['val_loss'], label="Validation Loss")
            plt.title("Loss Over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            st.pyplot(plt)

    add_feedback_section()
    add_dynamic_footer()

    st.markdown("""
    <div class="social-icons">
        <a href="https://github.com/mgilang56/TugasBesarDeeplearningKel1" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub">
        </a>
        <a href="https://wa.me/6285157725574" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg" alt="WhatsApp">
        </a>
        <a href="https://instagram.com" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" alt="Instagram">
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

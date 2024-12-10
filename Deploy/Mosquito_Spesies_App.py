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

# Google Drive URLs
MODEL_URL = "https://drive.google.com/uc?id=1rbfhPOQLBKxyRvrSUS5jpHjjVBGgCKqx"
HISTORY_URL = "https://drive.google.com/uc?id=1tl_NtfvabLha3-hrwYIaQmPu3hrxYgYv"

# File Names
MODEL_FILE = "trained_model.keras"
HISTORY_FILE = "training_history.json"

# Utility Functions
def download_file(url, output):
    """Download file from Google Drive if not already downloaded."""
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False)

def load_model():
    """Load the trained model."""
    download_file(MODEL_URL, MODEL_FILE)
    return tf.keras.models.load_model(MODEL_FILE)

def load_training_history(file_path=HISTORY_FILE):
    """Load the training history from a JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None

def load_and_preprocess_file(audio_file):
    """Load and preprocess the audio file."""
    try:
        y, sr = librosa.load(audio_file, sr=None)

        # Extract Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Resize features
        target_shape = (180, 180)
        mel_spectrogram_resized = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)

        # Combine features
        X_test = np.concatenate((mel_spectrogram_resized, mfcc_resized), axis=-1)
        return np.expand_dims(X_test, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error during audio preprocessing: {e}")
        return None

def model_prediction(X_test, model):
    """Perform prediction using the model."""
    try:
        prediction = model.predict(X_test)
        return np.argmax(prediction, axis=1)[0]  # Return predicted class index
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return None

def show_prediction_result(audio_file, model):
    """Display the prediction result."""
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

# UI Styling Functions
def add_bg_from_url():
    """Add background from a URL."""
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
        .center-content {
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
        }
        .social-icons {
            position: fixed;
            bottom: 10px;
            left: 10px;
            display: flex;
            gap: 15px;
        }
        .social-icons img {
            width: 40px;
            height: 40px;
            cursor: pointer;
        }
        .custom-button {
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 16px;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def add_header_logo():
    """Add header logo and title."""
    st.markdown(
        """
        <div class="center-content">
            <div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 20px;">
                <div style="border: 2px solid white; border-radius: 10px; padding: 5px;">
                    <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" alt="Logo 1" width="100" height="100">
                </div>
                <div style="border: 2px solid white; border-radius: 10px; padding: 5px;">
                    <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" alt="Logo 2" width="100" height="100">
                </div>
                <div style="border: 2px solid white; border-radius: 10px; padding: 5px;">
                    <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" alt="Logo 3" width="100" height="100">
                </div>
            </div>
            <h1>Klasifikasi Suara Nyamuk Berdasarkan Spesies Berbasis CNN untuk Inovasi Pengendalian Hama dan Penyakit</h1>
            <h3>Upload file suara nyamuk untuk memprediksi spesiesnya</h3>
        </div>
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
        """,
        unsafe_allow_html=True
    )

def add_user_guide():
    """Add user guide section."""
    st.markdown("""
    <div style="margin-top: 30px; padding: 20px; background-color: rgba(0, 0, 0, 0.6); border-radius: 10px; color: white;">
        <h2>Panduan Penggunaan</h2>
        <ol>
            <li>Upload file audio nyamuk dalam format <strong>.wav</strong> atau <strong>.mp3</strong>.</li>
            <li>Tekan tombol "Prediksi" untuk mengetahui spesies nyamuk.</li>
            <li>Tekan tombol "Show Training History" untuk melihat riwayat pelatihan model.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def add_dynamic_footer():
    """Add footer with real-time clock in WIB."""
    wib = pytz.timezone('Asia/Jakarta')  # Zona waktu WIB
    current_time = datetime.now(wib).strftime('%H:%M:%S')
    st.markdown(f"""
    <div class="footer">
        <span>© Developer: Kelompok 1 Deep Learning | Jam: {current_time} WIB</span>
    </div>
    """, unsafe_allow_html=True)

def add_animation():
    """Add mosquito animation."""
    animation_url = "https://assets9.lottiefiles.com/packages/lf20_9pnbs7tv.json"  # Animasi nyamuk
    r = requests.get(animation_url)
    if r.status_code == 200:
        lottie_animation = r.json()
        st_lottie(lottie_animation, height=300, key="mosquito-animation")

def main():
    add_bg_from_url()
    add_header_logo()
    add_animation()  # Animasi nyamuk
    add_user_guide()

    # Load Model
    model = load_model()

    # File Uploader
    audio_file = st.file_uploader("Pilih file audio untuk diprediksi", type=["wav", "mp3"], help="Drag & Drop atau klik untuk upload file.")
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        show_prediction_result(audio_file, model)

    # Show Training History
    if st.button("Show Training History", help="Lihat riwayat pelatihan model."):
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

    add_dynamic_footer()

if __name__ == "__main__":
    main()

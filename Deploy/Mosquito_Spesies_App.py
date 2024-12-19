import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
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
import soundfile as sf

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
        # Membaca file audio yang di-upload sebagai byte stream
        audio_bytes = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio using librosa from byte stream
        y, sr = librosa.load(audio_buffer, sr=None)

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
        print(f"Error during audio preprocessing: {e}")
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
    """Display the prediction result along with the spectrogram and species image."""
    X_test = load_and_preprocess_file(audio_file)
    if X_test is not None:
        result_index = model_prediction(X_test, model)
        if result_index is not None:
            labels = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]
            species_images = {
                "Aedes Aegypti": "https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/Aedes%20Aegypti.png?raw=true",
                "Anopheles Stephensi": "https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/Anopeles.png?raw=true",
                "Culex Pipiens": "https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/Culex%20Pipiens.png?raw=true"
            }

            predicted_species = labels[result_index]
            st.markdown(f"**Predicted Species:** {predicted_species}")

            # Plotting Mel-Spectrogram
            audio_file.seek(0)  # Reset file pointer to the start
            y, sr = librosa.load(audio_file, sr=None)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            plt.figure(figsize=(10, 6))  # Mel-Spectrogram plot size
            librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-Spectrogram')
            st.pyplot(plt)

            # Display the species image with the same size as the Mel-Spectrogram
            st.image(species_images[predicted_species], use_column_width=True)

        else:
            st.error("Model failed to provide a prediction.")
    else:
        st.error("Failed to process the audio file.")

# UI Styling Functions
def add_bg_from_url():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/backgroundcoba2.jpg?raw=true");
        background-size: cover;
        background-position: top center;
        color: white;
    }
    .social-icons {
        position: fixed;
        bottom: 10px;
        left: 10px;
        display: flex;
        gap: 10px;
    }
    .social-icons img {
        width: 40px;
        height: 40px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

def add_header_logo():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 15px;">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" width="80" height="80">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" width="80" height="80">
            <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" width="80" height="80">
            <img src="https://github.com/mgilang56/TugasBesarDeeplearningKel1/blob/main/Deploy/Logo%20MosquID.png?raw=true" width="80" height="80">
        </div>
        <h1 style="font-size: 30px; color: white;">Klasifikasi Suara Nyamuk Berdasarkan Spesies Berbasis CNN</h1>
        <h2 style="font-size: 22px; color: white; margin-top: -10px;">untuk Inovasi Pengendalian Hama dan Penyakit</h2>
    </div>
    """, unsafe_allow_html=True)

def add_user_guide():
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <h3 style="font-size: 20px; color: white;">Panduan Penggunaan:</h3>
        <ul style="display: inline-block; text-align: left; font-size: 18px; color: white;">
            <li>Unggah file audio dari nyamuk yang ingin diklasifikasikan.</li>
            <li>Klik tombol untuk memprediksi spesies nyamuk berdasarkan suara.</li>
            <li>Hasil prediksi akan ditampilkan beserta visualisasi Mel-spectrogram.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def add_social_icons():
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
    
def add_github_rating_section():
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
        <h3 style="font-size: 20px; color: white;">Dukung Proyek Kami dengan Memberikan Star di GitHub</h3>
        <p style="font-size: 16px; color: white;">Jika Anda menyukai aplikasi ini, silakan beri kami star di repo GitHub kami:</p>
        <a href="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit" target="_blank" style="text-decoration: none;">
            <button style="
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;">
                Berikan Star di GitHub
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)


# Main Streamlit App
def main():
    """Main function for the Streamlit app."""
    add_bg_from_url()
    add_header_logo()
    add_user_guide()
    add_social_icons()
    add_github_rating_section()
    
    st.markdown("### Upload File Audio Nyamuk:")
    
    # Upload file audio
    audio_file = st.file_uploader("Pilih file audio (.wav, .mp3)", type=["wav", "mp3"])
    
    if audio_file is not None:
        model = load_model()  # Load the model
        show_prediction_result(audio_file, model)  # Show the prediction result

# Run the app
if __name__ == "__main__":
    main()

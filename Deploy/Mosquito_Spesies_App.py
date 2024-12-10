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
            right: 0;
            font-size: 14px;
            color: white;
            margin: 10px;
        }
        .center-content {
            display: flex;
            justify-content: center;
            flex-direction: column;
            align-items: center;
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
            <div>
                <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo1.png?raw=true" alt="Logo 1" width="65" height="65">
                <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo2.png?raw=true" alt="Logo 2" width="65" height="65">
                <img src="https://github.com/sains-data/Klasifikasi-Suara-Nyamuk-Berbasis-CNN-untuk-Inovasi-Pengendalian-Hama-dan-Penyakit/blob/main/Deploy/Logo3.png?raw=true" alt="Logo 3" width="65" height="65">
            </div>
            <h1>Klasifikasi Suara Nyamuk Berdasarkan Spesies</h1>
            <h3>Upload file suara nyamuk untuk memprediksi spesiesnya</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


def add_footer():
    """Add footer."""
    st.markdown(
        """
        <div class="footer">
            <h4>© Developer: Kelompok 1 Deep Learning</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Application
def main():
    add_bg_from_url()
    add_header_logo()

    # Load Model
    model = load_model()

    # File Uploader
    audio_file = st.file_uploader("Pilih file audio untuk diprediksi", type=["wav", "mp3"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")
        show_prediction_result(audio_file, model)

    # Show Training History
    if st.button("Show Training History"):
        history = load_training_history()
        if history:
            st.write("**Training History:**")
            st.json(history)

            # Plot accuracy and loss
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

    add_footer()

if __name__ == "__main__":
    main()

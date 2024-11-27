import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize

# Function to load the model
def load_model():
    model = tf.keras.models.load_model("Trained_model.h5")
    return model

# Load and preprocess the audio file
def load_and_preprocess_file(file_path, target_shape=(180, 180)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    chunk_duration = 2
    overlap_duration = 1

    # convert duration to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    # iterate over the chunks
    for i in range(num_chunks):
        # calculate the start and end of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        # extract the chunk
        chunk = audio_data[start:end]

        # mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        # resize the mel spectrogram
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)

        # Compute the MFCC
        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=13)
        mfcc_resized = resize(np.expand_dims(mfcc, axis=-1), target_shape)

        # Concatenate Mel Spectrogram and MFCC along the depth axis
        combined_features = np.concatenate([mel_spectrogram, mfcc_resized], axis=-1)

        # Append the mel spectrogram to the data list
        data.append(combined_features)

    return np.array(data)

def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]

    return max_elements[0]

# Streamlit UI
st.sidebar.title("Dashboard")

app_node = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])


# Main page
if(app_node == "Home"):
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #181646;
            color: white;
        }
        h2, h3 {
            color: white;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(''' ## Welcome to the, \n
    ## Mosquito Species Identification App! ''')

    st.markdown(''' 
### This app is designed to predict the species of mosquito based on the audio file provided
                
### How It Works:
1. **Upload Audio:** Go to the **Prediction** page in sidebar and upload an audio file of a mosquito
2. **Analysis:** Our system will analyze the audio file and predict the species of the mosquito
3. **Result:** The predicted species will be displayed on the screen
                
### The model has been trained on the following mosquito species:
1. Aedes Aegypti
2. Anopheles Stephensi
3. Culex Pipiens 
                
### Get Started
Click on the **Prediction** page in the sidebar to upload an audio file and get the prediction.
                
### About Us
Learn more about the project and the team behind it by clicking on the **About Project** page in the sidebar.
''')
    
elif(app_node == "About Project"):
    st.markdown("""
        ### About Project
        This project is a part of the capstone project for the Deep Learning Course at Institute of Technology Sumatera (ITERA).
        The goal of this project is to develop a model that can predict the species of mosquitoes based on the audio files of their sounds.
        The model is trained on a dataset of audio files of three different species of mosquitoes: Aedes Aegypti, Anopheles Stephensi, and Culex Pipiens.
        
        ### About Dataset
        #### Content
        The dataset consists of audio files of mosquito sounds. Each audio file is labeled with the species of the mosquito.
        
        ### About Team
        The team behind this project consists of the following members:
        - Ardoni Yeriko Rifana Gultom (121140141)
        - 
        - 
        - 
        - 
        -
    """)

# Prediction page
elif(app_node == "Prediction"):
    st.header("Model Prediction")
    st.markdown(''' ### Upload an audio file of a mosquito to predict its species ''')

    test_wav = st.file_uploader("Choose an audio file...", type=["wav"])

    if test_wav is not None:
        filepath = "Test_App/"+test_wav.name

    # Play Audio Button
    if(st.button("Play Audio")):
        st.audio(test_wav)

    # Predict Button
    if(st.button("Predict")):
        with st.spinner("Predicting..."):
            X_test = load_and_preprocess_file(filepath)
            result_index = model_prediction(X_test)
            label = ["Aedes Aegypti", "Anopheles Stephensi", "Culex Pipiens"]

            st.markdown("Predicted Species: {}".format(label[result_index]))
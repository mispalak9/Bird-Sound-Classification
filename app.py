import os
import json
import librosa
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
import sounddevice as sd
import wavio

filterwarnings('ignore')

def streamlit_config():
    # page configuration
    st.set_page_config(
        page_title='Bird Sound Classification',
        page_icon='https://www.shareicon.net/data/2015/08/07/81274_music_512x512.png',  # Replace with your favicon image path
        layout='centered'
    )

    # page header transparent color
    page_background_color = """
    <style>
    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position with logo
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center;">
            <img src="https://is5-ssl.mzstatic.com/image/thumb/Purple117/v4/2d/38/05/2d380587-7a1a-6367-8216-ea7f79fc1731/source/512x512bb.jpg" style="width: 100px; height: 100px; margin-right: 10px;">
            <h1 style="text-align: center;">Bird Sound Classification</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    add_vertical_space(4)

def box_with_message_and_icon(message, icon_url):
    st.markdown(
        f"""
        <div style="
            display: flex; 
            align-items: center; 
            justify-content: center; 
            flex-direction: column;
            padding: 20px; 
            border: 2px solid #ccc; 
            background-color: #f0f0f0; 
            border-radius: 10px;
            width: 60%;
            margin: 0 auto;
        ">
            <img src="{icon_url}" style="width: 50px; height: 50px; margin-bottom: 10px;">
            <h3 style="text-align: center;">{message}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    add_vertical_space(2)

def record_audio(duration=5, fs=44100):
    box_with_message_and_icon("Recording In Progress...", "https://cdn-icons-png.flaticon.com/128/1069/1069283.png")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    box_with_message_and_icon("Analysing the Sound...", "https://cdn-icons-png.flaticon.com/128/5540/5540022.png")
    wavio.write("recorded_audio.wav", recording, fs, sampwidth=2)
    return "recorded_audio.wav"

def prediction(audio_file):
    # Load the Prediction JSON File to Predict Target_Label
    with open('prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    # Extract the Audio_Signal and Sample_Rate from Input Audio
    audio, sample_rate = librosa.load(audio_file)

    # Extract the MFCC Features and Aggregate
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)

    # Reshape MFCC features to match the expected input shape for Conv1D both batch & feature dimension
    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)

    # Convert into Tensors
    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    # Load the Model and Prediction
    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(mfccs_tensors)

    # Find the Maximum Probability Value
    target_label = np.argmax(prediction)

    # Find the Target_Label Name using Prediction_dict
    predicted_class = prediction_dict[str(target_label)]
    confidence = round(np.max(prediction) * 100, 2)

    add_vertical_space(1)
    st.markdown(f'<h4 style="text-align: center; color: orange;">{confidence}% Match Found</h4>',
                unsafe_allow_html=True)

    # Display the Image
    image_path = os.path.join('Inference_Images', f'{predicted_class}.jpg')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (350, 300))

    _, col2, _ = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.image(img)

    st.markdown(f'<h3 style="text-align: center; color: green;">{predicted_class}</h3>',
                unsafe_allow_html=True)

    return predicted_class, confidence

def save_feedback(audio_file, predicted_class, confidence, feedback):
    feedback_data = {
        "audio_file": audio_file,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "feedback": feedback
    }

    if os.path.exists('feedback.json'):
        with open('feedback.json', 'r') as f:
            feedback_list = json.load(f)
    else:
        feedback_list = []

    feedback_list.append(feedback_data)

    with open('feedback.json', 'w') as f:
        json.dump(feedback_list, f, indent=4)

def display_about_section():
    st.markdown(
        """
        <hr>
        <div style="text-align: center;">
            <p>&copy; 2024 Palak Mishra. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Streamlit Configuration Setup
streamlit_config()

_, col2, _ = st.columns([0.1, 0.9, 0.1])
with col2:
    input_audio = st.file_uploader(label='Upload the Audio', type=['mp3', 'wav'])
    record_button = st.button("Record Audio")

if record_button:
    recorded_file = record_audio()
    predicted_class, confidence = prediction(recorded_file)
    feedback = st.text_area("Provide your feedback:", key="feedback_record")
    if st.button("Submit Feedback"):
        save_feedback(recorded_file, predicted_class, confidence, feedback)
        st.success("Thank you for your feedback!")

if input_audio is not None:
    _, col2, _ = st.columns([0.2, 0.8, 0.2])
    with col2:
        predicted_class, confidence = prediction(input_audio)
        feedback = st.text_area("Provide your feedback:", key="feedback_upload")
        if st.button("Submit Feedback"):
            save_feedback(input_audio.name, predicted_class, confidence, feedback)
            st.success("Thank you for your feedback!")

# Add the "About" section at the end of the app
display_about_section()

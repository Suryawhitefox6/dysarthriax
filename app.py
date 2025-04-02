import streamlit as st
from streamlit_option_menu import option_menu
import json
import librosa
import numpy as np
import tensorflow as tf
from audiorecorder import audiorecorder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import whisper
from gtts import gTTS

def speech_to_text_to_speech(audio_file):
    # tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    # wave2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
    whisper_model = whisper.load_model("base")
    options = {"language": "en", "task": "transcribe"}

    audio, sampling_rate = librosa.load(audio_file, sr=16000)
    # input_values = tokenizer(audio, return_tensors = 'pt').input_values
    # logits = wave2vec_model(input_values).logits
    # predicted_ids = torch.argmax(logits, dim =-1)
    # transcription = tokenizer.decode(predicted_ids[0])
    
    transcription = whisper_model.transcribe(audio, **options)["text"]

    tts = gTTS(transcription)
    tts.save("enhanced_audio.mp3")


def predict_dysarthria(audio_file):
    with st.spinner("Analyzing..."):
        y, sr = librosa.load(audio_file)

        mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=256), axis=1)
        mfcc_features = mfcc_features.reshape(-1, 16, 8, 1)

        model = tf.keras.models.load_model("gru_model.h5")

        prediction = model.predict(mfcc_features)
        print(prediction)
        prediction_class = int(np.round(prediction[0][0]))

        return prediction_class

st.set_page_config(layout="wide")

if 'user_credentials' not in st.session_state:
    st.session_state.user_credentials = {}

try:
    with open('user_credentials.json', 'r') as file:
        st.session_state.user_credentials = json.load(file)
except FileNotFoundError:
        st.session_state.user_credentials = {}

if 'login' not in st.session_state:
    st.session_state.login = 0

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/premium-photo/grunge-texture-dark-wallpaper_1258-14137.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

with st.sidebar:
        page = option_menu(
            "Main Menu",
            [
                "Home",
                "Signup/Login",
                "Detection",
                "Enhancement",
            ],
            icons=["house", "shield lock", "soundwave",
                   "stars"],
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "menu-title": {"color": "#333333"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                    "color": "#333333",
                },
                "nav-link-selected": {"background-color": "#fafafa"},
            },
        )


if page == "Home":
    st.markdown("<h1 style='text-align: center;'>Dysarthria Speech Defect Recognition</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([0.3,0.8])
    col1.image('home.png')
    with open('writeup.txt','r') as writeup:
        summary = writeup.read()
    col2.markdown(f"<p style='text-align: justify;'>{summary}</p>", unsafe_allow_html=True)


elif page == "Detection":
    if st.session_state.login == 1:
        st.title("Dysarthria Detection")
        st.write("To perform dysarthria detection, you can either record audio or upload an audio file.")

        selected_option = st.radio("Select an option", ["Upload a File", "Record Audio"])

        if selected_option == "Record Audio":
            uploaded_file = None
            st.write("Audio Recorder:")
            audio = audiorecorder("Click to record", "Click to stop recording")

            if len(audio) > 0:
                audio.export("recorded_audio.wav", format="wav", bitrate="256k")
                st.audio(audio.export().read())

                prediction = predict_dysarthria("recorded_audio.wav")
                print(prediction)

                if prediction is not None:
                    if prediction == 1:
                        st.success("Dysarthria Detected")
                        speech_to_text_to_speech("recorded_audio.wav")
                        st.audio("enhanced_audio.mp3")
                    else:
                        st.success("No Dysarthria Detected")
                prediction = None

        if selected_option == "Upload a File":
            uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

            if uploaded_file:
                with open("uploaded_audio.wav", 'wb') as f:
                    f.write(uploaded_file.read())

                st.audio("uploaded_audio.wav")
                prediction = predict_dysarthria("uploaded_audio.wav")
                print(prediction)

                if prediction is not None:
                    if prediction == 1:
                        st.success("Dysarthria Detected")
                        speech_to_text_to_speech("uploaded_audio.wav")
                        st.audio("enhanced_audio.mp3")
                    else:
                        st.success("No Dysarthria Detected")
                prediction = None
                uploaded_file = None
    
    else:
        st.warning("Please login to continue")

elif page == "Signup/Login":
    st.title("Login or Signup")
    st.write("To access the detection page, please login or sign up.")

    login_or_signup = st.radio("Select an option", ["Login", "Signup"])

    if login_or_signup == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in st.session_state.user_credentials and st.session_state.user_credentials[username] == password:
                st.session_state.login = 1
                st.success("Login successful!")
                st.write("You can now access the Detection page.")
            else:
                st.error("Login failed. Please check your credentials.")

    if login_or_signup == "Signup":
        st.subheader("Signup")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")

        if st.button("Signup"):
            if new_username and new_password:
                st.session_state.user_credentials[new_username] = new_password
                with open('user_credentials.json', 'w') as file:
                    json.dump(st.session_state.user_credentials, file)
                st.success("Signup successful! You can now login.")

elif page == "Enhancement":
    if st.session_state.login == 1:
        st.title("Speech Enhancement")
        st.write("To perform speech enhancement, you can either record audio or upload an audio file.")

        selected_option = st.radio("Select an option", ["Upload a File", "Record Audio"])

        if selected_option == "Record Audio":
            uploaded_file = None
            st.write("Audio Recorder:")
            audio = audiorecorder("Click to record", "Click to stop recording")

            if len(audio) > 0:
                audio.export("recorded_audio.wav", format="wav", bitrate="256k")
                st.audio(audio.export().read())

                speech_to_text_to_speech("recorded_audio.wav")
                st.audio("enhanced_audio.mp3")

        if selected_option == "Upload a File":
            uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

            if uploaded_file:
                with open("uploaded_audio.wav", 'wb') as f:
                    f.write(uploaded_file.read())

                st.audio("uploaded_audio.wav")
                speech_to_text_to_speech("uploaded_audio.wav")
                st.audio("enhanced_audio.mp3")
                uploaded_file = None
    
    else:
        st.warning("Please login to continue")

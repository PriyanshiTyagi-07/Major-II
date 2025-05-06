import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

# Load the trained model
MODEL_PATH = 'speech_emotion_recognition_tess.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels
EMOTION_LABELS = [
    'OAF_Fear', 'OAF_Pleasant_surprise', 'OAF_Sad', 'OAF_angry',
    'OAF_disgust', 'OAF_happy', 'OAF_neutral', 'YAF_angry',
    'YAF_disgust', 'YAF_fear', 'YAF_happy', 'YAF_neutral',
    'YAF_pleasant_surprised', 'YAF_sad'
]

# Feature extraction
def extract_features(data, sample_rate):
    features = []
    features.append(np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0))
    features.append(np.mean(librosa.feature.chroma_stft(y=data, sr=sample_rate).T, axis=0))
    features.append(np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0))
    features.append(np.mean(librosa.feature.rms(y=data).T, axis=0))
    features.append(np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0))
    return np.hstack(features)

# Streamlit UI
st.title("🎤 Voice Emotion Detector")

if st.button("Record Audio"):
    duration = 2.5
    sample_rate = 22050
    st.info("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording Complete")

    audio_data = audio_data.flatten()
    features = extract_features(audio_data, sample_rate)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    prediction = model.predict(features)
    predicted_emotion = EMOTION_LABELS[np.argmax(prediction)]

    st.subheader("Predicted Emotion:")
    st.markdown(f"### 🧠 {predicted_emotion}")

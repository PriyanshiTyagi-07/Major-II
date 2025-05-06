import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

# Load model
MODEL_PATH = 'speech_emotion_recognition_tess.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels used during training
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']

# Extract 128x128 Mel spectrogram
def extract_features(data, sample_rate=22050, max_pad_len=128):
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_pad_len]

    return mel_spec_db

# Record audio from mic
def record_audio(duration=3, sample_rate=22050):
    st.info("🎙️ Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Recording finished.")
    return audio_data.flatten()

# Predict emotion
def predict_emotion(audio_data, sample_rate=22050):
    features = extract_features(audio_data, sample_rate)
    features = features.reshape(1, 128, 128, 1)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    return EMOTION_LABELS[predicted_index]

# Streamlit UI
st.title("🎤 Voice Emotion Detector")

if st.button("Start Recording"):
    audio_data = record_audio()
    predicted_emotion = predict_emotion(audio_data)
    st.subheader("🧠 Predicted Emotion:")
    st.success(f"### {predicted_emotion.capitalize()}")

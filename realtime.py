import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf


# Load trained model
MODEL_PATH = 'speech_emotion_recognition_IESC.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Correct Emotion labels used during training
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']

# Feature extraction: 128x128 Mel spectrogram
def extract_features(data, sample_rate=22050, max_pad_len=128):
    mel_spec = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_pad_len]

    return mel_spec_db

# Record audio
def record_audio(duration=3, sample_rate=22050):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return audio_data.flatten()

# Predict emotion
def predict_emotion(audio_data, sample_rate=22050):
    features = extract_features(audio_data, sample_rate)
    features = features.reshape(1, 128, 128, 1)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    return EMOTION_LABELS[predicted_index]

# Main loop
if __name__ == "__main__":
    while True:
        audio_data = record_audio()
        emotion = predict_emotion(audio_data)
        print(f"Predicted Emotion: {emotion}")
        cont = input("Press Enter to record again or type 'exit' to stop: ")
        if cont.lower() == 'exit':
            break

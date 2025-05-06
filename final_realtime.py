import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
from keras.models import load_model
import joblib  # For loading scaler and encoder

# Load your trained model
model = load_model('final_model.keras')  # Adjust if using 'model.h5'

# Load the scaler and encoder
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Feature extraction function (same as training pipeline)
def extract_features(data, sample_rate):
    result = np.array([])
    
    # Zero-crossing rate
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, mean_zero))

    # Chroma feature
    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC feature
    mfcc_out = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc_out))

    # Root Mean Square (RMS) feature
    root_mean_out = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, root_mean_out))

    # Mel Spectrogram feature
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel_spectrogram))

    return result

# Real-time recording and prediction loop
def record_and_predict(duration=2.5, fs=22050):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait for the recording to finish
    audio = audio.flatten()  # Flatten the audio for feature extraction

    # Extract features from the recorded audio
    features = extract_features(audio, fs).reshape(1, -1)

    # Scale the features using the saved scaler
    features_scaled = scaler.transform(features)

    # Expand dimensions to match model input
    features_scaled = np.expand_dims(features_scaled, axis=2)

    # Predict emotion using the trained model
    prediction = model.predict(features_scaled)
    
    # Inverse transform to get the original emotion label
    emotion = encoder.inverse_transform(prediction)
    
    print("Predicted Emotion:", emotion[0][0])

# Main loop for continuous recording and prediction
def start_realtime_detection():
    print("Starting real-time emotion detection. Press 'Ctrl + C' to stop.")

    while True:
        record_and_predict()
        if input("Press Enter to continue or type 'q' to quit: ") == 'q':
            break


# Run the real-time detection
start_realtime_detection()



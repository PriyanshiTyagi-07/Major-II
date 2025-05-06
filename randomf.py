import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib  # For saving ML models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Path to TESS dataset
DATASET_PATH = "/home/priyanshi/Documents/project/TESS Toronto emotional speech set data"

# Define emotions based on file names
EMOTIONS = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "pleasant_surprise",
    "sad": "sad"
}

# Function to extract MFCC features (more suitable for ML models)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Take mean along time-axis
    return mfcc_mean

# Load dataset
X, y = [], []
for subdir, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            label = next((EMOTIONS[key] for key in EMOTIONS if key in file), None)
            if label:
                file_path = os.path.join(subdir, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluate models
svm_pred = svm_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Save models
joblib.dump(svm_model, "svm_speech_emotion.pkl")
joblib.dump(rf_model, "rf_speech_emotion.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

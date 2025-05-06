import os
import numpy as np
import librosa
import joblib  # For saving models
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Path to TESS dataset
DATASET_PATH = "/Users/priyanshityagi/Documents/project/TESS Toronto emotional speech set data"

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

# Function to extract MFCC features
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

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train MLP (Neural Network) model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500)
mlp_model.fit(X_train, y_train)

# Evaluate models
knn_pred = knn_model.predict(X_test)
mlp_pred = mlp_model.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)
mlp_acc = accuracy_score(y_test, mlp_pred)

print(f"KNN Accuracy: {knn_acc:.4f}")
print(f"MLP Accuracy: {mlp_acc:.4f}")

# Save models
joblib.dump(knn_model, "knn_speech_emotion.pkl")
joblib.dump(mlp_model, "mlp_speech_emotion.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

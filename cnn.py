# import os
# import librosa
# import numpy as np
# import librosa.display
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, TimeDistributed
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam

# # Path to TESS dataset
# DATASET_PATH = "/Users/priyanshityagi/Documents/project/TESS Toronto emotional speech set data"

# # Define emotions based on file names
# EMOTIONS = {
#     "angry": "angry",
#     "disgust": "disgust",
#     "fear": "fear",
#     "happy": "happy",
#     "neutral": "neutral",
#     "ps": "pleasant_surprise",
#     "sad": "sad"
# }

# # Function to extract features
# def extract_features(file_path, max_pad_len=128):
#     y, sr = librosa.load(file_path, sr=22050)
#     mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
#     if mel_spec_db.shape[1] < max_pad_len:
#         pad_width = max_pad_len - mel_spec_db.shape[1]
#         mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mel_spec_db = mel_spec_db[:, :max_pad_len]
    
#     return mel_spec_db

# # Load dataset
# X, y = [], []
# for subdir, _, files in os.walk(DATASET_PATH):
#     for file in files:
#         if file.endswith(".wav"):
#             label = next((EMOTIONS[key] for key in EMOTIONS if key in file), None)
#             if label:
#                 file_path = os.path.join(subdir, file)
#                 features = extract_features(file_path)
#                 X.append(features)
#                 y.append(label)

# # Convert to NumPy arrays
# X = np.array(X)
# y = np.array([list(EMOTIONS.values()).index(emotion) for emotion in y])

# # Reshape data for CNN-LSTM
# X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
# y = to_categorical(y, num_classes=len(EMOTIONS))

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Build CNN-LSTM model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),  # Input should be (height, width, channels)
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.3),
#     Dense(len(EMOTIONS), activation='softmax')
# ])


# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# # Train model
# history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# # Evaluate model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.4f}")

# # Save model
# model.save("speech_emotion_recognition_tess.h5")
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Path to IESC dataset
DATASET_PATH = "/Users/priyanshityagi/Documents/project/IESC"

# Define emotion mapping based on filename prefix
EMOTIONS = {
    "A": "angry",
    "F": "fear",
    "H": "happy",
    "N": "neutral",
    "S": "sad"
}

# Function to extract mel-spectrogram features
def extract_features(file_path, max_pad_len=128):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_pad_len]
    return mel_spec_db

# Load dataset and labels
X, y = [], []
for subdir, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_key = file[0]  # First character of filename
            if emotion_key in EMOTIONS:
                label = EMOTIONS[emotion_key]
                file_path = os.path.join(subdir, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array([list(EMOTIONS.values()).index(emotion) for emotion in y])

# Reshape and encode
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = to_categorical(y, num_classes=len(EMOTIONS))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(EMOTIONS), activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
model.save("speech_emotion_recognition_IESC.h5")

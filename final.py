import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import joblib

# Path setup
input_folder = '/Users/priyanshityagi/Documents/Emotion_Recog/TESS Toronto emotional speech set data/'
Main_WAV_Path = Path(input_folder)
Wav_Path = list(Main_WAV_Path.glob(r'**/*.wav'))
Wav_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], Wav_Path))
Wav_Path_Series = pd.Series(Wav_Path, name='WAV').astype(str)
Wav_Labels_Series = pd.Series(Wav_Labels, name='EMOTION')
Main_Wav_Data = pd.concat([Wav_Path_Series, Wav_Labels_Series], axis=1)
Main_Wav_Data = Main_Wav_Data.sample(frac=1).reset_index(drop=True)

# Data augmentation
def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])
    return data

def stretch_process(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift_process(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch_process(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def extract_process(data, sample_rate):
    output_result = np.array([])
    mean_zero = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    output_result = np.hstack((output_result, mean_zero))

    stft_out = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft_out, sr=sample_rate).T, axis=0)
    output_result = np.hstack((output_result, chroma_stft))

    mfcc_out = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    output_result = np.hstack((output_result, mfcc_out))

    root_mean_out = np.mean(librosa.feature.rms(y=data).T, axis=0)
    output_result = np.hstack((output_result, root_mean_out))

    mel_spectogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    output_result = np.hstack((output_result, mel_spectogram))

    return output_result

def export_process(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    result = np.array(extract_process(data, sample_rate))
    noise_out = add_noise(data)
    result = np.vstack((result, extract_process(noise_out, sample_rate)))

    new_out = stretch_process(data)
    strectch_pitch = pitch_process(new_out, sample_rate)
    result = np.vstack((result, extract_process(strectch_pitch, sample_rate)))

    return result

# Feature extraction
X_train, y_train = [], []
for i, (path, emotion) in enumerate(zip(Main_Wav_Data.WAV, Main_Wav_Data.EMOTION)):
    features = export_process(path)
    for element in features:
        X_train.append(element)
        y_train.append(emotion)
    print(f"\rProcessed {i+1} files", end='')

New_Features_Wav = pd.DataFrame(X_train)
New_Features_Wav['EMOTIONS'] = y_train

# Preprocessing
encoder_label = OneHotEncoder()
scaler_data = StandardScaler()
X = New_Features_Wav.iloc[:, :-1].values
Y = New_Features_Wav['EMOTIONS'].values

# OneHotEncoding & StandardScaling
Y = encoder_label.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=42, shuffle=True)

X_train = scaler_data.fit_transform(X_train)
X_test = scaler_data.transform(X_test)

# Reshape for Conv1D
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Model
Model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),
    tf.keras.layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),
    tf.keras.layers.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(units=14, activation='softmax')  # Update units based on number of emotions
])

Model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('final_model.keras', verbose=1, save_best_only=True),  # Corrected file extension
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1)
]

# Training
history = Model.fit(X_train, y_train, batch_size=64, epochs=50, callbacks=callbacks, validation_data=(X_test, y_test))

# Plot Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(15,5))
plt.plot(epochs, acc, color='#ff0066')
plt.plot(epochs, val_acc, color='#00ccff')
plt.title('Train and Test Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.ylabel('accuracy')
plt.xlabel('epoch')

# Plot Loss
plt.figure(figsize=(15,5))
plt.plot(epochs, loss, color='#ff0066')
plt.plot(epochs, val_loss, color='#00ccff')
plt.legend(['train', 'test'], loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Training and Validation Loss')

# Evaluation
results = Model.evaluate(X_test, y_test)
print('Test loss:', results[0])
print('Test Accuracy:', results[1]*100)

# Predictions & Confusion Matrix
prediction_test = Model.predict(X_test)
y_prediction = encoder_label.inverse_transform(prediction_test)
y_test = encoder_label.inverse_transform(y_test)

conf_matrix = confusion_matrix(y_test, y_prediction)
plt.figure(figsize=(13,6))
sns.heatmap(conf_matrix, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
print(classification_report(y_test, y_prediction))

# Save Model and Preprocessors
joblib.dump(scaler_data, 'scaler.pkl')
joblib.dump(encoder_label, 'encoder.pkl')

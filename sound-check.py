import tensorflow as tf
import numpy as np
import librosa
import os

# Function to extract features from audio clips
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with open(file_path, "rb") as f:
        signal, sr = librosa.load(f, sr=None)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
        features.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr).T,axis=0)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T,axis=0)
        features.extend(mel)
    return features

# Function to load dataset
def load_dataset(data_dir):
    X = []
    y = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Prepare dataset
data_dir = "Users/rakesh.rai/code/whistles/dataset"
X, y = load_dataset(data_dir)

# Split dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Function to detect pattern occurrences in a sound clip
def detect_pattern(file_path, threshold=0.5, window_size=1.0, hop_length=512):
    with open(file_path, "rb") as f:
        signal, sr = librosa.load(f, sr=None)
    
    features = []
    for i in range(0, len(signal) - hop_length, hop_length):
        segment = signal[i:i+hop_length]
        if len(segment) == hop_length:
            segment_features = extract_features(segment)
            features.append(segment_features)
    features = np.array(features)

    predictions = model.predict(features).flatten()
    occurrences = np.sum(predictions >= threshold)
    return occurrences

# Example usage
sound_clip_path = "/Users/rakesh.rai/code/whistles/full_tracks/pressure-cooker-5431124yes.wav"
pattern_count = detect_pattern(sound_clip_path)
print(f"Number of pattern occurrences: {pattern_count}")

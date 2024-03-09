import pandas as pd
import numpy as np
import librosa

# Step 1: Read the CSV file containing sound intervals and labels
csv_file = './whistles_sound/air-or-steam-pressure-release-29600.csv'
df = pd.read_csv(csv_file)

# Step 2: Segment the audio file based on sound intervals
audio_file = './whistles_sound/air-or-steam-pressure-release-29600.wav'
audio_data, _ = librosa.load(audio_file, sr=None)

X = []
y = []

for index, row in df.iterrows():
    print(row)
    start_time = int(row['Start Time'] * sample_rate)  # Convert start time from seconds to samples
    end_time = int(row['End Time'] * sample_rate)      # Convert end time from seconds to samples
    print(row)
    audio_segment = audio_data[start_time:end_time]
    X.append(audio_segment)
    y.append(row['Label'])

# Step 3: Feature extraction (you may need to use different features depending on your model)
# For example, here we'll use Mel-Frequency Cepstral Coefficients (MFCCs)
X_mfcc = [librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=20) for segment in X]

# Step 4: Label encoding (assuming labels are strings)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 5: Prepare input-output pairs
X_train = np.array(X_mfcc)
y_train = np.array(y_encoded)

# Now, you can use X_train and y_train to train your audio classifier model

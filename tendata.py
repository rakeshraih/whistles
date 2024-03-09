import tensorflow as tf
import pandas as pd
import numpy as np

# Function to load and slice audio files based on start time, end time, and label
def load_sliced_audio(file_path, csv_file):
    # Load CSV file containing start times, end times, and labels
    df = pd.read_csv(csv_file, names=["start", "end", "label"])

    # Initialize lists to store sliced audio segments and their labels
    audio_segments = []
    labels = []

    # Iterate through each row in the CSV file
    for index, row in df.iterrows():
        start_time = row.start
        end_time = row.end
        label = row.label
        print(row.start,row.label)
        # Read audio file as binary
        audio_binary = tf.io.read_file(file_path)

        # Decode audio file
        audio_waveform, sample_rate = tf.audio.decode_wav(audio_binary)
        sample_rate = sample_rate.numpy()
        # Convert start and end times from seconds to samples
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Slice audio segment based on start and end times
        audio_segment = audio_waveform[start_sample:end_sample, :]

        # Append sliced audio segment and label to lists
        audio_segments.append(audio_segment)
        labels.append(label)

    return audio_segments, labels

# Example usage
audio_file = './whistles_sound/air-or-steam-pressure-release-29600.wav'
csv_file = './whistles_sound/air-or-steam-pressure-release-29600.csv'

audio_segments, labels = load_sliced_audio(audio_file, csv_file)
print(len(audio_segments))
print(len(labels))
print(audio_segments[:5])
print(labels[:5])

x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([0, 1, 0, 1, 0])

# Create a dataset from tensors or NumPy arrays
# dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

# Convert lists to TensorFlow dataset
# dataset = tf.data.Dataset.from_tensor_slices((audio_segments, labels))


# Example arrays
array1 = [1, 2, 3, 4, 5]
array2 = ['a', 'b', 'c', 'd', 'e']

# Create a dataset from the arrays
dataset = tf.data.Dataset.from_tensor_slices((audio_segments, labels))

# Print the dataset
for item in dataset:
    print(item)
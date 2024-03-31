import os
from pydub import AudioSegment

# Function to convert mp3 to wav
def mp3_to_wav(mp3_file):
    # Load mp3 audio file
    audio = AudioSegment.from_mp3(mp3_file)
    # Define output wav file path
    wav_file = os.path.splitext(mp3_file)[0] + '.wav'
    # Export audio in wav format
    audio.export(wav_file, format="wav")

# Function to convert mp3 to wav
def m4a_to_wav(m4a_file):
    # Load mp3 audio file
    audio = AudioSegment.from_file(m4a_file, format='m4a')
    # Define output wav file path
    wav_file = os.path.splitext(m4a_file)[0] + '.wav'
    # Export audio in wav format
    audio.export(wav_file, format="wav")
    
# Directory containing Python files
directory = './whistles_sound'

# Iterate through all files in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    # Check if the file is a Python file and has a .mp3 extension
    if os.path.isfile(filepath) and filename.endswith('.mp3'):
        # Convert mp3 to wav
        mp3_to_wav(filepath)
    if os.path.isfile(filepath) and filename.endswith('.m4a'):
              # Convert mp3 to wav
        m4a_to_wav(filepath)  
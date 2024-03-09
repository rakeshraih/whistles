from pydub import AudioSegment
import pandas as pd
import os

def split_audio(input_file, output_prefix, start_times, end_times, label):
    # Load the audio file
    audio = AudioSegment.from_wav('/Users/raihera/project/whistles/whistles_sound/'+input_file)
    snippet = audio[float(start_times)*1000:float(end_times)*1000]
    output_file = f"{output_prefix}.wav"
    if label.endswith('yes'):
      snippet.export('./yes/'+output_file, format="wav")
    else :
       snippet.export('./no/'+output_file, format="wav")  
#     # Iterate over the start and end times
#     for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
#         # Calculate start and end positions in milliseconds
#         start_pos = int(start_time * 1000)
#         end_pos = int(end_time * 1000)

#         # Extract the audio snippet
#         snippet = audio[start_pos:end_pos]

#         # Save the snippet to a new file
#         output_file = f"{output_prefix}_{i+1}.wav"
#         snippet.export(output_file, format="wav")

# Example usage
input_file = "input.wav"
output_prefix = "snippet"
start_times = [0.0, 3.0, 6.0]  # Example start times in seconds
end_times = [2.5, 5.5, 8.0]    # Example end times in seconds
POS = os.path.join('/Users','raihera','project','whistles', 'whistles_sound')

for file in os.listdir(POS):
        if file.endswith('.wav'):
            fileName = os.path.splitext(file)[0]
            abalone_train = pd.read_csv('/Users/raihera/project/whistles/whistles_sound/'+fileName+'.csv', names=["start", "end", "label"])
            for index, row in abalone_train.iterrows():
                  split_audio(file, fileName+str(index)+row.label, row.start, row.end, row.label)

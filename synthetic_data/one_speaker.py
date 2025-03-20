import os
import csv
from melo.api import TTS
import torch

# Create output directory if it doesn't exist
output_dir = "./wavs"
os.makedirs(output_dir, exist_ok=True)

# Check if CUDA (GPU) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the TTS model
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id
# speaker_ids = {'EN-US': 0, 'EN-BR': 1, 'EN_INDIA': 2, 'EN-AU': 3, 'EN-Default': 4}
accent = list(speaker_ids.keys())
speaker_id = accent[1]  # Using the second speaker ID
print(f"Speaker ID: {speaker_id}")

# Text-to-speech conversion function
def text2speech(text, speed, speaker_id, output_path):
    model.tts_to_file(text, speaker_ids[speaker_id], output_path, speed=speed)
    return output_path

# Parameters
speed = 1.0  # You can adjust this if needed

# Process the metadata.csv file
with open('metadata.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter='|')
    for row in csv_reader:
        if len(row) >= 2:
            # Extract file identifier and text
            file_id = row[0].strip()
            text = row[1].strip()
            
            # Create output path
            output_path = os.path.join(output_dir, f"{file_id}.wav")
            
            # Generate the WAV file
            print(f"Generating WAV for {file_id}: '{text}'")
            text2speech(text, speed, speaker_id, output_path)
            print(f"Saved to: {output_path}")
        else:
            print(f"Skipping malformed line: {row}")

print("All WAV files have been generated successfully!")
import os
import csv
import random
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

# Define available speaker IDs (typically these would come from the model)
speaker_ids = model.hps.data.spk2id
#speaker_ids = {'EN-US': 0, 'EN-BR': 1, 'EN_INDIA': 2, 'EN-AU': 3, 'EN-Default': 4}
accent_keys = list(speaker_ids.keys())

# For demonstration purposes only - print available accents
print(f"Available speaker accents: {accent_keys}")
print(f"Using device: {device}")

# Text-to-speech conversion function
def text2speech(text, speed, speaker_id_value, output_path):
    model.tts_to_file(text, speaker_id_value, output_path, speed=speed)
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
            
            # Randomly select a speaker accent
            random_accent = random.choice(accent_keys)
            speaker_id_value = speaker_ids[random_accent]
            
            # Generate the WAV file
            print(f"Generating WAV for {file_id}: '{text}' using accent: {random_accent}")
            text2speech(text, speed, speaker_id_value, output_path)
            print(f"Saved to: {output_path}")
        else:
            print(f"Skipping malformed line: {row}")

print("All WAV files have been generated successfully!")
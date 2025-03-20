from melo.api import TTS
import torch  # Make sure you have PyTorch installed for GPU support

# Check if CUDA (GPU) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Force GPU if available, otherwise fallback to CPU

# Initialize the TTS model
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id  # assuming this is a dictionary

# Example: If you want to choose the second speaker
accent = list(speaker_ids.keys())  # Get the list of speaker names/IDs
speaker_id = accent[1]  # Set the second speaker ID (adjust index as needed)

print(f"Using device: {device}")
print(f"Speaker ID: {speaker_id}")

# Text-to-speech conversion function
def text2speech(text, speed, speaker_id, output_path="./output.wav"):
    model.tts_to_file(text, speaker_ids[speaker_id], output_path, speed=speed)
    return output_path

# Define the text and other parameters
text = """
This is a exploratory test to develop an ASR from scratch for the aumotive industry
"""
speed = 1.0

# Convert text to speech
wav_path = text2speech(text, speed, speaker_id)
print(f"Output saved to: {wav_path}")
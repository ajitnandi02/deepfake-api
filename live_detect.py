import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile

from preprocessing.noise_removal import clean_audio
from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

# ---------------- CONFIG ----------------
DURATION = 5        # seconds
SAMPLE_RATE = 16000
# ----------------------------------------

device = torch.device("cpu")

# Load trained model
model = DeepfakeClassifier().to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.eval()

print("\n🎤 Live Audio Deepfake Detection")
print("Recording will start now...")
print(f"Speak for {DURATION} seconds...\n")

# Record audio
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()

print("Recording finished.\n")

# Save temp wav
temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
write(temp_wav, SAMPLE_RATE, audio)

# Preprocess + extract features
audio_clean = clean_audio(temp_wav)
features = extract_features(audio_clean).to(device)
features = features.mean(dim=0, keepdim=True)
# Predict
with torch.no_grad():
    output = model(features)
    prediction = torch.argmax(output, dim=1).item()

# Terminal output
print("=========== LIVE DETECTION RESULT ===========")
if prediction == 0:
    print("Prediction: REAL audio ✅")
else:
    print("Prediction: FAKE audio ❌")
print("===========================================\n")

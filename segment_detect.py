import torch
import numpy as np
import librosa
import tkinter as tk
from tkinter import filedialog

from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

# ================= CONFIG =================
SAMPLE_RATE = 16000
WINDOW_SEC = 4
HOP_SEC = 2

SILENCE_RMS_THRESHOLD = 0.01
CONFIDENCE_THRESHOLD = 0.85
STRONG_FAKE_CONF = 0.95
MIN_CONTINUOUS_FAKE_SEC = 30

BATCH_DURATION_SEC = 600

device = torch.device("cpu")

# ================= LOAD MODEL =================
try:
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ================= FILE PICKER =================
root = tk.Tk()
root.withdraw()

audio_path = filedialog.askopenfilename(
    title="Select Audio File",
    filetypes=[("Audio files", "*.wav *.ogg *.mp3")]
)

if not audio_path:
    print("❌ No file selected")
    exit()

print(f"\n📁 Selected file: {audio_path}")

# ================= LOAD AUDIO IN BATCHES =================
try:
    audio_full, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_duration = len(audio_full) / sr
    print(f"Total audio duration: {total_duration:.2f}s")
except Exception as e:
    print(f"❌ Error loading audio: {e}")
    exit()

window_len = int(WINDOW_SEC * sr)
hop_len = int(HOP_SEC * sr)

segments = []
continuous_fake = 0
max_continuous_fake = 0
total_fake_p = 0
num_segments = 0

# Process in batches
batch_size = int(BATCH_DURATION_SEC * sr)
for batch_start in range(0, len(audio_full), batch_size):
    batch_end = min(batch_start + batch_size, len(audio_full))
    audio = audio_full[batch_start:batch_end]

    # ================= SEGMENT LOOP =================
    for start in range(0, len(audio) - window_len + 1, hop_len):
        end = start + window_len
        chunk = audio[start:end]

        # -------- Silence check --------
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < SILENCE_RMS_THRESHOLD:
            continue

        # -------- Feature extraction --------
        chunk = chunk.astype("float32")
        try:
            features = extract_features(chunk).to(device)
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            continue

        try:
            with torch.no_grad():
                logits = model(features)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            continue

        real_p, fake_p = probs
        confidence = max(real_p, fake_p)

        # -------- Low confidence ignore --------
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        label = "FAKE" if fake_p > real_p else "REAL"

        segments.append((label, fake_p))
        total_fake_p += fake_p
        num_segments += 1

        # -------- Update continuous fake counter --------
        if label == "FAKE" and fake_p >= STRONG_FAKE_CONF:
            continuous_fake += HOP_SEC
            max_continuous_fake = max(max_continuous_fake, continuous_fake)
        else:
            continuous_fake = 0

# ================= OVERALL PREDICTION =================
print("\n🔍 OVERALL PREDICTION")
if num_segments > 0:
    overall_fake_p = (total_fake_p / num_segments) * 100
else:
    overall_fake_p = 0.0
print(f"Overall FAKE Percentage: {overall_fake_p:.1f}%")

# ================= GLOBAL DECISION =================
print("\n🧠 GLOBAL DECISION")
print("=" * 30)
print(f"Total segments analyzed: {len(segments)}")
print(f"Max continuous fake duration: {max_continuous_fake:.2f}s")

if max_continuous_fake >= MIN_CONTINUOUS_FAKE_SEC:
    print(f"❌ AUDIO LIKELY FAKE (continuous fake ≈ {max_continuous_fake}s)")
else:
    print("✅ AUDIO LIKELY REAL")

print("\n✅ Analysis complete.")
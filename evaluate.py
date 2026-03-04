import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import librosa

from preprocessing.noise_removal import clean_audio  # Ensure this exists
from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
CALIBRATION_BIAS = -0.1  # Same as in live_stream_detect.py
device = torch.device("cpu")

# Paths to test dataset (adjust if your folder structure is different)
TEST_REAL_DIR = "dataset/training/real/"  # Folder with real audio files
TEST_FAKE_DIR = "dataset/training/fake/"  # Folder with fake audio files

# Load model
try:
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Function to load and process audio
def process_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        audio_clean = clean_audio(audio)  # Assuming clean_audio takes numpy array
        features = extract_features(audio_clean).to(device)
        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)[0]
        real_percentage = probs[1].item() * 100
        fake_percentage = (probs[0].item() + CALIBRATION_BIAS) * 100
        fake_percentage = max(0.0, min(100.0, fake_percentage))
        prediction = 1 if fake_percentage > 50 else 0  # 1 = FAKE, 0 = REAL
        return prediction
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Load test data
true_labels = []
predictions = []

print("Loading test data...")

# Load real audio (label = 0)
for file in os.listdir(TEST_REAL_DIR):
    if file.endswith(('.wav', '.mp3', '.ogg')):
        file_path = os.path.join(TEST_REAL_DIR, file)
        pred = process_audio(file_path)
        if pred is not None:
            predictions.append(pred)
            true_labels.append(0)  # REAL

# Load fake audio (label = 1)
for file in os.listdir(TEST_FAKE_DIR):
    if file.endswith(('.wav', '.mp3', '.ogg')):
        file_path = os.path.join(TEST_FAKE_DIR, file)
        pred = process_audio(file_path)
        if pred is not None:
            predictions.append(pred)
            true_labels.append(1)  # FAKE

# Calculate metrics
if len(predictions) == 0:
    print("❌ No test data found. Check your dataset paths.")
    exit()

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, pos_label=1)  # FAKE
recall = recall_score(true_labels, predictions, pos_label=1)
f1 = f1_score(true_labels, predictions, pos_label=1)

print("\n📊 MODEL EVALUATION REPORT")
print("=" * 40)
print(f"Total test samples: {len(predictions)}")
print(f"Accuracy: {accuracy:.2f} ({accuracy*100:.1f}%)")
print(f"Precision (FAKE): {precision:.2f}")
print(f"Recall (FAKE): {recall:.2f}")
print(f"F1-Score (FAKE): {f1:.2f}")
print("\nDetailed Report:")
print(classification_report(true_labels, predictions, target_names=['REAL', 'FAKE']))

# Save report
with open("evaluation_report.txt", "w") as f:
    f.write("MODEL EVALUATION REPORT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Total test samples: {len(predictions)}\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision (FAKE): {precision:.2f}\n")
    f.write(f"Recall (FAKE): {recall:.2f}\n")
    f.write(f"F1-Score (FAKE): {f1:.2f}\n")
    f.write("\nDetailed Report:\n")
    f.write(classification_report(true_labels, predictions, target_names=['REAL', 'FAKE']))

print("\n✅ Report saved to 'evaluation_report.txt'")
import torch
from preprocessing.noise_removal import clean_audio
from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

# Device
device = torch.device("cpu")

# Load trained model
model = DeepfakeClassifier().to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.eval()

# Test audio path (change filename if needed)
audio_path = "dataset/training/real/file9.wav"

# Preprocess audio
audio = clean_audio(audio_path)

# Feature extraction
features = extract_features(audio).to(device)
features = features.mean(dim=0, keepdim=True)

# Prediction
with torch.no_grad():
    output = model(features)
    prediction = torch.argmax(output, dim=1).item()

# Output
if prediction == 0:
    print("✅ Prediction: REAL audio")
else:
    print("❌ Prediction: FAKE audio")

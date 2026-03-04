import os
import torch
import librosa
import torch.nn as nn
import torch.optim as optim

from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

SAMPLE_RATE = 16000
REAL_PATH = "dataset/training/real"
FAKE_PATH = "dataset/training/fake"

device = torch.device("cpu")
model = DeepfakeClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    return audio

epochs = 10   # 🔥 increase for stronger model

for epoch in range(epochs):
    total_loss = 0
    samples = 0

    for label, folder in [(0, REAL_PATH), (1, FAKE_PATH)]:
        for file in os.listdir(folder):
            if not file.lower().endswith((".wav", ".mp3", ".ogg")):
                continue

            path = os.path.join(folder, file)
            try:
                audio = load_audio(path)
            except:
                continue

            feats = extract_features(audio).to(device)
            y = torch.tensor([label]).to(device)

            optimizer.zero_grad()
            out = model(feats)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            samples += 1

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/samples:.4f}")

torch.save(model.state_dict(), "deepfake_model.pth")
print("✅ Training complete")

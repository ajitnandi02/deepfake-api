from fastapi import FastAPI, File, UploadFile
import torch
import numpy as np
import librosa
import tempfile
import os

from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

# ================= INIT APP =================
app = FastAPI(title="Audio Deepfake Detection API")

# ================= CONFIG =================
SAMPLE_RATE = 16000
WINDOW_SEC = 8
HOP_SEC = 4
device = torch.device("cpu")

# ================= LOAD MODEL =================
try:
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    raise e


# ================= ROOT ROUTE =================
@app.get("/")
def home():
    return {"message": "Audio Deepfake Detection API is running 🚀"}


# ================= DETECTION ROUTE =================
@app.post("/detect")
async def detect_audio(file: UploadFile = File(...)):

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # Load audio
        audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
        os.remove(temp_path)

        if len(audio) < SAMPLE_RATE * 2:
            return {"error": "Audio too short. Minimum 2 seconds required."}

        window_len = int(WINDOW_SEC * sr)
        hop_len = int(HOP_SEC * sr)

        total_real = 0.0
        total_fake = 0.0
        segment_count = 0

        # Segment-wise analysis
        for start in range(0, len(audio) - window_len + 1, hop_len):
            chunk = audio[start:start + window_len].astype("float32")

            features = extract_features(chunk).to(device)

            with torch.no_grad():
                logits = model(features)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            real_p, fake_p = float(probs[0]), float(probs[1])

            total_real += real_p
            total_fake += fake_p
            segment_count += 1

        if segment_count == 0:
            return {"error": "Audio too short for segmentation."}

        # Final percentage calculation
        avg_real = float((total_real / segment_count) * 100)
        avg_fake = float((total_fake / segment_count) * 100)

        result = "FAKE" if avg_fake > avg_real else "REAL"

        return {
            "real_percentage": round(avg_real, 2),
            "fake_percentage": round(avg_fake, 2),
            "final_prediction": result
        }

    except Exception as e:
        return {"error": str(e)}
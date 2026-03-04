from fastapi import FastAPI, UploadFile, File
import shutil
import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

app = FastAPI(title="Audio Deepfake Detection API")

# Load wav2vec2 from HuggingFace
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Dummy classifier (replace with your trained classifier)
classifier = torch.nn.Linear(768, 2)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Audio Deepfake Detection API running"}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    try:

        file_path = os.path.join(UPLOAD_DIR, audio.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        # Load audio
        waveform, sr = librosa.load(file_path, sr=16000)

        input_values = processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).input_values

        with torch.no_grad():
            outputs = wav2vec_model(input_values)

        features = outputs.last_hidden_state.mean(dim=1)

        logits = classifier(features)

        probs = torch.softmax(logits, dim=1)

        confidence, pred_class = torch.max(probs, dim=1)

        prediction = "Fake" if pred_class.item() == 1 else "Real"

        os.remove(file_path)

        return {
            "filename": audio.filename,
            "prediction": prediction,
            "confidence": round(confidence.item(), 4)
        }

    except Exception as e:
        return {"error": str(e)}
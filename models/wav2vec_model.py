import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

MODEL_PATH = "pretrained/wav2vec2-base"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
wav2vec = Wav2Vec2Model.from_pretrained(MODEL_PATH)

wav2vec.eval()

def extract_features(audio, sr=16000):
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = wav2vec(**inputs)

    # ✅ mean pooling (IMPORTANT)
    features = outputs.last_hidden_state.mean(dim=1)
    return features

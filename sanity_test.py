import torch
from models.classifier import DeepfakeClassifier

model = DeepfakeClassifier()
state = torch.load("deepfake_model.pth", map_location="cpu")
model.load_state_dict(state)

# check weights
for name, param in model.named_parameters():
    print(name, param.abs().mean().item())
    break

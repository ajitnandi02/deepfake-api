import torch.nn as nn

class DeepfakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 2)
    def forward(self, x):
        return self.fc(x)

# Audio Deepfake Detection System

## Project Objective
To detect whether an audio signal is REAL or DEEPFAKE using a deep learning model.
The system supports uploaded audio, video files, and real-time microphone input.

## Dataset Structure
dataset/
├── training/
│   ├── real/
│   └── fake/
├── validation/
│   ├── real/
│   └── fake/
└── testing/
    ├── real/
    └── fake/

## Model Architecture
- Pretrained Wav2Vec2 for feature extraction
- Custom neural network classifier (2-class: real/fake)
- Softmax for confidence score

## Features
- Audio file detection
- Video file audio extraction and detection
- Real-time microphone-based live detection
- Confidence score output (real vs fake)
- Noise and silence handling
- Continuous live stream detection
- Automatic result logging

## How to Run

### Train model
```bash
python train.py

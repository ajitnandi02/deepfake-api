import librosa
import numpy as np

def clean_audio(path):
    audio, sr = librosa.load(path, sr=16000)
    audio = librosa.effects.preemphasis(audio)
    return audio

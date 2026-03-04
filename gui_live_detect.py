import tkinter as tk
from tkinter import scrolledtext
import threading
import time
from datetime import datetime
import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import os
import winsound

from preprocessing.noise_removal import clean_audio
from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

# ---------------- CONFIG (same as live_stream_detect.py) ----------------
SAMPLE_RATE = 16000
DURATION = 5
SILENCE_THRESHOLD = 0.01
NOISE_THRESHOLD = 0.02
ZCR_THRESHOLD = 0.05
FAKE_CONFIDENCE_THRESHOLD = 85.0
MIN_CONTINUOUS_FAKE_SEC = 30
CALIBRATION_BIAS = -0.1
LOG_FILE = "live_results.log"
device = torch.device("cpu")

# Load model
try:
    model = DeepfakeClassifier().to(device)
    model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
    model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Global variables for GUI
running = False
total_chunks = 0
total_real_p = 0
total_fake_p = 0
continuous_fake = 0
max_continuous_fake = 0
session_start = datetime.now()

# Function to run live detection in a thread
def run_live_detection(output_text):
    global running, total_chunks, total_real_p, total_fake_p, continuous_fake, max_continuous_fake
    while running:
        try:
            output_text.insert(tk.END, "Listening...\n")
            output_text.see(tk.END)

            # Record audio
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()

            # Calculate RMS and ZCR
            rms = np.sqrt(np.mean(audio ** 2))
            zcr = np.mean(np.abs(np.diff(np.sign(audio.squeeze())))) / 2

            if rms < SILENCE_THRESHOLD:
                output_text.insert(tk.END, "Status: SILENT\nREAL 0.0% | FAKE 0.0%\n\n")
                continuous_fake = 0
                total_chunks += 1
                continue

            if rms < NOISE_THRESHOLD and zcr < ZCR_THRESHOLD:
                output_text.insert(tk.END, "Status: NOISY\nREAL 0.0% | FAKE 0.0%\n\n")
                continuous_fake = 0
                total_chunks += 1
                continue

            # Process audio
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            write(temp_wav, SAMPLE_RATE, audio)
            audio_clean = clean_audio(temp_wav)
            features = extract_features(audio_clean).to(device)

            with torch.no_grad():
                output = model(features)
                probs = torch.softmax(output, dim=1)[0]

            real_percentage = probs[1].item() * 100
            fake_percentage = (probs[0].item() + CALIBRATION_BIAS) * 100
            fake_percentage = max(0.0, min(100.0, fake_percentage))

            output_text.insert(tk.END, f"REAL {real_percentage:.1f}% | FAKE {fake_percentage:.1f}%\n")

            if fake_percentage > 85:
                winsound.Beep(1000, 500)
                output_text.insert(tk.END, "🚨 ALERT: High FAKE detected!\n")

            # Update stats
            total_real_p += real_percentage
            total_fake_p += fake_percentage
            total_chunks += 1

            if fake_percentage > FAKE_CONFIDENCE_THRESHOLD:
                continuous_fake += DURATION
                max_continuous_fake = max(max_continuous_fake, continuous_fake)
            else:
                continuous_fake = 0

            avg_real_p = total_real_p / total_chunks if total_chunks > 0 else 0.0
            avg_fake_p = total_fake_p / total_chunks if total_chunks > 0 else 0.0
            output_text.insert(tk.END, f"Running Average: REAL {avg_real_p:.1f}% | FAKE {avg_fake_p:.1f}%\n\n")
            output_text.see(tk.END)

            os.unlink(temp_wav)
            time.sleep(1)

        except Exception as e:
            output_text.insert(tk.END, f"❌ Error: {e}\n")
            time.sleep(1)

# GUI Functions
def start_detection():
    global running
    if not running:
        running = True
        output_text.delete(1.0, tk.END)  # Clear previous output
        output_text.insert(tk.END, "🎤 LIVE AUDIO DEEPFAKE DETECTION STARTED\n\n")
        threading.Thread(target=run_live_detection, args=(output_text,)).start()

def stop_detection():
    global running
    running = False
    output_text.insert(tk.END, "\n🧠 GLOBAL SESSION DECISION\n" + "=" * 40 + "\n")
    session_duration = (datetime.now() - session_start).total_seconds()
    avg_real_p = total_real_p / total_chunks if total_chunks > 0 else 0.0
    avg_fake_p = total_fake_p / total_chunks if total_chunks > 0 else 0.0
    output_text.insert(tk.END, f"Total chunks: {total_chunks}\nSession duration: {session_duration:.1f}s\nAverage: REAL {avg_real_p:.1f}% | FAKE {avg_fake_p:.1f}%\nMax Continuous Fake: {max_continuous_fake:.1f}s\n")
    if max_continuous_fake >= MIN_CONTINUOUS_FAKE_SEC:
        output_text.insert(tk.END, f"❌ SESSION LIKELY FAKE\n")
    else:
        output_text.insert(tk.END, "✅ SESSION LIKELY REAL\n")
    output_text.insert(tk.END, "\nLive detection stopped.\n")

# Create GUI Window
root = tk.Tk()
root.title("Audio Deepfake Detector")
root.geometry("600x400")

start_btn = tk.Button(root, text="Start Live Detection", command=start_detection, bg="green", fg="white")
start_btn.pack(pady=10)

stop_btn = tk.Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white")
stop_btn.pack(pady=10)

output_text = scrolledtext.ScrolledText(root, width=70, height=15)
output_text.pack(pady=10)

root.mainloop()
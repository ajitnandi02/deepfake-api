import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile
import time
from datetime import datetime
import os
import winsound
import librosa

from preprocessing.noise_removal import clean_audio
from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
DURATION = 5

SILENCE_THRESHOLD = 0.3  # Increased for noisy environments
NOISE_THRESHOLD = 0.05  # For noise detection
ZCR_THRESHOLD = 0.1  # For speech vs noise
FAKE_CONFIDENCE_THRESHOLD = 85.0
MIN_CONTINUOUS_FAKE_SEC = 30
CALIBRATION_BIAS = 0.2  # Balanced bias
AMPLIFY_FACTOR = 2.0

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

print("\n🎤 LIVE AUDIO DEEPFAKE DETECTION (CTRL + C to stop)")
print("=================================================\n")

# Baseline RMS calculation for dynamic silence threshold
print("Calibrating silence... (stay silent for 2 seconds)")
baseline_audio = sd.rec(int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()
baseline_rms = np.sqrt(np.mean(baseline_audio ** 2))
SILENCE_THRESHOLD = max(SILENCE_THRESHOLD, baseline_rms + 0.1)  # Dynamic threshold
print(f"Baseline RMS: {baseline_rms:.4f}, Adjusted Silence Threshold: {SILENCE_THRESHOLD:.4f}")
print("Calibration done.\n")

# Session tracking
total_chunks = 0
total_real_p = 0
total_fake_p = 0
continuous_fake = 0
max_continuous_fake = 0
session_start = datetime.now()

while True:
    try:
        print("Listening...")

        # Record audio
        try:
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
        except Exception as e:
            print(f"❌ Recording error: {e}\n")
            time.sleep(1)
            continue

        # Volume fix: Amplify and normalize
        audio = audio * AMPLIFY_FACTOR
        audio = librosa.util.normalize(audio)

        # Calculate RMS, ZCR, and variance
        rms = np.sqrt(np.mean(audio ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio.squeeze())))) / 2
        variance = np.var(audio)

        # Debug: Print RMS, ZCR, and variance
        print(f"RMS: {rms:.4f}, ZCR: {zcr:.4f}, Variance: {variance:.6f}")

        # Silence check (with variance for better detection)
        if rms < SILENCE_THRESHOLD and zcr < 0.05 and variance < 0.001:
            print("Status: SILENT (True silence detected - speak or adjust mic)")
            fake_percentage = 0.0
            is_fake = False
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.now()} | SILENT | FAKE 0.0%\n")
            print("FAKE Percentage: 0.0%\n")
            continuous_fake = 0
            total_chunks += 1
            continue

        # Noise check
        if rms < NOISE_THRESHOLD and zcr < ZCR_THRESHOLD:
            print("Status: NOISY (Background noise - try quieter environment)")
            fake_percentage = 0.0
            is_fake = False
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.now()} | NOISY | FAKE 0.0%\n")
            print("FAKE Percentage: 0.0%\n")
            continuous_fake = 0
            total_chunks += 1
            continue

        # Speech detected
        try:
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            write(temp_wav, SAMPLE_RATE, audio)

            audio_clean = clean_audio(temp_wav)
            features = extract_features(audio_clean).to(device)

            with torch.no_grad():
                output = model(features)
                probs = torch.softmax(output, dim=1)[0]

            # Swapped probabilities
            real_percentage = probs[1].item() * 100
            fake_percentage = (probs[0].item() + CALIBRATION_BIAS) * 100
            fake_percentage = max(0.0, min(100.0, fake_percentage))

            is_fake = fake_percentage > FAKE_CONFIDENCE_THRESHOLD

            print("=========== RESULT ===========")
            print(f"REAL {real_percentage:.1f}% | FAKE {fake_percentage:.1f}%")
            print("==============================\n")

            if fake_percentage > 85:
                winsound.Beep(1000, 500)
                print("🚨 ALERT: High FAKE detected!")

            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.now()} | DETECTED | REAL {real_percentage:.1f}% | FAKE {fake_percentage:.1f}%\n")

            total_real_p += real_percentage
            total_fake_p += fake_percentage
            total_chunks += 1

            if is_fake:
                continuous_fake += DURATION
                max_continuous_fake = max(max_continuous_fake, continuous_fake)
            else:
                continuous_fake = 0

            avg_real_p = total_real_p / total_chunks if total_chunks > 0 else 0.0
            avg_fake_p = total_fake_p / total_chunks if total_chunks > 0 else 0.0
            print(
                f"Running Average: REAL {avg_real_p:.1f}% | FAKE {avg_fake_p:.1f}% | Max Continuous Fake: {max_continuous_fake:.1f}s\n")

        except Exception as e:
            print(f"❌ Processing error: {e}")
            fake_percentage = 0.0
            is_fake = False
            continuous_fake = 0
            total_chunks += 1
            print("REAL 0.0% | FAKE 0.0%\n")
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.now()} | ERROR | REAL 0.0% | FAKE 0.0%\n")

        os.unlink(temp_wav)
        time.sleep(1)

    except KeyboardInterrupt:
        print("\n🧠 GLOBAL SESSION DECISION")
        print("=" * 40)
        session_duration = (datetime.now() - session_start).total_seconds()
        avg_real_p = total_real_p / total_chunks if total_chunks > 0 else 0.0
        avg_fake_p = total_fake_p / total_chunks if total_chunks > 0 else 0.0
        print(f"Total chunks analyzed: {total_chunks}")
        print(f"Session duration: {session_duration:.1f}s")
        print(f"Average: REAL {avg_real_p:.1f}% | FAKE {avg_fake_p:.1f}%")
        print(f"Max continuous fake duration: {max_continuous_fake:.1f}s")

        if max_continuous_fake >= MIN_CONTINUOUS_FAKE_SEC:
            print(f"❌ SESSION LIKELY FAKE (continuous fake ≈ {max_continuous_fake:.1f}s)")
        else:
            print("✅ SESSION LIKELY REAL")

        print("\nLive detection stopped.")
        break
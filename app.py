from flask import Flask, request, render_template, jsonify
import torch
import librosa
import numpy as np
import os
from werkzeug.utils import secure_filename

from models.wav2vec_model import extract_features
from models.classifier import DeepfakeClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model
device = torch.device("cpu")
model = DeepfakeClassifier().to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.eval()

CALIBRATION_BIAS = 0.2


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process audio
    try:
        audio, sr = librosa.load(filepath, sr=16000)
        audio = librosa.util.normalize(audio)

        # Extract features
        features = extract_features(audio).to(device)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)[0]

        real_percentage = probs[1].item() * 100
        fake_percentage = (probs[0].item() + CALIBRATION_BIAS) * 100
        fake_percentage = max(0.0, min(100.0, fake_percentage))

        result = "REAL" if real_percentage > fake_percentage else "FAKE"

        return jsonify({
            'result': result,
            'real_percentage': round(real_percentage, 2),
            'fake_percentage': round(fake_percentage, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
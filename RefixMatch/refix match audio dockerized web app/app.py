from flask import Flask, request, jsonify
import torch
import torchaudio
from torchvision import models, transforms
import torch.nn.functional as F
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("best_model_mobilenet_refixmatch_40.pt", map_location=device)
model.eval()

transform = transforms.Compose([
    Resample(orig_freq=16000, new_freq=16000),  
    MelSpectrogram(sample_rate=16000, n_mels=64),
    AmplitudeToDB(),
])

@app.route('/')
def home():
    return "Welcome to the Audio Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():

    audio_file = request.files['file']
    waveform, sample_rate = torchaudio.load(audio_file.stream)

    mel_spec = transform(waveform).unsqueeze(0).to(device)  

    with torch.no_grad():
        output = model(mel_spec)
        probs = F.softmax(output, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()

    return jsonify({
        'class': predicted_class,
        'probability': float(probs[0, predicted_class])
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

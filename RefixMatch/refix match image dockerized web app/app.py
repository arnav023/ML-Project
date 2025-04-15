from flask import Flask, request, jsonify
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, 200)
model.load_state_dict(torch.load("refixmatch_resnet18_450lpc.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():

    img_file = request.files['file']
    img = Image.open(img_file.stream)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(probs, 1)

    #returning the predicted class
    return jsonify({'class': int(predicted_class), 'probability': float(probs[0, predicted_class])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("best_model.pt", map_location=device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/')
def home():
    return "Welcome to the Text Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

    predicted_class = torch.argmax(probs, dim=1).item()

    return jsonify({
        'class': predicted_class,
        'probability': float(probs[0, predicted_class])
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

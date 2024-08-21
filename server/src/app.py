from flask import Flask, request, jsonify
import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
import torch.nn.functional as F
import time

app = Flask(__name__)

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DebertaV2ForSequenceClassification.from_pretrained('fine_tuned_deberta').to(device)
tokenizer = DebertaV2Tokenizer.from_pretrained('fine_tuned_deberta')
model.eval()

def predict(texts):
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return predictions, probabilities

@app.route('/api/predict', methods=['POST'])
def run_inference():
    time.sleep(1.2)
    data = request.json
    input_texts = [data['text']]
    _, probabilities = predict(input_texts)

    ai_probability = float(probabilities[0][1])

    if ai_probability < 0.01:
        ai_probability = 0.01
    elif ai_probability > 0.99:
        ai_probability = 0.99

    result = {
        "probability": ai_probability
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)

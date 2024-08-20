from flask import Flask, request, jsonify
import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
import torch.nn.functional as F

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

@app.route('/predict', methods=['POST'])
def run_inference():
    data = request.json
    input_texts = [data['text']]
    predictions, probabilities = predict(input_texts)

    result = {
        "probability": float(probabilities[0][1]) if predictions[0] == 1 else float(probabilities[0][0]),
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)

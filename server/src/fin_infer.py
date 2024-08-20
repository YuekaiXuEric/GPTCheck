import torch
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
import torch.nn.functional as F
import sys
import json

def predict(texts):
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return predictions, probabilities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DebertaV2ForSequenceClassification.from_pretrained('fine_tuned_deberta').to(device)
tokenizer = DebertaV2Tokenizer.from_pretrained('fine_tuned_deberta')
model.eval()

def run_inference(input_texts):
    predictions, probabilities = predict(input_texts)
    result = {
        "prediction": int(predictions[0]),
        "probability": float(probabilities[0][1]) if predictions[0] == 1 else float(probabilities[0][0]),
        "inferred_as": "AI" if predictions[0] == 1 else "Human"
    }
    return result

if __name__ == "__main__":
    input_text = sys.argv[1]
    input_texts = [input_text]
    result = run_inference(input_texts)
    print(json.dumps(result))
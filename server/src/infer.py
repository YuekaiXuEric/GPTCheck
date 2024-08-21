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
    input_texts = ['My experience with argumentative writing could just some maybe, I’m not pretty good at writing in my high school, but I like to sure my opinions, ideas, and views. Usually for the ideas about the genre of writing is states your view and trying the search and talk it deeply. For the Death and Life of Great American Cities, I think it can by kind of the argumentative writing, because Jacobs states the opposite view for the American cities, which are different to the others. Also she has made her points of why she thing is this, for example, the north end of Boston is actually not a Slum by going there.']
    result = run_inference(input_texts)
    print(json.dumps(result))
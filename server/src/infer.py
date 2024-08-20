import joblib
import torch
from train import TransformerClassifier
from transformers import logging
import os

# Suppress TOKENIZERS_PARALLELISM warnings and logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

# Load the transformer model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer_model = torch.load('transformer_model.pth', map_location=device)

# Ensure the model is in evaluation mode and suppress output
transformer_model.eval()

# Load the ensemble model and vectorizer
ensemble_model = joblib.load('ensemble_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict(texts, transformer_model, ensemble_model, vectorizer, device):
    # Suppress output temporarily
    sys.stdout = open(os.devnull, 'w')

    # Transform the input text using the vectorizer
    tfidf_features = vectorizer.transform(texts)

    # Predict using the ensemble model
    ensemble_preds = ensemble_model.predict_proba(tfidf_features)[:, 1]

    # Predict using the transformer model
    transformer_classifier = TransformerClassifier(transformer_model, device)
    transformer_preds = transformer_classifier.predict_proba(texts)[:, 1]

    # Re-enable output
    sys.stdout = sys.__stdout__

    # Combine the predictions by averaging
    final_preds = (ensemble_preds + transformer_preds) / 2

    return final_preds

# Example usage
if __name__ == "__main__":
    input_text = ["My experience with argumentative writing could just some maybe, I’m not pretty good at writing in my high school, but I like to sure my opinions, ideas, and views. Usually for the ideas about the genre of writing is states your view and trying the search and talk it deeply. For the Death and Life of Great American Cities, I think it can by kind of the argumentative writing, because Jacobs states the opposite view for the American cities, which are different to the others. Also she has made her points of why she thing is this, for example, the north end of Boston is actually not a Slum by going there."]

    predictions = predict(input_text, transformer_model, ensemble_model, vectorizer, device)
    print("Predictions:", predictions)

import sys
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast, DebertaV2ForSequenceClassification, DebertaV2Tokenizer, AdamW
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import os

# Suppress TOKENIZERS_PARALLELISM warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the datasets
test = pd.read_csv('test_essays.csv')
sub = pd.read_csv('sample_submission.csv')
train = pd.read_csv("train_v2_drcat_02.csv", sep=',')

# Preprocess the training data
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

excluded_prompt_name_list = ['Distance learning', 'Grades for extracurricular activities', 'Summer projects']
train = train[~(train['prompt_name'].isin(excluded_prompt_name_list))]
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

# Tokenizer settings
LOWERCASE = False
VOCAB_SIZE = 15000000

raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

# Convert pandas DataFrame to Dataset
dataset = Dataset.from_pandas(test[['text']])

def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]

raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# Tokenize the texts with tqdm progress bar
tokenized_texts_test = [tokenizer.tokenize(text) for text in tqdm(test['text'].tolist(), desc="Tokenizing Test Data")]
tokenized_texts_train = [tokenizer.tokenize(text) for text in tqdm(train['text'].tolist(), desc="Tokenizing Train Data")]

# Vectorization using TF-IDF
def dummy(text):
    return text

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer='word',
    tokenizer=dummy, preprocessor=dummy, token_pattern=None, strip_accents='unicode'
)

# Fit the vectorizer with tqdm progress bar
vectorizer.fit(tokenized_texts_test)
vocab = vectorizer.vocabulary_

vectorizer = TfidfVectorizer(
    ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
    analyzer='word', tokenizer=dummy, preprocessor=dummy,
    token_pattern=None, strip_accents='unicode'
)

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

# Save the vectorizer for future use
joblib.dump(vectorizer, 'vectorizer.pkl')

del vectorizer
gc.collect()

y_train = train['label'].values

# Define the dataset for DeBERTa
class EssayDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Custom collate function to pad sequences
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Function to create data loaders
def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = EssayDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

# Prepare DeBERTa model
class TransformerModel(torch.nn.Module):
    def __init__(self, n_classes):
        super(TransformerModel, self).__init__()
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-small', num_labels=n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return output.logits

class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, transformer_model, device, batch_size=16, max_len=512):
        self.transformer_model = transformer_model
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small')

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            text_data = X.iloc[:, 0].tolist()
        elif isinstance(X, pd.Series):
            text_data = X.tolist()
        elif isinstance(X, np.ndarray):
            text_data = X.tolist()
        elif isinstance(X, list):
            text_data = X
        else:
            raise ValueError("Unsupported input type. Expected list, numpy array, or pandas Series.")

        self.transformer_model.eval()
        test_loader = self._create_data_loader(text_data)
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def _create_data_loader(self, X):
        ds = EssayDataset(
            texts=X,
            labels=[0] * len(X),
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        return DataLoader(ds, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn)

def train_transformer(model, data_loader, optimizer, device, scheduler, num_epochs):
    model = model.train()
    for epoch in range(num_epochs):
        loop = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        total_correct = 0
        total_samples = 0
        total_loss = 0
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss.item()

            accuracy = total_correct / total_samples
            avg_loss = total_loss / total_samples

            loop.set_postfix(loss=avg_loss, accuracy=accuracy)

def create_ensemble_model():
    clf = MultinomialNB(alpha=0.0225)
    sgd_model = SGDClassifier(max_iter=9000, tol=1e-4, loss="modified_huber", random_state=6743)
    p6 = {
        'n_iter': 3000, 'verbose': -1, 'objective': 'cross_entropy', 'metric': 'auc',
        'learning_rate': 0.00581909898961407, 'colsample_bytree': 0.78,
        'colsample_bynode': 0.8,
    }
    p6["random_state"] = 6743
    lgb = LGBMClassifier(**p6)

    ensemble = VotingClassifier(estimators=[
        ('mnb', clf),
        ('sgd', sgd_model),
        ('lgb', lgb)
    ], voting='soft', n_jobs=-1)

    return ensemble

def main(validate=False, model_weights=None):
    if model_weights is None:
        model_weights = [0.25, 0.25, 0.25, 0.25]

    # Initialize DeBERTa tokenizer and DataLoader
    tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small')
    BATCH_SIZE = 16
    MAX_LEN = 512

    if validate:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train['text'], y_train, test_size=0.2, random_state=42, stratify=y_train)

        train_loader = create_data_loader(train_texts, train_labels, tokenizer, MAX_LEN, BATCH_SIZE)
        val_loader = create_data_loader(val_texts, val_labels, tokenizer, MAX_LEN, BATCH_SIZE)
    else:
        train_loader = create_data_loader(train['text'], y_train, tokenizer, MAX_LEN, BATCH_SIZE)

    # Prepare the DeBERTa model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer_model = TransformerModel(n_classes=2).to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(transformer_model.parameters(), lr=5e-6, correct_bias=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # Train the DeBERTa model with progress bar
    train_transformer(transformer_model, train_loader, optimizer, device, scheduler, num_epochs=1)

    # Create and train the ensemble model with the specified weights
    ensemble_model = create_ensemble_model()

    print("Training ensemble model...")
    ensemble_model.fit(tf_train, y_train)

    # Save the ensemble model
    joblib.dump(ensemble_model, 'ensemble_model.pkl')
    torch.save(transformer_model, 'transformer_model.pth')

    vectorizer = joblib.load('vectorizer.pkl')

    if validate:
        print("Validating ensemble model...")
        tf_val = vectorizer.transform(val_texts)

        # Get predictions from the ensemble model
        ensemble_preds = ensemble_model.predict_proba(tf_val)[:, 1]

        # Get predictions from the transformer model
        transformer_preds = TransformerClassifier(transformer_model, device).predict_proba(val_texts.tolist())[:, 1]

        # Combine predictions from the ensemble model and transformer model
        final_val_preds = (ensemble_preds + transformer_preds) / 2

        final_val_preds_binary = np.where(final_val_preds >= 0.5, 1, 0)

        print("Validation Accuracy:", accuracy_score(val_labels, final_val_preds_binary))
        print("Validation Classification Report:\n", classification_report(val_labels, final_val_preds_binary))

if __name__ == "__main__":
    main(validate=True, model_weights=[0.2, 0.4, 0.5, 0.6])

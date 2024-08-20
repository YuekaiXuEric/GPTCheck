import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import logging
logging.set_verbosity_error()

train = pd.read_csv("train_v2_drcat_02.csv", sep=',')
test = pd.read_csv('test_essays.csv')

train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

excluded_prompt_name_list = ['Distance learning', 'Grades for extracurricular activities', 'Summer projects']
train = train[~(train['prompt_name'].isin(excluded_prompt_name_list))]
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-small')
MAX_LEN = 512

class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
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

def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = EssayDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

def main(EPOCHS=3, do_validation=True):
    if do_validation:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train['text'], train['label'], test_size=0.2, random_state=42, stratify=train['label'])

        train_texts.reset_index(drop=True, inplace=True)
        train_labels.reset_index(drop=True, inplace=True)
        val_texts.reset_index(drop=True, inplace=True)
        val_labels.reset_index(drop=True, inplace=True)

        train_loader = create_data_loader(train_texts, train_labels, tokenizer, MAX_LEN, batch_size=16)
        val_loader = create_data_loader(val_texts, val_labels, tokenizer, MAX_LEN, batch_size=16)
    else:
        train_loader = create_data_loader(train['text'], train['label'], tokenizer, MAX_LEN, batch_size=16)
        val_loader = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-small', num_labels=2).to(device)

    optimizer = AdamW(model.parameters(), lr=4e-6, correct_bias=False)

    def train_epoch(model, data_loader, optimizer, device, epoch, EPOCHS):
        model = model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0

        loop = tqdm(data_loader, leave=True)

        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss.item()

            accuracy = total_correct / total_samples
            avg_loss = total_loss / total_samples

            loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
            loop.set_postfix(loss=avg_loss, accuracy=accuracy)

        return accuracy, avg_loss

    def eval_model(model, data_loader, device):
        model = model.eval()
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        loop = tqdm(data_loader, leave=True, desc="Validation")

        with torch.no_grad():
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                accuracy = total_correct / total_samples

                loop.set_postfix(accuracy=accuracy)

        return accuracy, all_preds, all_labels

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, epoch, EPOCHS)
        print(f"Train loss {train_loss}, accuracy {train_acc}")

        if do_validation:
            val_acc, val_preds, val_labels = eval_model(model, val_loader, device)
            print(f"Validation accuracy {val_acc}")
            print("Classification Report:\n", classification_report(val_labels, val_preds))

    model.save_pretrained('fine_tuned_deberta')
    tokenizer.save_pretrained('fine_tuned_deberta')

if __name__ == "__main__":
    main(EPOCHS=2, do_validation=True)

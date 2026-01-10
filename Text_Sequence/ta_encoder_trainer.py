import datetime
import time

from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import shutil
import random
from torch.utils.data import Dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(tokens_path, labels_path, location_path):
    with open(tokens_path, 'rb') as file1, open(labels_path, 'rb') as file2, open(location_path, 'rb') as file3:
        codes = pickle.load(file1)
        labels = pickle.load(file2)
        location = pickle.load(file3)

    combined = list(zip(codes, labels, location))
    random.shuffle(combined)
    codes, labels, location = zip(*combined)

    return codes, labels, location

def preprocess_data(codes, labels, tokenizer, max_length=512):
    inputs = tokenizer(codes, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels)
    return inputs

class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def extract_features(dataset, model, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            cls_embedding = hidden_states[:, 0, :]

            features.append(cls_embedding.cpu())
            labels.append(label.cpu())

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

def fine_tune_encoder(model, dataset, optimizer, num_epochs=10, batch_size=16, save_dir='./java_model'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state[:, 0, :]

            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx+1) % 100 == 0:
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time

                progress = (batch_idx + 1) / len(dataloader) * 100

        avg_loss = total_loss / len(dataloader)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        elapsed_time = epoch_end_time - start_time


        torch.save(model.state_dict(), f"{save_dir}/model_epoch_{epoch + 1}.pth")
        torch.save(optimizer.state_dict(), f"{save_dir}/optimizer_epoch_{epoch + 1}.pth")

    return model

def main():
    tokens_path = "java_ast_train/ast_data_tokens.pkl"
    labels_path = "java_ast_train/ast_data_labels.pkl"
    location_path = "java_ast_train/ast_data_location.pkl"
    out_put_features = 'java_token_features_train/'

    codes, labels, location = load_data(tokens_path, labels_path, location_path)

    tokenizer = RobertaTokenizer.from_pretrained('./unixcoder-base', use_fast=True)
    model = RobertaModel.from_pretrained('./unixcoder-base').to(device)

    processed_data = preprocess_data(codes, labels, tokenizer)

    dataset = CustomDataset(
        processed_data['input_ids'],
        processed_data['attention_mask'],
        processed_data['labels']
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model = fine_tune_encoder(model, dataset, optimizer, num_epochs=10, batch_size=16)

    features, labels = extract_features(dataset, model)

    features_path = os.path.join(out_put_features, "features.pkl")
    label_path = os.path.join(out_put_features, "labels.pkl")
    location_path = os.path.join(out_put_features, "location.txt")

    with open(features_path, "wb") as f_feat, open(label_path, "wb") as f_label:
        pickle.dump(features, f_feat)
        pickle.dump(labels, f_label)

    for lo in location:
        with open(location_path, 'a') as txt_file:
            txt_file.write(str(lo)+'\n')


if __name__ == "__main__":
    main()

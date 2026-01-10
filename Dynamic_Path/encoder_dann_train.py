import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
import numpy as np
import os
import pickle
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(source_tokens, source_labels, target_tokens, target_labels):

    with open(source_tokens, 'rb') as f1, open(source_labels, 'rb') as f2:
        src_codes = pickle.load(f1)
        src_vuln_labels = pickle.load(f2)
    src_domain_labels = [0] * len(src_codes) 
    with open(target_tokens, 'rb') as f3, open(target_labels, 'rb') as f4:
        tgt_codes = pickle.load(f3)
        tgt_vuln_labels = pickle.load(f4)
    tgt_domain_labels = [1] * len(tgt_codes)  
    all_codes = src_codes + tgt_codes
    all_vuln_labels = src_vuln_labels + tgt_vuln_labels
    all_domain_labels = src_domain_labels + tgt_domain_labels

    combined = list(zip(all_codes, all_vuln_labels, all_domain_labels))
    random.shuffle(combined) 
    codes, vuln_labels, domain_labels = zip(*combined) 

    return codes, vuln_labels, domain_labels


def preprocess_data(codes, vuln_labels, domain_labels, tokenizer, max_length=512):
    inputs = tokenizer(
        codes,
        truncation=True, 
        padding="max_length", 
        max_length=max_length,  
        return_tensors="pt"  
    )
    inputs['vuln_labels'] = torch.tensor(vuln_labels) 
    inputs['domain_labels'] = torch.tensor(domain_labels)  
    return inputs


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, vuln_labels, domain_labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.vuln_labels = vuln_labels
        self.domain_labels = domain_labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'vuln_labels': self.vuln_labels[idx],
            'domain_labels': self.domain_labels[idx]
        }


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class TransformerModel(nn.Module):
    def __init__(self, input_dim=50265, hidden_dim=768, num_heads=8, num_layers=6, max_length=512):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        seq_len = input_ids.size(1)
        embedded += self.positional_encoding[:, :seq_len, :]

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        transformer_output = self.encoder(
            embedded,
            src_key_padding_mask=key_padding_mask
        )

        cls_embedding = transformer_output[:, 0, :]
        return cls_embedding


class EnhancedDANN(nn.Module):
    def __init__(self, encoder, hidden_size=768):
        super().__init__()
        self.encoder = encoder

        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(alpha=2.0),
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask=attention_mask)
        task_output = self.task_classifier(features)
        domain_output = self.domain_classifier(features)
        return task_output, domain_output


def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def train_dann(model, dataset, optimizer, num_epochs=50, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()

    best_domain_acc = 0
    for epoch in range(num_epochs):
        total_loss = 0
        domain_correct = 0
        task_correct = 0
        total_samples = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vuln_labels = batch['vuln_labels'].to(device)
            domain_labels = batch['domain_labels'].to(device)
            task_logits, domain_logits = model(input_ids, attention_mask)
            task_loss = F.cross_entropy(task_logits, vuln_labels)
            domain_loss = F.cross_entropy(domain_logits, domain_labels)
            total_batch_loss = task_loss + domain_loss
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
            task_preds = torch.argmax(task_logits, dim=1)
            domain_preds = torch.argmax(domain_logits, dim=1)
            task_correct += (task_preds == vuln_labels).sum().item()
            domain_correct += (domain_preds == domain_labels).sum().item()
            total_samples += len(vuln_labels)

        task_acc = task_correct / total_samples
        domain_acc = domain_correct / total_samples
        avg_loss = total_loss / len(dataloader)

        if domain_acc > best_domain_acc:
            torch.save(model.state_dict(), "dann_model/en_best_dann_model.pth")
            best_domain_acc = domain_acc

        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, f"dann_model/checkpoint_epoch_{epoch + 1}.pth")

    return model


def extract_features(model, dataset, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    source_features = []
    source_vuln_labels = []
    target_features = []
    target_vuln_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            domain_labels = batch['domain_labels'].cpu().numpy()

            features = model.encoder(input_ids, attention_mask=attention_mask).cpu()

            for i in range(len(features)):
                if domain_labels[i] == 0: 
                    source_features.append(features[i])
                    source_vuln_labels.append(batch['vuln_labels'][i].item())
                else: 
                    target_features.append(features[i])
                    target_vuln_labels.append(batch['vuln_labels'][i].item())

    source_features = torch.stack(source_features) if source_features else torch.tensor([])
    target_features = torch.stack(target_features) if target_features else torch.tensor([])
    source_vuln_labels = torch.tensor(source_vuln_labels) if source_vuln_labels else torch.tensor([])
    target_vuln_labels = torch.tensor(target_vuln_labels) if target_vuln_labels else torch.tensor([])

    return source_features, source_vuln_labels, target_features, target_vuln_labels


def main():
    source_tokens = "dann_data_aug/s2/dy_data_tokens.pkl"
    source_labels = "dann_data_aug/s2/dy_data_labels.pkl"
    target_tokens = "dann_data_aug/t2/dy_data_tokens.pkl"
    target_labels = "dann_data_aug/t2/dy_data_labels.pkl"

    codes, vuln_labels, domain_labels = load_data(
        source_tokens, source_labels, target_tokens, target_labels)

    tokenizer = RobertaTokenizer.from_pretrained('../Sequence_AST/unixcoder-base', use_fast=True)

    encoder = TransformerModel(
        input_dim=tokenizer.vocab_size,
        hidden_dim=768, 
        num_heads=8, 
        num_layers=6, 
    ).to(device)

    model = EnhancedDANN(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    processed_data = preprocess_data(codes, vuln_labels, domain_labels, tokenizer)
    dataset = CustomDataset(
        processed_data['input_ids'],
        processed_data['attention_mask'],
        processed_data['vuln_labels'],
        processed_data['domain_labels']
    )

    model = train_dann(model, dataset, optimizer, num_epochs=10, batch_size=16)

    source_features, source_vuln_labels, target_features, target_vuln_labels = extract_features(model, dataset)

    output_dir = "dann_features_aug_2"
    os.makedirs(output_dir, exist_ok=True)

    source_dir = os.path.join(output_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    with open(os.path.join(source_dir, "features.pkl"), "wb") as f_feat, \
            open(os.path.join(source_dir, "vuln_labels.pkl"), "wb") as f_vuln:
        pickle.dump(source_features, f_feat)
        pickle.dump(source_vuln_labels, f_vuln)

    target_dir = os.path.join(output_dir, "target")
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, "features.pkl"), "wb") as f_feat, \
            open(os.path.join(target_dir, "vuln_labels.pkl"), "wb") as f_vuln:
        pickle.dump(target_features, f_feat)
        pickle.dump(target_vuln_labels, f_vuln)



if __name__ == "__main__":
    main()

import os
import time

import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import RobertaTokenizer, RobertaModel

import numpy as np
import random
random.seed(8)
np.random.seed(8)
torch.manual_seed(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_raw_data(tokens_path, labels_path):
    with open(tokens_path, 'rb') as f_tokens, open(labels_path, 'rb') as f_labels:
        codes = pickle.load(f_tokens)
        labels = pickle.load(f_labels)
    return codes, labels

def preprocess_data(codes, labels, tokenizer, max_length=512):
    inputs = tokenizer(codes, truncation=True, padding="max_length",
                      max_length=max_length, return_tensors="pt")
    inputs['labels'] = torch.tensor(labels).long()
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

            outputs = model(input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            features.append(cls_embedding.cpu())
            labels.append(batch['labels'].cpu())

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def load_encoder_model(model_path):
    model = RobertaModel.from_pretrained('./unixcoder-base').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128], dropout_rate=0.3):
        super(MLPClassifier, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_classifier(model_path, input_dim, num_classes):
    model = MLPClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def calculate_metrics(conf_matrix):
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    TNR = TN / (FP + TN) if (FP + TN) != 0 else 0
    FNR = FN / (TP + FN) if (TP + FN) != 0 else 0

    return TPR, FPR, TNR, FNR

def evaluate_classifier(model, features, labels):
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        predictions = model(features).argmax(dim=1).cpu()

        accuracy = accuracy_score(labels.cpu(), predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predictions, average='binary')

        conf_matrix = torch.zeros(2, 2).to(device)
        for t, p in zip(labels.view(-1), predictions.view(-1)):
            conf_matrix[t.long(), p.long()] += 1

        TPR, FPR, TNR, FNR = calculate_metrics(conf_matrix)


    if not os.path.exists('res'):
        os.mkdir('res')

    with open('res/evaluation_results.txt', 'a') as fwrite:
        fwrite.write("Accuracy: {:.4f}\n".format(accuracy))
        fwrite.write("Precision: {:.4f}\n".format(precision))
        fwrite.write("Recall: {:.4f}\n".format(recall))
        fwrite.write("F1: {:.4f}\n".format(f1))
        fwrite.write("TPR: {:.4f}\n".format(TPR))
        fwrite.write("FPR: {:.4f}\n".format(FPR))
        fwrite.write("TNR: {:.4f}\n".format(TNR))
        fwrite.write("FNR: {:.4f}\n".format(FNR))
        fwrite.write("-=" * 50 + '\n')

def main():
    tokens_path = "java_ast_train/ast_data_tokens.pkl"
    labels_path = "java_ast_train/ast_data_labels.pkl"
    codes, labels = load_raw_data(tokens_path, labels_path)

    tokenizer = RobertaTokenizer.from_pretrained('./unixcoder-base', use_fast=True)
    processed_data = preprocess_data(codes, labels, tokenizer)
    dataset = CustomDataset(
        processed_data['input_ids'],
        processed_data['attention_mask'],
        processed_data['labels']
    )

    encoder = load_encoder_model("model/model_epoch_10.pth")
    features, true_labels = extract_features(dataset, encoder)

    classifier = load_classifier("model/best_model.model",
                                 input_dim=768, num_classes=2)
    evaluate_classifier(classifier, features, true_labels)

if __name__ == "__main__":
    main()

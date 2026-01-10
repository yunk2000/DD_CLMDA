import os
import time
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
import numpy as np
import random

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dann_features_with_target_labels(src_features_path, src_labels_path, tgt_features_path, tgt_labels_path):
    with open(src_features_path, "rb") as s_feat, \
            open(src_labels_path, "rb") as s_vuln, \
            open(tgt_features_path, "rb") as t_feat, \
            open(tgt_labels_path, "rb") as t_vuln:
        src_features = pickle.load(s_feat).numpy()
        src_labels = pickle.load(s_vuln).numpy()
        tgt_features = pickle.load(t_feat).numpy()
        tgt_labels = pickle.load(t_vuln).numpy()

    src_features = torch.tensor(src_features).float()
    src_labels = torch.tensor(src_labels).long()
    tgt_features = torch.tensor(tgt_features).float()
    tgt_labels = torch.tensor(tgt_labels).long()

    return src_features, src_labels, tgt_features, tgt_labels


def calculate_class_weights(labels):
    class_counts = np.bincount(labels.numpy())
    total = class_counts.sum()
    weights = [total / (len(class_counts) * count) for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32).to(device)


class SimplifiedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, features):
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-8 

    def transform(self, features):
        return (features - self.mean) / self.std


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss) 
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='mean')

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, class_weights=None, lambda_focal=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.weighted_ce_loss = WeightedCrossEntropyLoss(class_weights=class_weights)
        self.lambda_focal = lambda_focal 

    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        weighted_ce_loss = self.weighted_ce_loss(inputs, targets)
        total_loss = self.lambda_focal * focal_loss + (1 - self.lambda_focal) * weighted_ce_loss
        return total_loss


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


def save_best_model(model, f1, best_f1, model_path):
    if f1 > best_f1:
        torch.save(model.state_dict(), model_path) 
        return f1 
    return best_f1


def train_improved_classifier(src_features, src_labels,
                              num_epochs=50, batch_size=512, lr=1e-4):
    normalizer = FeatureNormalizer()
    normalizer.fit(src_features)
    src_features = normalizer.transform(src_features)

    src_features = src_features.to(device)
    src_labels = src_labels.to(device)

    model = MLPClassifier(
        input_dim=src_features.shape[1],
        num_classes=2
    ).to(device) 
    class_weights = torch.tensor([0.2, 0.8]).to(device)  
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    train_dataset = TensorDataset(src_features, src_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    best_f1 = 0.0
    best_model_path = "./model/e3_S_D_C_best_dann_model.model"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )

        best_f1 = save_best_model(model, f1, best_f1, model_path=best_model_path)

    return model, best_f1


def evaluate_classifier(model, features, labels):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1)[:, 1].cpu().numpy() 
        predictions = outputs.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        accuracy = accuracy_score(labels_np, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, predictions, average='binary', zero_division=0
        )

        auc = roc_auc_score(labels_np, probabilities)
        mcc = matthews_corrcoef(labels_np, predictions)
        conf_matrix = torch.zeros(2, 2)
        for t, p in zip(labels.view(-1), torch.tensor(predictions)):
            conf_matrix[t.long(), p.long()] += 1
        TPR, FPR, TNR, FNR = calculate_metrics(conf_matrix)
        true_positives = conf_matrix[1, 1].item()  # TP
        false_positives = conf_matrix[0, 1].item()  # FP
        true_negatives = conf_matrix[0, 0].item()  # TN
        false_negatives = conf_matrix[1, 0].item()  # FN

        total_samples = len(labels)
        positive_samples = int(labels.sum().item()) 
        negative_samples = total_samples - positive_samples 
        eval_time = time.time() - start_time

  






def calculate_mmd(src_feat, tgt_feat):
    XX = rbf_kernel(src_feat, src_feat, gamma=1.0)
    YY = rbf_kernel(tgt_feat, tgt_feat, gamma=1.0)
    XY = rbf_kernel(src_feat, tgt_feat, gamma=1.0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def main():
    src_features, src_labels, tgt_features, tgt_labels = load_dann_features_with_target_labels(
        src_features_path="./fuse_zyl/e3_S+D+C/source/features.pkl",
        src_labels_path="./fuse_zyl/e3_S+D+C/source/labels.pkl",
        tgt_features_path="./fuse_zyl/e3_S+D+C/target/features.pkl",
        tgt_labels_path="./fuse_zyl/e3_S+D+C/target/labels.pkl"
    )
    model, best_f1 = train_improved_classifier(
        src_features, src_labels,
        num_epochs=100,
        batch_size=32,
        lr=1e-4
    )

    c_best_model_path = "./model/e3_S_D_C_best_dann_model.model"
    model.load_state_dict(torch.load(c_best_model_path))

    evaluate_classifier(model, tgt_features, tgt_labels)

    

if __name__ == "__main__":
    main()

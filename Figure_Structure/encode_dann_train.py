import datetime
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, GATv2Conv
import numpy as np
import os
from torch.cuda import amp
import pickle
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAT2DCNN(nn.Module):
    def __init__(self, node_features=256, edge_features=2, hidden_dim=768, num_heads=8):
        super(GAT2DCNN, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.gat1 = GATv2Conv(node_features, hidden_dim, heads=num_heads, concat=True, edge_dim=edge_features)
        self.gat2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=False, edge_dim=edge_features)

        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten_dim = self._calculate_flatten_dim(hidden_dim)
        self.fc_h_seq = nn.Linear(self.flatten_dim, 768)
        self.fc = nn.Linear(hidden_dim * 2 + 768, hidden_dim)

    def _calculate_flatten_dim(self, input_dim):
        dim = input_dim
        dim = math.floor((dim - 3 + 2 * 1) / 1 + 1) 
        dim = math.floor(dim / 2)  
        dim = math.floor((dim - 3 + 2 * 1) / 1 + 1) 
        dim = math.floor(dim / 2)  
        return dim * 64  

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()

        h = F.relu(self.gat1(x, edge_index, edge_attr))
        h = F.relu(self.gat2(h, edge_index, edge_attr))

        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)

        h_seq = h.unsqueeze(1)
        h_seq = self.pool1(F.relu(self.conv1d_1(h_seq)))
        h_seq = self.pool2(F.relu(self.conv1d_2(h_seq)))
        h_seq = h_seq.view(h_seq.size(0), -1)
        h_seq = self.fc_h_seq(h_seq)

        h_seq_graph = global_mean_pool(h_seq, batch)

        graph_features = torch.cat([mean_pool, max_pool, h_seq_graph], dim=-1)
        return self.fc(graph_features)


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


class GATDANN(nn.Module):
    def __init__(self, gat_extractor, hidden_dim=768, actual_node_features=128):
        super().__init__()
        self.gat_extractor = gat_extractor

        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2) 
        )

        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(alpha=1.0),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  
        )

        self.input_adapter = nn.Linear(actual_node_features, 256)

    def forward(self, batch, alpha=1.0):
        self.domain_classifier[0].alpha = alpha

        adapted_x = self.input_adapter(batch.x.float())

        features = self.gat_extractor(
            adapted_x, 
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        task_output = self.task_classifier(features)
        domain_output = self.domain_classifier(features)

        return task_output, domain_output, features


def load_graph_data(data_path, labels_path, domain_label=0):
    with open(data_path, "rb") as f1, open(labels_path, "rb") as f2:
        graph_datas = pickle.load(f1)
        labels = pickle.load(f2)

    for data, label in zip(graph_datas, labels):
        data.y = torch.tensor([label])
        data.domain_label = torch.tensor([domain_label]) 

    return graph_datas


def create_combined_dataset(source_data, target_data):
    combined_data = source_data + target_data
    random.shuffle(combined_data)
    return combined_data


def train_dann(model, source_loader, target_loader, optimizer, num_epochs=100, save_dir='./model_gat_dann'):
    os.makedirs(save_dir, exist_ok=True)
    model.train()

    scaler = amp.GradScaler()  
    accumulation_steps = 2  

    for epoch in range(num_epochs):
        start_time = time.time()
        total_loss = 0
        task_loss_total = 0
        domain_loss_total = 0
        step_count = 0 

        p = epoch / num_epochs
        alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0

        target_iter = iter(target_loader)
        optimizer.zero_grad() 

        for source_batch in source_loader:
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            step_count += 1

            source_batch = source_batch.to(device)
            target_batch = target_batch.to(device)
            with amp.autocast():
                source_task_output, source_domain_output, _ = model(source_batch, alpha)
                target_task_output, target_domain_output, _ = model(target_batch, alpha)
                task_loss = F.cross_entropy(source_task_output, source_batch.y.squeeze().long())
                source_domain_labels = torch.zeros(len(source_batch)).long().to(device)
                target_domain_labels = torch.ones(len(target_batch)).long().to(device)
                domain_loss_source = F.cross_entropy(source_domain_output, source_domain_labels)
                domain_loss_target = F.cross_entropy(target_domain_output, target_domain_labels)
                domain_loss = (domain_loss_source + domain_loss_target) / 2
                total_step_loss = (task_loss + domain_loss) / accumulation_steps

            scaler.scale(total_step_loss).backward()
            total_loss += (task_loss + domain_loss).item()
            task_loss_total += task_loss.item()
            domain_loss_total += domain_loss.item()
            if step_count % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if step_count % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.empty_cache()
        avg_total_loss = total_loss / len(source_loader)
        avg_task_loss = task_loss_total / len(source_loader)
        avg_domain_loss = domain_loss_total / len(source_loader)

        epoch_time = time.time() - start_time
       
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/dann_model_epoch_{epoch + 1}.pth")

    return model


def extract_features(model, dataloader, device):
    model.eval()

    all_features = []
    all_vuln_labels = []
    all_domain_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            _, _, features = model(batch, alpha=1.0)

            all_features.append(features.cpu())
            all_vuln_labels.append(batch.y.cpu())
            all_domain_labels.append(batch.domain_label.cpu())

    features = torch.cat(all_features, dim=0)  
    vuln_labels = torch.cat(all_vuln_labels, dim=0).squeeze()  
    domain_labels = torch.cat(all_domain_labels, dim=0).squeeze()  

    return features, vuln_labels, domain_labels


def save_features_as_pkl(features, vuln_labels, domain_labels, file_path, e):
    if e:
        with open(os.path.join(file_path, "src_features.pkl"), "wb") as f_feat, \
                open(os.path.join(file_path, "src_vuln_labels.pkl"), "wb") as f_vuln, \
                open(os.path.join(file_path, "src_domain_labels.pkl"), "wb") as f_domain:
            pickle.dump(features, f_feat)
            pickle.dump(vuln_labels, f_vuln)
            pickle.dump(domain_labels, f_domain)
    else:
        with open(os.path.join(file_path, "tar_features.pkl"), "wb") as f_feat, \
                open(os.path.join(file_path, "tar_vuln_labels.pkl"), "wb") as f_vuln, \
                open(os.path.join(file_path, "tar_domain_labels.pkl"), "wb") as f_domain:
            pickle.dump(features, f_feat)
            pickle.dump(vuln_labels, f_vuln)
            pickle.dump(domain_labels, f_domain)


def extract_and_save_all_features(model, source_loader, target_loader, epoch, save_dir='./gat_dann_features'):
    source_features, source_vuln_labels, source_domain_labels = extract_features(
        model, source_loader, device
    )

    save_features_as_pkl(
        source_features,
        source_vuln_labels,
        source_domain_labels,
        save_dir,
        1
    )

    target_features, target_vuln_labels, target_domain_labels = extract_features(
        model, target_loader, device
    )

    save_features_as_pkl(
        target_features,
        target_vuln_labels,
        source_domain_labels,
        save_dir,
        0
    )

    return source_features, target_features, source_domain_labels, source_domain_labels


def main():
    source_data = load_graph_data(
        "cpg_graph/graph_datas.pkl",
        "cpg_graph/labels.pkl",
        domain_label=0 
    )

    target_data = load_graph_data(
        "java_cpg_graph/graph_datas.pkl",
        "java_cpg_graph/labels.pkl",
        domain_label=1 
    )

    source_sample = source_data[0]
    target_sample = target_data[0]

    min_node_features = min(source_sample.x.shape[1], target_sample.x.shape[1])
    min_edge_features = min(source_sample.edge_attr.shape[1], target_sample.edge_attr.shape[1])

    for data in source_data + target_data:
        data.x = data.x[:, :min_node_features]
        data.edge_attr = data.edge_attr[:, :min_edge_features]

    batch_size = 8
    source_loader = PyGDataLoader(source_data, batch_size=batch_size, shuffle=True)
    target_loader = PyGDataLoader(target_data, batch_size=batch_size, shuffle=True)

    actual_node_features = min_node_features  
    actual_edge_features = min_edge_features 

    gat_extractor = GAT2DCNN(
        node_features=256, 
        edge_features=actual_edge_features,
        hidden_dim=768,
        num_heads=8
    ).to(device)

    model = GATDANN(
        gat_extractor,
        hidden_dim=768,
        actual_node_features=actual_node_features 
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    trained_model = train_dann(
        model,
        source_loader,
        target_loader,
        optimizer,
        num_epochs=10,
        save_dir='./model_gat_dann'
    )

    source_feats, target_feats, source_domains, target_domains = extract_and_save_all_features(
        trained_model, source_loader, target_loader, "final", './gat_dann_features'
    )



if __name__ == "__main__":
    main()

import datetime
import time
import math
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import shutil
import random
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, GATConv, GATv2Conv
import torch.nn.functional as F


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(graph_data_path, labels_path, location_path):
    with open(graph_data_path, "rb") as f1:
        graph_datas = pickle.load(f1)
    with open(labels_path, "rb") as f2:
        labels = pickle.load(f2)
    with open(location_path, "rb") as f3:
        locations = pickle.load(f3)

    combined = list(zip(graph_datas, labels, locations))
    random.shuffle(combined)
    graph_datas, labels, location = zip(*combined)

    return graph_datas, labels, location


def create_data_loader(graph_datas, batch_size=16, shuffle=True):

    data_loader = DataLoader(graph_datas, batch_size=batch_size, shuffle=shuffle)
    return data_loader


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

        output = self.fc(graph_features)

        return output


def train_model(model, dataset, optimizer, num_epochs=10, batch_size=16, save_dir='./model_gat'):
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time() 
        total_loss = 0
        h_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time() 

            batch = batch.to(device)  
            optimizer.zero_grad()

            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch) 
            logits = outputs

            loss = torch.nn.CrossEntropyLoss()(logits, batch.y.long())
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            h_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time

                progress = (batch_idx + 1) / len(dataloader) * 100  
                v_loss = h_loss / 100

                h_loss = 0

        avg_loss = total_loss / len(dataloader)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        elapsed_time = epoch_end_time - start_time

        torch.save(model.state_dict(), f"{save_dir}/gat_model.pth")
        torch.save(optimizer.state_dict(), f"{save_dir}/gat_optimizer.pth")

    return model


def extract_features(dataset, model, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    features = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            feature = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            features.append(feature.cpu())
            labels.append(batch.y.cpu())

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def main():
    graph_data_path = "java_cpg_graph/graph_datas.pkl"
    labels_path = "java_cpg_graph/labels.pkl"
    location_path = "java_cpg_graph/locations.pkl"
    out_put_features = 'java_gat_features_train_cpg/'
    save_dir = './java_model_gat'

    graph_datas, labels, location = load_data(graph_data_path, labels_path, location_path)

    model = GAT2DCNN(
        node_features=graph_datas[0].x.shape[1], 
        edge_features=graph_datas[0].edge_attr.shape[1],  
        hidden_dim=256,
        num_heads=8
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model = train_model(model, graph_datas, optimizer, num_epochs=10, batch_size=16, save_dir=save_dir)

    features, labels = extract_features(graph_datas, model)

    i, j = 0, 0
    for label in labels:
        if int(label) == 0:
            i += 1
        elif int(label) == 1:
            j += 1

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

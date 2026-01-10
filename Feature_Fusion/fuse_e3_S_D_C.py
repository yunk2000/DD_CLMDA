import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)


def load_features_and_labels(path):
    with open(os.path.join(path, "features.pkl"), "rb") as f_feat, \
            open(os.path.join(path, "labels.pkl"), "rb") as f_label:
        features = pickle.load(f_feat)
        labels = pickle.load(f_label)
    return features, labels


def save_fused(features, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "features.pkl"), "wb") as f_feat, \
            open(os.path.join(out_dir, "labels.pkl"), "wb") as f_label:
        pickle.dump(features, f_feat)
        pickle.dump(labels, f_label)


class AttentionFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=128):
        super(AttentionFusion, self).__init__()
        self.projections = nn.ModuleList([
            nn.Linear(d, hidden_dim) for d in input_dims
        ])
        self.attention_vector = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, feat_seq, feat_dy, feat_cpg):
        feats = [proj(f) for proj, f in zip(self.projections, [feat_seq, feat_dy, feat_cpg])]
        feats = torch.stack(feats, dim=1) 

        scores = torch.matmul(feats, self.attention_vector) 
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   

        fused = torch.sum(feats * weights, dim=1)  
        return fused


def main():
    seq_path = "./e1_seq/source"
    dy_path = "./e1_dy/source"
    cpg_path = "./e1_cpg/source"
    java_seq_path = "./e1_seq/target"
    java_dy_path = "./e1_dy/target"
    java_cpg_path = "./e1_cpg/target"
    output_dir = "./fuse_zyl/e3_S+D+C"

    feat_seq, label_seq = load_features_and_labels(seq_path)
    feat_dy, label_dy = load_features_and_labels(dy_path)
    feat_cpg, label_cpg = load_features_and_labels(cpg_path)
    java_feat_seq, java_label_seq = load_features_and_labels(java_seq_path)
    java_feat_dy, java_label_dy = load_features_and_labels(java_dy_path)
    java_feat_cpg, java_label_cpg = load_features_and_labels(java_cpg_path)

    feat_seq, feat_dy, feat_cpg = map(torch.tensor, [feat_seq, feat_dy, feat_cpg])
    java_feat_seq, java_feat_dy, java_feat_cpg = map(torch.tensor, [java_feat_seq, java_feat_dy, java_feat_cpg])


    input_dims = [feat_seq.shape[1], feat_dy.shape[1], feat_cpg.shape[1]]
    fusion_model = AttentionFusion(input_dims=input_dims, hidden_dim=768)

    fused_features = fusion_model(feat_seq, feat_dy, feat_cpg)
    java_fused_features = fusion_model(java_feat_seq, java_feat_dy, java_feat_cpg)

    save_fused(fused_features.detach(), label_seq, out_dir=os.path.join(output_dir, "source/"))
    save_fused(java_fused_features.detach(), java_label_seq, out_dir=os.path.join(output_dir, "target/"))



if __name__ == "__main__":
    main()

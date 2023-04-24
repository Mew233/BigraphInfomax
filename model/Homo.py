import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import SAGEConv
import numpy as np
import torch.nn as nn
from model.GAT import GAT
import math
    

class Homo(torch.nn.Module):
    def __init__(self,opt):
        super(Homo, self).__init__()
        self.opt=opt
        self.conv1 = SAGEConv(opt['feature_dim'], 128)
        self.relu = nn.ReLU()

        self.mlp1 = torch.nn.Linear(128, 64)
        self.mlp2 = torch.nn.Linear(64, 32)
        self.mlp3 = torch.nn.Linear(32, 2)

    def get_link_labels(self, pos_edge_index, neg_edges):
        E = pos_edge_index.size(1) + neg_edges.size(1)

        link_labels = torch.zeros(E, dtype=torch.float)

        link_labels[:pos_edge_index.size(1)] = 1.

        return link_labels

    def forward(self,drug_hidden_out, protein_hidden_out, all_edges): 
    
        n_fea = torch.cat((drug_hidden_out, protein_hidden_out), dim = 0)
        edges = []
        for i in range(len(all_edges)):
            #对称
            if i % 2 == 0:
                edges.append(list(all_edges[i]))
        edges = np.array(edges)
        edges = torch.from_numpy(edges).t().type(torch.long)

        neg_edges = negative_sampling(
            edge_index=edges, num_nodes=drug_hidden_out.shape[0] + protein_hidden_out.shape[0],
            num_neg_samples=edges.size(1),  # 负采样数量根据正样本
            force_undirected=True,
        )
        # original paper implements a homogenous graph
        edge_index = torch.cat([edges,neg_edges], dim=-1)

        logits = (n_fea[edge_index[0]]*n_fea[edge_index[1]])
        logits = self.conv1(logits, edge_index)
        logits = self.relu(logits)
        logits = self.mlp1(logits)
        logits = self.relu(logits)
        logits = self.mlp2(logits)
        logits = self.relu(logits)
        logits = self.mlp3(logits)

        real_sub_Two = logits[:edges.size(1)]    
        fake_sub_Two = logits[edges.size(1):]
        
        real_sub_prob = real_sub_Two
        fake_sub_prob = fake_sub_Two

        prob = F.log_softmax(logits, dim=-1)
        label = torch.cat((torch.ones_like(real_sub_prob), torch.zeros_like(fake_sub_prob)))

        return prob, label


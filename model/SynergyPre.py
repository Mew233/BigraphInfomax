import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import SAGEConv
import numpy as np
import torch.nn as nn
from model.GAT import GAT
import math
    

class SynergyPre(torch.nn.Module):
    def __init__(self,opt):
        super(SynergyPre, self).__init__()
        self.opt=opt
        self.proj = torch.nn.Linear(1000, 128)
        self.icnn = nn.Conv2d(1, 3, 3, padding = 0)

        self.NN = nn.Sequential(
            nn.Linear(128, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=opt['dropout']),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )


    def predict(self,drug_hidden_out, cell_hidden_out, all_edges):
        cell_hidden_out = self.proj(cell_hidden_out)
        edges = []
        for i in range(len(all_edges)):
            edges.append(list(all_edges[i]))
        edges = np.array(edges)
        edges = torch.from_numpy(edges).t().type(torch.long)

        # original paper implements a homogenous graph
        label = edges[3]
        edge_index = edges[:3]

        # logits = (n_fea[edge_index[0]])
        drug1_fea = drug_hidden_out[edge_index[0]]
        drug2_fea = drug_hidden_out[edge_index[1]]
        cell_fea = drug_hidden_out[edge_index[2]]

        # logits = torch.cat([drug1_fea,drug2_fea, cell_fea], dim=-1)
        # x = self.NN(logits)

        logits = drug1_fea*drug2_fea*cell_fea
        x = self.NN(logits)

        # logits = torch.cat([drug1_fea*drug2_fea, cell_fea], dim=-1)
        # x = self.NN(logits)

        # logits = drug1_fea*drug2_fea*cell_fea
        # i_v = logits.view(128, -1, 128,128) 
        # # batch_size x embed size x max_drug_seq_len x max_protein_seq_len
        # i_v = torch.sum(i_v, dim = 1)
        # #print(i_v.shape)
        # i_v = torch.unsqueeze(i_v, 1)
        # #print(i_v.shape)
        
        # i_v = F.dropout(i_v, p = self.dropout_rate)   
        # score = self.decoder(f)

        return x.detach().numpy(), label
    


    def forward(self,drug_hidden_out, cell_hidden_out, all_edges): 
        
        cell_hidden_out = self.proj(cell_hidden_out)
        edges = []
        for i in range(len(all_edges)):
            edges.append(list(all_edges[i]))
        edges = np.array(edges)
        edges = torch.from_numpy(edges).t().type(torch.long)

        # original paper implements a homogenous graph
        label = edges[3]
        edge_index = edges[:3]

        # logits = (n_fea[edge_index[0]])
        drug1_fea = drug_hidden_out[edge_index[0]]
        drug2_fea = drug_hidden_out[edge_index[1]]
        cell_fea = drug_hidden_out[edge_index[2]]

        # logits = torch.cat([drug1_fea,drug2_fea, cell_fea], dim=-1)
        logits = drug1_fea*drug2_fea*cell_fea
        # logits = torch.cat([drug1_fea*drug2_fea, cell_fea], dim=-1)
        x = self.NN(logits)

        return x, label


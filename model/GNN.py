import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GCN import GCN
from torch.autograd import Variable

class GNN(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, opt):
        super(GNN, self).__init__()
        self.opt=opt
        self.layer_number = opt["GNN"]
        self.encoder = []
        for i in range(self.layer_number):
            self.encoder.append(DGCNLayer(opt))
        self.encoder = nn.ModuleList(self.encoder)
        self.dropout = opt["dropout"]
        self.score_function1 = nn.Linear(opt["hidden_dim"]+opt["hidden_dim"],10)
        self.score_function2 = nn.Linear(10,1)

    def forward(self, ufea, vfea, UV_adj, VU_adj,adj):
        learn_drug = ufea
        learn_protein = vfea
        for layer in self.encoder:
            learn_drug = F.dropout(learn_drug, self.dropout, training=self.training)
            learn_protein = F.dropout(learn_protein, self.dropout, training=self.training)
            learn_drug, learn_protein = layer(learn_drug, learn_protein, UV_adj, VU_adj)
        return learn_drug, learn_protein

class DGCNLayer(nn.Module):
    """
        DGCN Module layer
    """
    def __init__(self, opt):
        super(DGCNLayer, self).__init__()
        self.opt=opt
        self.dropout = opt["dropout"]
        self.gc1 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc2 = GCN(
            nfeat=opt["feature_dim"],
            nhid=opt["hidden_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.gc3 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )

        self.gc4 = GCN(
            nfeat=opt["hidden_dim"], # change
            nhid=opt["feature_dim"],
            dropout=opt["dropout"],
            alpha=opt["leakey"]
        )
        self.drug_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])
        self.protein_union = nn.Linear(opt["feature_dim"] + opt["feature_dim"], opt["feature_dim"])

    def forward(self, ufea, vfea, UV_adj,VU_adj):
        drug_ho = self.gc1(ufea, VU_adj)
        protein_ho = self.gc2(vfea, UV_adj)
        drug_ho = self.gc3(drug_ho, UV_adj)
        protein_ho = self.gc4(protein_ho, VU_adj)
        drug = torch.cat((drug_ho, ufea), dim=1)
        protein = torch.cat((protein_ho, vfea), dim=1)
        drug = self.drug_union(drug)
        protein = self.protein_union(protein)
        return F.relu(drug), F.relu(protein)
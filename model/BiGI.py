import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN import GNN
from model.AttDGI import AttDGI
from model.myDGI import myDGI
from model.Homo import Homo

class BiGI(nn.Module):
    def __init__(self, opt):
        super(BiGI, self).__init__()
        self.opt=opt
        self.GNN = GNN(opt) # fast mode(GNN), slow mode(GNN2)
        if self.opt["number_drug"] * self.opt["number_protein"] > 10000000:
            self.DGI = AttDGI(opt) # Since pytorch is not support sparse matrix well
        else :
            self.DGI = myDGI(opt) # Since pytorch is not support sparse matrix well
        self.dropout = opt["dropout"]
        self.Homo = Homo(opt)

        self.drug_embedding = nn.Embedding(opt["number_drug"], opt["feature_dim"])
        self.protein_embedding = nn.Embedding(opt["number_protein"], opt["feature_dim"])
        self.protein_index = torch.arange(0, self.opt["number_protein"], 1)
        self.drug_index = torch.arange(0, self.opt["number_drug"], 1)
        if self.opt["cuda"]:
            self.protein_index = self.protein_index.cuda()
            self.drug_index = self.drug_index.cuda()

    def score_predict(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(out.size()[0], -1)

    def score(self, fea):
        out = self.GNN.score_function1(fea)
        out = F.relu(out)
        out = self.GNN.score_function2(out)
        out = torch.sigmoid(out)
        return out.view(-1)

    def forward(self, ufea, vfea, UV_adj, VU_adj, adj):
        learn_drug,learn_protein = self.GNN(ufea,vfea,UV_adj,VU_adj,adj)
        return learn_drug,learn_protein

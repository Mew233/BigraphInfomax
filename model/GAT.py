import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GAT(nn.Module):
    def __init__(self, opt):
        super(GAT, self).__init__()
        self.att = Attention(opt)
        self.dropout = opt["dropout"]
        self.leakyrelu = nn.LeakyReLU(opt["leakey"])

    def forward(self, ufea, vfea, UV_adj, VU_adj, adj=None):
        learn_drug = ufea
        learn_protein = vfea

        learn_drug = F.dropout(learn_drug, self.dropout, training=self.training)
        learn_protein = F.dropout(learn_protein, self.dropout, training=self.training)
        learn_drug, learn_protein = self.att(learn_drug, learn_protein, UV_adj, VU_adj)

        return learn_drug, learn_protein

class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention, self).__init__()
        self.lin_u = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.lin_v = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])
        self.opt = opt

    def forward(self, drug, protein, UV_adj, VU_adj):
        drug = self.lin_u(drug)
        protein = self.lin_v(protein)

        query = drug
        key = protein
        # import pdb
        # pdb.set_trace()
        value = torch.mm(query, key.transpose(0,1)) # drug * protein
        value = UV_adj.to_dense()*value  # drug * protein fuck pytorch!!!
        value /= math.sqrt(self.opt["hidden_dim"]) # drug * protein
        value = F.softmax(value,dim=1) # drug * protein
        learn_drug = torch.matmul(value,key) + drug

        query = protein
        key = drug
        value = torch.mm(query, key.transpose(0,1))  # protein * drug
        value = VU_adj.to_dense()*value  # protein * drug
        value /= math.sqrt(self.opt["hidden_dim"])  # protein * drug
        value = F.softmax(value, dim=1)  # protein * drug
        learn_protein = torch.matmul(value, key) + protein

        return learn_drug, learn_protein





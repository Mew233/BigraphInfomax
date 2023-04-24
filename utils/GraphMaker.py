import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def Bernoulli(rate):
    num = random.randint(0,1000000)
    p = num/1000000
    if p < rate:
        return 1
    else :
        return 0

def struct_corruption(opt, original_dict, rate):
    adj_dict = copy.deepcopy(original_dict)
    UV_edges = []
    VU_edges = []
    all_edges = []
    drug_fake_dict = {}
    protein_fake_dict = {}
    corruption_edges = int(opt["number_drug"] * opt["number_protein"] * rate)+1
    print("corruption_edges: ", corruption_edges)

    for k in range(corruption_edges):
        i = random.randint(0, opt["number_drug"]-1)
        j = random.randint(0, opt["number_protein"]-1)

        if adj_dict[i].get(j, "zxczxc") is "zxczxc":  # if 1 : adj is 0
            adj_dict[i][j] = 1
        else :
            del adj_dict[i][j]


    for i in adj_dict.keys():
        drug_fake_dict[i] = set()
        for j in adj_dict[i].keys():
            UV_edges.append([i,j])
            all_edges.append([i,j + opt["number_drug"]])
            all_edges.append([j + opt["number_drug"] , i])
            drug_fake_dict[i].add(j)
            VU_edges.append([j, i])
            if j not in protein_fake_dict.keys():
                protein_fake_dict[j] = set()
            protein_fake_dict[j].add(i)

    UV_edges = np.array(UV_edges)
    VU_edges = np.array(VU_edges)
    all_edges = np.array(all_edges)
    UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                           shape=(opt["number_drug"], opt["number_protein"]),
                           dtype=np.float32)
    VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                           shape=(opt["number_protein"], opt["number_drug"]),
                           dtype=np.float32)

    all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),
                            shape=(opt["number_protein"] + opt["number_drug"], opt["number_protein"] + opt["number_drug"]),
                            dtype=np.float32)

    UV_adj = normalize(UV_adj)
    VU_adj = normalize(VU_adj)
    all_adj = normalize(all_adj)
    UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
    VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
    all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)
    return UV_adj,VU_adj,all_adj,drug_fake_dict,protein_fake_dict

class GraphMaker(object):
    def __init__(self, opt, filename):
        self.opt = opt
        self.drug = set()
        self.protein = set()
        data=[]
        with codecs.open(opt["data_dir"] + "train.txt") as infile:
            for line in infile:
                line = line.strip().split("\t")
                data.append((int(line[0]),int(line[1]),float(line[2])))
                self.drug.add(int(line[0]))
                self.protein.add(int(line[1]))

        opt["number_drug"] = len(self.drug)
        opt["number_protein"] = len(self.protein)
        self.raw_data = data
        self.UV,self.VU, self.adj, self.all_edges, self.corruption_UV,self.corruption_VU, self.fake_adj = self.preprocess(data,opt)

    def preprocess(self,data,opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        real_adj = {}

        drug_real_dict = {}
        protein_real_dict = {}
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            if edge[0] not in drug_real_dict.keys():
                drug_real_dict[edge[0]] = set()
            drug_real_dict[edge[0]].add(edge[1])

            VU_edges.append([edge[1], edge[0]])
            if edge[1] not in protein_real_dict.keys():
                protein_real_dict[edge[1]] = set()
            protein_real_dict[edge[1]].add(edge[0])

            all_edges.append([edge[0],edge[1] + opt["number_drug"]])
            all_edges.append([edge[1] + opt["number_drug"], edge[0]])
            if edge[0] not in real_adj :
                real_adj[edge[0]] = {}
            real_adj[edge[0]][edge[1]] = 1

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(opt["number_drug"], opt["number_protein"]),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(opt["number_protein"], opt["number_drug"]),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(opt["number_protein"]+opt["number_drug"], opt["number_protein"]+opt["number_drug"]),dtype=np.float32)
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

        print("real graph loaded!")
        corruption_UV_adj, corruption_VU_adj, fake_adj, drug_fake_dict,protein_fake_dict = struct_corruption(opt,real_adj,opt["struct_rate"])

        self.drug_real_dict = drug_real_dict
        self.drug_fake_dict = drug_fake_dict

        self.protein_real_dict = protein_real_dict
        self.protein_fake_dict = protein_fake_dict
        print("fake graph loaded!")
        return UV_adj,VU_adj, all_adj, all_edges, corruption_UV_adj, corruption_VU_adj,fake_adj


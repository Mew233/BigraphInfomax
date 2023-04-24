import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.BiGI import BiGI
from model.SynergyPre import SynergyPre

class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class DGITrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = BiGI(opt)
        self.criterion = nn.BCELoss()
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])
        self.epoch_rec_loss = []
        self.epoch_dgi_loss = []

        self.SynergyPre = SynergyPre(opt)

    def unpack_batch_predict(self, batch, cuda):
        batch = batch[0]
        if cuda:
            drug_index = batch.cuda()
        else:
            drug_index = batch
        return drug_index

    def unpack_batch(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch]
            drug_index = inputs[0]
            protein_index = inputs[1]
            negative_protein_index = inputs[2]
        else:
            inputs = [Variable(b) for b in batch]
            drug_index = inputs[0]
            protein_index = inputs[1]
            negative_protein_index = inputs[2]
        return drug_index, protein_index, negative_protein_index

    def unpack_batch_DGI(self, batch, cuda):

        drug_index = batch[0]
        protein_index = batch[1]
        negative_protein_index = batch[2]
        drug_index_One = batch[3]
        protein_index_One = batch[4]
        real_drug_index_id_Two = batch[5]
        fake_drug_index_id_Two = batch[6]
        real_protein_index_id_Two = batch[7]
        fake_protein_index_id_Two = batch[8]
        return drug_index, protein_index, negative_protein_index, drug_index_One, protein_index_One, real_drug_index_id_Two, fake_drug_index_id_Two, real_protein_index_id_Two, fake_protein_index_id_Two

    def predict(self, batch):
        drug_One = self.unpack_batch_predict(batch, self.opt["cuda"])  # 1

        protein_feature = torch.index_select(self.protein_hidden_out, 0, self.model.protein_index) # protein_num * hidden_dim
        drug_feature = torch.index_select(self.drug_hidden_out, 0, drug_One) # drug_num * hidden_dim
        drug_feature = drug_feature.unsqueeze(1)
        drug_feature = drug_feature.repeat(1, self.opt["number_protein"], 1)
        protein_feature = protein_feature.unsqueeze(0)
        protein_feature = protein_feature.repeat(drug_feature.size()[0], 1, 1)
        Feature = torch.cat((drug_feature, protein_feature),
                            dim=-1)
        output = self.model.score_predict(Feature)
        output_list, recommendation_list = output.sort(descending=True)
        return recommendation_list.cpu().numpy()

    def feature_corruption(self):
        drug_index = torch.randperm(self.opt["number_drug"], device=self.model.drug_index.device)
        protein_index = torch.randperm(self.opt["number_protein"], device=self.model.drug_index.device)
        drug_feature = self.model.drug_embedding(drug_index)
        protein_feature = self.model.protein_embedding(protein_index)
        return drug_feature, protein_feature

    def update_bipartite(self, static_drug_feature, static_protein_feature, UV_adj, VU_adj, adj,fake = 0):
        # We do not use any side information. if have side information, modify following codes.
        if fake:
            drug_feature, protein_feature = self.feature_corruption()
            drug_feature = drug_feature.detach()
            protein_feature = protein_feature.detach()
        else :
            drug_feature = self.model.drug_embedding(self.model.drug_index)
            protein_feature = self.model.protein_embedding(self.model.protein_index)

        self.drug_hidden_out, self.protein_hidden_out = self.model(drug_feature, protein_feature, UV_adj, VU_adj, adj)

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        # import pdb
        # pdb.set_trace()
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def reconstruct(self, UV, VU, adj,  all_edges, CUV, CVU, fake_adj, drug_feature, protein_feature, batch):
        self.model.train()
        self.optimizer.zero_grad()

        self.update_bipartite(drug_feature, protein_feature, CUV, CVU, fake_adj, fake = 1)
        fake_drug_hidden_out = self.drug_hidden_out
        fake_protein_hidden_out = self.protein_hidden_out

        self.update_bipartite(drug_feature,protein_feature, UV, VU, adj)
        drug_hidden_out = self.drug_hidden_out
        protein_hidden_out = self.protein_hidden_out


        drug_One, protein_One, neg_protein_One = self.unpack_batch(batch, self.opt[
                "cuda"])

        drug_feature_Two = self.my_index_select(drug_hidden_out, drug_One)
        protein_feature_Two = self.my_index_select(protein_hidden_out, protein_One)
        neg_protein_feature_Two = self.my_index_select(protein_hidden_out, neg_protein_One)

        pos_One = self.model.score(torch.cat((drug_feature_Two, protein_feature_Two), dim=1))
        neg_One = self.model.score(torch.cat((drug_feature_Two, neg_protein_feature_Two), dim=1))


        reconstruct_loss = self.HingeLoss(pos_One, neg_One)


        Prob, Label = self.model.DGI(self.drug_hidden_out, self.protein_hidden_out, fake_drug_hidden_out,
                                        fake_protein_hidden_out, UV, VU, CUV, CVU, drug_One, protein_One)
        dgi_loss = self.criterion(Prob, Label)


        hprob, hlabel =  self.model.Homo(self.drug_hidden_out, self.protein_hidden_out, all_edges)
        homo_loss = F.cross_entropy(hprob, hlabel)

        #Loss function for the link prediciton task
        loss = (1 - 2*self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * dgi_loss + self.opt["lambda"] * homo_loss
        # loss = (1 - 2*self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * homo_loss
        self.epoch_rec_loss.append((1 - 2*self.opt["lambda"]) * reconstruct_loss.item())
        self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())

        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def reconstruct_copy(self, UV, VU, adj,  all_edges, all_syns, CUV, CVU, fake_adj, drug_feature, protein_feature, cell_feature, batch):
        self.model.train()
        self.optimizer.zero_grad()

        # self.update_bipartite(drug_feature, protein_feature, CUV, CVU, fake_adj, fake = 1)
        # fake_drug_hidden_out = self.drug_hidden_out
        # fake_protein_hidden_out = self.protein_hidden_out

        self.update_bipartite(drug_feature,protein_feature, UV, VU, adj)
        # drug_hidden_out = self.drug_hidden_out
        # protein_hidden_out = self.protein_hidden_out


        # drug_One, protein_One, neg_protein_One = self.unpack_batch(batch, self.opt[
        #         "cuda"])

        # drug_feature_Two = self.my_index_select(drug_hidden_out, drug_One)
        # protein_feature_Two = self.my_index_select(protein_hidden_out, protein_One)
        # neg_protein_feature_Two = self.my_index_select(protein_hidden_out, neg_protein_One)

        # pos_One = self.model.score(torch.cat((drug_feature_Two, protein_feature_Two), dim=1))
        # neg_One = self.model.score(torch.cat((drug_feature_Two, neg_protein_feature_Two), dim=1))


        # reconstruct_loss = self.HingeLoss(pos_One, neg_One)

        # Prob, Label = self.model.DGI(self.drug_hidden_out, self.protein_hidden_out, fake_drug_hidden_out,
        #                                 fake_protein_hidden_out, UV, VU, CUV, CVU, drug_One, protein_One)
        # dgi_loss = self.criterion(Prob, Label)


        # Downstream Task
        hprob, hlabel =  self.SynergyPre(self.drug_hidden_out, cell_feature, all_syns)
        homo_loss = self.criterion(hprob.view(-1), hlabel.float())

        #Loss function for the link prediciton task
        # loss = (1 - 2*self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * dgi_loss + self.opt["lambda"] * homo_loss'
        loss = homo_loss
        # loss = (1 - 2*self.opt["lambda"]) * reconstruct_loss + self.opt["lambda"] * homo_loss
        # self.epoch_rec_loss.append((1 - 2*self.opt["lambda"]) * reconstruct_loss.item())
        # self.epoch_dgi_loss.append(self.opt["lambda"] * dgi_loss.item())

        loss.backward()
        self.optimizer.step()
        return loss.item()
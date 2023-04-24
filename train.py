import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.trainer import DGITrainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker
from utils import torch_utils, helper
from utils.scorer import *
import json
import codecs
import joblib
import pandas as pd
import re
from rdkit.Chem import AllChem
import rdkit
# torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_dir', type=str, default='dataset/drug_protein/')
# parser.add_argument('--weight', action='store_true', default=False, help='Using weight graph?')

# model part
# parser.add_argument('--sparse', action='store_true', default=False, help='GNN with sparse version or not.')
parser.add_argument('--GNN', type=int, default=2, help="The layer of encoder.")
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001*10, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--lambda', type=float, default=0.1)
parser.add_argument('--DGI', type=bool, default=False)
parser.add_argument('--attention', action='store_true', default=True, help='Using attention in sub-graph?')
parser.add_argument('--struct', action='store_true', default=True, help='Using struct corruption in graph?')
parser.add_argument('--struct_rate', type=float, default=0.0001)
parser.add_argument('--early_stop', type=int, default=20)
# train part
parser.add_argument('--num_epoch', type=int, default=40, help='Number of total training epochs.')
parser.add_argument('--min_neighbor', type=int, default=100, help='Number of max neighbor per node')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--topn', type=int, default=10, help='Recommendation top-n protein for drug in test session')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=True,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str,default='/best_model.pt',help='Filename of the pretrained model.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--negative', type=int, default=1, help='negative sampling rate')

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
opt = vars(args)
seed_everything(opt["seed"])

G = GraphMaker(opt,"train.txt")
UV = G.UV
VU = G.VU
adj = G.adj
all_edges = G.all_edges

fake_adj = G.fake_adj
corruption_UV = G.corruption_UV
corruption_VU = G.corruption_VU
drug_real_dict = G.drug_real_dict
drug_fake_dict = G.drug_fake_dict
protein_real_dict = G.protein_real_dict
protein_fake_dict = G.protein_fake_dict

print("graph loaded!")

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)


# random feature; Now numpy
# split DB -- unique identifier
def split_it(compound):
    return int(re.split('\d*\D+',compound)[1])

def process_fingerprint():
    # load fingerprint data
    ## column is drug, index is morgan bits
    # Read SDF File
    supplier = rdkit.Chem.SDMolSupplier(os.path.join(opt['data_dir'], 'drug_protein', 'structures.sdf'))
    molecules = [mol for mol in supplier if mol is not None]

    fingerprints = dict()
    for mol in molecules:
        drugbank_id = mol.GetProp('DATABASE_ID')
        ## or use MACCS: (MACCSkeys.GenMACCSKeys(molecules[0])
        ## Here is morgan
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=128).ToBitString()
        fingerprint = [int(i) for i in bitvect]
        fingerprints[split_it(drugbank_id)] = fingerprint

    fingerprints = pd.DataFrame(fingerprints)
    fingerprints.columns = fingerprints.columns.astype(int)
    return fingerprints

fingerprints = process_fingerprint()
fingerprints_select = fingerprints.loc[selected_drugs:,]
drug_feature = fingerprints_select.to_numpy()
# drug_feature = np.random.randn(opt["number_drug"], opt["feature_dim"])
protein_feature = np.random.uniform(-1, 1, (opt["number_protein"], opt["feature_dim"]))
drug_feature = torch.FloatTensor(drug_feature)
protein_feature = torch.FloatTensor(protein_feature)


if opt["cuda"]:
    drug_feature = drug_feature.cuda()
    protein_feature = protein_feature.cuda()
    UV = UV.cuda()
    VU = VU.cuda()
    adj = adj.cuda()
    fake_adj = fake_adj.cuda()
    corruption_UV = corruption_UV.cuda()
    corruption_VU = corruption_VU.cuda()

print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + 'train.txt', opt['batch_size'], opt,
                         drug_real_dict,drug_fake_dict,protein_real_dict,protein_fake_dict, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + 'test.txt', 1000000, opt, drug_real_dict,drug_fake_dict,protein_real_dict,protein_fake_dict, evaluation=True)


# model
if not opt['load']:
    trainer = DGITrainer(opt)
else:
    # load pretrained model
    model_file = model_save_dir + opt['model_file']
    # model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = DGITrainer(opt)
    trainer.load(model_file)

def concat_feature(feature, id, cuda):
    x = id[:, 0]
    y = id[:, 1]
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    if cuda:
        x = x.cuda()
        y = y.cuda()
    x = torch.index_select(feature, 0, x)  # batch * hidden_dim
    y =torch.index_select(feature, 0, y)  # batch * hidden_dim
    ans = torch.cat((x, y), dim=1)
    return x, y, ans

link_dataset = []
link_label = []
for i, batch in enumerate(train_batch):
    drug_index = batch[0].cpu().detach().numpy()
    protein_index = batch[1].cpu().detach().numpy()
    negative_protein_index = batch[2].cpu().detach().numpy()
    for j in range(len(drug_index)):
        link_dataset.append([drug_index[j],protein_index[j] + opt["number_drug"]])
        link_dataset.append([drug_index[j], negative_protein_index[j] + opt["number_drug"]])
        link_label.append(1)
        link_label.append(0)

link_dataset_test = []
link_label_test = []
for i, batch in enumerate(dev_batch):
    drug_index = batch[0].cpu().detach().numpy()
    protein_index = batch[1].cpu().detach().numpy()
    link_label_test = batch[2]
    for j in range(len(drug_index)):
        link_dataset_test.append([drug_index[j],protein_index[j]])
link_dataset = np.array(link_dataset)
link_dataset_test = np.array(link_dataset_test)

print("train set:",len(link_dataset))
print("test set:",len(link_dataset_test))

def training(link_dataset,link_dataset_test):
    dev_score_history = []
    current_lr = opt['lr']
    global_step = 0
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_batch) * opt['num_epoch']


    # start training


    for epoch in range(1, opt['num_epoch'] + 1):
        train_loss = 0
        start_time = time.time()
        for i, batch in enumerate(train_batch):
            global_step += 1
            loss = trainer.reconstruct(UV, VU, adj, all_edges, corruption_UV, corruption_VU, fake_adj, drug_feature, protein_feature, batch)  # [ [drug_list], [protein_list], [neg_protein_list] ]
            train_loss += loss
        duration = time.time() - start_time
        print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                opt['num_epoch'], train_loss / len(train_batch), duration, current_lr))
        print("batch_rec_loss: ", sum(trainer.epoch_rec_loss) / len(trainer.epoch_rec_loss))
        print("batch_dgi_loss: ", sum(trainer.epoch_dgi_loss) / len(trainer.epoch_dgi_loss))
        trainer.epoch_rec_loss = []
        trainer.epoch_dgi_loss = []
        # eval model
        print("Evaluating on dev set...")

        trainer.model.eval()
        trainer.update_bipartite(drug_feature, protein_feature, UV, VU, adj)

        drug_hidden_out = trainer.drug_hidden_out
        protein_hidden_out = trainer.protein_hidden_out
        bi_feature = torch.cat((drug_hidden_out,protein_hidden_out),dim=0)


        train_x, train_y, train_feature = concat_feature(bi_feature, link_dataset, opt["cuda"])
        test_x, test_y, test_feature = concat_feature(bi_feature, link_dataset_test, opt["cuda"])
        #auc_roc, auc_pr, recall, pr, f1,  predict_label, y_true,pred
        auc_roc, auc_pr, recall, pr, f1, predict_label = link_prediction_logistic(train_feature.cpu().detach().numpy(), link_label,
                                                test_feature.cpu().detach().numpy(), link_label_test)
        train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
        print("epoch {}: train_loss = {:.6f}, auc_roc = {:.6f}, auc_pr = {:.6f}, recall = {:.6f}, pr = {:.6f}, f1 = {:.6f}".format(epoch, \
                                                                                        train_loss, auc_roc, auc_pr, recall, pr, f1))

        dev_score = auc_pr
        file_logger.log(
            "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max([dev_score] + dev_score_history)))

        # save
        model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        trainer.save(model_file, epoch)
        if epoch == 1 or dev_score > max(dev_score_history):
            copyfile(model_file, model_save_dir + '/best_model.pt')
            print("new best model saved.")

            metric_name = opt["id"] + ".json"
            json.dump(predict_label, codecs.open(metric_name, "w", encoding="utf-8"))

            file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}" \
                            .format(epoch, auc_roc * 100, auc_pr * 100))

        if epoch % opt['save_epoch'] != 0:
            os.remove(model_file)

        # lr schedule
        if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
                opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
            current_lr *= opt['lr_decay']
            trainer.update_lr(current_lr)

        dev_score_history += [dev_score]
        print("")


def fine_tuning():
# Fine tune for the downstream task
    """ 
    Fine tune basically requires:

    current best checkpoint
    a new optimizer
    a new scheduler
    """

    dev_score_history = []
    current_lr = opt['lr']
    global_step = 0
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_batch) * opt['num_epoch']



# load synergy dataset
    syn = pd.read_csv(os.path.join(opt['data_dir'], 'synergy','processed_synergydf_DrugComb_forbipartitegraph.csv'))
    syn['score'] = (syn['score']>0).astype(int).values
    syn = syn[["drug1","drug2","cell","score"]]
    
    #Drug's name map 
    drug_id = joblib.load(os.path.join(opt['data_dir'], 'drug_id.pkl'))
    # split DB -- unique identifier
    def split_it(compound):
        return int(re.split('\d*\D+',compound)[1]) 
    new_drug_id_dict = dict()
    for item,key in enumerate(drug_id):
        new_drug_id_dict[split_it(key)] = item
    syn['drug1'] = syn.drug1.map(new_drug_id_dict)
    syn['drug2'] = syn.drug2.map(new_drug_id_dict)

    # split gene
    def split_it_gene(compound):
        return int(re.search(r'\((.*?)\)', compound).group(1))

# load pre-trained model weights
    temp = pd.read_csv(os.path.join(opt['data_dir'], 'synergy','CCLE_exp.csv'), index_col=0)
    temp.columns = ['Entrez gene id']+[split_it_gene(_) for _ in list(temp.columns)[1:]]
    df_transpose = temp.T
    df_transpose.columns = df_transpose.iloc[0]
    processed_data = df_transpose.drop(df_transpose.index[0])
    var_df = processed_data.var(axis=1)
    selected_genes = list(var_df.sort_values(ascending=False).iloc[:1000].index)

    selected_cells = np.unique(syn.cell)
    # 106 cells versus 1000 genes
    temp = temp.loc[selected_cells,selected_genes]

    mapping = dict(zip(selected_cells,np.arange(len(selected_cells))))
    syn['cell'] = syn.cell.map(mapping)

    cell_feature = torch.FloatTensor(temp.to_numpy())

    all_syns = syn.to_numpy()

    random.shuffle(all_syns)
    train_len = int(len(all_syns)*(float(9)/10))+1

    train_set = all_syns[:train_len]
    test_set = all_syns[train_len:]


#Start training
    for epoch in range(1, opt['num_epoch'] + 1):
        train_loss = 0
        start_time = time.time()
        for i, batch in enumerate(train_batch):
            global_step += 1
            loss = trainer.reconstruct_copy(UV, VU, adj, all_edges, train_set, corruption_UV, corruption_VU, fake_adj, drug_feature, protein_feature, cell_feature, batch)  # [ [drug_list], [protein_list], [neg_protein_list] ]
            train_loss += loss
        duration = time.time() - start_time
        print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                opt['num_epoch'], train_loss / len(train_batch), duration, current_lr))
        # print("batch_rec_loss: ", sum(trainer.epoch_rec_loss) / len(trainer.epoch_rec_loss))
        # print("batch_dgi_loss: ", sum(trainer.epoch_dgi_loss) / len(trainer.epoch_dgi_loss))
        # trainer.epoch_rec_loss = []
        # trainer.epoch_dgi_loss = []
        # eval model
        print("Evaluating on dev set...")

        trainer.model.eval()
        trainer.update_bipartite(drug_feature, protein_feature, UV, VU, adj)

        drug_hidden_out = trainer.drug_hidden_out
        protein_hidden_out = trainer.protein_hidden_out


        prob, y_test = trainer.SynergyPre.predict(drug_hidden_out, cell_feature, test_set)
        lg_y_pred_est = prob
        pred = np.round(prob)
        fpr,tpr,thresholds = metrics.roc_curve(y_test,lg_y_pred_est)
        average_precision = average_precision_score(y_test, lg_y_pred_est)
        recall = metrics.recall_score(y_test,pred)
        f1 = metrics.f1_score(y_test,pred)
        pr = precision_score(y_test,pred)
        auc_roc = metrics.auc(fpr,tpr)

        print("epoch {}: train_loss = {:.6f}, auc_roc = {:.6f}, auc_pr = {:.6f}, recall = {:.6f}, pr = {:.6f}, f1 = {:.6f}".format(epoch, \
                                                                                        train_loss, auc_roc, average_precision, recall, pr, f1))

        #auc_roc, auc_pr, recall, pr, f1, 

        # bi_feature = torch.cat((drug_hidden_out,protein_hidden_out),dim=0)


        # train_x, train_y, train_feature = concat_feature(bi_feature, link_dataset, opt["cuda"])
        # test_x, test_y, test_feature = concat_feature(bi_feature, link_dataset_test, opt["cuda"])
        # #auc_roc, auc_pr, recall, pr, f1,  predict_label, y_true,pred
        # auc_roc, auc_pr, recall, pr, f1, predict_label = link_prediction_logistic(train_feature.cpu().detach().numpy(), link_label,
        #                                         test_feature.cpu().detach().numpy(), link_label_test)
        # train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
        # print("epoch {}: train_loss = {:.6f}, auc_roc = {:.6f}, auc_pr = {:.6f}, recall = {:.6f}, pr = {:.6f}, f1 = {:.6f}".format(epoch, \
        #                                                                                 train_loss, auc_roc, auc_pr, recall, pr, f1))

        # dev_score = auc_pr
        # file_logger.log(
        #     "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max([dev_score] + dev_score_history)))

        # # save
        # model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
        # trainer.save(model_file, epoch)
        # if epoch == 1 or dev_score > max(dev_score_history):
        #     copyfile(model_file, model_save_dir + '/best_model.pt')
        #     print("new best model saved.")

        #     metric_name = opt["id"] + ".json"
        #     json.dump(predict_label, codecs.open(metric_name, "w", encoding="utf-8"))

        #     file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}" \
        #                     .format(epoch, auc_roc * 100, auc_pr * 100))

        # if epoch % opt['save_epoch'] != 0:
        #     os.remove(model_file)

        # # lr schedule
        # if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
        #         opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        #     current_lr *= opt['lr_decay']
        #     trainer.update_lr(current_lr)

        # dev_score_history += [dev_score]
        # print("")




    return None





def testing(generate_fingerprints=True):
     # eval model
    print("Testing on set... and ...")

    #load_model
    trainer.model.eval()
    trainer.update_bipartite(drug_feature, protein_feature, UV, VU, adj)

    drug_hidden_out = trainer.drug_hidden_out
    protein_hidden_out = trainer.protein_hidden_out
    bi_feature = torch.cat((drug_hidden_out,protein_hidden_out),dim=0)




    if generate_fingerprints:
        """ 
         Generate the fingerprints.
         return: fingerprints for the input drug
        """
        def get_data(feature, id):
            #x is drug, y is protein
            x = np.unique(id[:, 0])
            y = np.unique(id[:, 1])
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)

            x = torch.index_select(feature, 0, x)  # batch * hidden_dim
            y =torch.index_select(feature, 0, y)  # batch * hidden_dim
            # ans = torch.cat((x), dim=1)
            return x, y
        
        all_dataset = np.concatenate((link_dataset,link_dataset_test))
        test_x, test_y = get_data(bi_feature, all_dataset)
        fps = test_x


        # save_path = os.path.join(model_save_dir + 'fps.pkl')
        # joblib.dump(fps,save_path)

        drug_id = joblib.load(os.path.join(opt['data_dir'], 'drug_id.pkl'))
        drug_ids = drug_id.keys()
        fp_df = pd.DataFrame(fps.detach().numpy()).T
        fp_df.columns = drug_ids
        print("Drug FPs are generated")
        save_path = os.path.join(model_save_dir,'fps.pkl')
        joblib.dump(fp_df,save_path)

    else:
        train_x, train_y, train_feature = concat_feature(bi_feature, link_dataset, opt["cuda"])
        test_x, test_y, test_feature = concat_feature(bi_feature, link_dataset_test, opt["cuda"])
        auc_roc, auc_pr, recall, pr, f1, predict_label = link_prediction_logistic(train_feature.cpu().detach().numpy(), link_label,
                                                test_feature.cpu().detach().numpy(), link_label_test)
        print("auc_roc = {:.6f}, auc_pr = {:.6f}, recall = {:.6f}, pr = {:.6f}, f1 = {:.6f}".format(\
                                                                                        auc_roc, auc_pr, recall, pr, f1))
    return None

if not opt["load"]:
    training()
else:
    # testing(generate_fingerprints=True)
    fine_tuning()

# print("Training ended with {} epochs.".format(epoch))
# if opt["save_node_feature"]:
#     np.savetxt("Bipartite_feature.txt" + str(opt["id"]), bi_feature.detach().numpy())


"""
CUDA_VISIBLE_DEVICES=1 nohup python -u train_lp.py --id wiki5 --struct_rate 0.0001 --GNN 2 > BiGIwiki5.log 2>&1&

CUDA_VISIBLE_DEVICES=1 nohup python -u train_lp.py --data_dir dataset/wiki/4/ --id wiki4 --struct_rate 0.0001 --GNN 2 > BiGIwiki4.log 2>&1&
"""

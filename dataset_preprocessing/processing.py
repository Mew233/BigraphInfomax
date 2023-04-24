import pandas as pd
import numpy as np
import random
import codecs 
import os
import joblib
import torch

def seed_everything(seed=2040):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()

cwd = os.getcwd()
def divide_dataset(div,dataset):
    random.shuffle(dataset)
    train_len = int(len(dataset)*(float(div)/10))+1

    train_set = dataset[:train_len]
    test_set = dataset[train_len:]
    adj={}
    protein_id= {}
    drug_id = {}
    for mytuple in train_set:
        if mytuple[0] not in drug_id:
            drug_id[mytuple[0]]=len(drug_id)
            adj[ drug_id[mytuple[0]]]={}
        if mytuple[1] not in protein_id:
            protein_id[mytuple[1]]=len(protein_id)
        adj[drug_id[mytuple[0]]][protein_id[mytuple[1]]] = 1
    for mytuple in test_set:
        if mytuple[0] not in drug_id:
            continue
        if mytuple[1] not in protein_id:
            continue
        adj[drug_id[mytuple[0]]][protein_id[mytuple[1]]] = 1

    print(len(test_set))
    print("drug_number: ", len(drug_id))
    print("protein_number: ", len(protein_id))
    with codecs.open("dataset/drug_protein/train.txt", "w", encoding="utf-8") as fw:
        for mytuple in train_set:

            fw.write("{}\t{}\t{}\n".format(drug_id[mytuple[0]], protein_id[mytuple[1]], int(mytuple[2])))

    with codecs.open("dataset/drug_protein/test.txt", "w", encoding="utf-8") as fw:
        for mytuple in test_set:
            if mytuple[0] not in drug_id:
                continue
            if mytuple[1] not in protein_id:
                continue
            fw.write("{}\t{}\t{}\n".format(drug_id[mytuple[0]], protein_id[mytuple[1]], int(mytuple[2])))

            neg=random.randint(0,len(protein_id)-1)
            while adj[drug_id[mytuple[0]]].get(neg,"0") == 1:
                neg = random.randint(0, len(protein_id)-1)
            fw.write("{}\t{}\t{}\n".format(drug_id[mytuple[0]], neg, 0))

    with codecs.open("test_T.txt", "w", encoding="utf-8") as fw:
        neg1 = dict(zip(protein_id.values(),protein_id.keys()))
        for mytuple in test_set:
            if mytuple[0] not in drug_id:
                continue
            if mytuple[1] not in protein_id:
                continue
            fw.write("{}\t{}\t{}\n".format(mytuple[0], mytuple[1], int(mytuple[2])))

            neg=random.randint(0,len(protein_id)-1)
            while adj[drug_id[mytuple[0]]].get(neg,"0") == 1:
                neg = random.randint(0, len(protein_id)-1)
            fw.write("{}\t{}\t{}\n".format(mytuple[0], neg1[neg], 0))

    save_path = os.path.join("dataset/drug_protein/",'drug_id.pkl')
    joblib.dump(drug_id,save_path)
    print("Drug ID is dumped")

# data = pd.read_excel('dataset/NPInter2.xlsx')
data = pd.read_csv(cwd+'/dataset_preprocessing/dataset/ChG-Miner_miner-chem-gene.tsv',sep='\t')
data = np.array(data)
divide_dataset(9, data)
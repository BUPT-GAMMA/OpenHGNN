import torch.nn as nn  
import torch  
import dgl
from scipy.io import loadmat
import numpy as np
from . import register_flow  
from . import BaseFlow
from ..utils import extract_embed, get_nodes_dict  
from torch.utils.data import Dataset, DataLoader
from ..models import BPHGNN 
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
import scipy.io as sio
import pickle as pkl
import time
from abc import ABC
from ..dataset import build_dataset
import os

data_dir = ''

class LogReg(nn.Module):  
    """
    Logical classifier
    """

    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()  
        self.fc = nn.Linear(ft_in, nb_classes)  

        for m in self.modules():  
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):  
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)  
        return ret
def sparse_mx_to_torch_sparse_tensor(sparse_mx):  
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def load_our_data(dataset_str, cuda=True):  

    global data_dir
    data = loadmat(os.path.join(data_dir,'alibaba_small.mat'))
    # label
    try:
        labels = data['label']  
    except:
        labels = data['labelmat']  
    N = labels.shape[0]
    try:
        labels = labels.todense()
    except:
        pass

    # # # alibaba_small
    t_v_t=loadmat(os.path.join(data_dir,'alibaba_small_20.mat'))
    idx_train = t_v_t['train_idx'].ravel()
    idx_val = t_v_t['valid_idx'].ravel()
    idx_test = t_v_t['test_idx'].ravel()

    try:
        node_features = data['full_feature'].toarray()
    except:
        try:
            node_features = data['feature']
        except:
            try:
                node_features = data['node_feature']
            except:
                node_features = data['features']
    features = csr_matrix(node_features)

    # edges to adj
    if dataset_str == 'small_alibaba_1_10':
        num_nodes = data['IUI_buy'].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['IUI_buy'] + data['IUI_cart'] + data["IUI_clk"] + data['IUI_collect']
    elif dataset_str == 'Aminer_10k_4class':
        num_nodes = 10000
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['PAP'] + data['PCP'] + data["PTP"]

        idx_test = idx_test - 1
        idx_train = idx_train - 1
        idx_val = idx_val - 1
    elif dataset_str == 'imdb_1_10':
        edges = data['edges'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        for edge in edges:
            adj += edge
    elif dataset_str == 'dblp_small':
        edges = data['edge'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
    elif dataset_str == 'imdb_small':
        edges = data['edge'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
    elif dataset_str == 'alibaba_large':
        edges = data['edge'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
    elif dataset_str == 'alibaba_small':
        edges = data['edge'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
    else:
        num_nodes = data['A'][0][0].toarray().shape[0]
        adj = data['A'][0][0] + data['A'][0][1] + data['A'][0][2]

    print('{} node number: {}'.format(dataset_str, num_nodes))

    try:
        features = features.astype(np.int16)
    except:
        pass
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train.astype(np.int16))
    idx_val = torch.LongTensor(idx_val.astype(np.int16))
    idx_test = torch.LongTensor(idx_test.astype(np.int16))

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test
def load_data(dataset, datasetfile_type):
    """"Get the label of node classification, training set, verification machine and test set"""
    global data_dir

    if datasetfile_type == 'mat':
        data = sio.loadmat(os.path.join(data_dir,'alibaba_small.mat'))
    else:
        pass
    try:
        labels = data['label']
    except:
        labels = data['labelmat']


    t_v_t=sio.loadmat(os.path.join(data_dir,'alibaba_small_20.mat'))
    idx_train = t_v_t['train_idx'].ravel()
    idx_val = t_v_t['valid_idx'].ravel()
    idx_test = t_v_t['test_idx'].ravel()

    return labels, idx_train.astype(np.int32) - 1, idx_val.astype(np.int32) - 1, idx_test.astype(np.int32) - 1


@register_flow("BPHGNN_trainer")
class BPHGNN_trainer(ABC):

    def __init__(self, args):
        super(BPHGNN_trainer, self).__init__()

        self.args = args
        
        #   在flow中构造数据集
        self.dataset = build_dataset(args.dataset, 'node_classification',  # 数据集名称  和  任务名称  是必要参数，其他都是 额外 关键字参数
                                     args = self.args , logger = args.logger)  

        #  返回的dataset包含两个成员 zip_file（压缩文件） 和  base_dir（所有数据内容）
        global data_dir
        data_dir = os.path.join(self.dataset.base_dir,args.dataset)

        self.encode=torch.tensor(np.loadtxt(os.path.join(data_dir,'alibaba_small_encode.txt')))
        mat=loadmat(os.path.join(data_dir,'alibaba_small.mat'))
        try:
            train = mat['A']
        except:
            try:
                train = mat['train']+mat['valid']+mat['test']
            except:
                try:
                    train = mat['train_full']+mat['valid_full']+mat['test_full']
                except:
                    try:
                        train = mat['edges']
                    except:
                        train = mat['edge']

        try:
            feature = mat['full_feature']
        except:
            try:
                feature = mat['feature']
            except:
                try:
                    feature = mat['features']
                except:
                    feature = mat['node_feature']
        feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
        self.feature=feature
        self.A = train


        self.adj, self.feature, self.labels, self.idx_train, self.idx_val, self.idx_test = load_our_data('alibaba_small',True)
        
        #   不用build_model_from_args，直接这样更方便
        self.model = BPHGNN(
            nfeat=self.feature.size(1),
            out=args.out,
            nhid=args.hidden_dim,               
            dropout=args.dropout
        )
        self.optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay))







    def train(self):
        embeds,near_embeds,far_embeds = self.model(self.feature,self.A,self.encode)

        labels, idx_train, idx_val, idx_test = load_data('alibaba_small', 'mat')

        try:
            labels = labels.todense()
        except:
            pass
        labels = labels.astype(np.int16)
        device=torch.device('cuda')
        embeds= torch.tensor(embeds[np.newaxis], dtype=torch.float32, device=device)

        labels = torch.FloatTensor(labels[np.newaxis]).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        idx_val = torch.LongTensor(idx_val).to(device)
        idx_test = torch.LongTensor(idx_test).to(device)

        hid_units = embeds.shape[2]
        nb_classes = labels.shape[2]
        xent = nn.CrossEntropyLoss()

        train_embs = embeds[0, idx_train]
        val_embs = embeds[0, idx_val]
        test_embs = embeds[0, idx_test]
        train_lbls = torch.argmax(labels[0, idx_train], dim=1)
        val_lbls = torch.argmax(labels[0, idx_val], dim=1)
        test_lbls = torch.argmax(labels[0, idx_test], dim=1)

        accs = []
        micro_f1s = []
        macro_f1s = []
        macro_f1s_val = []

        for _ in range(1):
            log = LogReg(hid_units, nb_classes)
            opt = torch.optim.Adam([{'params': self.model.parameters(), 'lr':self.args.lr}, {'params': log.parameters()}], lr=self.args.lr, weight_decay=self.args.weight_decay)
            log.to(device)

            val_accs = []
            test_accs = []
            val_micro_f1s = []
            test_micro_f1s = []
            val_macro_f1s = []
            test_macro_f1s = []

            starttime = time.time()
            for iter_ in range(200):
                embeds,near_embeds,far_embeds = self.model(self.feature, self.A,self.encode)
                # print(embeds)
                embeds= torch.tensor(embeds[np.newaxis], dtype=torch.float32, device=device)
                train_embs = embeds[0, idx_train]
                val_embs = embeds[0, idx_val]
                test_embs = embeds[0, idx_test]
                # train
                log.train()
                opt.zero_grad()

                logits = log(train_embs)

                loss = xent(logits, train_lbls)
                loss.backward()
                opt.step()

                logits_tra = log(train_embs)
                preds = torch.argmax(logits_tra, dim=1)

                tra_f1_macro = f1_score(train_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
                tra_f1_micro = f1_score(train_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
                print("===============================train{}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(),
                                                                                            tra_f1_macro,
                                                                                            tra_f1_micro))




                logits_val = log(val_embs)
                preds = torch.argmax(logits_val, dim=1)

                val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
                val_f1_macro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
                val_f1_micro = f1_score(val_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')

                print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(iter_ + 1, loss.item(), val_acc, val_f1_macro,
                                                                val_f1_micro))
                print("weight_b:{}".format(self.model.weight_b))

                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)

                # test
                logits_test = log(test_embs)
                preds = torch.argmax(logits_test, dim=1)

                test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                test_f1_macro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='macro')
                test_f1_micro = f1_score(test_lbls.cpu().detach().numpy(), preds.cpu().detach().numpy(), average='micro')
                print("test_f1-ma: {:.4f}\ttest_f1-mi: {:.4f}".format(test_f1_macro, test_f1_micro))

                test_accs.append(test_acc.item())
                test_macro_f1s.append(test_f1_macro)
                test_micro_f1s.append(test_f1_micro)

            endtime = time.time()

            print('time: {:.10f}'.format(endtime - starttime))

            max_iter = val_accs.index(max(val_accs))
            accs.append(test_accs[max_iter])

            max_iter = val_macro_f1s.index(max(val_macro_f1s))
            macro_f1s.append(test_macro_f1s[max_iter])

            max_iter = val_micro_f1s.index(max(val_micro_f1s))
            micro_f1s.append(test_micro_f1s[max_iter])



        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})".format(np.mean(macro_f1s),
                                                                                                    np.std(macro_f1s),
                                                                                                    np.mean(micro_f1s),
                                                                                                    np.std(micro_f1s)))

        return np.mean(macro_f1s), np.mean(micro_f1s)

    def _full_train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        h_dict = self.model.input_feature()
        loss = self.model(self.hg, h_dict, self.pos)
        loss.backward()
        self.optimizer.step()
        loss = loss.cpu()
        loss = loss.detach().numpy()
        return loss

    def evaluate(self, loader, is_test=False):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for features, labels in loader:
                features = torch.tensor(features).to(self.device)
                labels = torch.tensor(labels).to(self.device)

                outputs, _, _ = self.model(features, self.hg, self.pos)

                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        avg_loss = total_loss / len(loader)

        if is_test:
            all_labels = torch.tensor(all_labels)
            all_preds = torch.tensor(all_preds)

            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            micro_f1 = f1_score(all_labels, all_preds, average='micro')
            auc_score = roc_auc_score(all_labels, F.softmax(outputs, dim=1), multi_class='ovr')

            print(f"Test Loss: {avg_loss:.4f}")
            print(f"Macro-F1: {macro_f1:.4f}, Micro-F1: {micro_f1:.4f}, AUC: {auc_score:.4f}")
        else:
            return avg_loss
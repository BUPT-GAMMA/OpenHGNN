import scipy.io as scio
import numpy as np
import torch as th
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class evaluation:
    def __init__(self, seed):
        self.seed = seed

    def cluster(self, n, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        Y_pred = KMeans(n, random_state=self.seed).fit(X).predict(X)
        nmi = normalized_mutual_info_score(Y, Y_pred)
        ari = adjusted_rand_score(Y, Y_pred)
        return nmi, ari

    def classification(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.seed)
        LR = LogisticRegression(max_iter=10000)
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)

        macro_f1, micro_f1 = f1_node_classification(Y_test, Y_pred)
        return micro_f1, macro_f1


def f1_node_classification(y_label, y_pred):
    macro_f1 = f1_score(y_label, y_pred, average='macro')
    micro_f1 = f1_score(y_label, y_pred, average='micro')
    return macro_f1, micro_f1


def evaluate_(seed, X, Y, n):
    _evaluation = evaluation(seed)
    NMI, ARI = _evaluation.cluster(n, X, Y)
    micro, macro = _evaluation.classification(X, Y)

    print('<Cluster>        NMI = %.4f, ARI = %.4f' % (NMI, ARI))

    print('<Classification>     Micro-F1 = %.4f, Macro-F1 = %.4f' % (micro, macro))


def evaluate(seed, dataset, emd, g):
    if dataset == 'acm1':
        n = 3
        X = emd['paper'].detach().to('cpu')
        Y = g.nodes['paper'].data['label'].to('cpu')
    elif dataset == 'imdb':
        n = 3
        X = emd['movie'].detach().to('cpu')
        Y = g.nodes['movie'].data['label'].to('cpu')
    evaluate_(seed, X, Y, n)


def node_classification(y, node_data, mode):
    if mode not in ['train_mask', 'test_mask', 'valid_mask']:
        ValueError
    mask = node_data[mode]
    idx = th.nonzero(mask, as_tuple=False).squeeze()
    y_label = node_data['label'][idx].to('cpu')
    y_pred = th.argmax(y[idx], dim=1)

    macro_f1, micro_f1 = f1_node_classification(y_label, y_pred)
    return macro_f1, micro_f1


def cal_loss_f1(y, node_data, loss_func, mode):
    if mode not in ['train_mask', 'test_mask', 'valid_mask']:
        ValueError
    mask = node_data[mode]
    idx = th.nonzero(mask, as_tuple=False).squeeze()
    y_label = node_data['labels'][idx]
    y = y[idx]
    y_pred = th.argmax(y, dim=1)
    loss = loss_func(y, y_label)
    macro_f1, micro_f1 = f1_node_classification(y_label.cpu(), y_pred.cpu())
    return loss, macro_f1, micro_f1

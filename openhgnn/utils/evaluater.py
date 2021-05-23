import scipy.io as scio
import numpy as np
import torch as th
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as Metric


class Evaluator():
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

    def f1_node_classification(self, y_label, y_pred):
        macro_f1 = f1_score(y_label, y_pred, average='macro')
        micro_f1 = f1_score(y_label, y_pred, average='micro')
        return macro_f1, micro_f1

    def cal_acc(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def mrr_(self, embedding, w, train_triplets, valid_triplets, test_triplets, hits=[], eval_bz=100, eval_p="filtered"):
        if eval_p == "filtered":
            mrr = calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits)
            pass
        else:
            mrr = calc_raw_mrr(embedding, w, test_triplets, hits, eval_bz)
        return mrr


    ''''''
    # Used in HetGNN
    def LR_pred(self, train_X, train_Y, test_X):
        LR = LogisticRegression(max_iter=10000)
        LR.fit(train_X, train_Y)
        pred_Y = LR.predict(test_X)
        # AUC_score = Metric.roc_auc_score(test_Y, pred_Y)
        return pred_Y

    def link_prediction(self, train_X, train_Y, test_X, test_Y):
        pred_Y = self.LR_pred(train_X, train_Y, test_X)
        AUC_score = Metric.roc_auc_score(test_Y, pred_Y)
        print('\t<Link prediction> AUC=%.4f' % AUC_score, end='\t')
        macro_f1, micro_f1 = f1_node_classification(test_Y, pred_Y)
        print('macro_f1: {:.4f}, micro_f1: {:.4f}'.format(macro_f1, micro_f1))

    def author_link_prediction(self, x, train_batch, test_batch):
        train_u, train_v, train_Y = train_batch
        test_u, test_v, test_Y = test_batch

        train_X = concat_u_v(x, th.tensor(train_u), th.tensor(train_v))
        test_X = concat_u_v(x, th.tensor(test_u), th.tensor(test_v))
        train_Y = th.tensor(train_Y)
        test_Y = th.tensor(test_Y)
        self.link_prediction(train_X, train_Y, test_X, test_Y)
    ''''''
    # Given embedding and labels, train_idx and test_idx, training a LR.
    def nc_with_LR(slef, emd, labels, train_idx, test_idx):
        Y_train = labels[train_idx]
        Y_test = labels[test_idx]
        LR = LogisticRegression(max_iter=10000)
        X_train = emd[train_idx]
        X_test = emd[test_idx]
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)
        macro_f1, micro_f1 = f1_node_classification(Y_test, Y_pred)
        print('\t<node classification> macro_f1: {:.4f}, micro_f1: {:.4f}'.format(macro_f1, micro_f1))
        return micro_f1, macro_f1


def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=1000):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        #print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]



        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = th.bmm(emb_ar, emb_c) # size D x E x V
        score = th.sum(out_prod, dim=0) # size E x V

        # emb_ar = (embedding[batch_a] + w[batch_r]).unsqueeze(1).expand((batch_end - batch_start,embedding.shape[0], -1))
        # emb_c = embedding.unsqueeze(0)
        # out_prod = th.sub(emb_ar, emb_c) # size D x E x V
        # score = -th.norm(out_prod, p=1, dim=2) # size E x V

        score = th.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return th.cat(ranks)


def sort_and_rank(score, target):
    _, indices = th.sort(score, dim=1, descending=True)
    indices = th.nonzero(indices == target.view(-1, 1), as_tuple=False)
    indices = indices[:, 1].view(-1)
    return indices


# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
    with th.no_grad():
        s = test_triplets[0]
        r = test_triplets[1]
        o = test_triplets[2]
        test_size = test_triplets.shape[1]

        # perturb subject
        ranks = perturb_and_get_raw_rank(embedding, w, s, r, o, test_size, eval_bz)
        # perturb object
        #ranks_o = perturb_and_get_raw_rank(embedding, w, s, r, o, test_size, eval_bz)

        #ranks = th.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = th.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = th.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()


def perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb subject in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 1000 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities).to(target_s.device)
        target_s_idx = th.nonzero(filtered_s == target_s).item()
        emb_s = embedding[filtered_s]
        emb_r = w[target_r]
        emb_o = embedding[target_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = th.sigmoid(th.sum(emb_triplet, dim=1))
        _, indices = th.sort(scores, descending=True)
        rank = th.nonzero(indices == target_s_idx).item()
        ranks.append(rank)
    return th.LongTensor(ranks)


def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider a subject if it is part of a triplet to filter
    for s in range(num_entities):
        if (s, target_r, target_o) not in triplets_to_filter:
            filtered_s.append(s)
    return th.LongTensor(filtered_s)


def calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[]):
    with th.no_grad():
        s = test_triplets[0]
        r = test_triplets[1]
        o = test_triplets[2]
        test_size = test_triplets.shape[1]

        triplets_to_filter = th.transpose(th.cat([train_triplets, valid_triplets, test_triplets], dim=1), 0, 1).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        print('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)
        print('Perturbing object...')
        #ranks_o = perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)

        #ranks = th.cat([ranks_s, ranks_o])
        ranks = ranks_s
        ranks += 1 # change to 1-indexed

        mrr = th.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = th.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()


def concat_u_v(x, u_idx, v_idx):
    u = x[u_idx]
    v = x[v_idx]
    emd = th.cat((u, v), dim=1)
    return emd


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

def cal_acc(y_pred, y_true):
    return th.sum(y_pred.argmax(dim=1) == y_true).item() / len(y_true)

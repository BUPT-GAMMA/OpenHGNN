import numpy as np
import torch as th
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error,normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import f1_score, accuracy_score, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as Metric
from ogb.nodeproppred import Evaluator
from tqdm import tqdm

from sklearn import metrics, preprocessing
from sklearn.svm import SVC

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
        return dict(Macro_f1=macro_f1, Micro_f1=micro_f1)

    def cal_acc(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def cal_roc_auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def mrr_(self, n_embedding, r_embedding, train_triplets, valid_triplets, test_triplets, score_predictor, hits=[], filtered='raw', eval_mode='test'):
        if not hasattr(self, "triplets_to_filter"):
            triplets_to_filter = th.cat([train_triplets, valid_triplets, test_triplets]).tolist()
            self.triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        return cal_mrr(n_embedding, r_embedding, valid_triplets, test_triplets, self.triplets_to_filter, score_predictor, hits, filtered, eval_mode)

    def ndcg(self, y_score, y_true):
        return ndcg_score(y_true, y_score, 10)

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
        macro_f1, micro_f1 = f1_node_classification(test_Y, pred_Y)
        return AUC_score, macro_f1, micro_f1
    
    def author_link_prediction(self, x, train_batch, test_batch):
        train_u, train_v, train_Y = train_batch
        test_u, test_v, test_Y = test_batch

        train_X = concat_u_v(x, th.tensor(train_u), th.tensor(train_v))
        test_X = concat_u_v(x, th.tensor(test_u), th.tensor(test_v))
        train_Y = th.tensor(train_Y)
        test_Y = th.tensor(test_Y)
        return self.link_prediction(train_X, train_Y, test_X, test_Y)
    ''''''
    # Given embedding and labels, train_idx and test_idx, training a LR.
    def nc_with_LR(self, emd, labels, train_idx, test_idx):
        Y_train = labels[train_idx]
        Y_test = labels[test_idx]
        LR = LogisticRegression(max_iter=10000)
        X_train = emd[train_idx]
        X_test = emd[test_idx]
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)
        macro_f1, micro_f1 = f1_node_classification(Y_test, Y_pred)
        return micro_f1, macro_f1

    def ec_with_SVC(self, C, gamma, emd, labels, train_idx, test_idx):
        X_train = emd[train_idx]
        Y_train = labels[train_idx]
        X_test= emd[test_idx]
        Y_test = labels[test_idx]
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = SVC(C=C, gamma=gamma).fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        macro_f1 = metrics.f1_score(Y_test, Y_pred, average='macro')
        micro_f1 = metrics.f1_score(Y_test, Y_pred, average='micro')
        acc = metrics.accuracy_score(Y_test, Y_pred)
        return micro_f1, macro_f1, acc


    def prediction(self, real_score, pred_score):
        MAE = mean_absolute_error(real_score, pred_score)
        RMSE = math.sqrt(mean_squared_error(real_score, pred_score))
        return MAE, RMSE

    def dcg_at_k(self, scores):
        # assert scores
        return scores[0] + sum(
            sc / math.log(ind + 1, 2)
            for sc, ind in zip(scores[1:], range(2, len(scores) + 1))
        )

    def ndcg_at_k(self, real_scores, predicted_scores):
        idcg = self.dcg_at_k(sorted(real_scores, reverse=True))
        return (self.dcg_at_k(predicted_scores) / idcg) if idcg > 0.0 else 0.0

    def ranking(self, real_score, pred_score, k):
        # ndcg@k
        sorted_idx = sorted(
            np.argsort(real_score)[::-1][:k]
        )  # get the index of the top k real score
        r_s_at_k = real_score[sorted_idx]
        p_s_at_k = pred_score[sorted_idx]

        ndcg_5 = self.ndcg_at_k(r_s_at_k, p_s_at_k)

        return ndcg_5




def filter(triplets_to_filter, target_s, target_r, target_o, num_entities, mode):
    triplets_to_filter = triplets_to_filter.copy()
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider an object if it is part of a triplet to filter
    if mode == 's':
        for s in range(num_entities):
            if (s, target_r, target_o) not in triplets_to_filter:
                filtered.append(s)
    elif mode == 'o':
        for o in range(num_entities):
            if (target_s, target_r, o) not in triplets_to_filter:
                filtered.append(o)
    return th.LongTensor(filtered)

def perturb_and_get_rank(n_embedding, r_embedding, eval_triplets, triplets_to_filter, score_predictor, filtered, preturb_side):
    """ Perturb object in the triplets
    """
    ranks = []
    num_entities = n_embedding.shape[0]
    eval_range = tqdm(range(eval_triplets.shape[0]), ncols=100)
    for idx in eval_range:
        target_s = eval_triplets[idx, 0]
        target_r = eval_triplets[idx, 1]
        target_o = eval_triplets[idx, 2]

        if filtered == 'filtered':
            if preturb_side == 'o':
                select_s = target_s
                select_o = filter(triplets_to_filter, target_s, target_r, target_o, num_entities, 'o')
                target_idx = int((select_o == target_o).nonzero())
            elif preturb_side == 's':
                select_s = filter(triplets_to_filter, target_s, target_r, target_o, num_entities, 's')
                select_o = target_o
                target_idx = int((select_s == target_s).nonzero())
        elif filtered == 'raw':
            if preturb_side == 'o':
                select_s = target_s
                select_o = th.arange(num_entities)
                target_idx = target_o
            elif preturb_side == 's':
                select_o = target_o
                select_s = th.arange(num_entities)
                target_idx = target_s

        emb_s = n_embedding[select_s]
        emb_r = r_embedding[int(target_r)]
        emb_o = n_embedding[select_o]
        
        scores = score_predictor(emb_s, emb_r, emb_o)
        _, indices = th.sort(scores, descending=False)
        rank = int((indices == target_idx).nonzero())
        ranks.append(rank)
    return th.LongTensor(ranks)

def cal_mrr(n_embedding, r_embedding, valid_triplets, test_triplets, triplets_to_filter, score_predictor, hits=[], filtered='raw', eval_mode='test'):
    with th.no_grad():
        eval_triplets = test_triplets if eval_mode == 'test' else valid_triplets
        
        print('Perturbing subject...')
        ranks_s = perturb_and_get_rank(n_embedding, r_embedding, eval_triplets, triplets_to_filter, score_predictor, filtered, 's')
        print('Perturbing oubject...')
        ranks_o = perturb_and_get_rank(n_embedding, r_embedding, eval_triplets, triplets_to_filter, score_predictor, filtered, 'o')
        #get matrix
        ranks = th.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed
        mrr_matrix = {
            'Mode': filtered,
            'MR': th.mean(ranks.float()).item(),
            'MRR': th.mean(1.0 / ranks.float()).item(),
        }
        for hit in hits:
            mrr_matrix['Hits@'+str(hit)] = th.mean((ranks <= hit).float()).item()
        return mrr_matrix

# def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=100):
#     """ Perturb one element in the triplets
#     """
#     n_batch = (test_size + batch_size - 1) // batch_size
#     ranks = []
#     for idx in range(n_batch):
#         # print("batch {} / {}".format(idx, n_batch))
#         batch_start = idx * batch_size
#         batch_end = min(test_size, (idx + 1) * batch_size)
#         batch_a = a[batch_start: batch_end]
#         batch_r = r[batch_start: batch_end]
#         emb_ar = embedding[batch_a] * w[batch_r]
#         emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
#         emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
#         # out-prod and reduce sum
#         out_prod = th.bmm(emb_ar, emb_c) # size D x E x V
#         score = th.sum(out_prod, dim=0) # size E x V
#         score = th.sigmoid(score)
#         target = b[batch_start: batch_end].to(score.device)
#         ranks.append(sort_and_rank(score, target))
#     return th.cat(ranks)


# def sort_and_rank(score, target):
#     _, indices = th.sort(score, dim=1, descending=True)
#     indices = th.nonzero(indices == target.view(-1, 1), as_tuple=False)
#     indices = indices[:, 1].view(-1)
#     return indices


# return MRR (raw), and Hits @ (1, 3, 10)
# def calc_raw_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
#     with th.no_grad():
#         s = test_triplets[:, 0]
#         r = test_triplets[:, 1]
#         o = test_triplets[:, 2]
#         test_size = test_triplets.shape[0]

#         # perturb subject
#         ranks_s = perturb_and_get_raw_rank(embedding, w, o, r, s, test_size, eval_bz)
#         # perturb object
#         ranks_o = perturb_and_get_raw_rank(embedding, w, s, r, o, test_size, eval_bz)

#         ranks = th.cat([ranks_s, ranks_o])
#         ranks += 1 # change to 1-indexed

#         mrr = th.mean(1.0 / ranks.float())
#         print("MRR (raw): {:.6f}".format(mrr.item()))

#         for hit in hits:
#             avg_count = th.mean((ranks <= hit).float())
#             print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
#     return mrr.item()


# def perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
#     """ Perturb subject in the triplets
#     """
#     num_entities = embedding.shape[0]
#     ranks = []
#     for idx in range(test_size):
#         if idx % 1000 == 0:
#             print("test triplet {} / {}".format(idx, test_size))
#         target_s = s[idx]
#         target_r = r[idx]
#         target_o = o[idx]

#         filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities)
#         target_s_idx = int((filtered_s == target_s).nonzero())
#         emb_s = embedding[filtered_s]
#         emb_r = w[str(target_r.item())]
#         emb_o = embedding[target_o]
#         emb_triplet = emb_s * emb_r * emb_o
#         scores = th.sigmoid(th.sum(emb_triplet, dim=1))
#         _, indices = th.sort(scores, descending=True)
#         rank = int((indices == target_s_idx).nonzero())
#         ranks.append(rank)
#     return th.LongTensor(ranks)


# def perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
#     """ Perturb object in the triplets
#     """
#     num_entities = embedding.shape[0]
#     ranks = []
#     for idx in range(test_size):
#         if idx % 10000 == 0:
#             print("test triplet {} / {}".format(idx, test_size))
#         target_s = s[idx]
#         target_r = r[idx]
#         target_o = o[idx]
#         filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities)
#         target_o_idx = int((filtered_o == target_o).nonzero())
#         emb_s = embedding[target_s]
#         emb_r = w[str(target_r.item())]
#         emb_o = embedding[filtered_o]
#         emb_triplet = emb_s * emb_r * emb_o
#         scores = th.sigmoid(th.sum(emb_triplet, dim=1))
#         _, indices = th.sort(scores, descending=True)
#         rank = int((indices == target_o_idx).nonzero())
#         ranks.append(rank)
#     return th.LongTensor(ranks)


# def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
#     target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
#     filtered_o = []
#     # Do not filter out the test triplet, since we want to predict on it
#     if (target_s, target_r, target_o) in triplets_to_filter:
#         triplets_to_filter.remove((target_s, target_r, target_o))
#     # Do not consider an object if it is part of a triplet to filter
#     for o in range(num_entities):
#         if (target_s, target_r, o) not in triplets_to_filter:
#             filtered_o.append(o)
#     return th.LongTensor(filtered_o)


# def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
#     target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
#     filtered_s = []
#     # Do not filter out the test triplet, since we want to predict on it
#     if (target_s, target_r, target_o) in triplets_to_filter:
#         triplets_to_filter.remove((target_s, target_r, target_o))
#     # Do not consider a subject if it is part of a triplet to filter
#     for s in range(num_entities):
#         if (s, target_r, target_o) not in triplets_to_filter:
#             filtered_s.append(s)
#     return th.LongTensor(filtered_s)


# def calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[]):
#     with th.no_grad():
#         s = test_triplets[:, 0]
#         r = test_triplets[:, 1]
#         o = test_triplets[:, 2]
#         test_size = test_triplets.shape[0]

#         triplets_to_filter = th.cat([train_triplets, valid_triplets, test_triplets]).tolist()
#         triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
#         print('Perturbing subject...')
#         ranks_s = perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)
#         print('Perturbing object...')
#         ranks_o = perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)

#         ranks = th.cat([ranks_s, ranks_o])
#         ranks += 1 # change to 1-indexed

#         mrr = th.mean(1.0 / ranks.float())
#         print("MRR (filtered): {:.6f}".format(mrr.item()))

#         for hit in hits:
#             avg_count = th.mean((ranks <= hit).float())
#             print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
#     return mrr.item()


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

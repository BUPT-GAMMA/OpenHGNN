import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from openhgnn.models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
import scipy.sparse as ssp
from sklearn import metrics
from ..utils.Grail_utils import collate_dgl2,move_batch_to_device_dgl,ssp_multigraph_to_dgl
import dgl
import tqdm
import random
params = None

@register_flow('ComPILE_trainer')
class ComPILETrainer(BaseFlow):
    def __init__(self,args):
        super(ComPILETrainer, self).__init__(args)
        #self.train_hg = self.task.get_train()
        self.trainset = self.task.dataset.train
        self.valid = self.task.dataset.valid
        self.args.num_rels = self.trainset.num_rels
        self.args.aug_num_rels = self.trainset.aug_num_rels
        self.args.inp_dim = self.trainset.n_feat_dim

        self.args.collate_fn = collate_dgl2
        self.args.move_batch_to_device = move_batch_to_device_dgl
        self.args.max_label_value = self.trainset.max_n_label
        self.params = self.args
        self.params.adj_list = []
        self.params.dgl_adj_list = []
        self.params.triplets = []
        self.params.entity2id = []
        self.params.relation2id = []
        self.params.id2entity = []
        self.params.id2relation = []

        # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.

        self.model = build_model(self.model).build_model_from_args(self.args, self.task.dataset.relation2id).to(
            self.device)
        self.updates_counter = 0
        model_params = list(self.model.parameters())
        #logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if self.args.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=self.args.lr, momentum=self.args.momentum,
                                       weight_decay=self.args.l2)
        if self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=self.args.lr, weight_decay=self.args.l2)

        self.criterion = nn.MarginRankingLoss(self.args.margin, reduction='sum')

        self.reset_training_state()
        #graph_classifier = initialize_model(params, dgl_model, params.load_model)

        self.logger.info(f"Device: {args.device}")
        self.logger.info(
            f"Input dim : {args.inp_dim}, # Relations : {args.num_rels}, # Augmented relations : {args.aug_num_rels}")

        self.args.save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name
        self.valid_evaluator = Evaluator(self.args, self.model, self.valid)
        #self.save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name

        self.logger.info('Starting training with full batch...')




    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers, collate_fn=self.args.collate_fn)
        #      dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers)
        self.model.train()
        model_params = list(self.model.parameters())
        torch.multiprocessing.set_sharing_strategy('file_system')
        for b_idx, batch in enumerate(dataloader):
            (graphs_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = batch
            #(graphs_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = self.args.move_batch_to_device(batch, self.args.device)

            g_labels_pos = torch.LongTensor(g_labels_pos).to(device=self.args.device)
            r_labels_pos = torch.LongTensor(r_labels_pos).to(device=self.args.device)

            g_labels_neg = torch.LongTensor(g_labels_neg).to(device=self.args.device)
            r_labels_neg = torch.LongTensor(r_labels_neg).to(device=self.args.device)

            self.model.train()
            # data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()
            # print('batch size ', len(targets_pos), '     ', len(targets_neg))
            # print('r label pos ', len(data_pos[1]), '   r label neg  ', len(data_neg[1]))
            score_pos = self.model(graphs_pos)
            score_neg = self.model(graph_neg)
            loss = self.criterion(score_pos.squeeze(), score_neg.view(len(score_pos), -1).mean(dim=1),
                                  torch.Tensor([1]).to(device=self.args.device))
            # print(score_pos, score_neg, loss)
            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                # print(score_pos.shape, score_neg.shape)
                #  print(score_pos)
                all_scores += score_pos.squeeze(1).detach().cpu().tolist() + score_neg.squeeze(
                    1).detach().cpu().tolist()
                all_labels += g_labels_pos.tolist() + g_labels_neg.tolist()
                total_loss += loss

            if self.valid_evaluator and self.args.eval_every_iter and self.updates_counter % self.args.eval_every_iter == 0:
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info('\nPerformance:' + str(result) + 'in ' + str(time.time() - tic))

                if result['auc'] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result['auc']
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.args.early_stop:
                        logging.info(
                            f"Validation performance didn\'t improve for {self.args.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result['auc']

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.reset_training_state()
        for epoch in range(1, self.args.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            self.logger.info(
                f'Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}')

            if epoch % self.args.save_every == 0:
                #save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name
                torch.save(self.model, self.args.save_path + '/ComPILE_chk.pth')
        self.params.model_path = self.args.save_path + '/ComPILE_chk.pth'
        self.params.file_paths = {
            'graph': os.path.join(f'./openhgnn/dataset/data/{self.args.dataset}_ind/train.txt'),
            'links': os.path.join(f'./openhgnn/dataset/data/{self.args.dataset}_ind/test.txt')
        }
        global params
        params = self.params
        eval_rank(self.logger)
        return

    def save_classifier(self):
        #save_path = os.path.dirname(os.path.abspath('__file__')) + '/openhgnn/output/' + self.model_name
        torch.save(self.model, self.args.save_path + '/best.pth')
        self.logger.info('Better models found w.r.t accuracy. Saved it!')


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                (graphs_pos, r_labels_pos), g_labels_pos, (graph_neg, r_labels_neg), g_labels_neg = batch
                # data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                # print([self.data.id2relation[r.item()] for r in data_pos[1]])
                # pdb.set_trace()

                g_labels_pos = torch.LongTensor(g_labels_pos).to(device=self.params.device)
                r_labels_pos = torch.LongTensor(r_labels_pos).to(device=self.params.device)

                g_labels_neg = torch.LongTensor(g_labels_neg).to(device=self.params.device)
                r_labels_neg = torch.LongTensor(r_labels_neg).to(device=self.params.device)

                score_pos = self.graph_classifier(graphs_pos)
                score_neg = self.graph_classifier(graph_neg)

                # preds += torch.argmax(logits.detach().cpu(), dim=1).tolist()
                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += g_labels_pos.tolist()
                neg_labels += g_labels_neg.tolist()

        # acc = metrics.accuracy_score(labels, preds)
        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.save_path,
                                                  'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.save_path,
                                         'data/{}/grail_{}_predictions.txt'.format(self.params.dataset,
                                                                                   self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.save_path,
                                                  'data/{}/neg_{}_0.txt'.format(self.params.dataset,
                                                                                self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.save_path,
                                         'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset,
                                                                                          self.data.file_name,
                                                                                          self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}

def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_ = model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
    with open(ruleN_pred_path) as f:
        pred_data = [line.split() for line in f.read().split('\n')[:-1]]

    neg_triplets = []
    for i in range(len(pred_data) // 3):
        neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
        if pred_data[3 * i][1] in saved_relation2id:
            head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
            for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
                neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
                if entity2id[new_head] == head:
                    neg_triplet['head'][1] = j
            for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
                neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
                if entity2id[new_tail] == tail:
                    neg_triplet['tail'][1] = j

            neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
            neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

            neg_triplets.append(neg_triplet)

    return neg_triplets


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    # an implementation of the proposed double-radius node labeling (DRNd   L)
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes





def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph


def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value, id2entity, node_features=None, kge_entity2id=None):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params_.hop, enclosing_sub_graph=params.enclosing_sub_graph, max_node_label_value=max_node_label_value)

        subgraph = dgl_adj_list.subgraph(nodes)
        subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        # edges_btw_roots = subgraph.edge_id(0, 1)
        try:
            edges_btw_roots = subgraph.edge_ids(0, 1)
            edges_btw_roots = torch.tensor([edges_btw_roots])
        except:
            edges_btw_roots = torch.tensor([])
        edges_btw_roots = edges_btw_roots.numpy()
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)

        if rel_link.squeeze().nelement() == 0:
            # subgraph.add_edge(0, 1, {'type': torch.tensor([rel]), 'label': torch.tensor([rel])})
            subgraph = dgl.add_edges(subgraph, 0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        n_feats = node_features[kge_nodes] if node_features is not None else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        subgraphs.append(subgraph)
        r_labels.append(rel)

   # batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (subgraphs, r_labels)


def get_rank(neg_links):
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, model_.max_label_value, id2entity_, node_features_, kge_entity2id_)
        head_scores = model_(data[0]).squeeze(1).detach().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, params.max_label_value, id2entity_, node_features_, kge_entity2id_)
        tail_scores = model_(data[0]).squeeze(1).detach().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')

import json
def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id



def eval_rank(logger):
    # print(params.file_paths)
    model = torch.load(params.model_path, map_location='cpu')

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)

    node_features, kge_entity2id = None, None

    if params.mode == 'sample':
        neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)
    elif params.mode == 'ruleN':
        neg_triplets = get_neg_samples_replacing_head_tail_from_ruleN(params.ruleN_pred_path, entity2id, relation2id)
    print(len(neg_triplets))
    ranks = []
    all_head_scores = []
    all_tail_scores = []
    intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id)
    # with mp.Pool(processes=None, initializer=intialize_worker, initargs=(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id)) as p:
    # intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id)
    for link in neg_triplets:
        head_scores, head_rank, tail_scores, tail_rank = get_rank(link)
        ranks.append(head_rank)
        ranks.append(tail_rank)

        all_head_scores += head_scores.tolist()
        all_tail_scores += tail_scores.tolist()




    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))

    logger.info(f'MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}')
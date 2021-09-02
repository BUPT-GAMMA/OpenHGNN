import copy
import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, get_nodes_dict
from collections.abc import Mapping
from ..models import build_model
from ..layers.EmbedLayer import HeteroEmbedLayer
from ..layers.HeteroLinear import HeteroFeature


class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.etypes
        }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict


@register_flow("link_prediction")
class LinkPrediction(BaseFlow):
    """Link Prediction flows."""

    def __init__(self, args):
        super(LinkPrediction, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device

        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)
        #self.target_link = self.task.dataset.target_link
        self.loss_fn = self.task.get_loss_fn()
        self.args.has_feature = self.task.dataset.has_feature



        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

        self.evaluator = self.task.get_evaluator('mrr')

        self.optimizer = (
            th.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        )
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        if self.args.mini_batch_flag:
            self.hg = self.hg.to('cpu')
            train_eid_dict = {
                etype: self.hg.edges(etype=etype, form='eid')
                for etype in self.hg.canonical_etypes}

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.n_layers)
            self.dataloader = dgl.dataloading.EdgeDataLoader(
                self.hg, train_eid_dict, sampler, device=self.device,
                    negative_sampler=NegativeSampler(self.hg, 1), batch_size=100,
                shuffle=True, drop_last=False, num_workers=0
            )
        else:

            # self.train_eid_dict = {
            #     self.target_link: self.hg.edges(etype=self.target_link, form='eid')}
            self.train_eid_dict = {
                etype: self.hg.edges(etype=etype, form='eid')
                for etype in self.hg.canonical_etypes}
            self.positive_graph = self.hg
            self.negative_sampler = NegativeSampler(self.hg, 10)
            # self.pos_test_graph = self.task.dataset.pos_test_graph.to(self.device)
            # self.neg_test_graph = self.task.dataset.neg_test_graph.to(self.device)

    def preprocess(self):
        if type(self.hg.ndata['h']) == dict:
            self.input_feature = HeteroFeature(self.hg.ndata['h'], get_nodes_dict(self.hg), self.args.hidden_dim).to(self.device)
        self.optimizer.add_param_group({'params': self.input_feature.parameters()})
        return

    def train(self):
        self.preprocess()
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 100
        best_model = copy.deepcopy(self.model)

        for epoch in tqdm(range(self.max_epoch), ncols=80):
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
            if epoch % 2 == 0:
                metric = self._test_step()
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, NDCG: {metric:.4f}, Loss:{loss:.4f}"
                )
                if metric <= best_score:
                    best_score = metric
                    best_model = copy.deepcopy(self.model)
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience:
                        epoch_iter.close()
                        break
        print(f"Valid mrr = {best_score: .4f}")
        self.model = best_model
        test_mrr = self._test_step(split="test")
        val_mrr = self._test_step(split="val")
        print(f"Test mrr = {test_mrr:.4f}")
        return dict(Test_mrr=test_mrr, ValMrr=val_mrr)

    def _mini_train_step(self,):
        self.model.train()
        all_loss = 0
        for input_nodes, positive_graph, negative_graph, blocks in self.dataloader:
            blocks = [b.to(self.device) for b in blocks]
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)
            if type(input_nodes) == th.Tensor:
                input_nodes = {self.category: input_nodes}
            input_features = extract_embed(self.model.embed_layer(), input_nodes)
            logits = self.model(blocks, input_features)[self.category]
            loss = self.loss_calculation(positive_graph, negative_graph, logits)
            all_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return all_loss

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        p_score = self.ScorePredictor(positive_graph, embedding).repeat_interleave(10)
        n_score = self.ScorePredictor(negative_graph, embedding)
        label = th.ones(len(n_score), device=self.device)
        loss = F.binary_cross_entropy_with_logits(p_score - n_score, label)
        return loss

    def ScorePredictor(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            edge_subgraph.apply_edges(
                dgl.function.u_dot_v('x', 'x', 'score'))
            score = edge_subgraph.edata['score']
            return score.squeeze()

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.r_embedding.pow(2))

    def construct_negative_graph(self,):
        neg_srcdst = self.negative_sampler(self.positive_graph, self.train_eid_dict)
        if not isinstance(neg_srcdst, Mapping):
            assert len(self.hg.etypes) == 1, \
                'graph has multiple or no edge types; '\
                'please return a dict in negative sampler.'
            neg_srcdst = {self.hg.canonical_etypes[0]: neg_srcdst}
        # Get dtype from a tuple of tensors
        #dtype = F.dtype(list(neg_srcdst.values())[0][0])
        neg_edges = {
            etype: neg_srcdst.get(etype[1], (th.IntTensor([]), th.IntTensor([])))
            for etype in self.positive_graph.canonical_etypes}
        neg_pair_graph = dgl.heterograph(
            neg_edges, {ntype: self.positive_graph.number_of_nodes(ntype) for ntype in self.positive_graph.ntypes})
        return neg_pair_graph

    def _full_train_setp(self):
        self.model.train()
        embedding = self.model(self.hg)

        negative_graph = self.construct_negative_graph()
        loss = self.loss_calculation(self.positive_graph, negative_graph, embedding)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(loss.item())
        return loss.item()

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with th.no_grad():
            if self.has_feature == True:
                h = self.hg.ndata['h']
            else:
                h = self.input_feature()
            embedding = self.model(self.hg, h)
            p_score = self.ScorePredictor(self.pos_test_graph, embedding).unsqueeze(0)
            n_score = self.ScorePredictor(self.neg_test_graph, embedding)
            n_score = th.reshape(n_score, (99, -1))
            matrix = th.cat((p_score, n_score), 0).t().cpu().numpy()

            y_true = np.zeros_like(matrix)
            y_true[:, 0] = 1
            # _, indices = torch.sort(matrix, dim=0, descending=True)
            # rank = th.nonzero(indices == 0, as_tuple=True)[0]
            from sklearn.metrics import ndcg_score
            metric = ndcg_score(y_true, matrix, k=10)

        return metric

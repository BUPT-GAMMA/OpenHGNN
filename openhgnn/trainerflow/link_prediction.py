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
from ..models import build_model
from ..layers.HeteroLinear import HeteroFeature
from dgl.dataloading.negative_sampler import Uniform
from ..utils import extract_embed, EarlyStopping, get_nodes_dict


@register_flow("link_prediction")
class LinkPrediction(BaseFlow):
    """Link Prediction flows."""

    def __init__(self, args):
        super(LinkPrediction, self).__init__(args)

        self.target_link = self.task.dataset.target_link
        self.loss_fn = self.task.get_loss_fn()
        self.args.has_feature = self.task.dataset.has_feature

        self.args.out_node_type = self.task.dataset.ntypes
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

        self.evaluator = self.task.get_evaluator('mrr')

        self.optimizer = (
            th.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        )
        self.patience = args.patience
        self.max_epoch = args.max_epoch

    def preprocess(self):
        self.train_hg, self.val_hg, self.test_hg = self.task.get_idx()
        self.train_hg = self.train_hg.to(self.device)
        self.val_hg = self.val_hg.to(self.device)
        self.test_hg = self.test_hg.to(self.device)
        self.negative_sampler = Uniform(1)
        self.positive_graph = self.train_hg.edge_type_subgraph(self.target_link)
        self.preprocess_feature()


    def train(self):
        self.preprocess()
        epoch_iter = tqdm(range(self.max_epoch))
        best_model = copy.deepcopy(self.model)
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        for epoch in tqdm(range(self.max_epoch), ncols=80):
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
            if epoch % 2 == 0:
                metric = self._test_step()
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, roc_auc: {metric:.4f}, Loss:{loss:.4f}"
                )
                early_stop = stopper.step_score(metric, self.model)
                if early_stop:
                    print('Early Stop!\tEpoch:' + str(epoch))
                    break
        print(f"Valid_score_ = {stopper.best_score: .4f}")
        stopper.load_model(self.model)


        ############ TEST SCORE #########
        if self.args.dataset[:4] == 'HGBl':
            self.model.eval()
            with torch.no_grad():
                h_dict = self.input_feature()
                embedding = self.model(self.hg, h_dict)
                score = th.sigmoid(self.ScorePredictor(self.test_hg, embedding))
                self.task.dataset.save_results(logits=score, file_path=self.args.HGB_results_path)
            return
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
        p_score = self.ScorePredictor(positive_graph, embedding)
        n_score = self.ScorePredictor(negative_graph, embedding)

        p_label = th.ones(len(p_score), device=self.device)
        n_label = th.zeros(len(n_score), device=self.device)
        loss = F.binary_cross_entropy_with_logits(th.cat((p_score, n_score)), th.cat((p_label, n_label)))
        return loss

    def ScorePredictor(self, edge_subgraph, x):
        with edge_subgraph.local_scope():

            for ntype in edge_subgraph.ntypes:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            score = edge_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items():
                    result.append(value)
                score = th.cat(result)
            return score.squeeze()

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.r_embedding.pow(2))

    def construct_negative_graph(self, hg):
        e_dict = {
            etype: hg.edges(etype=etype, form='eid')
            for etype in hg.canonical_etypes}
        neg_srcdst = self.negative_sampler(hg, e_dict)
        neg_pair_graph = dgl.heterograph(neg_srcdst,
                                         {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes})
        return neg_pair_graph

    def _full_train_setp(self):
        self.model.train()
        h_dict = self.input_feature()
        embedding = self.model(self.train_hg, h_dict)
        negative_graph = self.construct_negative_graph(self.positive_graph)
        loss = self.loss_calculation(self.positive_graph, negative_graph, embedding)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(loss.item())
        return loss.item()

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with th.no_grad():
            h_dict = self.input_feature()
            embedding = self.model(self.hg, h_dict)
            negative_graph = self.construct_negative_graph(self.val_hg)
            p_score = th.sigmoid(self.ScorePredictor(self.val_hg, embedding))
            n_score = th.sigmoid(self.ScorePredictor(negative_graph, embedding))
            p_label = th.ones(len(p_score), device=self.device)
            n_label = th.zeros(len(n_score), device=self.device)

        from sklearn.metrics import f1_score, auc, roc_auc_score
        metric = roc_auc_score(th.cat((p_label, n_label)).cpu(), th.cat((p_score, n_score)).cpu() )

        return metric

import dgl
import torch as th
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..models import build_model
from dgl.dataloading.negative_sampler import Uniform
from ..utils import extract_embed, EarlyStopping, get_nodes_dict, add_reverse_edges


@register_flow("link_prediction")
class LinkPrediction(BaseFlow):
    """
    Link Prediction trainer flows.
    Here is a tutorial teach you how to train a GNN for link prediction. <https://docs.dgl.ai/en/latest/tutorials/blitz/4_link_predict.html>_

    When training, you will need to remove the edges in the test set from the original graph.
    DGL recommends you to treat the pairs of nodes as another graph, since you can describe a pair of nodes with an edge.
    In link prediction, you will have a positive graph consisting of all the positive examples as edges,
    and a negative graph consisting of all the negative examples.
    The positive graph and the negative graph will contain the same set of nodes as the original graph.
    This makes it easier to pass node features among multiple graphs for computation.
    As you will see later, you can directly feed the node representations computed on the entire graph to the positive
    and the negative graphs for computing pair-wise scores.
    """

    def __init__(self, args):
        super(LinkPrediction, self).__init__(args)

        self.target_link = self.task.dataset.target_link
        self.loss_fn = self.task.get_loss_fn()

        self.args.out_node_type = self.task.dataset.out_ntypes
        self.args.out_dim = self.args.hidden_dim
        self.train_hg, self.val_hg, self.test_hg = self.task.get_idx()
        self.train_hg = add_reverse_edges(self.train_hg)
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.train_hg).to(self.device)
        self.args.score_fn = 'distmult'
        if self.args.score_fn == 'distmult':
            self.r_embedding = nn.ParameterDict({etype[1]: nn.Parameter(th.Tensor(1, self.args.out_dim))
                                                for etype in self.hg.canonical_etypes}).to(self.device)
            for _, para in self.r_embedding.items():
                nn.init.xavier_uniform_(para)

        self.evaluator = self.task.get_evaluator('roc_auc')
        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if self.args.score_fn == 'distmult':
            self.optimizer.add_param_group({'params': self.r_embedding.parameters()})
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.train_hg = self.train_hg.to(self.device)
        self.val_hg = self.val_hg.to(self.device)
        self.test_hg = self.test_hg.to(self.device)
        self.negative_sampler = Uniform(1)
        self.positive_graph = self.train_hg.edge_type_subgraph(self.target_link)

    def preprocess(self):
        """
        In link prediction, you will have a positive graph consisting of all the positive examples as edges,
        and a negative graph consisting of all the negative examples.
        The positive graph and the negative graph will contain the same set of nodes as the original graph.
        Returns
        -------

        """
        self.preprocess_feature()

    def train(self):
        self.preprocess()
        epoch_iter = tqdm(range(self.max_epoch))
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        for epoch in tqdm(range(self.max_epoch), ncols=80):
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
            if epoch % 2 == 0:
                val_metric = self._test_step('valid')
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, roc_auc: {val_metric:.4f}, Loss:{loss:.4f}"
                )
                early_stop = stopper.step_score(val_metric, self.model)
                if early_stop:
                    print('Early Stop!\tEpoch:' + str(epoch))
                    break
        print(f"Valid_score_ = {stopper.best_score: .4f}")
        stopper.load_model(self.model)

        ############ TEST SCORE #########
        if self.args.dataset in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']:
            self.model.eval()
            with torch.no_grad():
                val_metric = self._test_step('valid')
                h_dict = self.model.input_feature()
                embedding = self.model(self.hg, h_dict)
                score = th.sigmoid(self.ScorePredictor(self.test_hg, embedding))
                self.task.dataset.save_results(hg=self.test_hg, score=score, file_path=self.args.HGB_results_path)
            return val_metric, val_metric, epoch
        test_mrr = self._test_step(split="test")
        val_mrr = self._test_step(split="valid")
        print(f"Test mrr = {test_mrr:.4f}")
        return dict(Test_mrr=test_mrr, Val_mrr=val_mrr)

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
        if self.args.score_fn == 'dot-product':
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
        elif self.args.score_fn == 'distmult':
            score_list = []
            with edge_subgraph.local_scope():
                for ntype in edge_subgraph.ntypes:
                    edge_subgraph.nodes[ntype].data['x'] = x[ntype]
                for etype in edge_subgraph.canonical_etypes:
                    e = self.r_embedding[etype[1]]
                    n = edge_subgraph.num_edges(etype)
                    if 1 == len(edge_subgraph.canonical_etypes):
                        edge_subgraph.edata['e'] = e.expand(n, -1)
                    else:
                        edge_subgraph.edata['e'] = {etype: e.expand(n, -1)}
                    edge_subgraph.apply_edges(
                        dgl.function.u_mul_e('x', 'e', 's'), etype=etype)
                    edge_subgraph.apply_edges(
                        dgl.function.e_mul_v('s', 'x', 'score'), etype=etype)
                    if 1 == len(edge_subgraph.canonical_etypes):
                        score = th.sum(edge_subgraph.edata['score'], dim=1)
                    else:
                        score = th.sum(edge_subgraph.edata['score'].pop(etype), dim=1)
                    #score = th.sum(th.mul(edge_subgraph.edata['score'].pop(etype), e), dim=1)
                    score_list.append(score)
                return th.cat(score_list)

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
        h_dict = self.model.input_feature()
        embedding = self.model(self.train_hg, h_dict)
        negative_graph = self.construct_negative_graph(self.positive_graph)
        loss = self.loss_calculation(self.positive_graph, negative_graph, embedding)
        # negative_graph = self.construct_negative_graph(self.train_hg)
        # loss = self.loss_calculation(self.train_hg, negative_graph, embedding)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #print(loss.item())
        return loss.item()

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            if split == 'valid':
                eval_hg = self.val_hg
                # label = self.task.dataset.val_label
            elif split == 'test':
                label = self.task.dataset.test_label
                score = th.sigmoid(self.ScorePredictor(self.test_hg, embedding))
                metric = self.evaluator(label.cpu(), score.cpu())
                return metric

            # score = th.sigmoid(self.ScorePredictor(eval_hg, embedding))
            # metric = self.evaluator(label.cpu(), score.cpu())
            negative_graph = self.construct_negative_graph(eval_hg)
            p_score = th.sigmoid(self.ScorePredictor(eval_hg, embedding))
            n_score = th.sigmoid(self.ScorePredictor(negative_graph, embedding))
            p_label = th.ones(len(p_score))
            n_label = th.zeros(len(n_score))
        metric = self.evaluator(th.cat((p_label, n_label)).cpu(), th.cat((p_score, n_score)).cpu())

        return metric

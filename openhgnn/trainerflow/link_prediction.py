import dgl
import torch as th
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import extract_embed, EarlyStopping, get_nodes_dict, add_reverse_edges


@register_flow("link_prediction")
class LinkPrediction(BaseFlow):
    """
    Link Prediction trainer flows.
    Here is a tutorial teach you how to train a GNN for
    `link prediction <https://docs.dgl.ai/en/latest/tutorials/blitz/4_link_predict.html>_`.

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
        """

        Parameters
        ----------
        args

        Attributes
        ------------
        target_link: list
            list of edge types which are target link type to be predicted

        score_fn: str
            score function used in calculating the scores of links, supported function: distmult[Default if not specified] & dot product

        r_embedding: nn. ParameterDict
            In DistMult, the representations of edge types are involving the calculation of score.
            General models do not generate the representations of edge types, so we generate the embeddings of edge types.
            The dimension of embedding is `self.args.hidden_dim`.

        """
        super(LinkPrediction, self).__init__(args)
        
        self.target_link = self.task.dataset.target_link
        self.args.out_node_type = self.task.get_out_ntype()
        self.train_hg = self.task.get_train().to(self.device)
        if hasattr(self.args, 'flag_add_reverse_edges'):
            self.train_hg = add_reverse_edges(self.train_hg)
        if not hasattr(self.args, 'out_dim'):
            self.args.out_dim = self.args.hidden_dim

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.train_hg).to(self.device)

        if not hasattr(self.args, 'score_fn'):
            self.args.score_fn = 'distmult'
        if self.args.score_fn == 'distmult':
            """
            In DistMult, the representations of edge types are involving the calculation of score.
            General models do not generate the representations of edge types, so we generate the embeddings of edge types.
            """
            self.r_embedding = nn.ParameterDict({etype[1]: nn.Parameter(th.Tensor(1, self.args.out_dim))
                                                for etype in self.hg.canonical_etypes}).to(self.device)
            for _, para in self.r_embedding.items():
                nn.init.xavier_uniform_(para)
        else:
            self.r_embedding = None

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        if self.args.score_fn == 'distmult':
            self.optimizer.add_param_group({'params': self.r_embedding.parameters()})
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        
        self.positive_graph = self.train_hg.edge_type_subgraph(self.target_link)

    def preprocess(self):
        """
        In link prediction, you will have a positive graph consisting of all the positive examples as edges,
        and a negative graph consisting of all the negative examples.
        The positive graph and the negative graph will contain the same set of nodes as the original graph.
        """
        super(LinkPrediction, self).preprocess()

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.patience, self._checkpoint)
        for epoch in tqdm(range(self.max_epoch)):
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
            if epoch % self.evaluate_interval == 0:
                val_metric = self._test_step('valid')
                self.logger.train_info(f"Epoch: {epoch:03d}, train loss: {loss:.4f}. " + self.logger.metric2str(val_metric))
                early_stop = stopper.step_score(val_metric['valid']['loss'], self.model)
                if early_stop:
                    self.logger.train_info(f'Early Stop!\tEpoch:{epoch:03d}.')
                    break
        self.logger.train_info(f"Valid score = {stopper.best_score: .4f}")
        stopper.load_model(self.model)

        # Test
        if self.args.dataset in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']:
            # Test in HGB datasets.
            self.model.eval()
            with torch.no_grad():
                val_metric = self._test_step('valid')
                h_dict = self.model.input_feature()
                embedding = self.model(self.hg, h_dict)
                score = th.sigmoid(self.task.ScorePredictor(self.task.test_hg, embedding, self.r_embedding))
                self.task.dataset.save_results(hg=self.task.test_hg, score=score, file_path=self.args.HGB_results_path)
            return val_metric['valid'], epoch
        test_score = self._test_step(split="test")
        # val_score = self._test_step(split="valid")
        self.logger.train_info(self.logger.metric2str(test_score))
        return test_score

    def _full_train_setp(self):
        self.model.train()
        h_dict = self.model.input_feature()
        embedding = self.model(self.train_hg, h_dict)
        # construct a negative graph according to the positive graph in each training epoch.
        negative_graph = self.task.construct_negative_graph(self.positive_graph)
        loss = self.loss_calculation(self.positive_graph, negative_graph, embedding)
        # negative_graph = self.construct_negative_graph(self.train_hg)
        # loss = self.loss_calculation(self.train_hg, negative_graph, embedding)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

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
        p_score = self.task.ScorePredictor(positive_graph, embedding, self.r_embedding)
        n_score = self.task.ScorePredictor(negative_graph, embedding, self.r_embedding)

        p_label = th.ones(len(p_score), device=self.device)
        n_label = th.zeros(len(n_score), device=self.device)
        loss = F.binary_cross_entropy_with_logits(th.cat((p_score, n_score)), th.cat((p_label, n_label)))
        return loss

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.r_embedding.pow(2))

    def _test_step(self, split=None):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            return {split: self.task.evaluate(embedding, self.r_embedding, mode=split)}

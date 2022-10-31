from ..sampler import HANSampler
import dgl
import torch as th
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping, add_reverse_edges, get_ntypes_from_canonical_etypes


@register_flow("han_nc_trainer")
class HANNodeClassification(BaseFlow):
    r"""
    HAN node classification flow.
    """

    def __init__(self, args):

        super(HANNodeClassification, self).__init__(args)
        self.args.category = self.task.dataset.category
        self.category = self.args.category

        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()

        if self.args.prediction_flag:
            self.pred_idx = self.task.dataset.pred_idx

        self.labels = self.task.get_labels().to(self.device)

        if self.args.mini_batch_flag:
            sampler = HANSampler(g=self.hg, seed_ntypes=[self.category], meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=20)
            if self.train_idx is not None:
                self.train_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.train_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.valid_idx is not None:
                self.val_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.valid_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.test_flag:
                self.test_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.test_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.prediction_flag:
                self.pred_loader = dgl.dataloading.DataLoader(
                    self.hg, {self.category: self.pred_idx.to(self.device)}, sampler,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)

    def preprocess(self):

        super(HANNodeClassification, self).preprocess()

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                modes = ['train', 'valid']
                if self.args.test_flag:
                    modes = modes + ['test']
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=modes)
                    # train_score, train_loss = self._mini_test_step(modes='train')
                    # val_score, val_loss = self._mini_test_step(modes='valid')
                else:
                    metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                                       + self.logger.metric2str(metric_dict))
                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break

        stopper.load_model(self.model)
        if self.args.prediction_flag:
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                indices, y_predicts = self._mini_prediction_step()
            else:
                y_predicts = self._full_prediction_step()
                indices = torch.arange(self.hg.num_nodes(self.category))
            return indices, y_predicts

        if self.args.test_flag:
            if self.args.dataset[:4] == 'HGBn':
                # save results for HGBn
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, val_loss = self._mini_test_step(modes=['valid'])
                else:
                    metric_dict, val_loss = self._full_test_step(modes=['valid'])
                self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
                self.model.eval()
                with torch.no_grad():
                    h_dict = self.model.input_feature()
                    logits = self.model(self.hg, h_dict)[self.category]
                    self.task.dataset.save_results(logits=logits, file_path=self.args.HGB_results_path)
                return dict(metric=metric_dict, epoch=epoch)
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                metric_dict, _ = self._mini_test_step(modes=['valid', 'test'])
            else:
                metric_dict, _ = self._full_test_step(modes=['valid', 'test'])
            self.logger.train_info('[Test Info]' + self.logger.metric2str(metric_dict))
            return dict(metric=metric_dict, epoch=epoch)

    def _full_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        logits = self.model(self.hg, h_dict)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self, ):
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
            seeds = seeds[self.category]
            mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
            emb_dict = {}
            for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({self.category: input_nodes})
            emb_dict = {self.category: emb_dict}
            logits = self.model(ntype_mp_name_block_dict, emb_dict)[self.category]
            lbl = self.labels[seeds].to(self.device)
            loss = self.loss_fn(logits, lbl)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / (i + 1)

    def _full_test_step(self, modes, logits=None):
        """
        Parameters
        ----------
        mode: list[str]
            `train`, 'test', 'valid' are optional in list.
        logits: dict[str, th.Tensor]
            given logits, default `None`.

        Returns
        -------
        metric_dict: dict[str, float]
            score of evaluation metric
        info: dict[str, str]
            evaluation information
        loss: dict[str, float]
            the loss item
        """
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = logits if logits else self.model(self.hg, h_dict)[self.category]
            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idx
                elif mode == "valid":
                    masks[mode] = self.valid_idx
                elif mode == "test":
                    masks[mode] = self.test_idx

            metric_dict = {key: self.task.evaluate(logits, mode=key) for key in masks}
            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metric_dict, loss_dict

    def _mini_test_step(self, modes):
        self.model.eval()
        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.train_loader, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.val_loader, ncols=120)
                elif mode == 'test':
                    loader_tqdm = tqdm(self.test_loader, ncols=120)
                y_trues = []
                y_predicts = []

                for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
                    seeds = seeds[self.category]
                    mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
                    emb_dict = {}
                    for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                        emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({self.category: input_nodes})
                    emb_dict = {self.category: emb_dict}
                    logits = self.model(ntype_mp_name_block_dict, emb_dict)[self.category]
                    lbl = self.labels[seeds].to(self.device)
                    loss = self.loss_fn(logits, lbl)
                    loss_all += loss.item()
                    y_trues.append(lbl.detach().cpu())
                    y_predicts.append(logits.detach().cpu())
                loss_all /= (i + 1)
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                evaluator = self.task.get_evaluator(name='f1')
                metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                loss_dict[mode] = loss
        return metric_dict, loss_dict

    def _full_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_feature()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = self.model(self.hg, h_dict)[self.category]
            return logits

    def _mini_prediction_step(self):
        self.model.eval()
        with torch.no_grad():
            loader_tqdm = tqdm(self.pred_loader, ncols=120)
            indices = []
            y_predicts = []
            for i, (input_nodes_dict, seeds, block_dict) in enumerate(loader_tqdm):
                seeds = seeds[self.category]
                emb_dict = {}
                for meta_path_name, input_nodes in input_nodes_dict.items():
                    emb_dict[meta_path_name] = self.model.input_feature.forward_nodes(input_nodes)
                logits = self.model(block_dict, emb_dict)[self.category]
                indices.append(seeds.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            indices = torch.cat(indices, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        return indices, y_predicts


@register_flow("han_lp_trainer")
class HANLinkPrediction(BaseFlow):
    """
    HAN link prediction trainer flows.
    """

    def __init__(self, args):

        super(HANLinkPrediction, self).__init__(args)

        self.target_link = self.task.dataset.target_link
        self.args.out_node_type = self.task.get_out_ntype()
        self.train_hg = self.task.get_train()
        if hasattr(self.args, 'flag_add_reverse_edges') \
                or self.args.dataset in ['ohgbl-MTWM', 'ohgbl-yelp1', 'ohgbl-yelp2']:
            self.train_hg = add_reverse_edges(self.train_hg)
        if not hasattr(self.args, 'out_dim'):
            self.args.out_dim = self.args.hidden_dim

        self.args.target_link = self.task.dataset.target_link
        self.model = build_model(self.model).build_model_from_args(self.args, self.train_hg).to(self.device)

        if not hasattr(self.args, 'score_fn'):
            self.args.score_fn = 'distmult'
        if self.args.score_fn == 'distmult':
            """
            In DistMult, the representations of edge types are involving the calculation of score.
            General models do not generate the representations of edge types, so we generate the embeddings of edge types.
            """
            self.r_embedding = nn.ParameterDict({etype[1]: nn.Parameter(th.Tensor(1, self.args.out_dim))
                                                 for etype in self.train_hg.canonical_etypes}).to(self.device)
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

        self.train_hg = self.train_hg.to(self.device)
        self.positive_graph = self.train_hg.edge_type_subgraph(self.target_link).to(self.device)
        if self.args.mini_batch_flag:
            ntypes = get_ntypes_from_canonical_etypes(self.target_link)
            train_eid_dict = {
                etype: self.train_hg.edges(etype=etype, form='eid')
                for etype in self.target_link}
            sampler = HANSampler(g=self.hg, seed_ntypes=ntypes, meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=20)
            negative_sampler = dgl.dataloading.negative_sampler.Uniform(2)
            sampler = dgl.dataloading.as_edge_prediction_sampler(sampler=sampler, negative_sampler=negative_sampler)
            self.dataloader = dgl.dataloading.DataLoader(
                self.train_hg, train_eid_dict, sampler,
                batch_size=self.args.batch_size,
                shuffle=True)
            self.category = self.hg.ntypes[0]

    def preprocess(self):
        """
        In link prediction, you will have a positive graph consisting of all the positive examples as edges,
        and a negative graph consisting of all the negative examples.
        The positive graph and the negative graph will contain the same set of nodes as the original graph.
        """
        super(HANLinkPrediction, self).preprocess()

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
                self.logger.train_info(
                    f"Epoch: {epoch:03d}, train loss: {loss:.4f}. " + self.logger.metric2str(val_metric))
                early_stop = stopper.loss_step(val_metric['valid']['loss'], self.model)
                if early_stop:
                    self.logger.train_info(f'Early Stop!\tEpoch:{epoch:03d}.')
                    break
        stopper.load_model(self.model)
        # Test
        if self.args.test_flag:
            if self.args.dataset in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']:
                # Test in HGB datasets.
                self.model.eval()
                with torch.no_grad():
                    val_metric = self._test_step('valid')
                    self.logger.train_info(self.logger.metric2str(val_metric))
                    h_dict = self.model.input_feature()
                    embedding = self.model(self.train_hg, h_dict)
                    score = th.sigmoid(self.task.ScorePredictor(self.task.test_hg, embedding, self.r_embedding))
                    self.task.dataset.save_results(hg=self.task.test_hg, score=score,
                                                   file_path=self.args.HGB_results_path)
                return dict(metric=val_metric, epoch=epoch)
            else:
                test_score = self._test_step(split="test")
                self.logger.train_info(self.logger.metric2str(test_score))
                return dict(metric=test_score, epoch=epoch)
        elif self.args.prediction_flag:
            if self.args.mini_batch_flag:
                prediction_res = self._mini_prediction_step()
            else:
                prediction_res = self._full_prediction_step()
            return prediction_res

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
        # construct a negative graph according to the positive graph in each training epoch.
        negative_graph = self.task.construct_negative_graph(self.positive_graph)
        loss = self.loss_calculation(self.positive_graph, negative_graph, embedding)
        # negative_graph = self.construct_negative_graph(self.train_hg)
        # loss = self.loss_calculation(self.train_hg, negative_graph, embedding)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self, ):
        self.model.train()
        all_loss = 0
        loader_tqdm = tqdm(self.dataloader, ncols=120)
        for ntype_mp_name_input_nodes_dict, positive_graph, negative_graph, ntype_mp_name_block_dict in loader_tqdm:

            positive_graph = positive_graph.edge_type_subgraph(self.target_link).to(self.device)
            negative_graph = negative_graph.edge_type_subgraph(self.target_link).to(self.device)
            emb_dict = {}
            for ntype, mp_name_input_nodes_dict in ntype_mp_name_input_nodes_dict.items():
                mp_name_emb_dict = {}
                for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                    mp_name_emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({ntype: input_nodes})
                emb_dict[ntype] = mp_name_emb_dict
            logits = self.model(ntype_mp_name_block_dict, emb_dict)
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
        if self.args.mini_batch_flag:
            return self._mini_test_step(split=split)
        else:
            return self._full_test_step(split=split)

    def _full_test_step(self, split=None):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            return {split: self.task.evaluate(embedding, self.r_embedding, mode=split)}

    def _mini_test_step(self, split=None):
        print('mini test...\n')
        self.model.eval()
        with th.no_grad():
            ntypes = get_ntypes_from_canonical_etypes(self.target_link)
            embedding = self._mini_embedding(model=self.model, fanouts=[20], g=self.train_hg,
                                             device=self.args.device, dim=self.model.out_dim, ntypes=ntypes,
                                             batch_size=self.args.batch_size)
            return {split: self.task.evaluate(embedding, self.r_embedding, mode=split)}

    def _full_prediction_step(self):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            return self.task.predict(embedding, self.r_embedding)

    def _mini_prediction_step(self):
        self.model.eval()
        with th.no_grad():
            ntypes = get_ntypes_from_canonical_etypes(self.target_link)
            embedding = self._mini_embedding(model=self.model, fanouts=[-1] * self.args.num_layers, g=self.train_hg,
                                             device=self.args.device, dim=self.model.out_dim, ntypes=ntypes,
                                             batch_size=self.args.batch_size)
            return self.task.predict(embedding, self.r_embedding)

    def _mini_embedding(self, model, fanouts, g, device, dim, ntypes, batch_size):
        model.eval()
        with th.no_grad():
            sampler = HANSampler(g=self.hg, seed_ntypes=ntypes, meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=fanouts[0])
            indices = {ntype: torch.arange(g.num_nodes(ntype)).to(device) for ntype in ntypes}
            embedding = {ntype: torch.zeros(g.num_nodes(ntype), dim).to(device) for ntype in ntypes}
            dataloader = dgl.dataloading.DataLoader(
                g, indices, sampler,
                device=device,
                batch_size=batch_size)
            loader_tqdm = tqdm(dataloader, ncols=120)
            for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
                emb_dict = {}
                for ntype, mp_name_input_nodes_dict in ntype_mp_name_input_nodes_dict.items():
                    mp_name_emb_dict = {}
                    for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                        mp_name_emb_dict[meta_path_name] = self.model.input_feature.forward_nodes({ntype: input_nodes})
                    emb_dict[ntype] = mp_name_emb_dict
                output_emb = self.model(ntype_mp_name_block_dict, emb_dict)
                for ntype, idx in seeds.items():
                    embedding[ntype][idx] = output_emb[ntype]
            return embedding

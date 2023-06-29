import dgl
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ..utils.sampler import get_node_data_loader
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping, to_hetero_idx, to_homo_feature, to_homo_idx
import warnings
from torch.utils.tensorboard import SummaryWriter
import math
from ..models.DSSL_openhgnn import LogisticRegression

import numpy as np

@register_flow("node_classification")
class NodeClassification(BaseFlow):
    r"""
    Node classification flow,
    The task is to classify the nodes of target nodes.
    Note: If the output dim is not equal the number of classes, we will modify the output dim with the number of classes.
    """

    def __init__(self, args):
        """

        Attributes
        ------------
        category: str
            The target node type to predict
        num_classes: int
            The number of classes for category node type

        """

        super(NodeClassification, self).__init__(args)
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

        self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        self.pred_idx = getattr(self.task.dataset, 'pred_idx', None)

        self.labels = self.task.get_labels().to(self.device)
        self.num_nodes_dict = {ntype: self.hg.num_nodes(ntype) for ntype in self.hg.ntypes}
        self.to_homo_flag = getattr(self.model, 'to_homo_flag', False)
        self.writer = SummaryWriter(f'./openhgnn/output/{self.model_name}/')

        if self.to_homo_flag:
            self.g = dgl.to_homogeneous(self.hg)

        if self.args.mini_batch_flag:
            if not hasattr(args, 'fanout'):
                warnings.warn("please set fanout when using mini batch training.")
                args.fanout = -1
            if isinstance(args.fanout, list):
                self.fanouts = args.fanout
            else:
                self.fanouts = [args.fanout] * self.args.num_layers
            sampler = dgl.dataloading.MultiLayerNeighborSampler(self.fanouts)
            use_uva = self.args.use_uva

            if self.to_homo_flag:
                loader_g = self.g
            else:
                loader_g = self.hg

            if self.train_idx is not None:
                if self.to_homo_flag:
                    loader_train_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                   {self.category: self.train_idx}).to(self.device)
                else:
                    loader_train_idx = {self.category: self.train_idx.to(self.device)}

                self.train_loader = dgl.dataloading.DataLoader(loader_g, loader_train_idx, sampler,
                                                               batch_size=self.args.batch_size, device=self.device,
                                                               shuffle=True, use_uva=use_uva)
            if self.train_idx is not None:
                if self.to_homo_flag:
                    loader_val_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict, {self.category: self.val_idx}).to(
                        self.device)
                else:
                    loader_val_idx = {self.category: self.val_idx.to(self.device)}
                self.val_loader = dgl.dataloading.DataLoader(loader_g, loader_val_idx, sampler,
                                                             batch_size=self.args.batch_size, device=self.device,
                                                             shuffle=True, use_uva=use_uva)
            if self.args.test_flag:
                if self.test_idx is not None:
                    if self.to_homo_flag:
                        loader_test_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                      {self.category: self.test_idx}).to(self.device)
                    else:
                        loader_test_idx = {self.category: self.test_idx.to(self.device)}
                    self.test_loader = dgl.dataloading.DataLoader(loader_g, loader_test_idx, sampler,
                                                                  batch_size=self.args.batch_size, device=self.device,
                                                                  shuffle=True, use_uva=use_uva)
            if self.args.prediction_flag:
                if self.pred_idx is not None:
                    if self.to_homo_flag:
                        loader_pred_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
                                                      {self.category: self.pred_idx}).to(self.device)
                    else:
                        loader_pred_idx = {self.category: self.pred_idx.to(self.device)}
                    self.pred_loader = dgl.dataloading.DataLoader(loader_g, loader_pred_idx, sampler,
                                                                  batch_size=self.args.batch_size, device=self.device,
                                                                  shuffle=True, use_uva=use_uva)

    def preprocess(self):
        r"""
        Preprocess for different models, e.g.: different optimizer for GTN.
        And prepare the dataloader foe train validation and test.
        Last, we will call preprocess_feature.
        """
        if self.args.model == 'GTN':
            if hasattr(self.args, 'adaptive_lr_flag') and self.args.adaptive_lr_flag == True:
                self.optimizer = torch.optim.Adam([{'params': self.model.gcn.parameters()},
                                                   {'params': self.model.linear1.parameters()},
                                                   {'params': self.model.linear2.parameters()},
                                                   {"params": self.model.layers.parameters(), "lr": 0.5}
                                                   ], lr=0.005, weight_decay=0.001)
            else:
                # self.model = MLP_follow_model(self.model, args.out_dim, self.num_classes)
                pass
        elif self.args.model == 'MHNF':
            if hasattr(self.args, 'adaptive_lr_flag') and self.args.adaptive_lr_flag == True:
                self.optimizer = torch.optim.Adam([{'params': self.model.HSAF.HLHIA_layer.gcn_list.parameters()},
                                                   {'params': self.model.HSAF.channel_attention.parameters()},
                                                   {'params': self.model.HSAF.layers_attention.parameters()},
                                                   {'params': self.model.linear.parameters()},
                                                   {"params": self.model.HSAF.HLHIA_layer.layers.parameters(),
                                                    "lr": 0.5}
                                                   ], lr=0.005, weight_decay=0.001)

            else:
                # self.model = MLP_follow_model(self.model, args.out_dim, self.num_classes)
                pass
        elif self.args.model == 'RHGNN':
            print(f'get node data loader...')
            self.train_loader, self.val_loader, self.test_loader = get_node_data_loader(
                self.args.node_neighbors_min_num,
                self.args.num_layers,
                self.hg.to(self.device),
                batch_size=self.args.batch_size,
                sampled_node_type=self.category,
                train_idx=self.train_idx.to(self.device),
                valid_idx=self.val_idx.to(self.device),
                test_idx=self.test_idx.to(self.device),
                device=self.device)

        super(NodeClassification, self).preprocess()

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
                self.writer.add_scalars('loss', {'train': train_loss, 'valid': val_loss}, global_step=epoch)
                for mode in modes:
                    self.writer.add_scalars(f'metric_{mode}', metric_dict[mode], global_step=epoch)
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
        self.writer.close()

    def _full_train_step(self):
        self.model.train()
        h_dict = self.model.input_feature()
        self.hg = self.hg.to(self.device)
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
        for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
            if self.to_homo_flag:
                # input_nodes = to_hetero_idx(self.g, self.hg, input_nodes)
                seeds = to_hetero_idx(self.g, self.hg, seeds)
            elif isinstance(input_nodes, dict):
                for key in input_nodes:
                    input_nodes[key] = input_nodes[key].to(self.device)
            # elif not isinstance(input_nodes, dict):
            #     input_nodes = {self.category: input_nodes}
            emb = self.model.input_feature.forward_nodes(input_nodes)
            # if self.to_homo_flag:
            #     emb = to_homo_feature(self.hg.ntypes, emb)
            lbl = self.labels[seeds[self.category]].to(self.device)
            logits = self.model(blocks, emb)[self.category]
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
                    masks[mode] = self.val_idx
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
                for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                    if self.to_homo_flag:
                        # input_nodes = to_hetero_idx(self.g, self.hg, input_nodes)
                        seeds = to_hetero_idx(self.g, self.hg, seeds)
                    elif not isinstance(input_nodes, dict):
                        input_nodes = {self.category: input_nodes}
                    emb = self.model.input_feature.forward_nodes(input_nodes)
                    # if self.to_homo_flag:
                    #     emb = to_homo_feature(self.hg.ntypes, emb)
                    lbl = self.labels[seeds[self.category]].to(self.device)
                    logits = self.model(blocks, emb)[self.category]
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
        """

        Returns
        -------
        """
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
            for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                if self.to_homo_flag:
                    input_nodes = to_hetero_idx(self.g, self.hg, input_nodes)
                    seeds = to_hetero_idx(self.g, self.hg, seeds)
                elif not isinstance(input_nodes, dict):
                    input_nodes = {self.category: input_nodes}
                emb = self.model.input_feature.forward_nodes(input_nodes)
                if self.to_homo_flag:
                    emb = to_homo_feature(self.hg.ntypes, emb)
                logits = self.model(blocks, emb)[self.category]
                seeds = seeds[self.category]
                indices.append(seeds.detach().cpu())
                y_predicts.append(logits.detach().cpu())
            indices = torch.cat(indices, dim=0)
            y_predicts = torch.cat(y_predicts, dim=0)
        return indices, y_predicts

@register_flow("DSSL_trainerflow")
class DSSL_trainerflow(BaseFlow):

    def __init__(self,args = None):
        super(DSSL_trainerflow, self).__init__(args)            # build task(dataset..)

        self.args.category = self.task.dataset.category
        self.category = self.args.category

        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]
        self.args.feature_dim = self.hg.ndata['feat'].shape[1]
        # print(self.args.feature_dim)

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        if args.rand_split == True or args.dataset in ['snap-patents','ogbn-proteins', 'wiki','Cora', 'PubMed','genius']:
            self.train_idx, self.val_idx, self.test_idx = self.task.get_rand_split()
        else:
            self.train_idx, self.val_idx, self.test_idx = self.task.get_split()

        self.split_idx_lst = [{'train':self.train_idx,'valid':self.val_idx,'test':self.test_idx}]
        self.pred_idx = getattr(self.task.dataset, 'pred_idx', None)

        self.labels = self.task.get_labels().to(self.device)
        self.num_nodes_dict = {ntype: self.hg.num_nodes(ntype) for ntype in self.hg.ntypes}
        self.to_homo_flag = getattr(self.model, 'to_homo_flag', False)

        self.sampled_neighborhoods = self.sample_neighborhood(self.task.dataset, self.device, args)


        # print(self.sampled_neighborhoods)

        # if self.args.mini_batch_flag:
        #     self.fanouts = [args.fanout] * self.args.num_layers
        #     sampler = dgl.dataloading.MultiLayerNeighborSampler(self.fanouts)
        #     use_uva = self.args.use_uva
        #
        #     if self.to_homo_flag:
        #         loader_g = self.g
        #     else:
        #         loader_g = self.hg
        #
        #     if self.train_idx is not None:
        #         if self.to_homo_flag:
        #             loader_train_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
        #                                            {self.category: self.train_idx}).to(self.device)
        #         else:
        #             loader_train_idx = {self.category: self.train_idx.to(self.device)}
        #
        #         self.train_loader = dgl.dataloading.DataLoader(loader_g, loader_train_idx, sampler,
        #                                                        batch_size=self.args.batch_size, device=self.device,
        #                                                        shuffle=True, use_uva = use_uva)
        #     if self.train_idx is not None:
        #         if self.to_homo_flag:
        #             loader_val_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict, {self.category: self.val_idx}).to(
        #                 self.device)
        #         else:
        #             loader_val_idx = {self.category: self.val_idx.to(self.device)}
        #         self.val_loader = dgl.dataloading.DataLoader(loader_g, loader_val_idx, sampler,
        #                                                      batch_size=self.args.batch_size, device=self.device,
        #                                                      shuffle=True, use_uva = use_uva)
        #     if self.args.test_flag:
        #         if self.test_idx is not None:
        #             if self.to_homo_flag:
        #                 loader_test_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
        #                                               {self.category: self.test_idx}).to(self.device)
        #             else:
        #                 loader_test_idx = {self.category: self.test_idx.to(self.device)}
        #             self.test_loader = dgl.dataloading.DataLoader(loader_g, loader_test_idx, sampler,
        #                                                           batch_size=self.args.batch_size, device=self.device,
        #                                                           shuffle=True, use_uva = use_uva)
        #     if self.args.prediction_flag:
        #         if self.pred_idx is not None:
        #             if self.to_homo_flag:
        #                 loader_pred_idx = to_homo_idx(self.hg.ntypes, self.num_nodes_dict,
        #                                               {self.category: self.pred_idx}).to(self.device)
        #             else:
        #                 loader_pred_idx = {self.category: self.pred_idx.to(self.device)}
        #             self.pred_loader = dgl.dataloading.DataLoader(loader_g, loader_pred_idx, sampler,
        #                                                           batch_size=self.args.batch_size, device=self.device,
        #                                                           shuffle=True, use_uva = use_uva)


    def train(self):

        print('MODEL:', self.model)

        # print (split_idx_lst)
        import datetime

        time_now = datetime.datetime.now()
        print('start training')
        print(time_now)

        meanAcc = 0

        split_idx = self.split_idx_lst[0]
        self.model.reset_parameters()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        best_val = float('-inf')
        loss_lst = []
        best_loss = float('inf')

        for epoch in range(self.args.epochs):  # args.epochs = 200
            # pre-training
            self.model.train()
            batch_size = self.args.batch_size  # args.batch_size = 1024
            perm = torch.randperm(self.task.dataset.g.num_nodes())  # 随机打乱后获得的数字序列
            epoch_loss = 0
            for batch in range(0, self.task.dataset.g.num_nodes(), batch_size):
                optimizer.zero_grad()
                online_embedding = self.model.online_encoder(self.task.dataset)
                target_embedding = self.model.target_encoder(self.task.dataset)
                batch_idx = perm[batch:batch + batch_size]  # 从图中随机取batch_size个结点的id
                batch_idx = batch_idx.to(self.device)
                batch_neighbor_index = self.sampled_neighborhoods[batch_idx]  # 每个随机取的原结点取出有neighbor_max的邻居结点
                batch_embedding = online_embedding[batch_idx].to(self.device)  # 随机选的batch_size个结点的online_embedding
                batch_embedding = F.normalize(batch_embedding, dim=-1, p=2)  # 归一化 尺寸为   batch_size*dim=【1024，64】

                # target_embedding 中拿出随机原结点的邻居节点
                batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neighbor_index]
                batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(self.device)
                batch_neighbor_embedding = F.normalize(batch_neighbor_embedding, dim=-1, p=2)
                # batch_size*neighbor_max*dim=【1024，5，64】

                main_loss, context_loss, entropy_loss, k_node = self.model(batch_embedding, batch_neighbor_embedding)
                tmp = F.one_hot(torch.argmax(k_node, dim=1), num_classes=self.args.cluster_num).type(
                    torch.FloatTensor).to(self.device)
                batch_sum = (torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                if self.args.neg_alpha:  # args.neg_alpha = 0
                    batch_neg_neighbor_index = sampled_neg_neighborhoods[batch_idx]
                    batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in
                                                batch_neg_neighbor_index]
                    batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(device)
                    batch_neighbor_embedding = F.normalize(batch_neighbor_embedding, dim=-1, p=2)
                    main_neg_loss, tmp, tmp, tmp = model(batch_embedding, batch_neighbor_embedding)
                    loss = main_loss + args.gamma * (context_loss + entropy_loss) + main_neg_loss

                else:
                    loss = main_loss + self.args.gamma * (context_loss + entropy_loss)
                print("batch : {}, main_loss: {}, context_loss: {}, entropy_loss: {}".format(batch,
                                                                                             main_loss,
                                                                                             context_loss,
                                                                                             entropy_loss))
                loss.backward()
                optimizer.step()
                self.model.update_moving_average()
                epoch_loss = epoch_loss + loss
            if epoch % 1 == 0:
                self.model.eval()
                for batch in range(0, self.task.dataset.g.num_nodes(), batch_size):
                    online_embedding = self.model.online_encoder(self.task.dataset)
                    target_embedding = self.model.target_encoder(self.task.dataset)
                    batch_idx = perm[batch:batch + batch_size]
                    batch_idx = batch_idx.to(self.device)
                    batch_neighbor_index = self.sampled_neighborhoods[batch_idx]
                    batch_target_embedding = target_embedding[batch_idx].to(self.device)
                    batch_embedding = online_embedding[batch_idx].to(self.device)
                    batch_neighbor_embedding = [target_embedding[i, :].unsqueeze(0) for i in batch_neighbor_index]
                    batch_neighbor_embedding = torch.cat(batch_neighbor_embedding, dim=0).to(self.device)
                    main_loss, context_loss, entropy_loss, k_node = self.model(batch_embedding, batch_neighbor_embedding)
                    tmp = F.one_hot(torch.argmax(k_node, dim=1), num_classes=self.args.cluster_num).type(
                        torch.FloatTensor).to(
                        self.device)
                    if batch == 0:
                        cluster = torch.matmul(batch_embedding.t(), tmp)
                        batch_sum = (torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                    else:
                        cluster += torch.matmul(batch_embedding.t(), tmp)
                        batch_sum += (torch.reshape(torch.sum(tmp, 0), (-1, 1)))
                cluster = F.normalize(cluster, dim=-1, p=2)
                self.model.update_cluster(cluster, batch_sum)
            print("epoch: {}, loss: {}".format(epoch, epoch_loss))

            if epoch % 10 == 0:
                time_epoch = datetime.datetime.now()
                print("-------------------------------------------------------------------------------")
                print('cost time')
                timetmp = time_epoch - time_now
                print(timetmp)
                train_acc, dev_acc, test_acc = self.evaluate(self.model, self.task, split_idx)
                print("train acc: {}, val acc: {}, test acc: {}".format(train_acc, dev_acc, test_acc))
                print("-------------------------------------------------------------------------------")

                if best_val < dev_acc:
                    best_val = dev_acc
                    besttest_acc = test_acc

        meanAcc += besttest_acc
        print("meanAcc: {}".format(meanAcc))

    def _full_train_step(self):
        pass

    def _mini_train_step(self):
        pass

    def _full_test_step(self):
        pass

    def _mini_test_step(self):
        pass

    def sample_neighborhood(self,dataset, device, args):
        neighbor_max = args.neighbor_max  # args.neighbor_max = 5
        adjacent_list = {}
        # print(dataset.g)
        # print(dataset.g.num_nodes())
        # print(dataset.g.edges())

        undirect_g = dataset.g
        undirect_g = dgl.add_self_loop(undirect_g)
        undirect_g = dgl.to_bidirected(undirect_g)

        edge_index = undirect_g.edges()

        # adjacent_matrix = g.adj().to_dense()

        for i in range(0, dataset.g.num_nodes()):
            # adjacent_list[i] = torch.tensor(tmp,dtype=torch.int64).to(device)
            adjacent_list[i] = edge_index[1][edge_index[0] == i].to(device)  # 邻接矩阵



        for i in range(0, dataset.g.num_nodes()):
            num_neighbor = adjacent_list[i].shape[0]  # 第i个结点的邻居数量
            if num_neighbor < neighbor_max:
                more = math.ceil(neighbor_max / num_neighbor)
                adjacent_list[i] = adjacent_list[i].repeat(1, more)[0, :neighbor_max]  # 邻居数量不够使用重复填充
            if num_neighbor > neighbor_max:
                perm = torch.randperm(num_neighbor)
                idx = perm[:neighbor_max]
                adjacent_list[i] = adjacent_list[i][idx]  # 邻居数量超过使用在原num_neighbor里随机取neighbor_max个
        adjacent_list = list(adjacent_list.values())
        adjacent_list = torch.vstack(adjacent_list)
        return adjacent_list.to(device)

    def evaluate(self, model, task, split_idx):
        self.model.eval()
        embedding = self.model.online_encoder(task.dataset)
        embedding = embedding.detach()
        emb_dim, num_class = embedding.shape[1], self.num_classes
        train_accs, dev_accs, test_accs = [], [], []

        for i in range(10):
            train_idx = np.array(split_idx['train'].cpu())
            valid_idx = np.array(split_idx['valid'].cpu())
            test_idx = np.array(split_idx['test'].cpu())

            task.labels = task.labels.type(torch.LongTensor).to(self.device)
            task.labels = task.labels.unsqueeze(1)
            train_label = task.labels[train_idx]
            valid_label = task.labels[valid_idx]
            test_label = task.labels[test_idx]

            classifier = LogisticRegression(emb_dim, num_class).to(self.device)
            optimizer_LR = torch.optim.AdamW(classifier.parameters(), lr=0.01, weight_decay=0.01)

            for epoch in range(100):  # epoch（LR） = 100
                classifier.train()
                logits, loss = classifier(embedding[train_idx, :], train_label.squeeze())
                # print ("finetune epoch: {}, finetune loss: {}".format(epoch, loss))
                optimizer_LR.zero_grad()
                loss.backward()
                optimizer_LR.step()

            train_logits, _ = classifier(embedding[train_idx, :], train_label.squeeze())
            dev_logits, _ = classifier(embedding[valid_idx, :], valid_label.squeeze())
            test_logits, _ = classifier(embedding[test_idx, :], test_label.squeeze())
            train_preds = torch.argmax(train_logits, dim=1)
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            train_acc = (torch.sum(train_preds == train_label.squeeze()).float() / train_label.squeeze().shape[
                0]).detach().cpu().numpy()
            dev_acc = (torch.sum(dev_preds == valid_label.squeeze()).float() / valid_label.squeeze().shape[
                0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == test_label.squeeze()).float() /
                        test_label.squeeze().shape[0]).detach().cpu().numpy()

            train_accs.append(train_acc * 100)
            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        train_accs = np.stack(train_accs)
        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        train_acc, train_std = train_accs.mean(), train_accs.std()
        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        return train_acc, dev_acc, test_acc


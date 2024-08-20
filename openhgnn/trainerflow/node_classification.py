import dgl
import torch
from tqdm import tqdm
from ..utils.sampler import get_node_data_loader
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping, to_hetero_idx, to_homo_feature, to_homo_idx
import warnings
from torch.utils.tensorboard import SummaryWriter
import dgl.graphbolt as gb

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
        self.use_distributed = args.use_distributed
        if self.use_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device], output_device=self.device, find_unused_parameters=True
            )

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
                                                               shuffle=True, use_uva=use_uva, use_ddp=self.use_distributed)
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

        def create_loader(Item_set,graph):
            
            datapipe = gb.ItemSampler(Item_set, batch_size=self.args.batch_size, shuffle=True)
            datapipe = datapipe.copy_to(self.device)
            datapipe = datapipe.sample_neighbor(graph, self.fanouts)
            return gb.DataLoader(datapipe)
        

        if self.args.mini_batch_flag and self.args.graphbolt:
            
            
            dataset = gb.OnDiskDataset(self.task.dataset_GB.base_dir).load()
            graph = dataset.graph.to(self.device)
            # feature = dataset.feature.to(self.device)
            tasks = dataset.tasks
            nc_task = tasks[0]
            self.train_GB_loader = create_loader(nc_task.train_set, graph)
            self.val_GB_loader = create_loader(nc_task.validation_set, graph)
            self.test_GB_loader = create_loader(nc_task.test_set, graph)






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
                device=self.device,
                use_distributed=self.use_distributed)

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
        if self.args.graphbolt:
            self.model.train()
            loss_all = 0.0
            for i, data in enumerate(self.train_GB_loader):

                input_nodes = data.input_nodes
                seeds = data.seeds
                for key in input_nodes:
                    input_nodes[key] = input_nodes[key].to(self.device)
                emb = self.model.input_feature.forward_nodes(input_nodes)
                label = data.labels[self.category].to(self.device)
                logits = self.model(data.blocks, emb)[self.category]
                loss = self.loss_fn(logits, label)
                loss_all += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return loss_all / (i + 1)

        else:
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
        if self.args.graphbolt:
            self.model.eval()
            with torch.no_grad():
                metric_dict = {}
                loss_dict = {}
                loss_all = 0.0
               
                for mode in modes:  
                    if mode == 'train':
                        loader = self.train_GB_loader
                    elif mode == 'valid':
                        loader = self.val_GB_loader
                    elif mode == 'test':
                        loader = self.test_GB_loader
                    y_trues = []
                    y_predicts = []
                    for i, data in enumerate(loader):                        
                        input_nodes = data.input_nodes
                        seeds = data.seeds   
                        if not isinstance(input_nodes, dict):
                            input_nodes = {self.category: input_nodes}
                        emb = self.model.input_feature.forward_nodes(input_nodes)
                        label = data.labels[self.category].to(self.device)
                        logits = self.model(data.blocks, emb)[self.category]   
                        loss = self.loss_fn(logits, label)
                        loss_all += loss.item() 
                        
                        y_trues.append(label.detach().cpu())
                        y_predicts.append(logits.detach().cpu())
                    loss_all /= (i + 1)
                    y_trues = torch.cat(y_trues, dim=0)
                    y_predicts = torch.cat(y_predicts, dim=0)
                    evaluator = self.task.get_evaluator(name='f1')
                    
                    metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                    loss_dict[mode] = loss_all
            return metric_dict, loss_dict

       
        else:
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

import dgl
import torch
from tqdm import tqdm
from ..utils.sampler import get_node_data_loader
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils.logger import printInfo, printMetric
from ..utils import extract_embed, EarlyStopping


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

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg).to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.train_idx, self.valid_idx, self.test_idx = self.task.get_split()
        self.labels = self.task.get_labels().to(self.device)

        if self.args.mini_batch_flag:
            # sampler = dgl.dataloading.MultiLayerNeighborSampler([self.args.fanout] * self.args.n_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.n_layers)
            self.train_loader = dgl.dataloading.NodeDataLoader(
                self.hg.cpu(), {self.category: self.train_idx.cpu()}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=8)
            self.val_loader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {self.category: self.valid_idx.to('cpu')}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=4)
            self.test_loader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {self.category: self.test_idx.to('cpu')}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=0)

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
                                                   {"params": self.model.HSAF.HLHIA_layer.layers.parameters(), "lr": 0.5}
                                                   ], lr=0.005, weight_decay=0.001)

            else:
                # self.model = MLP_follow_model(self.model, args.out_dim, self.num_classes)
                pass
        elif self.args.model == 'RHGNN':
            print(f'get node data loader...')
            self.train_loader, self.val_loader, self.test_loader = get_node_data_loader(self.args.node_neighbors_min_num,
                                                                         self.args.n_layers,
                                                                         self.hg.to('cpu'),
                                                                         batch_size=self.args.batch_size,
                                                                         sampled_node_type=self.category,
                                                                         train_idx=self.train_idx, valid_idx=self.valid_idx,
                                                                         test_idx=self.test_idx)

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
                if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=['train', 'valid', 'test'])
                    # train_score, train_loss = self._mini_test_step(modes='train')
                    # val_score, val_loss = self._mini_test_step(modes='valid')
                else:
                    metric_dict, losses = self._full_test_step(modes=['train', 'valid', 'test'])
                val_loss = losses['valid']
                self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                                       + self.logger.metric2str(metric_dict))
                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                    break

        stopper.load_model(self.model)
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

    def _mini_train_step(self,):
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
            blocks = [blk.to(self.device) for blk in blocks]
            seeds = seeds[self.category]  # out_nodes, we only predict the nodes with type "category"
            # batch_tic = time.time()
            emb = extract_embed(self.model.input_feature(), input_nodes)
            lbl = self.labels[seeds].to(self.device)
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
                for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                    blocks = [blk.to(self.device) for blk in blocks]
                    emb = extract_embed(self.model.input_feature(), input_nodes)
                    seeds = seeds[self.category]
                    lbl = self.labels[seeds].to(self.device)
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
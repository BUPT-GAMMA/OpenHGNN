import dgl
import torch
from tqdm import tqdm
from ..models import build_model
from ..layers.EmbedLayer import HeteroEmbedLayer
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, EarlyStopping, get_nodes_dict


@register_flow("entity_classification")
class EntityClassification(BaseFlow):
    r"""Node classification flows.
    Supported Model: RGCN/CompGCN/RSHN
    Supported Dataset：AIFB/MUTAG/BGS/AM
        Dataset description can be found in https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero
    The task is to classify the entity.
    """

    def __init__(self, args):
        super(EntityClassification, self).__init__(args)

        self.num_classes = self.task.dataset.num_classes

        # Build the model. If the output dim is not equal the number of classes, modify the dim.
        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            print('Modify the out_dim with num_classes')
            self.args.out_dim = self.num_classes

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.category = self.task.dataset.category
        self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
        self.labels = self.task.get_labels().to(self.device)
        if self.args.mini_batch_flag:
            self.sampler = dgl.dataloading.MultiLayerNeighborSampler([self.args.fanout] * self.args.n_layers)
            #self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.n_layers)
            self.train_loader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {self.category: self.train_idx.to('cpu')}, self.sampler,
                batch_size=self.args.batch_size, shuffle=True, num_workers=4
            )
            self.test_loader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {self.category: self.test_idx.to('cpu')}, self.sampler,
                batch_size=self.args.batch_size, shuffle=False, num_workers=0, drop_last=False,
            )

    def preprocess(self):
        self.preprocess_feature()
        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_step()
            if (epoch + 1) % self.evaluate_interval == 0:
                acc, losses = self._test_step()
                train_acc = acc["train"]
                val_acc = acc["val"]
                val_loss = losses["val"]
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Loss:{loss: .4f}, Train_acc: {train_acc:.4f}, Val_acc: {val_acc:.4f}, Val_loss: {val_loss:.4f}"
                )
                early_stop = stopper.loss_step(val_loss, self.model)
                if early_stop:
                    print('Early Stop!\tEpoch:' + str(epoch))
                    break

        print(f"Valid loss = {stopper.best_loss: .4f}")
        stopper.load_model(self.model)
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="val")
        print(f"Valid accuracy = {val_acc:.4f}, Test accuracy = {test_acc:.4f}")
        if self.args.dataset[:4] == 'HGBn':
            self.model.eval()
            with torch.no_grad():
                h_dict = self.input_feature()
                logits = self.model(self.hg, h_dict)[self.category]
                self.task.dataset.save_results(logits=logits, file_path=self.args.HGB_results_path)
            return
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _full_train_step(self):
        self.model.train()
        h_dict = self.input_feature()
        logits = self.model(self.hg, h_dict)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self):
        self.model.train()
        loss_all = 0
        for i, (input_nodes, seeds, blocks) in enumerate(self.train_loader):
            n = i + 1
            blocks = [blk.to(self.device) for blk in blocks]
            seeds = seeds[self.category]  # out_nodes, we only predict the nodes with type "category"
            # batch_tic = time.time()
            lbl = self.labels[seeds].to(self.device).squeeze()

            h = self.input_feature.forward_nodes(input_nodes)
            logits = self.model(blocks, h)[self.category]
            loss = self.loss_fn(logits, lbl)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / n

    def _mini_test(self, split=None, logits=None):
        self.model.eval()
        if split == "train":
            mask = self.train_idx
        elif split == "val":
            mask = self.val_idx
        elif split == "test":
            mask = self.test_idx
        else:
            mask = None
        if mask is not None:
            dataloader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {self.category: mask.to('cpu')}, self.sampler,
                batch_size=self.args.batch_size, shuffle=False, num_workers=0, drop_last=False,
            )
            preds = self._forward_model(dataloader)
            acc = self.task.evaluate(preds, 'acc-ogbn-mag')
        else:
            masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
            acc = {}
            for key, mask in masks:
                dataloader = dgl.dataloading.NodeDataLoader(
                    self.hg.to('cpu'), {self.category: mask.to('cpu')}, self.sampler,
                    batch_size=self.args.batch_size, shuffle=False, num_workers=0, drop_last=False,
                )
                preds = self._forward_model(dataloader)
                acc[key] = self.task.evaluate(preds, 'acc-ogbn-mag')
        return acc, 0

    def _forward_model(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            preds = []
            for i, (input_nodes, seeds, blocks) in enumerate(dataloader):
                blocks = [blk.to(self.device) for blk in blocks]
                # batch_tic = time.time()
                if self.has_feature:
                    h = blocks[0].srcdata['h']
                else:
                    h = self.input_feature.forward_nodes(input_nodes)
                preds.append(self.model(blocks, h)[self.category].argmax(dim=1).to('cpu'))
            preds = torch.cat(preds, dim=0)
        return preds

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.input_feature()
            logits = logits if logits else self.model(self.hg, h_dict)[self.category]
        if split == "train":
            mask = self.train_idx
        elif split == "val":
            mask = self.val_idx
        elif split == "test":
            mask = self.test_idx
        else:
            mask = None

        if mask is not None:
            loss = self.loss_fn(logits[mask], self.labels[mask]).item()
            metric = self.task.evaluate(logits[mask].argmax(dim=1).to('cpu'), 'acc', mask)
            return metric, loss
        else:
            masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
            metrics = {key: self.task.evaluate(logits[mask].argmax(dim=1).to('cpu'), 'acc', mask) for key, mask in
                       masks.items()}
            losses = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metrics, losses

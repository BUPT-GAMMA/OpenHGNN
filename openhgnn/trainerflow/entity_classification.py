import dgl
import torch
from tqdm import tqdm
from openhgnn.models import build_model

from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, EarlyStopping


@register_flow("entity_classification")
class EntityClassification(BaseFlow):

    """Node classification flows.
    Supported Model: RGCN/CompGCN/RSHN
    Supported Dataset：AIFB/MUTAG/BGS/AM
        Dataset description can be found in https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero
    The task is to classify the entity.

    Note: If the output dim is not equal the number of classes, a MLP will follow the gnn model.
    """

    def __init__(self, args):
        super(EntityClassification, self).__init__(args)

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.hg = self.task.get_graph().to(self.device)
        self.num_classes = self.task.dataset.num_classes

        if hasattr(self.task.dataset, 'in_dim'):
            self.args.in_dim = self.task.dataset.in_dim
        # Build the model. If the output dim is not equal the number of classes, modify the dim.
        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            print('Modify the out_dim with num_classes')
            self.args.out_dim = self.num_classes
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

        self.loss_fn = self.task.get_loss_fn()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.category = self.task.dataset.category
        self.train_idx, self.val_idx, self.test_idx = self.task.get_idx()
        self.labels = self.task.get_labels().to(self.device)
        if self.args.mini_batch_flag:
            # sampler = dgl.dataloading.MultiLayerNeighborSampler([self.args.fanout] * self.args.n_layers)
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.n_layers)
            self.loader = dgl.dataloading.NodeDataLoader(
                self.hg.to('cpu'), {self.category: self.train_idx.to('cpu')}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True, num_workers=0)

    def preprocess(self):
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
                print(f'Test_acc:{acc["test"]:.4f}')
                # print(
                #     f"Epoch: {epoch:03d}, Loss:{loss: .4f}, Train_acc: {train_acc:.4f}, Val_acc: {val_acc:.4f}, Val_loss: {val_loss:.4f}"
                # )
                early_stop = stopper.step(val_loss, val_acc, self.model)
                if early_stop:
                    print('Early Stop!\tEpoch:' + str(epoch))
                    break

        print(f"Valid accurracy = {stopper.best_score: .4f}")
        stopper.load_model(self.model)
        test_acc, _ = self._test_step(split="test")
        val_acc, _ = self._test_step(split="val")
        print(f"Test accuracy = {test_acc:.4f}")
        return dict(Acc=test_acc, ValAcc=val_acc)

    def _full_train_step(self):
        logits = self.model(self.hg)[self.category]
        loss = self.loss_fn(logits[self.train_idx], self.labels[self.train_idx])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self,):

        loss_all = 0
        for i, (input_nodes, seeds, blocks) in enumerate(self.loader):
            n = i +1
            blocks = [blk.to(self.device) for blk in blocks]
            seeds = seeds[self.category]  # out_nodes, we only predict the nodes with type "category"
            # batch_tic = time.time()
            emb = extract_embed(self.model.embed_layer(), input_nodes)
            lbl = self.labels[seeds].to(self.device)

            logits = self.model(blocks, emb)[self.category]
            loss = self.loss_fn(logits, lbl)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / n

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        with torch.no_grad():
            logits = logits if logits else self.model(self.hg)[self.category]
        if split == "train":
            mask = self.train_idx
        elif split == "val":
            mask = self.val_idx
        elif split == "test":
            mask = self.test_idx
        else:
            mask = None

        if mask is not None:
            loss = self.loss_fn(logits[mask], self.labels[mask])
            metric = self.task.evaluate(logits[mask].argmax(dim=1).to('cpu'), 'acc', mask)
            return metric, loss
        else:
            masks = {'train': self.train_idx, 'val': self.val_idx, 'test': self.test_idx}
            metrics = {key: self.task.evaluate(logits[mask].argmax(dim=1).to('cpu'), 'acc', mask) for key, mask in masks.items()}
            losses = {key: self.loss_fn(logits[mask], self.labels[mask]) for key, mask in masks.items()}
            return metrics, losses



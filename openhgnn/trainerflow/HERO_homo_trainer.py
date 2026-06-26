import torch
from tqdm import tqdm

from openhgnn.models import build_model
from openhgnn.tasks import build_task
from openhgnn.trainerflow import BaseFlow, register_flow

from ..models.HERO_homo import HEROHomoEvaluation


@register_flow("hero_homo_trainer")
class HEROHomoTrainer(BaseFlow):
    def __init__(self, args=None):
        super(HEROHomoTrainer, self).__init__(args)

        self.task = build_task(args)
        self.dataset_wrapper = self.task.dataset
        dataset = self.dataset_wrapper.dataset

        if not hasattr(dataset, "graph"):
            raise ValueError("Current dataset is not a HERO homogeneous dataset.")

        self.graph = dataset.graph.to(self.device)
        self.features = [x.to(self.device) for x in dataset.features]
        self.feature_distance = dataset.feature_distance.to(self.device)
        self.labels = dataset.labels.to(self.device)

        self.train_idx = dataset.train_idx.to(self.device)
        self.val_idx = dataset.val_idx.to(self.device)
        self.test_idx = dataset.test_idx.to(self.device)

        self.args.ft_size = self.features[0].shape[1]
        self.args.node_num = self.graph.number_of_nodes()

        self.model = build_model("HERO_homo").build_model_from_args(self.args).to(self.device)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr
        )

        self.best_epoch = -1
        self.best_loss = float("inf")
        self.best_state = None
        self.wait = 0
        self.patience = self.args.patience

    def train(self):
        for epoch in tqdm(range(self.max_epoch)):
            train_loss = self._full_train_step()

            self.logger.train_info(
                f"Epoch: {epoch:03d} | train_loss: {train_loss:.6f}"
            )

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.best_epoch = epoch
                self.wait = 0
                self.best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.wait += 1

            if self.wait >= self.patience:
                self.logger.train_info("Early Stop!")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        final_metric = self._full_test_step()
        self.logger.train_info(
            f"[Final] best_epoch={self.best_epoch}, "
            f"f1_micro={final_metric['f1_micro']:.5f}, "
            f"f1_macro={final_metric['f1_macro']:.5f}"
        )
        return final_metric

    def _full_train_step(self):
        self.model.train()

        loss_dict = self.model.compute_loss(
            self.graph, self.features, self.feature_distance
        )
        loss = loss_dict["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _full_test_step(self):
        self.model.eval()

        concat_emb = self.model.get_concat_embeddings(
            self.graph, self.features, self.feature_distance
        )
        concat_emb = concat_emb.detach().cpu().numpy()

        evaluator = HEROHomoEvaluation(
            concat_emb,
            self.labels.detach().cpu().numpy(),
            self.train_idx.detach().cpu().numpy(),
            self.val_idx.detach().cpu().numpy(),
            self.test_idx.detach().cpu().numpy(),
        )
        f1_micro, f1_macro = evaluator.evaluate()

        return {
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
        }
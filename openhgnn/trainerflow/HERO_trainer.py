import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from openhgnn.models import build_model
from openhgnn.tasks import build_task
from openhgnn.trainerflow import BaseFlow, register_flow

from ..models.HERO import HEROEvaluation


@register_flow("hero_trainer")
class HEROTrainer(BaseFlow):
 
    def __init__(self, args=None):
        super(HEROTrainer, self).__init__(args)

        # ===== 1. task / graph / labels =====
        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)

        self.category = getattr(self.args, "category", "paper")
        self.labels = self.task.get_labels().to(self.device)

        # ===== 2. split =====
        try:
            self.train_idx, self.val_idx, self.test_idx = self.task.get_split()
        except Exception:
            node_data = self.hg.nodes[self.category].data
            self.train_idx = torch.nonzero(node_data["train_mask"], as_tuple=True)[0]
            self.val_idx = torch.nonzero(node_data["val_mask"], as_tuple=True)[0]
            self.test_idx = torch.nonzero(node_data["test_mask"], as_tuple=True)[0]

        self.train_idx = self.train_idx.to(self.device)
        self.val_idx = self.val_idx.to(self.device)
        self.test_idx = self.test_idx.to(self.device)

        #  model 
        self.model = build_model("HERO").build_model_from_args(
            self.args, self.hg
        ).to(self.device)

      
        self.args.batch_size = 1

        self.g_proj = nn.Sequential(
            nn.Linear(self.args.out_ft, self.args.g_dim, bias=False),
            nn.ReLU(inplace=True)
        ).to(self.device)

        self.g_1 = nn.Sequential(
            nn.Linear(self.args.out_ft, self.args.g_equidim, bias=False),
            nn.ReLU(inplace=True)
        ).to(self.device)

        self.p_1 = nn.Sequential(
            nn.Linear(self.args.g_equidim, self.args.p_equidim, bias=False),
            nn.ReLU(inplace=True)
        ).to(self.device)

        # ===== 5. optimizer：严格只优化主模型 =====
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr
        )

        # ===== 6. early stopping =====
        self.best_epoch = -1
        self.best_loss = float("inf")
        self.best_state = None
        self.wait = 0
        self.patience = self.args.patience

    def _compute_loss(self):
        out = self.model(self.hg)
        emb_het = out["emb_het"]
        emb_hom = out["emb_hom"]

        # ==========================================================
        # consistency branch
        # ==========================================================
        embs_P1 = self.g_proj(emb_het)
        embs_P2 = self.g_proj(emb_hom)

        # The second term in Eq. (10): uniformity loss
        intra_c = embs_P1.T @ embs_P1.contiguous()
        intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
        loss_uni = torch.log(intra_c).mean()

        intra_c_2 = embs_P2.T @ embs_P2.contiguous()
        intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
        loss_uni += torch.log(intra_c_2).mean()

        # The first term in Eq. (10): invariance loss
        inter_c = embs_P1.T @ embs_P2
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_inv = -torch.diagonal(inter_c).sum()

        # ==========================================================
        # specificity branch
        # ==========================================================
        embs_Q2 = self.g_1(emb_het)
        embs_Q1 = self.g_1(emb_hom)
        embs_Q1_trans = self.p_1(embs_Q1)

        # The first term in Eq. (12)
        inter_c = embs_Q1_trans.T @ embs_Q2
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_spe_inv = -torch.diagonal(inter_c).sum()

        # The second term in Eq. (12)
        inter_c = embs_Q1_trans.T @ embs_Q1
        inter_c = F.normalize(inter_c, p=2, dim=1)
        loss_spe_nontrival_1 = torch.diagonal(inter_c).sum()

        # The third term in Eq. (12)
        inter_c_1 = embs_Q1_trans.T @ embs_P2
        inter_c_1 = F.normalize(inter_c_1, p=2, dim=1)
        loss_spe_nontrival_2 = torch.diagonal(inter_c_1).sum()

        # ==========================================================
        # final objective
        # ==========================================================
        loss_consistency = loss_inv + self.args.gamma * loss_uni
        loss_specificity = loss_spe_inv + self.args.eta * (
            loss_spe_nontrival_1 + loss_spe_nontrival_2
        )
        loss = loss_consistency + self.args.lambbda * loss_specificity

        return {
            "loss": loss,
            "loss_consistency": loss_consistency.detach(),
            "loss_specificity": loss_specificity.detach(),
            "loss_inv": loss_inv.detach(),
            "loss_uni": loss_uni.detach(),
            "loss_spe_inv": loss_spe_inv.detach(),
            "loss_spe_nontrival_1": loss_spe_nontrival_1.detach(),
            "loss_spe_nontrival_2": loss_spe_nontrival_2.detach(),
        }

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
        self.g_proj.train()
        self.g_1.train()
        self.p_1.train()

        loss_dict = self._compute_loss()
        loss = loss_dict["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _full_test_step(self):
        self.model.eval()

        concat_emb = self.model.get_concat_embeddings(self.hg)
        concat_emb = concat_emb.detach().cpu().numpy()

        labels = self.labels.detach().cpu().numpy()
        train_idx = self.train_idx.detach().cpu().numpy()
        val_idx = self.val_idx.detach().cpu().numpy()
        test_idx = self.test_idx.detach().cpu().numpy()

        evaluator = HEROEvaluation(
            concat_emb,
            labels,
            self.args,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx
        )
        f1_micro, f1_macro = evaluator.evaluate()

        return {
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
        }
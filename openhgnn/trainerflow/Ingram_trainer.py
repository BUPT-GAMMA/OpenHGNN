from scipy.sparse import csr_matrix
import numpy as np
import torch
import dgl
from tqdm import tqdm
import random
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
import os
from ..utils import Ingram_utils


def evaluate(my_model, target, epoch, init_emb_ent, init_emb_rel, relation_triplets):
    with torch.no_grad():
        my_model.eval()

        msg = torch.tensor(target.msg_triplets).cuda()
        sup = torch.tensor(target.sup_triplets).cuda()

        emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)

        head_ranks = []
        tail_ranks = []
        ranks = []
        for triplet in tqdm(sup):
            triplet = triplet.unsqueeze(dim=0)
            head_corrupt = triplet.repeat(target.num_ent, 3)
            head_corrupt[:, 0] = torch.arange(end=target.num_ent)
            head_scores = my_model.score(emb_ent, emb_rel, head_corrupt)
            head_filters = target.filter_dict[('_', int(triplet[0, 1].item()), int(triplet[0, 2].item()))]
            head_rank = Ingram_utils.get_rank(triplet, head_scores, head_filters, target=0)
            tail_corrupt = triplet.repeat(target.num_ent, 3)
            tail_corrupt[:, 2] = torch.arange(end=target.num_ent)
            tail_scores = my_model.score(emb_ent, emb_rel, tail_corrupt)
            tail_filters = target.filter_dict[(int(triplet[0, 0].item()), int(triplet[0, 1].item()), '_')]
            tail_rank = Ingram_utils.get_rank(triplet, tail_scores, tail_filters, target=2)
            ranks.append(head_rank)
            head_ranks.append(head_rank)
            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)

        print("--------LP--------")
        mr, mrr, hit10, hit3, hit1 = Ingram_utils.get_metrics(ranks)
        print(f"MR: {mr:.1f}")
        print(f"MRR: {mrr:.3f}")
        print(f"Hits@10: {hit10:.3f}")
        print(f"Hits@1: {hit1:.3f}")


@register_flow("Ingram_trainer")
class Ingram_Trainer(BaseFlow):
    """ingram flows."""
    OMP_NUM_THREADS = 8
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)
    torch.cuda.empty_cache()

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.output_dir = args.output_dir
        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)
        self.model = build_model(self.model_name).build_model_from_args(
            self.args).model
        print("build_model_finish")
        torch.cuda.set_device(args.device)
        self.model = self.model.to(self.device)
        self.loss_fn = torch.nn.MarginRankingLoss(margin=args.margin, reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.validation_epoch = args.validation_epoch
        self.num_epoch = args.num_epoch
        self.num_neg = args.num_neg
        self.num_bin = args.num_bin
        self.d_e = args.d_e
        self.d_r = args.d_r

    def train(self):
        my_model = self.model
        train = self.task.train_dataloader
        valid = self.task.valid_dataloader
        test = self.task.test_dataloader
        pbar = tqdm(range(self.num_epoch))
        valid_epochs = self.args.validation_epoch
        total_loss = 0
        file_format = f"lr_{self.args.lr}_dim_{self.args.d_e}_{self.args.d_r}" + \
                      f"_bin_{self.args.num_bin}_total_{self.args.num_epoch}_every_{self.args.validation_epoch}" + \
                      f"_neg_{self.args.num_neg}_layer_{self.args.nle}_{self.args.nlr}" + \
                      f"_hid_{self.args.hdr_e}_{self.args.hdr_r}" + \
                      f"_head_{self.args.num_head}_margin_{self.args.margin}"
        for epoch in pbar:
            self.optimizer.zero_grad()
            msg, sup = train.split_transductive(0.75)
            init_emb_ent, init_emb_rel, relation_triplets = Ingram_utils.initialize(train, msg, self.d_e, self.d_r,
                                                                                    self.num_bin)
            msg = torch.tensor(msg).cuda()
            sup = torch.tensor(sup).cuda()
            emb_ent, emb_rel = my_model(init_emb_ent, init_emb_rel, msg, relation_triplets)
            pos_scores = my_model.score(emb_ent, emb_rel, sup)
            neg_scores = my_model.score(emb_ent, emb_rel,
                                        Ingram_utils.generate_neg(sup, train.num_ent, num_neg=self.num_neg))
            loss = self.loss_fn(pos_scores.repeat(self.num_neg), neg_scores, torch.ones_like(neg_scores))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), 0.1, error_if_nonfinite=False)
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_description(f"loss {loss.item()}")

            if ((epoch + 1) % valid_epochs) == 0:
                print("Validation")
                my_model.eval()
                val_init_emb_ent, val_init_emb_rel, val_relation_triplets = Ingram_utils.initialize(valid,
                                                                                                    valid.msg_triplets, \
                                                                                                    self.d_e, self.d_r,
                                                                                                    self.num_bin)

                evaluate(my_model, valid, epoch, val_init_emb_ent, val_init_emb_rel, val_relation_triplets)
                path_ckp = self.output_dir
                print(path_ckp)
                folder = os.path.exists(path_ckp)

                if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                    os.makedirs(path_ckp)  # makedirs 创建文件时如果路径不存在会创建这个路径
                    print("---  new folder...  ---")

                else:
                    print("---  There is this folder!  ---")
                torch.save({'model_state_dict': my_model.state_dict(), \
                            'optimizer_state_dict': self.optimizer.state_dict(), \
                            'inf_emb_ent': val_init_emb_ent, \
                            'inf_emb_rel': val_init_emb_rel}, \
                           path_ckp + f"/{file_format}_{epoch + 1}.ckpt")

                my_model.train()

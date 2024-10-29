import torch
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
import random
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# python main.py -m MetaHIN -t coldstart_recommendation -d dbook -g 0

@register_flow("coldstart_recommendation")
class coldstart_recommendation(BaseFlow):
    def __init__(self, args):
        super(coldstart_recommendation, self).__init__(args)
        self.datahelper = self.task.dataloader
        self.states = [
            "meta_training",
            "user_cold_testing",
            "item_cold_testing",
            "user_and_item_cold_testing",
            "warm_up",
        ]
        self.data_set = "dbook"
        self.model = (
            build_model(self.model).build_model_from_args(self.args).to(self.device)
        )
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.device = "cuda"

    def preprocess(self):
        train_data = self.datahelper.load_data(
            data_set=self.data_set, state="meta_training", load_from_file=True
        )
        return train_data

    def full_test_step(self, model, device="cpu"):
        print("evaluating model...")
        if self.device != "cpu":
            model.cuda()
        model.eval()
        for state in self.states:
            if state == "meta_training":
                continue
            print(state + "...")
            test_data = self.datahelper.load_data(
                data_set=self.data_set, state=state, load_from_file=True
            )
            supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(
                *test_data
            )  # supp_um_s:(list,list,...,2553)
            loss, mae, rmse = [], [], []
            ndcg_at_5 = []

            for i in range(len(test_data)):  # each task
                _mae, _rmse, _ndcg_5 = model.evaluation(
                    supp_xs_s[i],
                    supp_ys_s[i],
                    supp_mps_s[i],
                    query_xs_s[i],
                    query_ys_s[i],
                    query_mps_s[i],
                    device,
                )
                mae.append(_mae)
                rmse.append(_rmse)
                ndcg_at_5.append(_ndcg_5)
            print(
                " ndcg@5: {:.5f}".format(
                    np.mean(mae), np.mean(rmse), np.mean(ndcg_at_5)
                )
            )

    def train(self):
        train_data = self.preprocess()
        num_epoch = self.num_epoch
        for i in range(num_epoch):  # 20
            self.full_train_step(i, train_data)
            if i % 10 == 0 and i != 0:
                self.full_test_step(self.model, self.device)
                self.model.train()
        self.full_test_step(self.model, self.device)

    def full_train_step(self, epoch, train_data):

        batch_size = self.batch_size

        loss, mae, rmse = [], [], []
        ndcg_at_5 = []
        start = time.time()
        random.shuffle(train_data)
        num_batch = int(len(train_data) / batch_size)  # ~80
        supp_xs_s, supp_ys_s, supp_mps_s, query_xs_s, query_ys_s, query_mps_s = zip(
            *train_data
        )  # supp_um_s:(list,list,...,2553)
        for i in range(
            num_batch
        ):  # each batch contains some tasks (each task contains a support set and a query set)
            support_xs = list(supp_xs_s[batch_size * i : batch_size * (i + 1)])
            support_ys = list(supp_ys_s[batch_size * i : batch_size * (i + 1)])
            support_mps = list(supp_mps_s[batch_size * i : batch_size * (i + 1)])
            query_xs = list(query_xs_s[batch_size * i : batch_size * (i + 1)])
            query_ys = list(query_ys_s[batch_size * i : batch_size * (i + 1)])
            query_mps = list(query_mps_s[batch_size * i : batch_size * (i + 1)])

            _loss, _mae, _rmse, _ndcg_5 = self.model.global_update(
                support_xs,
                support_ys,
                support_mps,
                query_xs,
                query_ys,
                query_mps,
                self.device,
            )
            loss.append(_loss)
            mae.append(_mae)
            rmse.append(_rmse)
            ndcg_at_5.append(_ndcg_5)

        print(
            "epoch: {}, loss: {:.6f}, cost time: {:.1f}s, mae: {:.5f}, rmse: {:.5f}, ndcg@5: {:.5f}".format(
                epoch,
                np.mean(loss),
                time.time() - start,
                np.mean(mae),
                np.mean(rmse),
                np.mean(ndcg_at_5),
            )
        )
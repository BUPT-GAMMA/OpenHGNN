import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from . import BaseModel, register_model
from torch.autograd import Variable
import numpy as np
from ..utils import Evaluator


@register_model("MetaHIN")
class MetaHIN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args, args.model_name)

    def __init__(self, config, model_name):
        super(MetaHIN, self).__init__()
        self.config = config
        self.mp = ["ub", "ubab", "ubub"]
        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")
        self.model_name = model_name

        self.item_emb = ItemEmbeddingDB(config)
        self.user_emb = UserEmbeddingDB(config)

        self.mp_learner = MetapathLearner(config)
        self.meta_learner = MetaLearner(config)

        self.mp_lr = config.mp_lr
        self.local_lr = config.local_lr
        self.emb_dim = self.config.embedding_dim

        self.cal_metrics = Evaluator(config.seed)

        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.mp_weight_len = len(self.mp_learner.update_parameters())
        self.mp_weight_name = list(self.mp_learner.update_parameters().keys())

        self.transformer_liners = self.transform_mp2task()

        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

    def transform_mp2task(self):
        liners = {}
        ml_parameters = self.meta_learner.update_parameters()
        # output_dim_of_mp = self.config['user_embedding_dim']
        output_dim_of_mp = 32  # movielens: lr=0.001, avg mp, 0.8081
        for w in self.ml_weight_name:
            liners[w.replace(".", "-")] = torch.nn.Linear(
                output_dim_of_mp, np.prod(ml_parameters[w].shape)
            )
        return torch.nn.ModuleDict(liners)

    def forward(
        self,
        support_user_emb,
        support_item_emb,
        support_set_y,
        support_mp_user_emb,
        vars_dict=None,
    ):
        """ """
        if vars_dict is None:
            vars_dict = self.meta_learner.update_parameters()

        support_set_y_pred = self.meta_learner(
            support_user_emb, support_item_emb, support_mp_user_emb, vars_dict
        )
        loss = F.mse_loss(support_set_y_pred, support_set_y)
        grad = torch.autograd.grad(loss, vars_dict.values(), create_graph=True)

        fast_weights = {}
        for i, w in enumerate(vars_dict.keys()):
            fast_weights[w] = vars_dict[w] - self.local_lr * grad[i]

        for idx in range(
            1, self.config.local_update
        ):  # for the current task, locally update
            support_set_y_pred = self.meta_learner(
                support_user_emb,
                support_item_emb,
                support_mp_user_emb,
                vars_dict=fast_weights,
            )
            loss = F.mse_loss(
                support_set_y_pred, support_set_y
            )  # calculate loss on support set
            grad = torch.autograd.grad(
                loss, fast_weights.values(), create_graph=True
            )  # calculate gradients w.r.t. model parameters

            for i, w in enumerate(fast_weights.keys()):
                fast_weights[w] = fast_weights[w] - self.local_lr * grad[i]

        return fast_weights

    def mp_update(
        self,
        support_set_x,
        support_set_y,
        support_set_mps,
        query_set_x,
        query_set_y,
        query_set_mps,
    ):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        """
        # each mp
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        mp_task_fast_weights_s = {}
        mp_task_loss_s = {}
        # 元路径学习器和元学习器（g与h）的初始权重
        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()
        # 提取出用户和物品的嵌入

        support_user_emb = self.user_emb(support_set_x[:, self.config.item_fea_len :])
        support_item_emb = self.item_emb(support_set_x[:, 0 : self.config.item_fea_len])
        query_user_emb = self.user_emb(query_set_x[:, self.config.item_fea_len :])
        query_item_emb = self.item_emb(query_set_x[:, 0 : self.config.item_fea_len])
        # 对每一个元路径
        for mp in self.mp:
            support_set_mp = list(support_set_mps[mp])
            query_set_mp = list(query_set_mps[mp])
            support_neighs_emb = self.item_emb(torch.cat(support_set_mp))
            support_index_list = list(map(lambda _: _.shape[0], support_set_mp))
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = list(map(lambda _: _.shape[0], query_set_mp))
            # 用元路径学习器计算用户增强的嵌入
            support_mp_enhanced_user_emb = self.mp_learner(  # 对应论文中的聚合过程：g
                support_user_emb,
                support_item_emb,
                support_neighs_emb,
                mp,
                support_index_list,
            )
            # 用元学习器来预测
            support_set_y_pred = self.meta_learner(  # 对应论文中的预测过程：h
                support_user_emb, support_item_emb, support_mp_enhanced_user_emb
            )
            # 损失和梯度
            loss = F.mse_loss(support_set_y_pred, support_set_y)
            grad = torch.autograd.grad(
                loss, mp_initial_weights.values(), create_graph=True
            )
            # 更新mp的参数
            fast_weights = {}
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                fast_weights[weight_name] = (
                    mp_initial_weights[weight_name] - self.mp_lr * grad[i]
                )

            # # 继续进行mp的元学习
            for idx in range(1, self.config.mp_update):
                support_mp_enhanced_user_emb = self.mp_learner(
                    support_user_emb,
                    support_item_emb,
                    support_neighs_emb,
                    mp,
                    support_index_list,
                    vars_dict=fast_weights,
                )
                support_set_y_pred = self.meta_learner(
                    support_user_emb, support_item_emb, support_mp_enhanced_user_emb
                )
                loss = F.mse_loss(support_set_y_pred, support_set_y)
                grad = torch.autograd.grad(
                    loss, fast_weights.values(), create_graph=True
                )

                for i in range(self.mp_weight_len):
                    weight_name = self.mp_weight_name[i]
                    fast_weights[weight_name] = (
                        fast_weights[weight_name] - self.mp_lr * grad[i]
                    )
            ########################################################
            # 上面完成语义级适应,下面做任务级适应
            support_mp_enhanced_user_emb = self.mp_learner(
                support_user_emb,
                support_item_emb,
                support_neighs_emb,
                mp,
                support_index_list,
                vars_dict=fast_weights,
            )
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb = self.mp_learner(
                query_user_emb,
                query_item_emb,
                query_neighs_emb,
                mp,
                query_index_list,
                vars_dict=fast_weights,
            )
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

            f_fast_weights = {}
            for w, liner in self.transformer_liners.items():
                w = w.replace("-", ".")
                f_fast_weights[w] = ml_initial_weights[w] * torch.sigmoid(
                    liner(support_mp_enhanced_user_emb.mean(0))
                ).view(ml_initial_weights[w].shape)
            # f_fast_weights = None
            # # the current mp ---> task update
            mp_task_fast_weights = self.forward(
                support_user_emb,
                support_item_emb,
                support_set_y,
                support_mp_enhanced_user_emb,
                vars_dict=f_fast_weights,
            )
            mp_task_fast_weights_s[mp] = mp_task_fast_weights

            query_set_y_pred = self.meta_learner(
                query_user_emb,
                query_item_emb,
                query_mp_enhanced_user_emb,
                vars_dict=mp_task_fast_weights,
            )
            q_loss = F.mse_loss(query_set_y_pred, query_set_y)
            mp_task_loss_s[mp] = q_loss.data  # movielens: 0.8126 dbook 0.6084
            # mp_task_loss_s[mp] = loss.data  # dbook 0.6144

        # mp_att = torch.FloatTensor(
        #     [l / sum(mp_task_loss_s.values()) for l in mp_task_loss_s.values()]
        # ).to(
        #     self.device
        # )  # movielens: 0.81
        mp_att = F.softmax(
            -torch.stack(list(mp_task_loss_s.values())), dim=0
        )  # movielens: 0.80781 lr0.001
        # mp_att = torch.FloatTensor([1.0 / len(self.config['mp'])] * len(self.config['mp'])).to(self.device)

        agg_task_fast_weights = self.aggregator(mp_task_fast_weights_s, mp_att)
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        # agg_mp_emb = torch.stack(support_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        query_y_pred = self.meta_learner(
            query_user_emb,
            query_item_emb,
            query_agg_enhanced_user_emb,
            vars_dict=agg_task_fast_weights,
        )

        loss = F.mse_loss(query_y_pred, query_set_y)
        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_y_pred.data.cpu().numpy()
        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)
        return loss, mae, rmse, ndcg_5

    def global_update(
        self,
        support_xs,
        support_ys,
        support_mps,
        query_xs,
        query_ys,
        query_mps,
        device="cpu",
    ):
        """ """
        batch_sz = len(support_xs)
        loss_s = []
        mae_s = []
        rmse_s = []
        ndcg_at_5_s = []

        for i in range(batch_sz):  # each task in a batch
            support_mp = dict(support_mps[i])  # must be dict!!!
            query_mp = dict(query_mps[i])

            for mp in self.mp:
                support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
                query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])
            _loss, _mae, _rmse, _ndcg_5 = self.mp_update(
                support_xs[i].to(device),
                support_ys[i].to(device),
                support_mp,
                query_xs[i].to(device),
                query_ys[i].to(device),
                query_mp,
            )
            loss_s.append(_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_5_s.append(_ndcg_5)

        loss = torch.stack(loss_s).mean(0)
        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)

        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        return loss.cpu().data.numpy(), mae, rmse, ndcg_at_5

    def evaluation(
        self, support_x, support_y, support_mp, query_x, query_y, query_mp, device="cpu"
    ):
        """ """
        support_mp = dict(support_mp)  # must be dict!!!
        query_mp = dict(query_mp)
        for mp in self.mp:
            support_mp[mp] = map(lambda x: x.to(device), support_mp[mp])
            query_mp[mp] = map(lambda x: x.to(device), query_mp[mp])

        _, mae, rmse, ndcg_5 = self.mp_update(
            support_x.to(device),
            support_y.to(device),
            support_mp,
            query_x.to(device),
            query_y.to(device),
            query_mp,
        )
        return mae, rmse, ndcg_5

    def aggregator(self, task_weights_s, att):
        for idx, mp in enumerate(self.mp):
            if idx == 0:
                att_task_weights = dict(
                    {k: v * att[idx] for k, v in task_weights_s[mp].items()}
                )
                continue
            tmp_att_task_weights = dict(
                {k: v * att[idx] for k, v in task_weights_s[mp].items()}
            )
            att_task_weights = dict(
                zip(
                    att_task_weights.keys(),
                    list(
                        map(
                            lambda x: x[0] + x[1],
                            zip(
                                att_task_weights.values(), tmp_att_task_weights.values()
                            ),
                        )
                    ),
                )
            )

        return att_task_weights

    def eval_no_MAML(self, query_set_x, query_set_y, query_set_mps):
        # each mp
        query_mp_enhanced_user_emb_s = []
        query_user_emb = self.user_emb(query_set_x[:, self.config.item_fea_len :])
        query_item_emb = self.item_emb(query_set_x[:, 0 : self.config.item_fea_len])

        for mp in self.mp:
            query_set_mp = list(query_set_mps[mp])
            query_neighs_emb = self.item_emb(torch.cat(query_set_mp))
            query_index_list = map(lambda _: _.shape[0], query_set_mp)
            query_mp_enhanced_user_emb = self.mp_learner(
                query_user_emb, query_item_emb, query_neighs_emb, mp, query_index_list
            )
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)

        mp_att = torch.FloatTensor([1.0 / len(self.mp)] * len(self.mp)).to(
            self.device
        )  # mean
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)

        query_y_pred = self.meta_learner(
            query_user_emb, query_item_emb, query_agg_enhanced_user_emb
        )
        query_mae, query_rmse = self.cal_metrics.prediction(
            query_set_y.data.cpu().numpy(), query_y_pred.data.cpu().numpy()
        )
        query_ndcg_5 = self.cal_metrics.ranking(
            query_set_y.data.cpu().numpy(), query_y_pred.data.cpu().numpy(), 5
        )

        return query_mae, query_rmse, query_ndcg_5


class MetaLearner(torch.nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.fc1_in_dim = 32 + config.item_embedding_dim
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.second_fc_hidden_dim
        self.use_cuda = config.use_cuda
        self.config = config

        # prediction parameters
        self.vars = torch.nn.ParameterDict()
        self.vars_bn = torch.nn.ParameterList()

        w1 = torch.nn.Parameter(
            torch.ones([self.fc2_in_dim, self.fc1_in_dim])
        )  # 64, 96
        torch.nn.init.xavier_normal_(w1)
        self.vars["ml_fc_w1"] = w1
        self.vars["ml_fc_b1"] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim, self.fc2_in_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.vars["ml_fc_w2"] = w2
        self.vars["ml_fc_b2"] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim]))
        torch.nn.init.xavier_normal_(w3)
        self.vars["ml_fc_w3"] = w3
        self.vars["ml_fc_b3"] = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user_emb, item_emb, user_neigh_emb, vars_dict=None):
        """ """
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_neigh_emb  # movielens: loss:12.14... up! ; dbook 20epoch: user_cold: mae 0.6051;

        x = torch.cat((x_i, x_u), 1)  # ?, item_emb_dim+user_emb_dim+user_emb_dim
        x = F.relu(F.linear(x, vars_dict["ml_fc_w1"], vars_dict["ml_fc_b1"]))
        x = F.relu(F.linear(x, vars_dict["ml_fc_w2"], vars_dict["ml_fc_b2"]))
        x = F.linear(x, vars_dict["ml_fc_w3"], vars_dict["ml_fc_b3"])
        return x.squeeze()

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class MetapathLearner(torch.nn.Module):
    def __init__(self, config):
        super(MetapathLearner, self).__init__()
        self.config = config

        # meta-path parameters
        self.vars = torch.nn.ParameterDict()
        neigh_w = torch.nn.Parameter(
            torch.ones([32, config.item_embedding_dim])
        )  # dim=32, movielens 0.81006
        torch.nn.init.xavier_normal_(neigh_w)
        self.vars["neigh_w"] = neigh_w
        self.vars["neigh_b"] = torch.nn.Parameter(torch.zeros(32))

    def forward(self, user_emb, item_emb, neighs_emb, mp, index_list, vars_dict=None):
        """ """
        if vars_dict is None:
            vars_dict = self.vars
        agg_neighbor_emb = F.linear(
            neighs_emb, vars_dict["neigh_w"], vars_dict["neigh_b"]
        )  # (#neighbors, item_emb_dim)
        output_emb = F.leaky_relu(torch.mean(agg_neighbor_emb, 0)).repeat(
            user_emb.shape[0], 1
        )  # (#sample, user_emb_dim)
        #
        # # each mean, then att agg
        # _emb = []
        # start = 0
        # for idx in index_list:
        #     end = start+idx
        #     _emb.append(F.leaky_relu(torch.mean(agg_neighbor_emb[start:end],0)))
        #     start = end
        # output_emb = torch.stack(_emb, 0)  # (#sample, dim)

        return output_emb

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class UserEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(UserEmbeddingDB, self).__init__()
        self.num_location = config.num_location
        self.embedding_dim = config.embedding_dim

        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location, embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: tensor, shape = [#sample, #user_fea]
        :return:
        """
        location_idx = Variable(user_fea[:, 0], requires_grad=False)  # [#sample]
        location_emb = self.embedding_location(location_idx)
        return location_emb  # (1, 1*32)


class ItemEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingDB, self).__init__()
        self.num_publisher = config.num_publisher
        self.embedding_dim = config.embedding_dim

        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher, embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """
        :param item_fea:
        :return:
        """
        publisher_idx = Variable(item_fea[:, 0], requires_grad=False)
        publisher_emb = self.embedding_publisher(publisher_idx)  # (1,32)
        return publisher_emb  # (1, 1*32)
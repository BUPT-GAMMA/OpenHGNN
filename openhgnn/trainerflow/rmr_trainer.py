import dgl.data
from openhgnn.models import HAN, build_model
from openhgnn.tasks import build_task
from openhgnn.trainerflow import register_flow
from openhgnn.trainerflow.base_flow import BaseFlow
import dgl
import torch.nn.functional as F
from tqdm import tqdm
from ..utils import EarlyStopping
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from openhgnn.sampler import HANSampler
import random
import time
import argparse


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--use_cuda', default=True, action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0.0)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--w', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=200)

    # model-specific parameters
    parser.add_argument('--attr1', type=float, default=0.0)
    parser.add_argument('--attr2', type=float, default=0.0)
    parser.add_argument('--feat', type=float, default=0.8)
    parser.add_argument('--r1', type=float, default=0.9)
    parser.add_argument('--r2', type=float, default=0.3)
    parser.add_argument('--r3', type=float, default=0.0)

    args, _ = parser.parse_known_args()
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0.0)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--w', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=4000)

    # model-specific parameters
    parser.add_argument('--attr1', type=float, default=0.0)
    parser.add_argument('--attr2', type=float, default=0.15)
    parser.add_argument('--feat', type=float, default=0.5)
    parser.add_argument('--r1', type=float, default=0.1)
    parser.add_argument('--r2', type=float, default=0.15)
    parser.add_argument('--r3', type=float, default=0.55)

    args, _ = parser.parse_known_args()
    return args


def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--seed', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0.0)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--w', type=float, default=5e-5)
    parser.add_argument('--epoch', type=int, default=5500)

    # model-specific parameters
    parser.add_argument('--attr1', type=float, default=0.0)
    parser.add_argument('--attr2', type=float, default=0.3)
    parser.add_argument('--feat', type=float, default=0.55)
    parser.add_argument('--r1', type=float, default=0.9)
    parser.add_argument('--r2', type=float, default=0.8)
    parser.add_argument('--r3', type=float, default=0.0)

    args, _ = parser.parse_known_args()
    return args

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# trainerflow中__init__使用build_flow(config,args)时将会实例化RMR_trainer类
@register_flow('rmr_trainer')
class RMR_trainer(BaseFlow):
    def __init__(self, args):
        super(RMR_trainer, self).__init__(args)
        self.args.category = self.task.dataset.category
        self.category = self.args.category

        self.num_classes = self.task.dataset.num_classes

        if not hasattr(self.task.dataset, 'out_dim') or args.out_dim != self.num_classes:
            self.logger.info('[NC Specific] Modify the out_dim with num_classes')
            args.out_dim = self.num_classes
        self.args.out_node_type = [self.category]


        # 构建模型
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.g = self.model.g
        self.main_node = self.model.g.graph_data['main_node']
    def train(self):
        seed_everything(0)
        start_time = time.time()
        self.model.eval()
        embeds = self.model.get_embed()
        args = 0
        if self.model.dataset_name == 'acm4RMR':
            args = acm_params()
        elif self.model.dataset_name == 'aminer4RMR':
            args = aminer_params()
        elif self.model.dataset_name == 'imdb4RMR':
            args = imdb_params()
        else:
            raise ValueError(f"Unsupported dataset name: {self.model.dataset_name}")
        if args.dataset != 'aminer':
            for ratio in args.ratio:
                evaluate(
                    embeds,
                    self.g.nodes[self.main_node].data[f'{ratio}_train_mask'],
                    self.g.nodes[self.main_node].data[f'{ratio}_val_mask'],
                    self.g.nodes[self.main_node].data[f'{ratio}_test_mask'],
                    self.g.nodes[self.main_node].data['y'].long(),
                    device,
                    self.g,
                    0.01,
                    0,
                    args.dataset,
                    self.logger
                )
        else:
            for ratio in args.ratio:
                evaluate(
                    embeds,
                    self.g.graph_data[f'{ratio}_train_mask'],
                    self.g.graph_data[f'{ratio}_val_mask'],
                    self.g.graph_data[f'{ratio}_test_mask'],
                    self.g.nodes[self.main_node].data['y'][:127202].long(),
                    device,
                    self.g,
                    0.03,
                    0,
                    args.dataset,
                    self.logger
                )



        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w)
        for epoch in range(1, int(args.epoch) + 1):

            self.model.train()
            self.optimizer.zero_grad()
            loss = self.model()
            loss.backward()
            self.optimizer.step()
            # print("Epoch:{}, Loss:{}".format(epoch, loss))
            current_time = time.time()
            self.logger.train_info(f"Epoch: {epoch}, Loss: {loss:.4f}, " + f"time: {current_time - start_time:.10f} .")

        self.model.eval()
        embeds = self.model.get_embed()
        if args.dataset != 'aminer':
            for ratio in args.ratio:
                evaluate(
                    embeds,
                    self.g.nodes[self.main_node].data[f'{ratio}_train_mask'],
                    self.g.nodes[self.main_node].data[f'{ratio}_val_mask'],
                    self.g.nodes[self.main_node].data[f'{ratio}_test_mask'],
                    self.g.nodes[self.main_node].data['y'].long(),
                    device,
                    self.g,
                    0.01,
                    0,
                    args.dataset,
                    self.logger
                )
        else:
            for ratio in args.ratio:
                evaluate(
                    embeds,
                    self.g.graph_data[f'{ratio}_train_mask'],
                    self.g.graph_data[f'{ratio}_val_mask'],
                    self.g.graph_data[f'{ratio}_test_mask'],
                    self.g.nodes[self.main_node].data['y'][:127202].long(),
                    device,
                    self.g,
                    0.03,
                    0,
                    args.dataset,
                    self.logger
                )




    def _full_train_step(self):
        r"""
        Train with a full_batch graph
        """
        raise NotImplementedError

    def _full_test_step(self):
        r"""
        Test with a full_batch graph
        """
        raise NotImplementedError



class LogReg(nn.Module):
    # 输入特征数量，总类别
    def __init__(self, num_features, num_classes):
        super(LogReg, self).__init__()
        # 一个全连接层完成分类任务
        self.fc = nn.Linear(num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 随机权重，偏置置为0
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, seq):
        return self.fc(seq)
        # return torch.log_softmax(self.fc(seq).squeeze(), dim=-1)


class EvaData(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.y = y

    def __getitem__(self, item):
        return self.data[item], self.y[item]

    def __len__(self):
        return len(self.data)


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]



def evaluate(embeds, train_mask, val_mask, test_mask, label, device, data, lr, wd, name,logger):
    # 特征数量 一般等于隐藏层维度 64
    num_features = embeds.shape[1]
    embeds = embeds.to(device)
    # data.lable size为[main_node_num,2],实际上形式为(main_node_index,对应label)例如[0,0],[1,0],[2,0]
    train_mask = train_mask.to(torch.bool)
    val_mask = val_mask.to(torch.bool)
    test_mask = test_mask.to(torch.bool)

    label = label.to(device)

    if name == 'aminer':
        #由于label从0开始，实际种类需要+1
        num_classes = label.max() + 1
        # xent代表损失函数
        xent = nn.CrossEntropyLoss()
        # 提取有label的节点对应的嵌入层特征

        embeds = embeds[data.nodes['P'].data['idx'][:127202]]
        # embeds = embeds[data['P'].y[:, 0]]
    elif name == 'dblpv1':
        num_classes = label.max() + 1
        xent = nn.CrossEntropyLoss()
        num_classes = label.size(1)
        embeds = embeds[data['P'].y[:, 0]]
        train_lbls = data['P'].y[:, 1][train_mask]
        val_lbls = data['P'].y[:, 1][val_mask]
        test_lbls = data['P'].y[:, 1][test_mask]
    elif name == 'pubmed':
        num_classes = label.size(1)
        xent = torch.nn.BCEWithLogitsLoss()
        embeds = embeds.to(device)
        label = label.to(device).float()
    elif name == 'acm':
        # 由于label从0开始，实际种类需要+1
        num_classes = label.max() + 1
        # xent代表损失函数
        xent = nn.CrossEntropyLoss()
        # 提取有label的节点对应的嵌入层特征
        embeds = embeds.to(device)
    elif name == 'imdb':
        # 由于label从0开始，实际种类需要+1
        num_classes = label.max() + 1
        # xent代表损失函数
        xent = nn.CrossEntropyLoss()
        # 提取有label的节点对应的嵌入层特征
        embeds = embeds.to(device)
    if name != 'dblpv1':
        train_lbls = label[train_mask]
        val_lbls = label[val_mask]
        test_lbls = label[test_mask]
    train_embs = embeds[train_mask]
    val_embs = embeds[val_mask]
    test_embs = embeds[test_mask]

    # if name == 'cite':
    #     train_lbls = train_lbls.float()
    #     # dataset = EvaData(train_embs, train_lbls)
    #     # da = DataLoader(dataset, batch_size=50000, shuffle=True, num_workers=0)
    #     val_lbls = val_lbls.float()
    #     test_lbls = test_lbls.float()

    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(1):
        # logReg为一个全连接层，输入待处理的特征矩阵，完成分类任务
        log = LogReg(num_features, num_classes).to(device)
        #设置优化器参数
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        info_f1s = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        logits_list = []

        train_embs = train_embs.to(device)
        train_lbls = train_lbls.to(device)
        for i in range(200):
            # train
            # for index, (train_x, train_y) in enumerate(da):
            #     log.train()
            #     opt.zero_grad()
            #     logits = log(train_x.to(device))
            #     loss = xent(logits, train_y.to(device))
            #     loss.backward()
            #     opt.step()

            log.train()
            opt.zero_grad()
            # logits 即为对应train_mask的节点的预测结果，size为[train_mask_num,label_num]
            logits = log(train_embs)
            # 求出损失值
            loss = xent(logits, train_lbls)
            # 完成反向传播
            loss.backward()
            # 参数更新
            opt.step()

            ##########################################################################
            # Val 验证集数据
            logits = log(val_embs.to(device))
            if name == 'pubmed':
                preds = (logits > 0).float().cpu()
            else:
                # pred 代表模型在验证集上的结果
                preds = torch.argmax(logits, dim=1).cpu()
            #分别计算f1_macro和f1_micro
            val_f1_macro = f1_score(val_lbls.cpu().numpy(), preds, average='macro')
            val_f1_micro = f1_score(val_lbls.cpu().numpy(), preds, average='micro')

            #将本轮计算结果加入队列
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # #####################################################################
            # Test
            logits = log(test_embs)



            if name == 'pubmed':
                preds = (logits > 0).float().cpu()
            else:
                preds = torch.argmax(logits, dim=1).cpu()
            test_f1_macro = f1_score(test_lbls.cpu(), preds, average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds, average='micro')
            try:
                if name == 'pubmed':
                    y_true = test_lbls.cpu().numpy()
                    y_score = logits.detach().cpu().numpy()
                    auc = roc_auc_score(y_true, y_score, average='macro')
                else:
                    y_true = test_lbls.cpu().numpy()
                    y_score = torch.softmax(logits, dim=1).detach().cpu().numpy()
                    auc = roc_auc_score(y_true, y_score, multi_class='ovr')
                auc_score_list.append(auc)
            except Exception as e:
                auc_score_list.append(0.0)
                # print(f"[Warning] AUC calculation failed: {e}")
            # 将测试集相关结果加入对应队列
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            info_f1s.append((test_f1_macro+test_f1_micro)/(val_f1_macro+val_f1_micro))
            logits_list.append(logits)
        #################################################################################
        # 找到当前 验证集测试最优结果对应的下标
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        # 将最优验证集的结果对应的测试集结果加入队列
        macro_f1s.append(test_macro_f1s[max_iter])
        # 将最优结果加入到新val结果队列中，相当于每200轮，只取一个最好成绩
        macro_f1s_val.append(val_macro_f1s[max_iter])
        # #################################################################################
        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])
    # 源码打印消息在循环外，我认为不太合理进行修改
    # print("Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f} var: {:.4f}"
    #     .format(
    #     np.mean(macro_f1s),
    #     np.std(macro_f1s),
    #     np.mean(micro_f1s),
    #     np.std(micro_f1s),
    #     np.mean(auc_score_list),
    #     np.std(auc_score_list)))

    logger.train_info("Macro-F1_mean: {:.4f} var: {:.4f}  Micro-F1_mean: {:.4f} var: {:.4f} auc {:.4f} var: {:.4f}"
        .format(
        np.mean(macro_f1s),
        np.std(macro_f1s),
        np.mean(micro_f1s),
        np.std(micro_f1s),
        np.mean(auc_score_list),
        np.std(auc_score_list))
    )
    return np.mean(micro_f1s)

import numpy as np
import torch
from numpy.core.defchararray import upper
from sklearn import decomposition

from ..layers.QGNNLayer import *
# from ..layers.quaternion_layer import to_tensor, get_Laplacian, get_Laplacian_from_weights
from ..models import BaseModel, register_model
from sklearn.cluster import KMeans

from ..utils.metrics import cal_clustering_metric


@register_model('QGRL')
class QGRL(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        # {'seed': 0,
        # 'patience': 1,
        # 'max_epoch': 50,
        # 'task': 'qgrl', 'model': 'QGRL', 'dataset': 'zoo', 'dataset_name': 'zoo', 'model_name': 'QGRL', 'optimizer': 'Adam',
        # 'layers': '[512, 256, 128]',
        # 'acts': [<function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>, <function relu at 0x00000212FAA4A0D0>],
        # 'lr': 0.0004, 'pretrain_learning_rate': 0.0001,
        # 'lamSC': 2.0,
        # 'coeff_reg': 0.0001,
        # 'max_iter': 1,
        # 'pre_iter': 10,
        # 'decomposition': "'symeig'",
        # 'device': device(type='cpu'), 'gpu': -1, 'use_best_config': True, 'load_from_pretrained': False, 'output_dir': './openhgnn/output\\QGRL', 'hpo_search_space': None,
        # 'hpo_trials': 100, 'logger': <openhgnn.utils.logger.Logger object at 0x00000212C2236E80>, 'test_flag': True, 'prediction_flag': False, 'use_uva': False,
        # 'meta_paths_dict': None}
        # 提取args中的参数
        name = args.model_name
        # print(args.__dict__)
        features = hg.fea
        adjacency = hg.adj
        labels = hg.lab
        layers = list(eval(args.layers))
        acts = [torch.nn.functional.relu] * len(layers)

        learning_rate = args.lr
        pretrain_learning_rate = args.pretrain_learning_rate
        lamSC = np.power(2.0, 1)
        coeff_reg = args.coeff_reg
        seed = args.seed

        max_epoch = args.max_epoch
        max_iter = args.max_iter
        pre_iter = args.pre_iter

        decomposition = args.decomposition  # symeig,svd,eigh

        is_norm = False
        device = torch.device('cpu') if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')

        # 构建模型
        model = cls(
            name=name,
            features=features,
            adjacency=adjacency,
            labels=labels,
            decomposition=decomposition,
            is_norm=is_norm,
            layers=layers,
            acts=acts,
            max_epoch=max_epoch,
            max_iter=max_iter,
            pre_iter=pre_iter,
            learning_rate=learning_rate,
            pretrain_learning_rate=pretrain_learning_rate,
            coeff_reg=coeff_reg,
            seed=seed,
            lam=lamSC,
            device=device
        )
        return model

    def __init__(self,
                 name,          # 名字
                 features,      # 特征矩阵
                 adjacency,     # 邻接矩阵
                 labels,        # 节点标签
                 decomposition='symeig',
                 is_norm=True,
                 layers=None,
                 acts=None,
                 max_epoch=10,
                 max_iter=50,
                 pre_iter=10,
                 learning_rate=10 ** -2,
                 pretrain_learning_rate=10 ** -2,
                 coeff_reg=10 ** -3,
                 seed=114514,
                 lam=-1,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(QGRL, self).__init__()
        self.name = name
        self.device = device
        self.X = to_tensor(features).to(self.device)  # 输入特征矩阵
        self.adjacency = to_tensor(adjacency).to(self.device)  # 图的邻接矩阵
        self.labels = to_tensor(labels).to(self.device)  # 节点标签

        self.decomposition = decomposition  # 特征分解方法
        self.is_norm = is_norm  # 是否对指示向量进行归一化

        self.n_clusters = self.labels.unique().shape[0]  # 聚类的类别数

        if layers is None:
            layers = [32, 16]  # 默认的网络层结构
        self.layers = layers

        self.acts = acts  # 每层的激活函数
        assert len(self.acts) == len(self.layers)
        self.max_iter = max_iter  # 每个epoch的最大迭代次数
        self.pre_iter = pre_iter
        self.max_epoch = max_epoch  # 最大epoch数
        self.learning_rate = learning_rate  # 学习率
        self.pretrain_learning_rate = pretrain_learning_rate
        self.coeff_reg = coeff_reg  # 正则化系数
        self.seed = seed  # 随机种子

        self.data_size = self.X.shape[0]  # 数据样本数
        self.input_dim = self.X.shape[1]  # 输入特征维度

        self.indicator = self.X  # 初始化指示向量
        self.embedding = self.X  # 初始化嵌入向量
        self.num_neighbors = 5  # 邻居节点数
        self.links = 0  # 初始化链接数
        self.lam = lam  # 谱聚类损失的权重系数
        self._build_up()  # 构建网络结构

        self.to(self.device)  # 将模型转移到指定设备（CPU或GPU）

    # 构建网络结构的方法，定义模型中的各个层
    def _build_up(self):

        # 定义四个线性层，用于将输入特征映射到四元数空间
        self.pro1 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro2 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro3 = torch.nn.Linear(self.input_dim, self.layers[0])
        self.pro4 = torch.nn.Linear(self.input_dim, self.layers[0])

        # 定义两个四元图神经网络层，用于图卷积操作
        self.qgnn1 = QGNNLayer(self.layers[0] * 4, self.layers[1] * 4, quaternion_ff=True, \
                               act=self.acts[0], init_criterion='he', weight_init='quaternion', seed=self.seed)
        self.qgnn2 = QGNNLayer(self.layers[1] * 4, self.layers[2] * 4, quaternion_ff=True, \
                               act=self.acts[1], init_criterion='he', weight_init='quaternion', seed=self.seed)

    def forward(self, Laplacian):
        # 将输入特征通过四个线性层映射到四元数空间的四个部分
        x1 = self.pro1(self.X)
        x2 = self.pro2(self.X)
        x3 = self.pro3(self.X)
        x4 = self.pro4(self.X)

        # 将四个部分拼接成一个四元数表示的输入张量
        input = torch.cat((x1, x2, x3, x4), dim=1)

        # 通过两个四元图神经网络层进行图卷积操作
        input = self.qgnn1(input, Laplacian)
        input = self.qgnn2(input, Laplacian)

        # 对输出进行重塑和平均，得到最终的嵌入向量
        input = input.reshape(self.data_size, 4, self.layers[2]).sum(dim=1) / 4.
        self.embedding = input

        # 计算重构的邻接矩阵
        recons_A = self.embedding.matmul(self.embedding.t())

        return recons_A

    def pretrain(self):
        pretrain_iter=self.pre_iter
        learning_rate=self.pretrain_learning_rate
        # 预训练方法，用于在正式训练前对模型进行预训练，以获得更好的初始化参数
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        print('Start pretraining (totally {} iterations) ......'.format(pretrain_iter))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        Laplacian = get_Laplacian(self.adjacency)
        # 执行预训练循环
        for i in range(pretrain_iter):
            optimizer.zero_grad()
            recons_A = self(Laplacian)
            loss = self.build_pretrain_loss(recons_A)
            loss.backward()
            optimizer.step()
        print(loss.item())

    # 构建预训练损失的方法，用于模型的预训练阶段
    def build_pretrain_loss(self, recons_A):
        epsilon = torch.tensor(10 ** -7).to(self.device)
        recons_A = recons_A - recons_A.diag().diag()  # 去除对角线元素
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()
        loss = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) + (1 - self.adjacency).mul(
            (1 / torch.max((1 - recons_A), epsilon)).log())
        loss = loss.sum() / (loss.shape[0] * loss.shape[1])
        loss_reg = self.build_loss_reg()
        loss = loss + self.coeff_reg * loss_reg
        return loss

    # 构建正则化损失的方法，计算模型中各层权重的L1正则化损失
    def build_loss_reg(self):
        loss_reg = 0

        for module in self.modules():
            if type(module) is torch.nn.Linear:
                loss_reg += torch.abs(module.weight).sum()
            if type(module) is QGNNLayer:
                loss_reg += (torch.abs(module.r) + torch.abs(module.i) + torch.abs(module.j) + torch.abs(module.k)).sum()

            return loss_reg

    # 构建总损失的方法，结合重构损失、正则化损失和谱聚类损失
    def build_loss(self, recons_A):
        size = self.X.shape[0]

        epsilon = torch.tensor(10 ** -7).to(self.device)  # 防止除零的极小值
        pos_weight = (self.data_size * self.data_size - self.adjacency.sum()) / self.adjacency.sum()  # 正样本权重

        # 计算重构损失，基于邻接矩阵和重构邻接矩阵的对数似然
        loss_1 = pos_weight * self.adjacency.mul((1 / torch.max(recons_A, epsilon)).log()) \
                     + (1 - self.adjacency).mul((1 / torch.max((1 - recons_A), epsilon)).log())
        loss_1 = loss_1.sum() / (self.data_size ** 2)

        # 计算正则化损失
        loss_reg = self.build_loss_reg()

        # 计算谱聚类损失，基于嵌入向量和重构邻接矩阵
        degree = recons_A.sum(dim=1)
        L = torch.diag(degree) - recons_A
        loss_SC = torch.trace(self.embedding.t().matmul(L).matmul(self.embedding)) / (size)

        # 总损失为重构损失、正则化损失和谱聚类损失的加权和
        loss = loss_1 + self.coeff_reg * loss_reg + self.lam * loss_SC
        return loss

    # 更新图结构的方法，根据当前嵌入向量计算新的权重矩阵
    def update_graph(self, embedding):
        weights = embedding.matmul(embedding.t())
        weights = weights.detach()
        return weights

    # 聚类方法，基于嵌入向量进行谱聚类并计算聚类指标
    def clustering(self, weights):
        degree = torch.sum(weights, dim=1).pow(-0.5)
        L = (weights * degree).t() * degree  # 计算归一化的拉普拉斯矩阵

        # 根据指定的分解方法计算特征向量
        if self.decomposition == 'symeig':
            _, vectors = torch.linalg.eigh(L, UPLO='U' if upper else 'L')
        elif self.decomposition == 'svd':
            vectors, _, __ = torch.svd(L)
        elif self.decomposition == 'eigh':
            _, vectors = torch.linalg.eigh(L)

        indicator = vectors[:, -self.n_clusters:].detach()  # 提取用于聚类的特征向量

        # 对指示向量进行归一化处理
        if self.is_norm:
            indicator = indicator / (indicator.norm(dim=1) + 10 ** -10).repeat(self.n_clusters, 1).t()

        indicator = indicator.cpu().numpy()  # 转换为NumPy数组
        km = KMeans(n_clusters=self.n_clusters).fit(indicator)  # 使用KMeans进行聚类
        prediction = km.predict(indicator)  # 获取聚类预测结果
        acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), prediction)  # 计算聚类指标

        return acc, nmi, ari, f1

    # 模型训练方法，执行模型的训练过程，包括前向传播、损失计算、反向传播和聚类评估
    def run(self):
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        # 初始化图结构和聚类评估指标
        weights = self.update_graph(self.embedding)
        weights = get_Laplacian_from_weights(weights)

        acc, nmi, ari, f1 = self.clustering(weights)

        best_acc, best_nmi, best_ari, best_f1 = acc, nmi, ari, f1

        print('Initial ACC: %.2f, NMI: %.2f, ARI: %.2f' % (acc * 100, nmi * 100, ari * 100))
        objs = []
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 定义优化器
        Laplacian = get_Laplacian(self.adjacency)  # 计算初始拉普拉斯矩阵
        # 执行训练循环
        for epoch in range(self.max_epoch):
            for i in range(self.max_iter):
                optimizer.zero_grad()  # 清除梯度
                recons_A = self(Laplacian)  # 前向传播，得到重构邻接矩阵
                loss = self.build_loss(recons_A)  # 计算损失
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 更新模型参数
                objs.append(loss.item())  # 记录损失值

            # 更新图结构并进行聚类评估
            weights = self.update_graph(self.embedding)
            weights = get_Laplacian_from_weights(weights)

            acc, nmi, ari, f1 = self.clustering(weights)
            loss = self.build_loss(recons_A)
            objs.append(loss.item())
            print('{}'.format(epoch) + 'loss: %.4f, ACC: %.2f, NMI: %.2f, ARI: %.2f, F1: %.2f' % (
            loss.item(), acc * 100, nmi * 100, ari * 100, f1 * 100))

            # 更新最佳聚类结果
            if acc >= best_acc:
                best_acc = acc
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)

        # 输出最终的聚类结果和统计指标
        print("best_acc{},best_nmi{},best_ari{},best_f1{}".format(best_acc, best_nmi, best_ari, best_f1))
        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)
        print(acc_list.mean(), "±", acc_list.std())
        print(nmi_list.mean(), "±", nmi_list.std())
        print(ari_list.mean(), "±", ari_list.std())
        print(f1_list.mean(), "±", f1_list.std())
        return best_acc, best_nmi, best_ari, best_f1

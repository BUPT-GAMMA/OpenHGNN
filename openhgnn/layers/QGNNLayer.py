import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import RandomState
from torch.nn.parameter import Parameter

from ..layers.quaternion_layer import *


class QGNNLayer(nn.Module):
    # 初始化方法，设置输入和输出特征维度、是否使用四元数全连接层、激活函数等参数，并初始化权重参数
    def __init__(self, in_features, out_features, quaternion_ff=True,
                 act=F.relu, init_criterion='he', weight_init='quaternion',
                 seed=None):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.quaternion_ff = quaternion_ff
        self.act = act

        if self.quaternion_ff:
            # 如果使用四元数全连接层，初始化四个权重参数张量（r、i、j、k）
            self.r = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
            self.i = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
            self.j = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
            self.k = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)
        else:
            # 否则初始化一个普通的权重参数张量
            self.commonLinear = Parameter(torch.Tensor(self.in_features, self.out_features), requires_grad=True)

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()  # 调用权重初始化方法

    # 权重初始化方法，根据指定的初始化标准和方式对权重参数进行初始化
    def reset_parameters(self):
        if self.quaternion_ff:
            winit = {'quaternion': quaternion_init,
                     'unitary': unitary_init}[self.weight_init]
            affect_init(self.r, self.i, self.j, self.k, winit,
                        self.rng, self.init_criterion)

        else:
            stdv = math.sqrt(6.0 / (self.commonLinear.size(0) + self.commonLinear.size(1)))
            self.commonLinear.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if self.quaternion_ff:
            # 构造四元数哈密顿矩阵，用于四元数乘法运算
            r1 = torch.cat([self.r, -self.i, -self.j, -self.k], dim=0)
            i1 = torch.cat([self.i, self.r, -self.k, self.j], dim=0)
            j1 = torch.cat([self.j, self.k, self.r, -self.i], dim=0)
            k1 = torch.cat([self.k, -self.j, self.i, self.r], dim=0)
            cat_kernels_4_quaternion = torch.cat([r1, i1, j1, k1], dim=1)

            # 执行输入张量与哈密顿矩阵的矩阵乘法，得到中间结果
            mid = torch.mm(x, cat_kernels_4_quaternion)
        else:
            # 如果不使用四元数全连接层，直接执行普通的矩阵乘法
            mid = torch.mm(x, self.commonLinear)

        # 执行图卷积操作，将中间结果与邻接矩阵相乘
        out = torch.mm(adj, mid)

        # 应用激活函数并返回结果
        return self.act(out)

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


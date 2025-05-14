import torch
import torch.nn as nn
import math
import dgl
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
# 确保从 OpenHGNN 正确导入所需的层
# 可能需要根据你的 OpenHGNN 版本和目录结构调整路径
try:
    # 尝试标准路径
    from openhgnn.layers.Fully_connect import FullyConnect
    from openhgnn.layers.Linear_layer import Linear_layer
except ImportError:
    print("警告：无法从标准 openhgnn.layers 导入层。请检查 OpenHGNN 安装或路径。")


from . import BaseModel, register_model # OpenHGNN 模型注册机制
import dgl.function as fn

VERY_SMALL_NUMBER = 1e-12
INF = 1e20

@register_model('HERO')
class HERO(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg) # 将 hg 传递给 __init__ 以便访问 ntypes/etypes

    def __init__(self, args, hg): # 接收 hg 以便获取类型信息
        super(HERO, self).__init__()
        self.args = args
        self.hg = hg # 保存图对象引用，主要用于获取 ntypes 和 etypes

        # --- 重要修改：不再自行计算 node_cnt 和 node_size ---
        # 这些值现在应该由 Herotrainer 通过 args 传入
        if not hasattr(args, 'node_cnt') or not hasattr(args, 'node_size') or not hasattr(args, 'ft_size'):
             raise ValueError("必要的参数 'node_cnt', 'node_size', 'ft_size' 未在 args 中提供！")

        self.node_cnt = args.node_cnt # 使用来自 trainer 的映射
        self.node_size = args.node_size # 使用来自 trainer 的全局节点数
        self.ft_size = args.ft_size   # 使用来自 trainer 的特征维度

        # --- 网络层初始化 ---
        self.bnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        # MLP 输入维度应为全局特征维度 self.ft_size
        self.mlp = MLP([self.ft_size, self.args.out_ft]) # out_ft 是同构嵌入的目标维度

        # 初始化 FC 层 (用于第二层聚合后的拼接)
        # 输入维度是 hid_units2 (第二层聚合输出) + ft_size (原始特征)
        fc_input_dim = args.hid_units2 + self.ft_size
        for ntype in hg.ntypes: # 遍历 DGL 图中的节点类型
            # 如果该类型在 node_cnt 中定义了 (即参与了全局排序)
            if ntype in self.node_cnt:
                 self.fc[ntype] = FullyConnect(fc_input_dim, args.out_ft) # 输出维度是异构嵌入的目标维度
            else:
                 print(f"警告: DGL 图中的节点类型 '{ntype}' 未在 node_cnt 中定义，未初始化 FC 层。")


        # 初始化 BNN 层 (用于 GNN 聚合)
        # 键名使用 DGL 的规范边类型字符串 (src_type, edge_name, dst_type)
        for canonical_etype in hg.canonical_etypes:
            etype_str = canonical_etype[1] # 获取中间的边类型名称字符串，例如 'pa' 或 'ap'
            # key = f"{src_type}-{dst_type}" # 或者使用原始的 p-a 格式，如果 Herotrainer 中处理了
            # 为了安全，我们使用 DGL 提供的 etype 名称字符串
            # BNN 层 0 (第一层聚合)，输入是原始特征维度 self.ft_size
            self.bnn['0' + etype_str] = Linear_layer(self.ft_size, args.hid_units, act=nn.ReLU(), isBias=False)
            # BNN 层 1 (第二层聚合)，输入是第一层聚合的隐藏维度 args.hid_units
            self.bnn['1' + etype_str] = Linear_layer(args.hid_units, args.hid_units2, act=nn.ReLU(), isBias=False)
        print("HERO 模型层初始化完成.")


    def forward(self, hg, features, distance):
        """
        模型的前向传播。

        Args:
            hg (dgl.DGLHeteroGraph): 输入的 DGL 图。
            features (torch.Tensor): 全局排序的节点特征张量。
            distance (torch.Tensor): 目标节点的特征距离矩阵。

        Returns:
            tuple: (emb_het, emb_hom) 异构和同构节点嵌入。
        """
        # 使用局部作用域，避免修改传入的原始 hg 图对象
        with hg.local_scope():
            # --- 准备工作 ---
            # 1. 初始化用于存储中间结果的全局张量 (使用正确的 node_size)
            # embs1 存储第一层聚合结果 (hid_units 维度)
            embs1 = torch.zeros((self.node_size, self.args.hid_units), device=features.device)
            # embs2 存储第二层聚合+FC后的结果 (out_ft 维度)
            embs2 = torch.zeros((self.node_size, self.args.out_ft), device=features.device)

            # 2. 将全局有序特征写入 DGL 图的节点数据中，以便消息传递使用
            #    使用 self.node_cnt 进行映射
            # print("将全局特征写入 DGL 节点数据...")
            for ntype in hg.ntypes:
                if ntype in self.node_cnt: # 只处理在全局映射中存在的类型
                    # 获取该类型节点的全局索引范围
                    idx = self.node_cnt[ntype]
                    # 从全局 features 张量中提取对应特征
                    features_slice = features[idx]
                    # 将特征赋给 DGL 图中该类型的节点
                    hg.nodes[ntype].data['h'] = features_slice
                    # print(f"  - 类型 '{ntype}': 分配特征，形状 {features_slice.shape}")
                else:
                    print(f"警告: 类型 '{ntype}' 未在 node_cnt 中，无法分配特征给 DGL 节点。")


            # 3. 构造反向图，用于从邻居向中心节点聚合信息
            #    共享节点和边数据以提高效率
            rev_hg = dgl.reverse(hg, share_ndata=True, share_edata=True)

            # --- HERO 异构嵌入计算 ---
            # print("开始第一层聚合...")
            # === 第一层聚合 ===
            # 按目标节点类型进行迭代聚合（DGL 的标准方式）
            for ntype in hg.ntypes:
                 if ntype not in self.node_cnt: continue # 跳过未参与全局映射的类型
                 # print(f"  处理目标类型: '{ntype}'")
                 # 用于收集来自不同关系类型的聚合结果
                 aggregated_results_l1 = []
                 # 遍历指向当前节点类型 ntype 的所有边类型 (在反向图中是发出边)
                 # etype 是 (src, name, dst)，这里 ntype 是 dst
                 for etype in rev_hg.canonical_etypes:
                      src_type, etype_name, dst_type = etype
                      if dst_type == ntype: # 如果当前类型是这条反向边的目标类型
                           # print(f"    - 通过关系 '{src_type}' --'{etype_name}'--> '{dst_type}' 进行聚合")
                           # 获取对应的 BNN 层 (使用边类型名称)
                           bnn_key = '0' + etype_name
                           if bnn_key not in self.bnn:
                                print(f"警告: 未找到 BNN 层 '{bnn_key}'，跳过此边类型。")
                                continue

                           # 在反向图上执行消息传递和聚合
                           # 从源节点 (src_type) 复制特征 'h' 到消息 'm'
                           # 对收到的消息 'm' 按目标节点 (dst_type) 求平均值 'agg_temp'
                           msg_func = fn.copy_u('h', 'm')
                           reduce_func = fn.mean('m', 'agg_temp')
                           rev_hg.update_all(msg_func, reduce_func, etype=etype)

                           # 检查聚合结果是否存在
                           if 'agg_temp' in rev_hg.nodes[dst_type].data:
                               # 获取聚合后的邻居特征 (形状: [num_dst_type, ft_size])
                               neigh_features = rev_hg.nodes[dst_type].data.pop('agg_temp')
                               # 通过 BNN 层进行变换 (形状: [num_dst_type, hid_units])
                               mapped_features = self.bnn[bnn_key](neigh_features)
                               aggregated_results_l1.append(mapped_features)
                           # else:
                           #      print(f"    - 注意: 类型 '{dst_type}' 没有收到来自 '{src_type}' 通过 '{etype_name}' 的消息。")

                 # 如果收到了来自至少一种关系类型的聚合结果
                 if aggregated_results_l1:
                      # 将来自不同关系类型的结果堆叠起来 (形状: [num_relations, num_dst_type, hid_units])
                      stacked_results_l1 = torch.stack(aggregated_results_l1, dim=0)
                      # 对不同关系类型的结果求平均 (形状: [num_dst_type, hid_units])
                      v_summary_l1 = torch.mean(stacked_results_l1, dim=0)
                      # --- 将第一层结果写入全局 embs1 和 DGL 节点数据 ---
                      # 获取当前类型 ntype 的全局索引
                      global_idx_l1 = self.node_cnt[ntype]
                      # 写入全局张量
                      embs1[global_idx_l1] = v_summary_l1
                      # 同时写入 DGL 节点数据，供下一层聚合使用
                      # （注意：需要写回原始图 hg 和反向图 rev_hg，因为下一层聚合还需要在 rev_hg 上进行）
                      hg.nodes[ntype].data['em1'] = v_summary_l1
                      rev_hg.nodes[ntype].data['em1'] = v_summary_l1
                      # print(f"  完成类型 '{ntype}' 的第一层聚合，结果形状: {v_summary_l1.shape}")
                 # else:
                 #      print(f"  类型 '{ntype}' 未收到任何第一层聚合结果。")


            # print("开始第二层聚合...")
            # === 第二层聚合 ===
            # 与第一层类似，但使用 'em1' 作为输入特征，并使用 '1'+etype_name 的 BNN 层
            for ntype in hg.ntypes:
                 if ntype not in self.node_cnt: continue
                 # print(f"  处理目标类型: '{ntype}'")
                 aggregated_results_l2 = []
                 for etype in rev_hg.canonical_etypes:
                      src_type, etype_name, dst_type = etype
                      if dst_type == ntype:
                           # print(f"    - 通过关系 '{src_type}' --'{etype_name}'--> '{dst_type}' 进行聚合 (L2)")
                           bnn_key = '1' + etype_name
                           if bnn_key not in self.bnn:
                                print(f"警告: 未找到 BNN 层 '{bnn_key}'，跳过此边类型。")
                                continue

                           # 检查源节点是否有 'em1' 特征
                           if 'em1' not in rev_hg.nodes[src_type].data:
                                # print(f"    - 警告: 源类型 '{src_type}' 缺少 'em1' 特征，无法进行第二层聚合。")
                                continue # 跳过这条边

                           # 执行聚合 (使用 em1 作为消息源)
                           msg_func = fn.copy_u('em1', 'm')
                           reduce_func = fn.mean('m', 'agg_temp_l2')
                           rev_hg.update_all(msg_func, reduce_func, etype=etype)

                           if 'agg_temp_l2' in rev_hg.nodes[dst_type].data:
                               neigh_features_l1 = rev_hg.nodes[dst_type].data.pop('agg_temp_l2')
                               mapped_features_l2 = self.bnn[bnn_key](neigh_features_l1)
                               aggregated_results_l2.append(mapped_features_l2)
                           # else:
                           #      print(f"    - 注意: 类型 '{dst_type}' 没有收到来自 '{src_type}' 通过 '{etype_name}' 的第二层消息。")

                 # 如果收到了第二层聚合结果
                 if aggregated_results_l2:
                      # 对不同关系类型结果求平均 (形状: [num_dst_type, hid_units2])
                      stacked_results_l2 = torch.stack(aggregated_results_l2, dim=0)
                      v_summary_l2 = torch.mean(stacked_results_l2, dim=0)

                      # --- 拼接原始特征并通过 FC 层 ---
                      # 获取当前类型 ntype 的全局索引和原始特征
                      global_idx_l2 = self.node_cnt[ntype]
                      original_features = features[global_idx_l2] # 从全局有序特征中提取

                      # 拼接第二层聚合结果和原始特征 (形状: [num_dst_type, hid_units2 + ft_size])
                      cat_features = torch.cat((v_summary_l2, original_features), dim=1)

                      # 通过特定于类型的 FC 层 (形状: [num_dst_type, out_ft])
                      if ntype in self.fc:
                           final_emb = self.fc[ntype](cat_features)
                           # 写入全局 embs2 张量
                           embs2[global_idx_l2] = final_emb
                           # print(f"  完成类型 '{ntype}' 的第二层聚合与FC，结果形状: {final_emb.shape}")
                      else:
                           print(f"警告: 未找到类型 '{ntype}' 的 FC 层，无法计算最终嵌入。")

                 # else:
                 #      print(f"  类型 '{ntype}' 未收到任何第二层聚合结果。")

            # --- 选择最终的异构嵌入 ---
            # 根据数据集选择使用第一层还是第二层的聚合结果
            # 注意：这里的 'acm4GTN' 是示例，需要根据实际数据集名称调整
            # 如果你的数据集名称是 'ACM' (来自 Herotrainer 的 args.dataset)，确保这里也用 'ACM'
            if self.args.dataset in ['acm4GTN', 'ACM']: # 如果是 ACM 数据集变体
                embs_het_full = embs1
            else:
                embs_het_full = embs2

            # --- HERO 同构嵌入计算 ---
            # print("计算同构嵌入...")
            # 使用 MLP 处理全局特征
            # 注意：只处理目标节点（前 args.node_num 个）
            emb_f_full = self.mlp(features) # 先对所有节点计算，后面再切片
            # emb_f = emb_f_full[:self.args.node_num] # 取出目标节点的 MLP 输出
            # !!! 修正：应该对距离矩阵对应的节点（前 args.node_num 个）的 MLP 输出进行计算
            emb_f_target = emb_f_full[:self.args.node_num]


            # 同构计算公式 (与原始代码一致)
            coe2 = 1.0 / (self.args.beta + 1e-9) # 防止 beta 为 0
            # H_target^T * H_target
            res = torch.mm(emb_f_target.T, emb_f_target)
            # 矩阵求逆部分 (I + c * H^T * H)^(-1)
            identity_matrix = torch.eye(emb_f_target.shape[1], device=features.device)
            inv = torch.inverse(identity_matrix + coe2 * res)
            # (I + c * H^T * H)^(-1) * (H^T * H)
            res = torch.mm(inv, res)
            # B = c*H - c^2*H*(I + c*H^T*H)^(-1)*(H^T*H)
            res = coe2 * emb_f_target - coe2**2 * torch.mm(emb_f_target, res)
            # H^T * B
            tmp = torch.mm(emb_f_target.T, res)
            # Part1 = H * (H^T * B)
            part1 = torch.mm(emb_f_target, tmp)

            # Part2 = (-alpha / 2) * D * B
            # distance 矩阵应该只包含目标节点之间的距离
            if distance.shape[0] != self.args.node_num or distance.shape[1] != self.args.node_num:
                 print(f"警告: 传入的 distance 矩阵形状 ({distance.shape}) 与目标节点数 ({self.args.node_num}) 不符！")
                 # 可能需要报错或采取默认行为
                 part2 = torch.zeros_like(part1) # 例如，将 part2 设为零
            else:
                 part2 = (-self.args.alpha / 2.0) * torch.mm(distance, res)

            # 同构嵌入 = Part1 + Part2
            embs_hom_target = part1 + part2

            # --- 返回目标节点的嵌入 ---
            # 从完整的异构嵌入中切片出目标节点的部分
            embs_het_target = embs_het_full[:self.args.node_num]

            return embs_het_target, embs_hom_target

    def embed(self, hg, features, distance):
        """
        生成节点嵌入（通常在评估时调用，不计算梯度）。
        逻辑与 forward 基本相同，但最后返回 .detach() 的结果。
        """
        # print("调用 embed 方法生成节点嵌入...")
        # 设置模型为评估模式 (虽然此方法本身不训练，但依赖的层如 Dropout 应关闭)
        self.eval()
        with torch.no_grad(): # 确保不计算梯度
            # 完全复用 forward 的逻辑来计算嵌入
            embs_het_target, embs_hom_target = self.forward(hg, features, distance)

            # 返回分离后的张量 (不带梯度信息)
            return embs_het_target.detach(), embs_hom_target.detach()


class MLP(nn.Module):
    """ 简单的多层感知机 """
    def __init__(self, dim, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        # 如果 args 中定义了 dropout，则使用它，否则用默认值
        dropout_rate = getattr(self.args, 'dropout', dropprob) if hasattr(self, 'args') else dropprob
        self.dropout = torch.nn.Dropout(dropout_rate)
        for i in range(len(dim) - 1):
            self.net.append(nn.Linear(dim[i], dim[i+1]))
        print(f"MLP 初始化: 维度={dim}, Dropout={dropout_rate}")


    def forward(self, x):
        # 遍历除最后一层外的所有线性层
        for i in range(len(self.net) - 1):
            x = self.net[i](x)
            x = F.relu(x) # ReLU 激活
            x = self.dropout(x) # Dropout

        # 最后一层线性变换 (通常不加激活和 Dropout)
        y = self.net[-1](x)
        return y
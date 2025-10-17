import torch.nn as nn
import torch as th
import torch
import copy as copy
import scipy.sparse as sp
import os
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import warnings
from tqdm import tqdm
from openhgnn.trainerflow import BaseFlow, register_flow
from openhgnn.models import build_model
import dgl  # 确保导入 dgl

warnings.filterwarnings("ignore")


@register_flow("herotrainer")
class Herotrainer(BaseFlow):
    def __init__(self, args):
        super(Herotrainer, self).__init__(args)
        self.args = args

        prepared_data = self._prepare_hero_data_and_distance(self.hg)
        self.features = prepared_data['features'].to(self.device)
        self.feature_distance = prepared_data['feature_distance'].to(self.device)
        self.node_cnt = prepared_data['node_cnt']

        self.args.node_cnt = self.node_cnt
        self.args.node_size = prepared_data['num_nodes_global']
        if self.features is not None and self.features.ndim == 2:
            self.args.ft_size = self.features.shape[1]
        else:
            print(
                f"警告: 处理后的 self.features 形状不正确或为None，无法设置 args.ft_size。将尝试使用配置中的 hidden_dim。")
            if hasattr(args, 'hidden_dim'):  # 假设配置中有 hidden_dim 作为特征原始维度
                self.args.ft_size = args.hidden_dim
            else:
                raise ValueError("无法确定特征维度 ft_size。")

        self.args.node_num = 3025  # HERO 目标节点数

        # 检查 p_equidim vs g_equidim (用于 loss_spe_nontrival_1 和 loss_spe_inv)
        if hasattr(self.args, 'p_equidim') and hasattr(self.args, 'g_equidim'):
            if self.args.p_equidim != self.args.g_equidim:
                print(
                    f"重要警告: args.p_equidim ({self.args.p_equidim}) 与 args.g_equidim ({self.args.g_equidim}) 不相等。")
                print("  这将导致 loss_spe_nontrival_1 和 loss_spe_inv 的对角线求和在非方阵上进行。")
                print("  请检查 HERO 论文/原始代码，确认这些维度是否需要相等，并在配置文件中修改。")
        else:
            print("警告: 缺少 p_equidim 或 g_equidim 参数，无法检查维度匹配性 (loss_spe_nontrival_1, loss_spe_inv)。")

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        self.g = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_dim, bias=False),
                               nn.ReLU(inplace=True)).to(self.device)
        self.g_1 = nn.Sequential(nn.Linear(self.args.out_ft, self.args.g_equidim, bias=False),
                                 nn.ReLU(inplace=True)).to(self.device)
        self.p_1 = nn.Sequential(nn.Linear(self.args.g_equidim, self.args.p_equidim, bias=False),
                                 nn.ReLU(inplace=True)).to(self.device)

        self.args.batch_size = 1
        print("Herotrainer 初始化完成.")
        if self.features is not None: print(f"  全局特征形状: {self.features.shape}")
        if self.feature_distance is not None: print(f"  特征距离形状: {self.feature_distance.shape}")
        print(f"  全局节点总数 (args.node_size): {self.args.node_size}")
        print(f"  目标节点数量 (args.node_num): {self.args.node_num}")

    def _prepare_hero_data_and_distance(self, hg):
        print(f"为 HERO 模型准备数据 (数据集: {self.args.dataset})...")
        if self.args.dataset in ['acm4GTN', 'acm4NSHE', 'acm4NARS', 'acm4HeCo', 'ACM']:
            node_order = ['paper', 'author', 'subject']
            expected_counts = {'paper': 3025, 'author': 5912, 'subject': 57}
            num_target_nodes = 3025
            print(f"使用 ACM 数据集的节点顺序: {node_order}")
        else:
            print(
                f"警告: 未为数据集 {self.args.dataset} 定义显式的全局节点顺序。将使用 DGL 图的默认 ntypes 顺序: {hg.ntypes}。")
            node_order = hg.ntypes
            expected_counts = {}
            if not node_order: raise ValueError("DGL 图中没有任何节点类型 (hg.ntypes is empty)。")
            num_target_nodes = hg.num_nodes(node_order[0]) if node_order else 0
            print(
                f"警告: 自动假设目标节点类型为 '{node_order[0] if node_order else 'None'}'，数量为 {num_target_nodes}。")

        actual_counts = {}
        valid_node_order = []
        for ntype in node_order:
            if ntype not in hg.ntypes:
                print(f"警告: 期望的节点类型 '{ntype}' 在图中未找到，将跳过。")
                continue
            valid_node_order.append(ntype)
            actual_counts[ntype] = hg.num_nodes(ntype)
            if ntype in expected_counts and actual_counts[ntype] != expected_counts[ntype]:
                print(
                    f"警告: 节点类型 '{ntype}' 的数量不匹配! 期望 {expected_counts[ntype]}, 实际 {actual_counts[ntype]}。")
        node_order = valid_node_order

        node_cnt = {}
        current_idx = 0
        print("创建 node_cnt 全局索引映射:")
        for ntype in node_order:
            if ntype not in actual_counts: continue
            num = actual_counts[ntype]
            node_cnt[ntype] = th.arange(current_idx, current_idx + num)
            print(f"  - 类型 '{ntype}': 全局索引 {current_idx} - {current_idx + num - 1}")
            current_idx += num
        num_nodes_global = current_idx
        print(f"计算出的全局节点总数: {num_nodes_global}")

        feature_list = []
        feature_dim = -1
        print("按全局顺序提取、拼接并二值化特征:")
        for ntype in node_order:
            if ntype not in actual_counts: continue
            feature_key = None
            if 'h' in hg.nodes[ntype].data:
                feature_key = 'h'
            elif 'feat' in hg.nodes[ntype].data: ## 鲁棒一点
                feature_key = 'feat'

            if feature_key:
                features_ntype_raw = hg.nodes[ntype].data[feature_key].clone()
                # print(f"  - 提取类型 '{ntype}' 原始特征，形状: {features_ntype_raw.shape}, 样本均值: {features_ntype_raw.float().mean().item():.4f}")
                features_ntype_binarized = features_ntype_raw
                features_ntype_binarized[features_ntype_binarized > 0] = 1.0
                # print(f"    类型 '{ntype}' 二值化后特征，形状: {features_ntype_binarized.shape}, 样本均值: {features_ntype_binarized.float().mean().item():.4f}")
                feature_list.append(features_ntype_binarized)
                if feature_dim == -1:
                    feature_dim = features_ntype_binarized.shape[1]
                elif feature_dim != features_ntype_binarized.shape[1]:
                    print(
                        f"警告: 类型 '{ntype}' 特征维度 ({features_ntype_binarized.shape[1]}) 与之前 ({feature_dim}) 不匹配。")
            else:
                num_nodes_ntype = actual_counts[ntype]
                if feature_dim != -1:
                    print(
                        f"警告: 未找到类型 '{ntype}' 的特征 ('h' 或 'feat')，将填充零特征，形状: ({num_nodes_ntype}, {feature_dim})。")
                    feature_list.append(th.zeros((num_nodes_ntype, feature_dim)))
                else:
                    raise ValueError(f"无法处理类型 '{ntype}' 的缺失特征，因为维度未知。")

        features_globally_ordered_binarized = th.cat(feature_list, dim=0)
        print(f"拼接后的全局【二值化】特征张量形状: {features_globally_ordered_binarized.shape}")
        if features_globally_ordered_binarized.shape[0] != num_nodes_global:
            print(f"警告: 最终特征行数 ({features_globally_ordered_binarized.shape[0]}) 与计算的全局节点总数 ({num_nodes_global}) 不匹配！")

        feature_distance = th.empty((0, 0), device=features_globally_ordered_binarized.device)  # 默认空
        if features_globally_ordered_binarized.shape[0] >= num_target_nodes:
            target_node_features_binarized = features_globally_ordered_binarized[0:num_target_nodes]
            print(
                f"为前 {num_target_nodes} 个目标节点计算特征距离 (使用二值化特征)，特征形状: {target_node_features_binarized.shape}")
            feature_distance = self.pairwise_distance(target_node_features_binarized)
            feature_distance = F.normalize(feature_distance, p=2, dim=1)
            num_elements = feature_distance.numel()
            if num_elements > 0 and hasattr(self.args,
                                            'edge_rate') and self.args.edge_rate > 0 and self.args.edge_rate < 1:
                k = int(num_elements * self.args.edge_rate)
                if k >= 1:
                    k = min(k, num_elements - 1)
                    if k == 0 and num_elements > 0: k = 1  # 确保 k 至少为1如果矩阵非空

                    if feature_distance.view(-1).shape[0] > 0 and k > 0 and k < feature_distance.view(-1).shape[
                        0]:  # k 必须小于元素总数
                        kth_val = th.kthvalue(feature_distance.view(-1), k)[0]
                        mask = (feature_distance > kth_val).float()
                        feature_distance = feature_distance * mask
                        print("特征距离已进行归一化和基于 edge_rate 的稀疏化处理。")
                    elif k >= feature_distance.view(-1).shape[0]:
                        print(
                            f"警告: k值({k}) 大于或等于元素数量({feature_distance.view(-1).shape[0]})，不进行kthvalue稀疏化。")
                    else:
                        print("警告: 无法有效进行kthvalue稀疏化 (k=0或矩阵为空)。")
                else:
                    print("警告: edge_rate 过低 (k<1)，无法稀疏化。")
            elif not hasattr(self.args, 'edge_rate'):
                print("警告: args 中未定义 edge_rate，跳过距离矩阵稀疏化。")
            else:
                print("警告: 距离矩阵为空或 edge_rate 无效，跳过稀疏化。")
        else:
            print(
                f"错误: 全局特征行数 ({features_globally_ordered_binarized.shape[0]}) 小于目标节点数 ({num_target_nodes})，无法计算距离。")

        print(f"最终特征距离矩阵形状: {feature_distance.shape}")
        return {'features': features_globally_ordered_binarized, 'feature_distance': feature_distance,
                'node_cnt': node_cnt, 'num_nodes_global': num_nodes_global}

    def pairwise_distance(self, x, y=None):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, min=0.0)

    def train(self):
        cnt_wait = 0;
        best = 1e9
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        print("开始训练...")
        for epoch in tqdm(range(self.args.nb_epochs), desc="训练进度"):
            self.model.train()
            self.optimizer.zero_grad()
            emb_het, emb_hom = self.model(self.hg, self.features, self.feature_distance)
            ## 得到的同质信息和异构信息
            embs_P1 = self.g(emb_het)
            embs_P2 = self.g(emb_hom)
            #######################################################################
            # The second term in Eq. (10): uniformity loss
            intra_c = (embs_P1).T @ (embs_P1).contiguous()
            intra_c = torch.exp(F.normalize(intra_c, p=2, dim=1)).sum()
            loss_uni = torch.log(intra_c).mean()

            intra_c_2 = (embs_P2).T @ (embs_P2).contiguous()
            intra_c_2 = torch.exp(F.normalize(intra_c_2, p=2, dim=1)).sum()
            loss_uni += torch.log(intra_c_2).mean()
            # C_P = embs_P2.T @ embs_P2
            # C_tilde_P = embs_P1.T @ embs_P1
            # loss_uni = torch.logsumexp(sum_of_cov_matrices.view(-1), dim=0)
            #######################################################################
            # The first term in Eq. (10): invariance loss
            # 这里不是MSE，但是也能起到类似的效果，而且效果好得多，实际使用行归一化后的整体嵌入集合的互协方差矩阵的对角线元素之和的负数
            inter_c = embs_P1.T @ embs_P2
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_inv = -torch.diagonal(inter_c).sum()
            # loss_inv = torch.nn.functional.mse_loss(embs_P2, embs_P1, reduction='sum')
            #######################################################################
            # Projection and Transformation
            embs_Q2 = self.g_1(emb_het)
            embs_Q1 = self.g_1(emb_hom)
            embs_Q1_trans = self.p_1(embs_Q1)

            # The first term in Eq. (11)
            inter_c = embs_Q1_trans.T @ embs_Q2
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_spe_inv = -torch.diagonal(inter_c).sum()

            #######################################################################
            # The second term in Eq. (11)
            inter_c = embs_Q1_trans.T @ embs_Q1
            inter_c = F.normalize(inter_c, p=2, dim=1)
            loss_spe_nontrival_1 = torch.diagonal(inter_c).sum()

            inter_c_1 = embs_Q1_trans.T @ embs_P2
            inter_c_1 = F.normalize(inter_c_1, p=2, dim=1)
            loss_spe_nontrival_2 = torch.diagonal(inter_c_1).sum()
            ########################################################################

            loss_consistency = loss_inv + self.args.gamma * loss_uni
            loss_specificity = loss_spe_inv - self.args.eta * (loss_spe_nontrival_1 + loss_spe_nontrival_2)

            loss = loss_consistency + self.args.lambbda * loss_specificity

            loss.backward()
            self.optimizer.step()
            train_loss = loss.item()

            if (train_loss < best):
                best = train_loss
                cnt_wait = 0
                args_to_save = {k: v for k, v in vars(self.args).items() if
                                not callable(v) and not isinstance(v, torch.nn.Module) and not isinstance(v,
                                                                                                          dgl.DGLGraph)}
                # 移除可能包含 RLock 的对象，例如 logger
                if 'logger' in args_to_save:
                    del args_to_save['logger']

                torch.save({
                    'epoch': epoch, 'model_state_dict': copy.deepcopy(self.model.state_dict()),
                    'g_state_dict': copy.deepcopy(self.g.state_dict()),
                    'g_1_state_dict': copy.deepcopy(self.g_1.state_dict()),
                    'p_1_state_dict': copy.deepcopy(self.p_1.state_dict()),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'loss': train_loss,
                    'hyperparameters': args_to_save,  # 保存处理后的参数字典
                    'node_cnt': self.node_cnt
                }, os.path.join(self.args.save_dir, f'hero_best_model_{self.args.dataset}.pth'))
            else:
                cnt_wait += 1

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.args.nb_epochs}: 训练损失 {train_loss:.4f}, 等待次数 {cnt_wait}/{self.args.patience}, 当前最佳 {best:.4f}")

            if cnt_wait >= self.args.patience:
                print(f"Epoch {epoch + 1}: 连续 {self.args.patience} 个 epoch 损失没有改善，触发早停！")
                break
        print("训练结束.")

        best_model_path = os.path.join(self.args.save_dir, f'hero_best_model_{self.args.dataset}.pth')
        if os.path.exists(best_model_path):
            print(f"加载最佳模型: {best_model_path}")
            ckpt = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.g.load_state_dict(ckpt['g_state_dict'])
            self.g_1.load_state_dict(ckpt['g_1_state_dict'])
            self.p_1.load_state_dict(ckpt['p_1_state_dict'])
            print("最佳模型参数已加载。")
            # 如果需要，可以从 ckpt['hyperparameters'] 恢复部分 args 设置
            # for key, value in ckpt['hyperparameters'].items():
            #     setattr(self.args, key, value)
            print("使用最佳模型进行评估...")
            test_micro_f1, test_macro_f1 = self.evaluate()
            print(f"最终测试结果 (最佳模型): Micro-F1 = {test_micro_f1:.5f}, Macro-F1 = {test_macro_f1:.5f}")
            return test_macro_f1
        else:
            print("错误：未找到保存的最佳模型文件。无法进行最终评估。")
            return None

    def _full_test_step(self):
        micro_f1, macro_f1 = self.evaluate()
        metric = {'test_micro_f1': micro_f1, 'test_macro_f1': macro_f1}
        return metric

    @torch.no_grad()
    def evaluate(self, split='test'):
        self.model.eval();
        self.g.eval();
        self.g_1.eval();
        self.p_1.eval()
        graph = self.hg
        eval_features = self.features.to(self.device)
        eval_feature_distance = self.feature_distance.to(self.device)

        if eval_features is None or eval_feature_distance is None:
            print(f"错误: 评估时 features 或 feature_distance 为 None。")
            return -1.0, -1.0

        embs_het, emb_hom = self.model.embed(graph, eval_features, eval_feature_distance)
        h_concat = torch.cat((embs_het, emb_hom), dim=1)
        embeddings_np = h_concat.detach().cpu().numpy()

        if not hasattr(self, 'task') or self.task is None:
            print("错误: 无法访问 'self.task' 来获取标签和划分。")
            return -1.0, -1.0
        try:
            labels_all = self.task.get_labels().cpu().numpy()
            idx_split = None
            if split == 'train':
                idx_split = self.train_idx if hasattr(self, 'train_idx') else self.task.dataset.train_idx
            elif split == 'val':
                idx_split = self.val_idx if hasattr(self, 'val_idx') and self.val_idx is not None else getattr(
                    self.task.dataset, 'val_idx', None)
                if idx_split is None:
                    print(f"警告: '{split}' 集索引未找到，无法评估。")
                    return 0.0, 0.0  # 或者其他默认值
            elif split == 'test':
                idx_split = self.test_idx if hasattr(self, 'test_idx') else self.task.dataset.test_idx

            if idx_split is None:
                print(f"错误: 未能获取 '{split}' 集的划分索引。")
                return -1.0, -1.0

            idx_split = np.array(idx_split.cpu().numpy() if torch.is_tensor(idx_split) else idx_split).astype(int)

            if idx_split.size == 0:
                print(f"警告: '{split}' 集索引为空。无法进行评估。")
                return 0.0, 0.0
            if idx_split.max() >= embeddings_np.shape[0] or idx_split.max() >= len(labels_all):
                max_emb_idx = embeddings_np.shape[0] - 1
                max_lbl_idx = len(labels_all) - 1
                print(
                    f"错误: '{split}' 集最大索引 ({idx_split.max()}) 超出嵌入数量 ({max_emb_idx}) 或标签数量 ({max_lbl_idx})!")
                return -1.0, -1.0

        except Exception as e:
            print(f"获取标签或划分时出错: {e}")
            import traceback
            traceback.print_exc()
            return -1.0, -1.0

        train_idx_for_lr = self.train_idx if hasattr(self, 'train_idx') else self.task.dataset.train_idx
        train_idx_for_lr = np.array(
            train_idx_for_lr.cpu().numpy() if torch.is_tensor(train_idx_for_lr) else train_idx_for_lr).astype(int)

        if train_idx_for_lr.size == 0:
            print("错误: 用于训练分类器的训练集索引为空。")
            return -1.0, -1.0
        if train_idx_for_lr.max() >= embeddings_np.shape[0] or train_idx_for_lr.max() >= len(labels_all):
            print(f"错误: 用于训练分类器的训练集索引 ({train_idx_for_lr.max()}) 越界！")
            return -1.0, -1.0

        train_embs_for_lr = embeddings_np[train_idx_for_lr]
        train_labels_for_lr = labels_all[train_idx_for_lr]
        eval_embs_target_split = embeddings_np[idx_split]
        eval_labels_target_split = labels_all[idx_split]

        total_micro_f1, total_macro_f1 = 0, 0;
        num_runs_actual = 0
        num_runs_total = 5
        for rs in range(num_runs_total):
            lr = LogisticRegression(max_iter=1000, random_state=rs, solver='liblinear', C=1.0, n_jobs=-1)
            try:
                # 检查训练标签是否只有一个类别
                if len(np.unique(train_labels_for_lr)) < 2:
                    print(f"警告 (Run {rs + 1}): Logistic Regression 的训练标签中只有一个类别，跳过此轮评估。")
                    continue  # 跳过这一轮

                lr.fit(train_embs_for_lr, train_labels_for_lr)
                Y_pred = lr.predict(eval_embs_target_split)
                f1_micro = metrics.f1_score(eval_labels_target_split, Y_pred, average='micro')
                f1_macro = metrics.f1_score(eval_labels_target_split, Y_pred, average='macro', zero_division=0)
                total_micro_f1 += f1_micro;
                total_macro_f1 += f1_macro
                num_runs_actual += 1
            except ValueError as ve:
                print(f"Logistic Regression 训练/预测时出错 (run {rs + 1}): {ve}")

        avg_micro_f1 = total_micro_f1 / num_runs_actual if num_runs_actual > 0 else 0.0
        avg_macro_f1 = total_macro_f1 / num_runs_actual if num_runs_actual > 0 else 0.0
        print(
            f"\t[{split.capitalize()} Classification Avg ({num_runs_actual} runs)] Macro-F1 = {avg_macro_f1:.5f}, Micro-F1 = {avg_micro_f1:.5f}")
        return avg_micro_f1, avg_macro_f1
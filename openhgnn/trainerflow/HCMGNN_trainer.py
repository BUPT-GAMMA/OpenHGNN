import numpy as np
from sklearn.model_selection import KFold

from . import BaseFlow, register_flow
import warnings
import torch as th
import torch.nn.functional as F
import time
import os
import pandas as pd  # For loading CSVs

from openhgnn.utils import HCMGNN_utils
from openhgnn.models import build_model
from ..tasks import build_task

# Import necessary functions from hcm_dataset for data loading helpers
from ..dataset.hcm_dataset import neg_data_generate, get_train_val_data, get_indep_data  # These are helper functions

warnings.filterwarnings("ignore")


@register_flow("HCMGNN_trainer")
class HCMGNN_trainer(BaseFlow):
    def __init__(self, args):
        super(HCMGNN_trainer, self).__init__(args)
        # 初始化任务，获取数据集实例
        self.task = build_task(args)
        # self.dataset 是 HCM_recommendation 实例，它已经处理并加载了所有数据
        self.dataset = self.task.dataset
        self.metapaths = [['g', 'm', 'd'], ['m', 'g', 'd'], ['d', 'g', 'm'], ['d', 'm', 'g'], ['g', 'd', 'm'],
                                 ['m', 'd', 'g']]
        self.etypes = [[0, 1], [2, 3], [4, 0], [5, 2], [3, 5], [1, 4]]
        # 从 dataset 获取 DGL 图、特征和独立测试集数据
        # HCM_recommendation.get_split() 返回 (graph, train_data_combined_indep, test_data_combined_indep)
        self.hg, self.train_data_indep, self.test_data_indep = self.dataset.get_split()

        # 获取特征字典和维度字典
        self.features = self.dataset.features
        self.in_size = self.dataset.in_size

        # 将特征数据和图数据移动到指定设备
        # self.features 是 PyTorch Tensor 字典，需要移动
        if isinstance(self.features, dict):
            for k in self.features:
                if hasattr(self.features[k], 'to'):
                    self.features[k] = self.features[k].to(self.device)
        elif hasattr(self.features, 'to'):
            self.features = self.features.to(self.device)

        # 初始化 HCMGNN 模型 (在 _run_experiment_loop 中会重新初始化以确保每次实验都是新的模型)
        # 这里只是一个占位符，用于获取模型类名
        self.model_name = self.model  # self.model 来自 BaseFlow，通常是模型名称字符串

        # 初始化损失函数和评估指标 (这些是通用的，可以在每次实验中重用)
        self.myloss = HCMGNN_utils.Myloss()
        self.mrr = HCMGNN_utils.MRR()
        self.matrix = HCMGNN_utils.Matrix()

        np.random.seed(self.args.seed)
        th.manual_seed(self.args.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(self.args.seed)

    def _run_experiment_loop(self, train_data_combined, test_data_combined, val_data_pos_for_eval,hg_graph,
                             experiment_name="Experiment"):
        """
        Helper to run a single training and evaluation cycle for a given dataset split.
        This method re-initializes the model and optimizer for each run.
        """
        print(f"--- Starting {experiment_name} Training Loop ---")

        # Re-initialize model and optimizer for a fresh start for each experiment/fold
        self.model = build_model(self.model_name).build_model_from_args(
            meta_paths=self.metapaths,
            test_data=val_data_pos_for_eval,  # Positive samples for evaluation metrics
            in_size=self.in_size,  # Use shared in_size from __init__
            hidden_size=self.args.hidden_size,
            num_heads=self.args.num_heads,
            dropout=self.args.dropout,
            etypes=self.etypes  # Use the pre-built graph's edge types
        ).to(self.device)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        # Reset tracking metrics for each run
        self.trainloss = []
        self.valloss = []
        self.result_list = []
        self.hits_max_matrix = np.zeros((1, 3))
        self.NDCG_max_matrix = np.zeros((1, 3))
        self.patience_num_matrix = np.zeros((1, 1))
        self.MRR_max_matrix = np.zeros((1, 1))
        self.epoch_max_matrix = np.zeros((1, 1))

        # Convert current train and test data to tensors
        train_data_tensor = th.from_numpy(train_data_combined).to(self.device)

        # For evaluation, create a shuffled index and tensor for the current test_data_combined
        shuffle_index_current = np.random.choice(range(len(test_data_combined)), len(test_data_combined), replace=False)
        task_test_data_tensor_current = th.from_numpy(test_data_combined[shuffle_index_current]).to(self.device)

        # 此处可以调整训练轮数
        max_epoch = self.args.num_epochs
        # max_epoch = 101
        for self.epoch in range(max_epoch):
            step_start = time.time()

            # --- Train Step ---
            self.model.train()  # Set model to training mode
            self.optimizer.zero_grad()  # Clear gradients

            # Forward pass: HCMGNN takes the graph, features, and training data tensor
            score_train_predict = self.model(hg_graph, self.features, train_data_tensor)
            train_label = th.unsqueeze(train_data_tensor[:, 3], 1)  # Extract labels and add dimension

            train_loss = self.loss_calculation(score_train_predict, train_label)

            train_loss.backward()  # Backpropagation
            self.optimizer.step()  # Update model parameters
            self.trainloss.append(train_loss.item())

            # --- Evaluate Step ---
            self.model.eval()  # Set model to evaluation mode
            with th.no_grad():  # Disable gradient calculation for evaluation
                score_val_predict = self.model(hg_graph, self.features, task_test_data_tensor_current)
                val_label = th.unsqueeze(task_test_data_tensor_current[:, 3], 1)
                val_label = val_label.to(th.float)  # Ensure labels are float type
                val_loss = self.loss_calculation(score_val_predict, val_label)

                predict_val = np.squeeze(score_val_predict.detach().cpu().numpy())

                # Calculate metrics (Hits@n, NDCG@n, MRR)
                hits5, ndcg5, _, _ = self.matrix(5, 30, predict_val, len(val_data_pos_for_eval), shuffle_index_current)
                hits3, ndcg3, _, _ = self.matrix(3, 30, predict_val, len(val_data_pos_for_eval), shuffle_index_current)
                hits1, ndcg1, _, _ = self.matrix(1, 30, predict_val, len(val_data_pos_for_eval), shuffle_index_current)
                MRR_num, _ = self.mrr(30, predict_val, len(val_data_pos_for_eval), shuffle_index_current)

                result = [val_loss.item(), hits5, hits3, hits1, ndcg5, ndcg3, ndcg1, MRR_num]
                self.result_list.append(result)

            step_end = time.time()
            print(f'Epoch: {self.epoch + 1}, '
                  f'Train loss: {train_loss.item():.4f}, '
                  f'Val Loss: {result[0]:.4f}, '
                  f'Hits@5: {result[1]:.6f}, '
                  f'Hits@3: {result[2]:.6f}, '
                  f'Hits@1: {result[3]:.6f}, '
                  f'NDCG@5: {result[4]:.6f}, '
                  f'NDCG@3: {result[5]:.6f}, '
                  f'NDCG@1: {result[6]:.6f}, '
                  f'MRR: {result[7]:.6f}')

            # Early stopping logic
            self.patience_num_matrix = HCMGNN_utils.ealy_stop(
                self.hits_max_matrix, self.NDCG_max_matrix, self.MRR_max_matrix,
                self.patience_num_matrix, self.epoch_max_matrix,
                self.epoch, hits1, hits3, hits5, ndcg1, ndcg3, ndcg5, MRR_num
            )
            if self.patience_num_matrix[0][0] >= self.args.patience:
                self.logger.info(f"Early stopping at epoch {self.epoch + 1} due to patience.")
                break

        optimal_epoch_index = int(self.epoch_max_matrix[0][0])
        print(f'Optimal epoch for this {experiment_name} run: {optimal_epoch_index + 1}')
        return self.result_list[optimal_epoch_index][1:]

    def _run_independent_test(self):
        """
        Executes the independent test experiment.
        """
        print('Starting the independent test experiment')

        # Re-extract positive test data for evaluation from the combined independent test data
        val_data_pos_for_eval = self.test_data_indep[np.where(self.test_data_indep[:, -1] == 1)]

        indep_result = self._run_experiment_loop(
            train_data_combined=self.train_data_indep,
            test_data_combined=self.test_data_indep,
            val_data_pos_for_eval=val_data_pos_for_eval,
            hg_graph=self.hg,
            experiment_name="Independent Test",
        )
        print('----------Independent Test Finished-----------')
        print(
            f'Independent test result: Hits@5:{indep_result[0]:.6f}, Hits@3:{indep_result[1]:.6f}, '
            f'Hits@1:{indep_result[2]:.6f}, NDCG@5:{indep_result[3]:.6f}, NDCG@3:{indep_result[4]:.6f}, '
            f'NDCG@1:{indep_result[5]:.6f}, MRR:{indep_result[6]:.6f}'
        )

        # Save result to file
        result_file_path = os.path.join(self.args.output_dir, 'HCMGNN_indep_print.txt')
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
        with open(result_file_path, 'a') as f:
            f.write('\t'.join(map(str, indep_result)) + '\n')

        return indep_result

    def _run_cross_validation(self):
        """
        Executes the 5-fold cross-validation experiment.
        """
        print('Starting the 5-fold CV experiment')
        Hits_5, Hits_3, Hits_1, NDCG_5, NDCG_3, NDCG_1, MRR = [list() for _ in range(7)]


        fold_num = 0
        for i in range(5):
            fold_num += 1
            print(f"--- Running CV Fold {fold_num} ---")

            # Load data for the current CV fold from CSVs generated by HCMDataset.process()
            # These paths are relative to args.output_dir
            cv_fold_dir = os.path.join(self.args.output_dir, 'CV_data', f'CV_{fold_num}')

            train_data_pos = np.loadtxt(os.path.join(cv_fold_dir, 'train_data_pos.csv'), delimiter=",")
            train_data_neg = np.loadtxt(os.path.join(cv_fold_dir, 'train_data_neg.csv'), delimiter=",")
            val_data_pos = np.loadtxt(os.path.join(cv_fold_dir, 'val_data_pos.csv'), delimiter=",")
            val_data_neg = np.loadtxt(os.path.join(cv_fold_dir, 'val_data_neg.csv'), delimiter=",")
            hg_graph = HCMGNN_utils.construct_hg(train_data_pos)
            # Combine positive and negative samples for training and validation
            train_data_combined_fold = np.vstack((train_data_pos, train_data_neg))
            np.random.shuffle(train_data_combined_fold)
            val_data_combined_fold = np.vstack((val_data_pos, val_data_neg))

            # Re-extract positive validation data for evaluation from the combined validation data
            val_data_pos_for_eval_fold = val_data_combined_fold[np.where(val_data_combined_fold[:, -1] == 1)]

            # Run the training loop for this fold
            fold_result = self._run_experiment_loop(
                train_data_combined=train_data_combined_fold,
                test_data_combined=val_data_combined_fold,  # For CV, val_data is the test data for this fold
                val_data_pos_for_eval=val_data_pos_for_eval_fold,
                hg_graph=hg_graph,
                experiment_name=f"CV Fold {fold_num}",
            )
            Hits_5.append(fold_result[0])
            Hits_3.append(fold_result[1])
            Hits_1.append(fold_result[2])
            NDCG_5.append(fold_result[3])
            NDCG_3.append(fold_result[4])
            NDCG_1.append(fold_result[5])
            MRR.append(fold_result[6])

        print('----------5-Fold CV Finished-----------')
        mean_hits5, mean_hits3, mean_hits1 = np.mean(Hits_5), np.mean(Hits_3), np.mean(Hits_1)
        mean_ndcg5, mean_ndcg3, mean_ndcg1 = np.mean(NDCG_5), np.mean(NDCG_3), np.mean(NDCG_1)
        mean_mrr = np.mean(MRR)

        print(
            f'5-fold CV result: Hits@5:{mean_hits5:.6f}, Hits@3:{mean_hits3:.6f}, '
            f'Hits@1:{mean_hits1:.6f}, NDCG@5:{mean_ndcg5:.6f}, NDCG@3:{mean_ndcg3:.6f}, '
            f'NDCG@1:{mean_ndcg1:.6f}, MRR:{mean_mrr:.6f}'
        )

        # Save result to file
        result_file_path = os.path.join(self.args.output_dir, 'HCMGNN_CV_print.txt')
        os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
        with open(result_file_path, 'a') as f:
            f.write(f'{mean_hits5}\t{mean_hits3}\t{mean_hits1}\t'
                    f'{mean_ndcg5}\t{mean_ndcg3}\t{mean_ndcg1}\t'
                    f'{mean_mrr}\n')

        return mean_hits5, mean_hits3, mean_hits1, mean_ndcg5, mean_ndcg3, mean_ndcg1, mean_mrr

    def train(self):
        """
        HCMGNN 模型的训练主循环，依次执行独立测试和 5 折交叉验证。
        """
        # Execute 5-Fold Cross Validation
        cv_results = self._run_cross_validation()
        # Execute Independent Test

        indep_results = self._run_independent_test()


        return indep_results


    def _mini_train_step(self):
        """
        此方法已弃用。其逻辑已集成到 _run_experiment_loop 中。
        """
        raise NotImplementedError("This method is deprecated. Use _run_experiment_loop for training steps.")

    def evaluate(self, data_type='eval'):
        """
        此方法已弃用。其逻辑已集成到 _run_experiment_loop 中。
        """
        raise NotImplementedError("This method is deprecated. Use _run_experiment_loop for evaluation steps.")

    def loss_calculation(self, scores, labels):
        """
        计算模型的损失。
        HCMGNN 使用自定义的 Myloss 函数。
        """
        # Myloss 实例在 __init__ 中已初始化
        # 损失计算不包含显式的 L2 正则化项，因为 Adam 优化器通过 weight_decay 隐式处理
        loss = self.myloss(scores, labels, self.args.loss_gamma)
        return loss

    def preprocess(self, dataIndex):
        """
        HCMGNN_trainer 不使用 DGL DataLoader 的 preprocess 模式，
        所有数据在 __init__ 中已加载并准备好。
        """
        pass


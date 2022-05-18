import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch
from openhgnn.models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping


@register_flow("recommendation")
class Recommendation(BaseFlow):
    """Recommendation flows."""

    def __init__(self, args=None):
        super(Recommendation, self).__init__(args)

        self.target_link = self.task.dataset.target_link
        self.args.out_node_type = self.task.dataset.out_ntypes
        self.args.out_dim = self.args.hidden_dim

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)
        self.reg_weight = 0.1

        self.metric = ['recall', 'ndcg']
        self.val_metric = 'recall'
        # self.topk_list = [5, 10, 20, 50, 100]
        self.topk = 20
        #self.evaluator = self.task.get_evaluator(self.metric)

        self.optimizer = (
            th.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        )
        self.patience = args.patience
        self.max_epoch = args.max_epoch
        
        self.num_neg = self.task.dataset.num_neg
        self.user_name = self.task.dataset.user_name
        self.item_name = self.task.dataset.item_name
        self.num_user = self.hg.num_nodes(self.user_name)
        self.num_item = self.hg.num_nodes(self.user_name)
        self.train_eid_dict = {
            etype: self.hg.edges(etype=etype, form='eid')
            for etype in self.hg.canonical_etypes}

    def preprocess(self):
        self.train_hg, self.val_hg, self.test_hg = self.task.get_split()
        self.train_neg_hg = self.task.dataset.construct_negative_graph(self.train_hg)
        self.train_hg = self.train_hg.to(self.device)
        self.val_hg = self.val_hg.to(self.device)
        self.test_hg = self.test_hg.to(self.device)
        self.negative_graph = self.train_neg_hg.to(self.device)
        self.positive_graph = self.train_hg.edge_type_subgraph([self.target_link])
        # generage complete user-item graph for evaluation
        # src, dst = th.arange(self.num_user), th.arange(self.num_item)
        # src = src.repeat_interleave(self.num_item)
        # dst = dst.repeat(self.num_user)
        # self.eval_graph = dgl.heterograph({('user', 'user-item', 'item'): (src, dst)}, {'user': self.num_user, 'item': self.num_item}).to(self.device)
        self.preprocess_feature()
        return

    def train(self):
        self.preprocess()
        epoch_iter = tqdm(range(self.max_epoch))
        stopper = EarlyStopping(self.args.patience, self._checkpoint)

        for epoch in tqdm(range(self.max_epoch), ncols=80):
            loss = 0
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                metric_dic = self._test_step(split='val')
                epoch_iter.set_description(
                    f"Epoch: {epoch:03d}, Recall@K: {metric_dic['recall']:.4f}, NDCG@K: {metric_dic['ndcg']:.4f}, Loss:{loss:.4f}"
                )
                early_stop = stopper.step_score(metric_dic[self.val_metric], self.model)
                if early_stop:
                    print('Early Stop!\tEpoch:' + str(epoch))
                    break

        print(f"Valid {self.val_metric} = {stopper.best_score: .4f}")
        stopper.load_model(self.model)
        test_metric_dic = self._test_step(split="test")
        #val_metric_dic = self._test_step(split="val")
        print(f"Test Recall@K = {test_metric_dic['recall']: .4f}, NDCG@K = {test_metric_dic['ndcg']: .4f}")
        # result = dict(Test_metric=test_metric_dic, Val_metric=val_metric_dic)
        # with open(self.args.results_path, 'w') as f:
        #     json.dump(result, f)
        #     f.write('\n')
        # self.task.dataset.save_results(result, self.args.results_path)
        return test_metric_dic['recall'], test_metric_dic['ndcg'], epoch
        # return dict(Test_metric=test_metric_dic, Val_metric=val_metric_dic)

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        p_score = self.ScorePredictor(positive_graph, embedding).repeat_interleave(self.num_neg)
        n_score = self.ScorePredictor(negative_graph, embedding)
        bpr_loss = -torch.log(torch.sigmoid(p_score - n_score)).mean()
        reg_loss = self.regularization_loss(embedding)
        return bpr_loss + self.reg_weight * reg_loss

    def ScorePredictor(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            for ntype in [self.user_name, self.item_name]:
                edge_subgraph.nodes[ntype].data['x'] = x[ntype]
            edge_subgraph.apply_edges(
                dgl.function.u_dot_v('x', 'x', 'score'), etype=self.target_link)
            score = edge_subgraph.edges[self.target_link].data['score']
            return score.squeeze()

    def regularization_loss(self, embedding):
        reg_loss = th.zeros(1, 1, device=self.device)
        for e in embedding.values():
            reg_loss += th.mean(e.pow(2))
        return reg_loss

    def _full_train_step(self):
        self.model.train()
        h_dict = self.input_feature()
        embedding = self.model(self.train_hg, h_dict)

        loss = self.loss_calculation(self.positive_graph, self.negative_graph, embedding)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print(loss.item())
        return loss.item()

    def _test_step(self, split=None, logits=None):
        self.model.eval()
        if split == 'val':
            test_graph = self.val_hg
        elif split == 'test':
            test_graph = self.test_hg
        else:
            raise ValueError('split must be in [val, test]')
        
        with th.no_grad():
            h_dict = self.input_feature()
            embedding = self.model(self.hg, h_dict)

            score_matrix = (embedding[self.user_name] @ embedding[self.item_name].T).detach().cpu().numpy()

            train_u, train_i = self.positive_graph.edges(etype=self.target_link)[0].cpu().numpy(), self.positive_graph.edges(etype=self.target_link)[1].cpu().numpy()
            score_matrix[train_u, train_i] = np.NINF
            ind = np.argpartition(score_matrix, -self.topk) # (num_users, num_items)
            ind = ind[:, -self.topk:] # (num_users, k), indicating non-ranked rec list 
            arr_ind = score_matrix[np.arange(self.num_user)[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(self.num_user), ::-1]
            pred_list = ind[np.arange(len(score_matrix))[:, None], arr_ind_argsort] # (num_uses, k)

            metric_dic = {}

            for m in self.metric:
                if m == 'recall':
                    metric_k = recall_at_k(pred_list, test_graph, self.topk, self.user_name, self.target_link)
                elif m == 'ndcg':
                    metric_k = ndcg_at_k(pred_list, test_graph, self.topk, self.user_name, self.target_link)
                else:
                    raise NotImplementedError
                metric_dic[m] = metric_k
            
            return metric_dic


def recall_at_k(pred_list, test_graph, k, user_name, target_link):
    sum = 0.0
    test_users = 0
    for user in range(test_graph.num_nodes(user_name)):
        test_items_set = set(test_graph.successors(user, etype=target_link).cpu().numpy())
        pred_items_set = set(pred_list[user][:k])
        if len(test_items_set) != 0:
            sum += len(test_items_set & pred_items_set) / float(len(test_items_set))
            test_users += 1
    return sum / test_users


def ndcg_at_k(pred_list, test_graph, k, user_name, target_link):
    ndcg = []
    for user in range(test_graph.num_nodes(user_name)):
        test_items_set = set(test_graph.successors(user, etype=target_link).cpu().numpy())
        pred_items_set = pred_list[user][:k]
        hit_list = [1 if i in pred_items_set else 0 for i in test_items_set]
        GT = len(test_items_set)
        if GT >= k:
            ideal_hit_list = [1] * k
        else:
            ideal_hit_list = [1] * GT + [0] * (k - GT)
        # idcg = compute_DCG(sorted(hit_list, reverse=True))
        idcg = compute_DCG(ideal_hit_list)
        if idcg:
            ndcg.append(compute_DCG(hit_list) / idcg)
    return np.mean(ndcg)


def compute_DCG(l):
    l = np.array(l)
    if l.size:
        return np.sum(np.subtract(np.power(2, l), 1) / np.log2(np.arange(2, l.size + 2)))
    else:
        return 0.0
# if  __name__ == '__main__':
#     dataset_name = 'Yelp'
#     rec_dataset = TestRecData(dataset_name)
#     print(rec_dataset)

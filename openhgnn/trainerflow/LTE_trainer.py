
from . import BaseFlow, register_flow
from ..models import build_model


@register_flow("LTE_trainer")
class LTETrainer(BaseFlow):

    def __init__(self, args):
        print("2")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device('cuda:0')
        self.device = device
        args.device= device
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        runner = Runner(args)
        runner.fit()


    def train(self):
       print("1")


import os
import argparse
import time
import logging
from pprint import pprint
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import dgl
from ..utils.lte_knowledge_graph import load_data

from ..utils.lte_data_set import  TrainDataset, TestDataset
from ..utils.lte_process_data import process





class Runner(object):
    def __init__(self, params):
        params.embed_dim=None
        params.r_ops=""
        self.p = params
        self.prj_path = Path(__file__).parent.resolve()
        self.p.dataset_name=self.p.data
        self.data = load_data(self.p.dataset_name)
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels
        self.triplets = process({'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data},
                                self.num_rels)

        self.p.embed_dim = self.p.k_w * \
            self.p.k_h if self.p.embed_dim is None else self.p.embed_dim  # output dim of gnn
        self.data_iter = self.get_data_iter()

        if self.p.gpu >= 0:
            self.g = self.build_graph().to(self.p.device)
        else:
            self.g = self.build_graph()
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()
        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}
        os.makedirs('./logs', exist_ok=True)
        self.logger = logging.getLogger(__name__)
        pprint(vars(self.p))

    def fit(self):
        save_root = self.prj_path / 'checkpoints'

        if not save_root.exists():
            save_root.mkdir()
        save_path = save_root / (self.p.name + '.pt')

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train()
            val_results = self.evaluate('valid')
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val_results = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                # self.save_model(save_path)
            print(f"hits@1 = {val_results['hits@1']:.5}")
            print(f"hits@3 = {val_results['hits@3']:.5}")
            print(f"hits@10 = {val_results['hits@10']:.5}")
            print(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
        self.logger.info(vars(self.p))
        # self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results = self.evaluate('test')
        end = time.time()
        self.logger.info(
            f"MRR: Tail {test_results['left_mrr']:.5}, Head {test_results['right_mrr']:.5}, Avg {test_results['mrr']:.5}")
        self.logger.info(
            f"MR: Tail {test_results['left_mr']:.5}, Head {test_results['right_mr']:.5}, Avg {test_results['mr']:.5}")
        self.logger.info(f"hits@1 = {test_results['hits@1']:.5}")
        self.logger.info(f"hits@3 = {test_results['hits@3']:.5}")
        self.logger.info(f"hits@10 = {test_results['hits@10']:.5}")
        self.logger.info("time ={}".format(end-start))

    def train(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            if self.p.gpu >= 0:
                triplets, labels = triplets.to(self.p.device), labels.to(self.p.device)
            subj, rel = triplets[:, 0], triplets[:, 1]
            # print(subj)
            pred = self.model(self.g, subj, rel)  # [batch_size, num_ent]
            loss = self.model.calc_loss(pred, labels)
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)
        return loss

    def evaluate(self, split):
        """
        Function to evaluate the model on validation or test set
        :param split: valid or test, set which data-set to evaluate on
        :return: results['mr']: Average of ranks_left and ranks_right
                 results['mrr']: Mean Reciprocal Rank
                 results['hits@k']: Probability of getting the correct prediction in top-k ranks based on predicted score
                 results['left_mrr'], results['left_mr'], results['right_mrr'], results['right_mr']
                 results['left_hits@k'], results['right_hits@k']
        """

        def get_combined_results(left, right):
            results = dict()
            assert left['count'] == right['count']
            count = float(left['count'])
            results['left_mr'] = round(left['mr'] / count, 5)
            results['left_mrr'] = round(left['mrr'] / count, 5)
            results['right_mr'] = round(right['mr'] / count, 5)
            results['right_mrr'] = round(right['mrr'] / count, 5)
            results['mr'] = round((left['mr'] + right['mr']) / (2 * count), 5)
            results['mrr'] = round(
                (left['mrr'] + right['mrr']) / (2 * count), 5)
            for k in [1, 3, 10]:
                results[f'left_hits@{k}'] = round(left[f'hits@{k}'] / count, 5)
                results[f'right_hits@{k}'] = round(
                    right[f'hits@{k}'] / count, 5)
                results[f'hits@{k}'] = round(
                    (results[f'left_hits@{k}'] + results[f'right_hits@{k}']) / 2, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        right_result = self.predict(split, 'head')
        res = get_combined_results(left_result, right_result)
        return res

    def predict(self, split='valid', mode='tail'):
        """
        Function to run model evaluation for a given mode
        :param split: valid or test, set which data-set to evaluate on
        :param mode: head or tail
        :return: results['mr']: Sum of ranks
                 results['mrr']: Sum of Reciprocal Rank
                 results['hits@k']: counts of getting the correct prediction in top-k ranks based on predicted score
                 results['count']: number of total predictions
        """
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels) in enumerate(test_iter):
                triplets, labels = triplets.to(self.p.device), labels.to(self.p.device)
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                pred = self.model(self.g, subj, rel)
                b_range = torch.arange(pred.shape[0], device=self.p.device)
                # [batch_size, 1], get the predictive score of obj
                target_pred = pred[b_range, obj]
                # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair
                pred = torch.where(
                    labels.bool(), -torch.ones_like(pred) * 10000000, pred)
                # copy predictive score of obj to new pred
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]  # get the rank of each (sub, rel, obj)
                ranks = ranks.float()
                results['count'] = torch.numel(
                    ranks) + results.get('count', 0)  # number of predictions
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(
                    1.0 / ranks).item() + results.get('mrr', 0)

                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(
                        ranks[ranks <= k]) + results.get(f'hits@{k}', 0)
        return results

    def save_model(self, path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, path)

    def load_model(self, path):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_val_mrr = self.best_val_results['mrr']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def build_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_ent)

        if not self.p.rat:
            g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
            g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
        else:
            if self.p.ss > 0:
                sampleSize = self.p.ss
            else:
                sampleSize = self.num_ent - 1
            g.add_edges(self.train_data[:, 0], np.random.randint(
                low=0, high=sampleSize, size=self.train_data[:, 2].shape))
            g.add_edges(self.train_data[:, 2], np.random.randint(
                low=0, high=sampleSize, size=self.train_data[:, 0].shape))
        return g

    def get_data_iter(self):
        """
        get data loader for train, valid and test section
        :return: dict
        """

        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers
            )

        return {
            'train': get_data_loader(TrainDataset, 'train'),
            'valid_head': get_data_loader(TestDataset, 'valid_head'),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),
            'test_head': get_data_loader(TestDataset, 'test_head'),
            'test_tail': get_data_loader(TestDataset, 'test_tail')
        }

    def get_edge_dir_and_norm(self):
        """
        :return: edge_type: indicates type of each edge: [E]
        """
        in_deg = self.g.in_degrees(range(self.g.number_of_nodes())).float()
        norm = in_deg ** -0.5
        norm[torch.isinf(norm).bool()] = 0
        self.g.ndata['xxx'] = norm
        self.g.apply_edges(
            lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        if self.p.gpu >= 0:
            norm = self.g.edata.pop('xxx').squeeze().to(self.p.device)
            edge_type = torch.tensor(np.concatenate(
                [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels])).to(self.p.device)
        else:
            norm = self.g.edata.pop('xxx').squeeze()
            edge_type = torch.tensor(np.concatenate(
                [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels]))
        return edge_type, norm

    def get_model(self):
        args = self.p
        args.num_ents = self.num_ent
        args.num_rels= self.num_rels
        if self.p.n_layer > 0:
            if self.p.score_func.lower() == 'transe':
                model = build_model(args.model_name_GCN).build_model_from_args(
                    args).model
                # model = GCN_TransE(args)
            else:
                raise KeyError(
                    f'score function {self.p.score_func} not recognized.')
        else:
            if self.p.score_func.lower() == 'transe':
                model = build_model(args.model_name).build_model_from_args(
                    args).model
                # model = TransE(self.num_ent, self.num_rels, params=self.p)
            else:
                raise NotImplementedError

        if self.p.gpu >= 0:
            model.to(self.p.device)
        return model




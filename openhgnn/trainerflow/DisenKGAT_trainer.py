from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
import os
import sys
import torch
from .demo import * 
from ..models import DisenKGAT  
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
import numpy as np, sys, os,  json, time
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet
import traceback
from torch.utils.data import DataLoader
np.set_printoptions(precision=4)
from torch.utils.data import Dataset






def get_combined_results(left_results, right_results):
    results = {}
    count = float(left_results['count'])

    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mr']) / (2 * count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2 * count), 5)

    for k in range(10):
        results['left_hits@{}'.format(k + 1)] = round(left_results['hits@{}'.format(k + 1)] / count, 5)
        results['right_hits@{}'.format(k + 1)] = round(right_results['hits@{}'.format(k + 1)] / count, 5)
        results['hits@{}'.format(k + 1)] = round(
            (left_results['hits@{}'.format(k + 1)] + right_results['hits@{}'.format(k + 1)]) / (2 * count), 5)
    return results




@register_flow("DisenKGAT_trainer")
class Runner(BaseFlow):

    def load_data(self): 
        ent_set, rel_set = OrderedSet(), OrderedSet()


        for split in ['train', 'test', 'valid']:
            path = os.path.join(self.raw_dir, split + ".txt")
            for line in open(path):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim
        # self.logger.info('num_ent {} num_rel {}'.format(self.p.num_ent, self.p.num_rel))
        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'test', 'valid']:
            
            path = os.path.join(self.raw_dir, split + ".txt")
            for line in open(path):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
        # self.data: all origin train + valid + test triplets
        self.data = dict(self.data)
        # self.sr2o: train origin edges and reverse edges
        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        # for (sub, rel), obj in self.sr2o.items():
        #     self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        if self.p.strategy == 'one_to_n':
            for (sub, rel), obj in self.sr2o.items():
                self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
        else:
            for sub, rel, obj in self.data['train']:
                rel_inv = rel + self.p.num_rel
                sub_samp = len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
                sub_samp = np.sqrt(1 / sub_samp)

                self.triples['train'].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o[(sub, rel)], 'sub_samp': sub_samp})
                self.triples['train'].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)], 'sub_samp': sub_samp})

        for split in ['test', 'valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

            # self.logger.info('{}_{} num is {}'.format(split, 'tail', len(self.triples['{}_{}'.format(split, 'tail')])))
            # self.logger.info('{}_{} num is {}'.format(split, 'head', len(self.triples['{}_{}'.format(split, 'head')])))

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.test_batch),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.test_batch),
            'test_head': get_data_loader(TestDataset, 'test_head', self.p.test_batch),
            'test_tail': get_data_loader(TestDataset, 'test_tail', self.p.test_batch),
        }
        # self.logger.info('num_ent {} num_rel {}\n'.format(self.p.num_ent, self.p.num_rel))
        # self.logger.info('train set num is {}\n'.format(len(self.triples['train'])))
        # self.logger.info('{}_{} num is {}\n'.format('test', 'tail', len(self.triples['{}_{}'.format('test', 'tail')])))
        # self.logger.info('{}_{} num is {}\n'.format('valid', 'tail', len(self.triples['{}_{}'.format('valid', 'tail')])))
        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)
        # edge_index: 2 * 2E, edge_type: 2E * 1
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, args):  # args == self.config

        args.task = args.model +"_" +args.task 
        self.args = args  
        self.model_name = args.model
        self.device = args.device
        self.hg = None
        self.task = build_task(self.args)
        self.raw_dir = self.task.dataset.raw_dir 
        self.process_dir = self.task.dataset.raw_dir
        self.p = args  
        self.logger = logging.getLogger(__file__)
        pprint(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data() 
        self.model = self.add_model(self.p.model, self.p.score_func)  # disenkgat , interacte
        self.optimizer, self.optimizer_mi = self.add_optimizer(self.model) 
        if not args.restore:
            args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        set_gpu(args.gpu)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():   
            torch.cuda.manual_seed_all(args.seed)  
        

    def add_model(self, model, score_func):
        """
        """
        
        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'disenkgat_transe':
            model = DisenKGAT.DisenKGAT_TransE(edge_index=self.edge_index,
                                      edge_type=self.edge_type, 
                                      params=self.p)
            
        elif model_name.lower() == 'disenkgat_distmult':
            model = DisenKGAT.DisenKGAT_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'disenkgat_conve':
            model = DisenKGAT.DisenKGAT_ConvE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'disenkgat_interacte':
            model = DisenKGAT.DisenKGAT_InteractE(self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, model):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        if self.p.mi_train and self.p.mi_method.startswith('club'):
            mi_disc_params = list(map(id,model.mi_Discs.parameters()))
            rest_params = filter(lambda x:id(x) not in mi_disc_params, model.parameters())
            for m in model.mi_Discs.modules():
                self.logger.info(m)
            for name, parameters in model.named_parameters():
                print(name,':',parameters.size())
            return torch.optim.Adam(rest_params, lr=self.p.lr, weight_decay=self.p.l2), torch.optim.Adam(model.mi_Discs.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        else:
            return torch.optim.Adam(model.parameters(), lr=self.p.lr, weight_decay=self.p.l2), None

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch:      the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        # if split == 'train':
        #     triple, label = [_.to(self.device) for _ in batch]
        #     return triple[:, 0], triple[:, 1], triple[:, 2], label
        # else:
        #     triple, label = [_.to(self.device) for _ in batch]
        #     return triple[:, 0], triple[:, 1], triple[:, 2], label
        if split == 'train':
            if self.p.strategy == 'one_to_x':
                triple, label, neg_ent, sub_samp = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, neg_ent, sub_samp
            else:
                triple, label = [_.to(self.device) for _ in batch]
                return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:            The evaluation results containing the following:
            results['mr']:          Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        left_results = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)
        res_mrr = '\n\tMRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mrr'],
                                                                              results['right_mrr'],
                                                                              results['mrr'])
        res_mr = '\tMR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_mr'],
                                                                          results['right_mr'],
                                                                          results['mr'])
        res_hit1 = '\tHit-1: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@1'],
                                                                               results['right_hits@1'],
                                                                               results['hits@1'])
        res_hit3 = '\tHit-3: Tail : {:.5}, Head : {:.5}, Avg : {:.5}\n'.format(results['left_hits@3'],
                                                                               results['right_hits@3'],
                                                                               results['hits@3'])
        res_hit10 = '\tHit-10: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(results['left_hits@10'],
                                                                               results['right_hits@10'],
                                                                               results['hits@10'])
        log_res = res_mrr + res_mr + res_hit1 + res_hit3 + res_hit10
        if (epoch + 1) % 10 == 0 or split == 'test':
            self.logger.info(
                '[Evaluating Epoch {} {}]: {}'.format(epoch, split, log_res))
        else:
            self.logger.info(
                '[Evaluating Epoch {} {}]: {}'.format(epoch, split, res_mrr))

        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string)     If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):     Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:            The evaluation results containing the following:
            results['mr']:          Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred, _ = self.model.forward(sub, rel, None, split)
                b_range = torch.arange(pred.size()[0], device=self.device)
                target_pred = pred[b_range, obj]
                # filter setting
                pred = torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                # if step % 100 == 0:
                #     self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

        return results

    def run_epoch(self, epoch, val_mrr=0):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        losses_train = []
        corr_losses = []
        lld_losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):  
            self.optimizer.zero_grad()
            if self.p.mi_train and self.p.mi_method.startswith('club'):
                self.model.mi_Discs.eval()
            sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')

            pred, corr = self.model.forward(sub, rel, neg_ent, 'train')  

            loss = self.model.loss(pred, label)   
            if self.p.mi_train:
                losses_train.append(loss.item())
                loss = loss + self.p.alpha * corr 
                corr_losses.append(corr.item())

            loss.backward()     
            self.optimizer.step() 
            losses.append(loss.item())
            # start to compute mi_loss
            if self.p.mi_train and self.p.mi_method.startswith('club'):
                for i in range(self.p.mi_epoch):
                    self.model.mi_Discs.train()
                    lld_loss = self.model.lld_best(sub, rel)
                    self.optimizer_mi.zero_grad()
                    lld_loss.backward()
                    self.optimizer_mi.step()
                    lld_losses.append(lld_loss.item())

            if step % 100 == 0:
                if self.p.mi_train:
                    self.logger.info('[E:{}| {}]: total Loss:{:.5}, Train Loss:{:.5}, Corr Loss:{:.5}, Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses),
                                                                                           np.mean(losses_train), np.mean(corr_losses), self.best_val_mrr,
                                                                                           self.p.name))                     
                else:
                    self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses),
                                                                                           self.best_val_mrr,
                                                                                           self.p.name))

        loss = np.mean(losses_train) if self.p.mi_train else np.mean(losses)
        if self.p.mi_train:
            loss_corr = np.mean(corr_losses)
            if self.p.mi_method.startswith('club') and self.p.mi_epoch == 1:
                loss_lld = np.mean(lld_losses)
                return loss, loss_corr, loss_lld
            return loss, loss_corr, 0.
        # self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
        return loss, 0., 0.

    def train(self):
        """
        Function to run training and evaluation of model

        """
        try:
            self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
            save_path = os.path.join('./save_models', self.p.name)

            if not os.path.exists('./save_models'):   # 如果 节点目录不存在，则创建目录
                os.makedirs('./save_models')

            # if self.p.restore:
            #     self.load_model(save_path)
            #     self.logger.info('Successfully Loaded previous model')

            val_results = {}
            val_results['mrr'] = 0
            kill_cnt = 0
            for epoch in range(self.p.epoch):
                train_loss, corr_loss, lld_loss = self.run_epoch(epoch, val_mrr) 
                val_results = self.evaluate('valid', epoch) 
        
                if val_results['mrr'] > self.best_val_mrr:
                    self.best_val = val_results
                    self.best_val_mrr = val_results['mrr']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    kill_cnt = 0
                else:
                    kill_cnt += 1
                    if kill_cnt % 10 == 0 and self.p.gamma > self.p.max_gamma:
                        self.p.gamma -= 5 
                        # self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                    if kill_cnt > self.p.early_stop: 
                        self.logger.info("Early Stopping!!")
                        break
                if self.p.mi_train:
                    if self.p.mi_method == 'club_s' or self.p.mi_method == 'club_b':
                        self.logger.info(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, lld loss :{:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss, corr_loss,
                                                                                             lld_loss, self.best_val_mrr))
                    else:
                        self.logger.info(
                            '[Epoch {}]: Training Loss: {:.5}, corr Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss, corr_loss,
                                                                                             self.best_val_mrr))
                else:
                    self.logger.info(
                        '[Epoch {}]: Training Loss: {:.5}, Best valid MRR: {:.5}\n\n'.format(epoch, train_loss,
                                                                                         self.best_val_mrr))


            self.logger.info('Loading best model, Evaluating on Test data')
            self.load_model(save_path)
            test_results = self.evaluate('test', self.best_epoch)
        except Exception as e:
            self.logger.debug("%s____%s\n"
                              "traceback.format_exc():____%s" % (Exception, e, traceback.format_exc()))






def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger



def set_gpu(gpus):
    """
    Sets the GPU to be used for the run

    Parameters
    ----------
    gpus:           List of GPUs to be used for the run

    Returns
    -------

    """

    gpus = str(gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus



class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:    The triples used for training the model
    params:     Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label, sub_samp = torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
        trp_label = self.get_label(label)

        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)

        if self.p.strategy == 'one_to_n':
            return triple, trp_label, None, None

        elif self.p.strategy == 'one_to_x':
            sub_samp = torch.FloatTensor([sub_samp])
            neg_ent = torch.LongTensor(self.get_neg_ent(triple, label))
            return triple, trp_label, neg_ent, sub_samp
        else:
            raise NotImplementedError

        # return triple, trp_label, None, None

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        # triple: (batch-size) * 3(sub, rel, -1) trp_label (batch-size) * num entity
        # return triple, trp_label
        if not data[0][2] is None:  # one_to_x
            neg_ent = torch.stack([_[2] for _ in data], dim=0)
            sub_samp = torch.cat([_[3] for _ in data], dim=0)
            return triple, trp_label, neg_ent, sub_samp
        else:
            return triple, trp_label

    # def get_neg_ent(self, triple, label):
    #     def get(triple, label):
    #         pos_obj = label
    #         mask = np.ones([self.p.num_ent], dtype=np.bool)
    #         mask[label] = 0
    #         neg_ent = np.int32(
    #             np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
    #         neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))
    #
    #         return neg_ent
    #
    #     neg_ent = get(triple, label)
    #     return neg_ent
    def get_neg_ent(self, triple, label):
        def get(triple, label):
            if self.p.strategy == 'one_to_x':
                pos_obj = triple[2]
                mask = np.ones([self.p.num_ent], dtype=np.bool)
                mask[label] = 0
                neg_ent = np.int32(np.random.choice(self.entities[mask], self.p.neg_num, replace=False)).reshape([-1])
                neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))
            else:
                pos_obj = label
                mask = np.ones([self.p.num_ent], dtype=np.bool)
                mask[label] = 0
                neg_ent = np.int32(
                    np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
                neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))

                if len(neg_ent) > self.p.neg_num:
                    import pdb;
                    pdb.set_trace()

            return neg_ent

        neg_ent = get(triple, label)
        return neg_ent

    def get_label(self, label):
        # y = np.zeros([self.p.num_ent], dtype=np.float32)
        # for e2 in label: y[e2] = 1.0
        # return torch.FloatTensor(y)
        if self.p.strategy == 'one_to_n':
            y = np.zeros([self.p.num_ent], dtype=np.float32)
            for e2 in label: y[e2] = 1.0
        elif self.p.strategy == 'one_to_x':
            y = [1] + [0] * self.p.neg_num
        else:
            raise NotImplementedError
        return torch.FloatTensor(y)

class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:    The triples used for evaluating the model
    params:     Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)


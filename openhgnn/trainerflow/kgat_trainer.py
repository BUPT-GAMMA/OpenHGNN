import random
from time import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from . import BaseFlow,register_flow
from ..models import build_model
from ..tasks import build_task

@register_flow("KGAT_trainer")
class KGAT_Trainer(BaseFlow):
    def __init__(self, args):
        super(KGAT_Trainer, self).__init__(args)
        self.args = args
        self.device = args.device
        self.task = build_task(args)
        self.logger=self.task.logger

        #parameter for building model
        self.n_users= self.task.dataset.n_users
        self.n_entities= self.task.dataset.n_entities
        self.n_relations= self.task.dataset.n_relations
        self.n_items= self.task.dataset.n_items
        self.n_users_entities=self.n_users+self.n_entities

        #load pretrained embedding
        if self.args.use_pretrain == 1:
            pretrain_data = np.load(self.task.dataset.pretrain_embedding_dir)
            self.user_pre_embed = torch.tensor(pretrain_data['user_embed'])
            self.item_pre_embed = torch.tensor(pretrain_data['item_embed'])
        else:
            self.user_pre_embed, self.item_pre_embed = None, None

        #build model
        self.model_name = self.args.model
        self.model = build_model(self.model_name).build_model_from_args(self.args,None)
        self.model.set_parameters(self.n_users,self.n_entities,self.n_relations,self.user_pre_embed,self.item_pre_embed)

        # Using multiple GPUs for training
        if self.args.multi_gpu == True:
            n_gpu=torch.cuda.device_count()
            if n_gpu > 1:
                self.model= torch.nn.parallel.DistributedDataParallel(self.model)

        #Load trained model from pretrain_model_path
        if self.args.use_pretrain == 2:
            self.logger.info(f"Load trained model from {args.pretrain_model_path}")
            self.model.load_state_dict(torch.load(args.pretrain_model_path))

        print("build_model_finish")
        self.logger.info(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.max_epoch = args.max_epoch
        self.preprocess()

    def preprocess(self):
        # load other information for sampling and training
        self.g = self.task.dataset.g.to(self.device)
        self.train_user_dict = self.task.dataset.train_user_dict
        self.test_user_dict = self.task.dataset.test_user_dict
        self.train_kg_dict = self.task.dataset.train_kg_dict
        self.n_cf_train = self.task.dataset.n_cf_train
        self.n_kg_train = self.task.dataset.n_kg_train

        self.cf_batch_size=self.args.cf_batch_size
        self.kg_batch_size = self.args.kg_batch_size
        self.test_batch_size = self.args.test_batch_size

        user_ids = list(self.test_user_dict.keys())
        user_ids_batches = [user_ids[i: i + self.args.test_batch_size] for i in
                                range(0, len(user_ids), self.args.test_batch_size)]
        self.user_ids_batches = [torch.LongTensor(d).to(self.device) for d in user_ids_batches]
        self.item_ids = torch.arange(self.n_items, dtype=torch.long).to(self.device)


    def train(self):
        # seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        epoch_list = []
        precision_list = []
        recall_list = []
        ndcg_list = []

        for epoch in range(1, self.max_epoch + 1):
            time0 = time()
            self.model.train()

            # update attention scores

            with torch.no_grad():
                att = self.model('calc_att', self.g)

            self.g.edata['att'] = att
            print('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

            # train cf
            time1 = time()
            cf_total_loss = 0
            n_cf_batch = self.n_cf_train // self.args.cf_batch_size + 1

            for iter in range(1, n_cf_batch + 1):
                time2 = time()
                cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = self.generate_cf_batch(self.train_user_dict)
                cf_batch_user = cf_batch_user.to(self.device)
                cf_batch_pos_item = cf_batch_pos_item.to(self.device)
                cf_batch_neg_item = cf_batch_neg_item.to(self.device)
                cf_batch_loss = self.model('calc_cf_loss', self.g, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
                cf_batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                cf_total_loss += cf_batch_loss.item()

                print('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'
                        .format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))

            # train kg
            time1 = time()
            kg_total_loss = 0
            n_kg_batch = self.n_kg_train //self.args.kg_batch_size + 1

            for iter in range(1, n_kg_batch + 1):
                time2 = time()
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = self.generate_kg_batch(self.train_kg_dict)
                kg_batch_head = kg_batch_head.to(self.device)
                kg_batch_relation = kg_batch_relation.to(self.device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(self.device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(self.device)
                kg_batch_loss = self.model('calc_kg_loss', kg_batch_head, kg_batch_relation, kg_batch_pos_tail,kg_batch_neg_tail)

                kg_batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                kg_total_loss += kg_batch_loss.item()
                print('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'
                    .format(epoch, iter, n_kg_batch, time() - time2, kg_batch_loss.item(), kg_total_loss / iter))
            print('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'
                    .format(epoch, n_kg_batch, time() - time1, kg_total_loss / n_kg_batch))
            print('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

            # evaluate cf
            time1 = time()
            _, precision, recall, ndcg = self.evaluate(self.model, self.g, self.train_user_dict, self.test_user_dict, self.user_ids_batches, self.item_ids, self.args.K)

            self.logger.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(
                epoch, time() - time1, precision, recall, ndcg))

            epoch_list.append(epoch)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            best_recall, should_stop = self.early_stopping(recall_list, self.args.stopping_steps)
            if should_stop:
                break

            if recall_list.index(best_recall) == len(recall_list) - 1:
                torch.save(self.model.state_dict(), f"{self.args.output_dir}/{self.model_name}_{self.args.dataset_name[0:-5]}_{self.args.aggregation_type}.pth")
                print('Save model on epoch {:04d}!'.format(epoch))



    def evaluate(self,model, train_graph, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
        model.eval()

        with torch.no_grad():
            att = model.compute_attention(train_graph)
        train_graph.edata['att'] = att

        n_users = len(test_user_dict.keys())
        # item_ids_batch = item_ids
        item_ids_batch = item_ids.cpu().numpy()

        cf_scores = []
        precision = []
        recall = []
        ndcg = []

        with torch.no_grad():
            # for user_ids_batch in user_ids_batches:
            for user_ids_batch in tqdm(user_ids_batches, desc='Evaluating Iteration'):
                cf_scores_batch = model('predict', train_graph, user_ids_batch,
                                        item_ids)  # (n_batch_users, n_eval_items)

                cf_scores_batch = cf_scores_batch.cpu()
                user_ids_batch = user_ids_batch.cpu().numpy()
                precision_batch, recall_batch, ndcg_batch = self.calc_metrics_at_k(cf_scores_batch, train_user_dict,
                                                                              test_user_dict, user_ids_batch,
                                                                              item_ids_batch, K)

                cf_scores.append(cf_scores_batch.numpy())
                precision.append(precision_batch)
                recall.append(recall_batch)
                ndcg.append(ndcg_batch)

        cf_scores = cf_scores[0]
        precision_k = sum(np.concatenate(precision)) / n_users
        recall_k = sum(np.concatenate(recall)) / n_users
        ndcg_k = sum(np.concatenate(ndcg)) / n_users
        return cf_scores, precision_k, recall_k, ndcg_k

    #sample

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):

        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items


    def generate_cf_batch(self, user_dict):
        exist_users = user_dict.keys()
        if self.cf_batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, self.cf_batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(self.cf_batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]
        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break
            tail = np.random.randint(low=0, high=self.n_users_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(self.kg_batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    #early stopping

    def early_stopping(self,recall_list, stopping_steps):
        best_recall = max(recall_list)
        best_step = recall_list.index(best_recall)
        if len(recall_list) - best_step - 1 >= stopping_steps:
            should_stop = True
        else:
            should_stop = False
        return best_recall, should_stop

    #metrics

    def calc_metrics_at_k(self,cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, K):
        """
        cf_scores: (n_eval_users, n_eval_items)
        """

        def precision_at_k_batch(hits, k):
            """
            calculate Precision@k
            hits: array, element is binary (0 / 1), 2-dim
            """
            res = hits[:, :k].mean(axis=1)
            return res

        def recall_at_k_batch(hits, k):
            """
            calculate Recall@k
            hits: array, element is binary (0 / 1), 2-dim
            """
            res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
            return res

        def ndcg_at_k_batch(hits, k):
            """
            calculate NDCG@k
            hits: array, element is binary (0 / 1), 2-dim
            """
            hits_k = hits[:, :k]
            dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
            sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
            idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
            idcg[idcg == 0] = np.inf
            res = (dcg / idcg)
            return res

        test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
        for idx, u in enumerate(user_ids):
            train_pos_item_list = train_user_dict[u]
            test_pos_item_list = test_user_dict[u]
            cf_scores[idx][train_pos_item_list] = 0
            test_pos_item_binary[idx][test_pos_item_list] = 1
        try:
            _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)
        except:
            _, rank_indices = torch.sort(cf_scores, descending=True)
        rank_indices = rank_indices.cpu()

        binary_hit = []
        for i in range(len(user_ids)):
            binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
        binary_hit = np.array(binary_hit, dtype=np.float32)

        precision = precision_at_k_batch(binary_hit, K)
        recall = recall_at_k_batch(binary_hit, K)
        ndcg = ndcg_at_k_batch(binary_hit, K)
        return precision, recall, ndcg

import os

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from ..models import build_model
from . import BaseFlow, register_flow
from ..sampler import mg2vec_sampler
import numpy as np


@register_flow('mg2vec_trainer')
class Mg2vecTrainer(BaseFlow):
    def __init__(self, args):
        super(Mg2vecTrainer, self).__init__(args)
        self.mg2vec_sampler = None
        self.dataloader = None
        self.model = None
        self.embeddings_file_path = os.path.join(self.args.output_dir, self.args.dataset + '_mg2vec_embeddings.npy')
        self.embeddings_file_path2 = os.path.join(self.args.output_dir, self.args.dataset + '_mg2vec_embeddings.txt')
        self.load_trained_embeddings = False

    def preprocess(self):
        input_file = "./openhgnn/dataset/{}/meta.txt".format(self.args.dataset)
        block_size = self.args.batch_size * 100000
        self.mg2vec_sampler = mg2vec_sampler.Mg2vecSampler(input_file, block_size, self.args.alpha)
        self.dataloader = DataLoader(self.mg2vec_sampler, batch_size=self.args.batch_size, shuffle=True,
                                     num_workers=self.args.num_workers,
                                     )
        self.args.node_num = self.mg2vec_sampler.data.node_count
        self.args.mg_num = self.mg2vec_sampler.data.mg_count
        self.args.unigram = self.mg2vec_sampler.data.unigram
        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg).to(self.device)

    def train(self):
        emb = self.load_embeddings()
        emb_dict = dict()
        for nId, node in self.mg2vec_sampler.data.node_reverse_dict.items():
            emb_dict[int(node)] = emb[nId]
        # todo: only supports edge classification now
        metric = {
            'test': self.task.downstream_evaluate(logits=self.get_edge_embed(emb=emb_dict), evaluation_metric='acc_f1')}
        self.logger.train_info(self.logger.metric2str(metric))
        # metric = {'test': self.task.evaluate(emb_dict)}
        # self.logger.train_info(self.logger.metric2str(metric))

    def load_embeddings(self):
        if not self.load_trained_embeddings or not os.path.exists(self.embeddings_file_path):
            self.train_embeddings()
        emb = np.load(self.embeddings_file_path)
        return emb

    def train_embeddings(self):
        self.preprocess()
        epoch_index = 1
        optimizer = optim.Adam(list(self.model.parameters()), lr=self.args.lr)
        average_loss = 0.0
        step = 0
        print("train start")
        while True:
            for i, sampled_batch in enumerate(self.dataloader):
                if len(sampled_batch) > 0:
                    train_a = sampled_batch[0].to(self.device)
                    train_b = sampled_batch[1].to(self.device)
                    train_label = sampled_batch[2].to(self.device)
                    train_freq = sampled_batch[3].reshape(-1, 1).to(self.device)
                    train_weight = sampled_batch[4].reshape(-1, 1).to(self.device)

                    optimizer.zero_grad()
                    loss = self.model.forward(train_a, train_b, train_label, train_freq, train_weight, self.device)
                    loss.backward()
                    optimizer.step()

                    average_loss += loss.item()
                    step += 1
                    if step > 0 and step % 10000 == 0:
                        average_loss /= 10000
                        print('Average loss at step ', step, ': ', average_loss)
                        average_loss = 0.0
            if self.mg2vec_sampler.data.epoch_end:
                print("epoch %d end" % epoch_index)
                epoch_index += 1
                self.mg2vec_sampler.data.epoch_end = False
                if epoch_index > self.args.max_epoch:
                    break
            self.mg2vec_sampler.data.read_block()

        print("total step: ", step)
        self.model.save_embedding_np(self.embeddings_file_path)
        self.model.save_embedding(self.mg2vec_sampler.data.node_reverse_dict, self.embeddings_file_path2)

    def get_edge_embed(self, emb):
        edge_embed = []
        g = self.hg
        u, v = g.edges()
        core1_dict = g.nodes['core1'].data['id2node'].cpu()
        core2_dict = g.nodes['core2'].data['id2node'].cpu()
        for i in range(len(u)):
            edge_embed.append(np.hstack([emb[int(core1_dict[u[i]])], emb[int(core2_dict[v[i]])]]))
        x = np.array(edge_embed)
        return x

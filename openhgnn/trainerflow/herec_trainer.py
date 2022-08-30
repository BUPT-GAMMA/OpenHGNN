import os
import numpy
from tqdm import tqdm
import torch.sparse as sparse
import torch.optim as optim
from torch.utils.data import DataLoader
from ..models import build_model
from . import BaseFlow, register_flow
from ..sampler import random_walk_sampler
import dgl


@register_flow("herec_trainer")
class HERecTrainer(BaseFlow):
    def __init__(self, args):
        super(HERecTrainer, self).__init__(args)
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.random_walk_sampler = None

        self.dataloader = None

        self.metapath = self.task.dataset.meta_paths_dict[self.args.meta_path_key]
        self.embeddings_file_path = os.path.join(self.args.output_dir, self.args.dataset_name + '_' +
                                                 self.args.meta_path_key + '_herec_embeddings.npy')
        self.load_trained_embeddings = False

    def preprocess(self):

        for i, elem in enumerate(self.metapath):
            if i == 0:
                adj = self.hg.adj(etype=elem)
            else:
                adj = sparse.mm(adj, self.hg.adj(etype=elem))
        adj = adj.coalesce()

        g = dgl.graph(data=(adj.indices()[0], adj.indices()[1]))
        g.edata['rw_prob'] = adj.values()

        self.random_walk_sampler = random_walk_sampler.RandomWalkSampler(g=g.to('cpu'),
                                                                         rw_length=self.args.rw_length,
                                                                         rw_walks=self.args.rw_walks,
                                                                         window_size=self.args.window_size,
                                                                         neg_size=self.args.neg_size, rw_prob='rw_prob')

        self.dataloader = DataLoader(self.random_walk_sampler, batch_size=self.args.batch_size,
                                     shuffle=True, num_workers=self.args.num_workers,
                                     collate_fn=self.random_walk_sampler.collate)

    def train(self):
        emb = self.load_embeddings()

        # if node classification, evaluate and return metric
        if self.args.task == 'node_classification':
            metric = {'test': self.task.downstream_evaluate(logits=emb, evaluation_metric='f1_lr')}
            self.logger.train_info(self.logger.metric2str(metric))
            return metric
        # otherwise, return emb
        return emb

    def load_embeddings(self):
        if not self.load_trained_embeddings or not os.path.exists(self.embeddings_file_path):
            self.train_embeddings()
        emb = numpy.load(self.embeddings_file_path)
        return emb

    def train_embeddings(self):
        self.preprocess()

        optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for epoch in range(self.max_epoch):
            self.logger.train_info('\n\n\nEpoch: ' + str(epoch + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 50 == 0:
                        self.logger.train_info(' Loss: ' + str(running_loss))
        self.model.save_embedding(self.embeddings_file_path)

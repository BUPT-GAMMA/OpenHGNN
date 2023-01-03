import os.path
import numpy
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from ..models import build_model
from . import BaseFlow, register_flow
from ..sampler import random_walk_sampler


@register_flow("mp2vec_trainer")
class Metapath2VecTrainer(BaseFlow):
    def __init__(self, args):
        super(Metapath2VecTrainer, self).__init__(args)
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.model = self.model.to(self.device)
        self.mp2vec_sampler = None
        self.dataloader = None
        self.embeddings_file_path = os.path.join(self.args.output_dir, self.args.dataset_name + '_' +
                                                 self.args.meta_path_key + '_mp2vec_embeddings.npy')
        self.load_trained_embeddings = False

    def preprocess(self):
        metapath = self.task.dataset.meta_paths_dict[self.args.meta_path_key]
        self.mp2vec_sampler = random_walk_sampler.RandomWalkSampler(g=self.hg.to('cpu'),
                                                                    metapath=metapath * self.args.rw_length,
                                                                    rw_walks=self.args.rw_walks,
                                                                    window_size=self.args.window_size,
                                                                    neg_size=self.args.neg_size)

        self.dataloader = DataLoader(self.mp2vec_sampler, batch_size=self.args.batch_size,
                                     shuffle=True, num_workers=self.args.num_workers,
                                     collate_fn=self.mp2vec_sampler.collate)

    def train(self):
        emb = self.load_embeddings()

        # if node classification, evaluate and return metric
        if self.args.task == 'node_classification':
            start_idx, end_idx = self.get_ntype_range(self.task.dataset.category)
            metric = {'test': self.task.downstream_evaluate(logits=emb[start_idx:end_idx], evaluation_metric='f1_lr')}
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
            self.logger.info('Epoch: ' + str(epoch + 1))
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
                        self.logger.info(' Loss: ' + str(running_loss))
        self.model.save_embedding(self.embeddings_file_path)

    def get_ntype_range(self, target_ntype):
        start_idx = 0
        for ntype in self.hg.ntypes:
            if ntype == target_ntype:
                end_idx = start_idx + self.hg.num_nodes(ntype)
                return start_idx, end_idx
            start_idx += self.hg.num_nodes(ntype)

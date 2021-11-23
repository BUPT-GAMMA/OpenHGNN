import numpy
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..sampler import mp2vec_sampler


@register_flow("mp2vec_trainer")
class Metapath2VecTrainer(BaseFlow):
    def __init__(self, args):
        super(Metapath2VecTrainer, self).__init__(args)
        self.args = args
        self.model_name = args.model
        self.device = args.device

        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg).to(self.device)
        self.model = self.model.to(self.device)
        self.mp2vec_sampler = None
        self.dataloader = None
        self.embeddings_file_name = self.args.dataset + '_mp2vec_embeddings'

    def preprocess(self):
        metapath = self.task.dataset.meta_paths[0]
        start_ntype = metapath[0][0]
        metapath_edges = [elem[1] for elem in metapath]
        self.mp2vec_sampler = mp2vec_sampler.Metapath2VecSampler(hg=self.hg, metapath=metapath_edges,
                                                                 start_ntype=start_ntype, rw_length=self.args.rw_length,
                                                                 rw_walks=self.args.rw_walks,
                                                                 window_size=self.args.window_size,
                                                                 neg_size=self.args.neg_size)

        self.dataloader = DataLoader(self.mp2vec_sampler, batch_size=self.args.batch_size,
                                     shuffle=True, num_workers=self.args.num_workers,
                                     collate_fn=self.mp2vec_sampler.collate)

    def train(self):
        emb = self.load_embeddings()

        # todo: only supports node classification now
        start_idx, end_idx = self.get_ntype_range(self.task.dataset.category)
        self.task.evaluate(logits=emb[start_idx:end_idx], name='f1_lr')

    def load_embeddings(self):
        try:
            emb = numpy.load(self.embeddings_file_name + '.npy')
        except Exception:
            self.train_embeddings()
            emb = numpy.load(self.embeddings_file_name + '.npy')
        return emb

    def train_embeddings(self):
        self.preprocess()

        optimizer = optim.SparseAdam(list(self.model.parameters()), lr=self.args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

        for epoch in range(self.max_epoch):
            print('\n\n\nEpoch: ' + str(epoch + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 50 == 0:
                        print(' Loss: ' + str(running_loss))
        self.model.save_embedding(self.embeddings_file_name)

    def get_ntype_range(self, target_ntype):
        start_idx = 0
        for ntype in self.hg.ntypes:
            if ntype == target_ntype:
                end_idx = start_idx + self.hg.num_nodes(ntype)
                return start_idx, end_idx
            start_idx += self.hg.num_nodes(ntype)

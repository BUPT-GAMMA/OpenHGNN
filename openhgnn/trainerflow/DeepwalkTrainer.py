import copy
import dgl
import torch as th
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..sampler.HetGNN_sampler import SkipGramBatchSampler, HetGNNCollator, NeighborSampler, hetgnn_graph
from ..utils import EarlyStopping


@register_flow("deepwalktrainer")
class DeepwalkTrainer(BaseFlow):
    """DeepwalkTrainer flows.

    Supported Model: DeepWalk
    Supported Dataset：Academic4HetGNN
    Dataset description can be found in HetGNN paper.
    The trainerflow supports node classification and author link prediction.

    """

    def __init__(self, args):
        super(DeepwalkTrainer, self).__init__(args)

        self.args = args
        self.dataset = DeepwalkDataset(
            net_file=args.data_file,
            map_file=args.map_file,
            walk_length=args.walk_length,
            window_size=args.window_size,
            num_walks=args.num_walks,
            batch_size=args.batch_size,
            negative=args.negative,
            gpus=args.gpus,
            fast_neg=args.fast_neg,
            ogbl_name=args.ogbl_name,
            load_from_ogbl=args.load_from_ogbl,
        )
        self.emb_size = self.dataset.G.number_of_nodes()
        self.emb_model = None

    def preprocess(self):

        if self.args.mini_batch_flag:
            if self.args.model == 'HetGNN':
                hetg = hetgnn_graph(self.hg, self.args.dataset)
                self.hg = self.hg.to('cpu')
                self.het_graph = hetg.get_hetgnn_graph(self.args.rw_length, self.args.rw_walks, self.args.rwr_prob).to('cpu')

                batch_sampler = SkipGramBatchSampler(self.hg, self.args.batch_size, self.args.window_size)
                neighbor_sampler = NeighborSampler(self.het_graph, self.hg.ntypes, batch_sampler.num_nodes, self.args.device)
                collator = HetGNNCollator(neighbor_sampler, self.hg)
                dataloader = DataLoader(
                    batch_sampler,
                    collate_fn=collator.collate_train,
                    num_workers=self.args.num_workers)
                self.dataloader_it = iter(dataloader)
                self.hg = self.hg.to(self.args.device)
                self.het_graph = self.het_graph.to(self.args.device)
            # elif self.args.model == 'Metapath2vec':
            #     batch_sampler = SkipGramBatchSampler(self.hg, self.args.batch_size, self.args.window_size, self.args.rw_length)
            #     collator = MP2vecCollator(self.hg.ntypes, batch_sampler.num_nodes)
            #     dataloader = DataLoader(batch_sampler, collate_fn=collator.collate_train, num_workers=self.args.num_workers)
            #     self.dataloader_it = iter(dataloader)

        return

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            if self.args.mini_batch_flag:
                loss = self._mini_train_step()
            else:
                loss = self._full_train_setp()
            epoch_iter.set_description('Epoch{}: Loss:{:.4f}'.format(epoch, loss))
            early_stop = stopper.loss_step(loss, self.model)
            if early_stop:
                print('Early Stop!\tEpoch:' + str(epoch))
                break
        stopper.load_model(self.model)
        metrics = self._test_step()
        return dict(metrics=metrics)

    def _full_train_setp(self):
        self.model.train()
        negative_graph = self.construct_negative_graph()
        x = self.model(self.het_graph)[self.category]
        loss = self.loss_calculation(self.ScorePredictor(self.hg, x), self.ScorePredictor(negative_graph, x))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def init_device_emb(self):
        """ set the device before training
        will be called once in fast_train_mp / fast_train
        """
        choices = sum([self.args.only_gpu, self.args.only_cpu, self.args.mix])
        assert choices == 1, "Must choose only *one* training mode in [only_cpu, only_gpu, mix]"

        # initializing embedding on CPU
        self.emb_model = SkipGramModel(
            emb_size=self.emb_size,
            emb_dimension=self.args.dim,
            walk_length=self.args.walk_length,
            window_size=self.args.window_size,
            batch_size=self.args.batch_size,
            only_cpu=self.args.only_cpu,
            only_gpu=self.args.only_gpu,
            mix=self.args.mix,
            neg_weight=self.args.neg_weight,
            negative=self.args.negative,
            lr=self.args.lr,
            lap_norm=self.args.lap_norm,
            fast_neg=self.args.fast_neg,
            record_loss=self.args.print_loss,
            norm=self.args.norm,
            use_context_weight=self.args.use_context_weight,
            async_update=self.args.async_update,
            num_threads=self.args.num_threads,
        )

        torch.set_num_threads(self.args.num_threads)
        if self.args.only_gpu:
            print("Run in 1 GPU")
            assert self.args.gpus[0] >= 0
            self.emb_model.all_to_device(self.args.gpus[0])
        elif self.args.mix:
            print("Mix CPU with %d GPU" % len(self.args.gpus))
            if len(self.args.gpus) == 1:
                assert self.args.gpus[0] >= 0, 'mix CPU with GPU should have available GPU'
                self.emb_model.set_device(self.args.gpus[0])
        else:
            print("Run in CPU process")
            self.args.gpus = [torch.device('cpu')]

    def fast_train(self):
        """ fast train with dataloader with only gpu / only cpu"""
        # the number of postive node pairs of a node sequence
        num_pos = 2 * self.args.walk_length * self.args.window_size \
                  - self.args.window_size * (self.args.window_size + 1)
        num_pos = int(num_pos)

        self.init_device_emb()

        if self.args.async_update:
            self.emb_model.share_memory()
            self.emb_model.create_async_update()

        sampler = self.dataset.create_sampler(0)

        dataloader = DataLoader(
            dataset=sampler.seeds,
            batch_size=self.args.batch_size,
            collate_fn=sampler.sample,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_sampler_threads,
        )

        num_batches = len(dataloader)
        print("num batchs: %d\n" % num_batches)

        start_all = time.time()
        start = time.time()
        with torch.no_grad():
            max_i = num_batches
            for i, walks in enumerate(dataloader):
                if self.args.fast_neg:
                    self.emb_model.fast_learn(walks)
                else:
                    # do negative sampling
                    bs = len(walks)
                    neg_nodes = torch.LongTensor(
                        np.random.choice(self.dataset.neg_table,
                                         bs * num_pos * self.args.negative,
                                         replace=True))
                    self.emb_model.fast_learn(walks, neg_nodes=neg_nodes)

                if i > 0 and i % self.args.print_interval == 0:
                    if self.args.print_loss:
                        print("Batch %d training time: %.2fs loss: %.4f" \
                              % (i, time.time() - start, -sum(self.emb_model.loss) / self.args.print_interval))
                        self.emb_model.loss = []
                    else:
                        print("Batch %d, training time: %.2fs" % (i, time.time() - start))
                    start = time.time()

            if self.args.async_update:
                self.emb_model.finish_async_update()

        print("Training used time: %.2fs" % (time.time() - start_all))
        if self.args.save_in_txt:
            self.emb_model.save_embedding_txt(self.dataset, self.args.output_emb_file)
        elif self.args.save_in_pt:
            self.emb_model.save_embedding_pt(self.dataset, self.args.output_emb_file)
        else:
            self.emb_model.save_embedding(self.dataset, self.args.output_emb_file)




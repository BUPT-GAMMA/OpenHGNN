import torch as th
from tqdm import tqdm
from . import BaseFlow, register_flow
from ..models import build_model
from ..models.GATNE import NSLoss
import torch
from tqdm.auto import tqdm
from numpy import random
import dgl
from ..sampler.GATNE_sampler import NeighborSampler, generate_pairs


@register_flow("GATNE_trainer")
class GATNE(BaseFlow):
    def __init__(self, args):
        super(GATNE, self).__init__(args)
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.train_pairs = None
        self.train_dataloader = None
        self.nsloss = None
        self.neighbor_sampler = None

        self.orig_val_hg = self.task.val_hg
        self.orig_test_hg = self.task.test_hg

        self.preprocess()
        
    def preprocess(self):
        assert len(self.hg.ntypes) == 1
        bidirected_hg = dgl.to_bidirected(dgl.to_simple(self.hg.to('cpu')))
        all_walks = []
        for etype in self.hg.etypes:
            nodes = torch.unique(bidirected_hg.edges(etype=etype)[0]).repeat(self.args.rw_walks)
            traces, types = dgl.sampling.random_walk(
                bidirected_hg, nodes, metapath=[etype] * (self.args.rw_length - 1)
            )
            all_walks.append(traces)
        self.train_pairs = generate_pairs(all_walks, self.args.window_size, self.args.num_workers)
        self.neighbor_sampler = NeighborSampler(bidirected_hg, [self.args.neighbor_samples])
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_pairs,
            batch_size=self.args.batch_size,
            collate_fn=self.neighbor_sampler.sample,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.nsloss = NSLoss(self.hg.num_nodes(), self.args.neg_size, self.args.dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            [{"params": self.model.parameters()}, {"params": self.nsloss.parameters()}], lr=self.args.learning_rate
        )
        return

    def train(self):
        best_score = 0
        patience = 0
        for self.epoch in range(self.args.max_epoch):
            self._full_train_step()
            cur_score = self._full_test_step()
            if cur_score > best_score:
                best_score = cur_score
                patience = 0
            else:
                patience += 1
                if patience > self.args.patience:
                    self.logger.train_info(f'Early Stop!\tEpoch:{self.epoch:03d}.')
                    break

    def _full_train_step(self):
        self.model.train()
        random.shuffle(self.train_pairs)
        data_iter = tqdm(
            self.train_dataloader,
            desc="epoch %d" % self.epoch,
            total=(len(self.train_pairs) + (self.args.batch_size - 1)) // self.args.batch_size,
        )
        avg_loss = 0.0

        for i, (block, head_invmap, tails, block_types) in enumerate(data_iter):
            self.optimizer.zero_grad()
            # embs: [batch_size, edge_type_count, embedding_size]
            block_types = block_types.to(self.device)
            embs = self.model(block[0].to(self.device))[head_invmap]
            embs = embs.gather(
                1, block_types.view(-1, 1, 1).expand(embs.shape[0], 1, embs.shape[2])
            )[:, 0]
            loss = self.nsloss(
                block[0].dstdata[dgl.NID][head_invmap].to(self.device),
                embs,
                tails.to(self.device),
            )
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()

            post_fix = {
                "epoch": self.epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item(),
            }
            data_iter.set_postfix(post_fix)

    def _full_test_step(self):
        self.model.eval()
        # {'1': {}, '2': {}}
        final_model = dict(
            zip(self.hg.etypes, [th.empty(self.hg.num_nodes(), self.args.dim) for _ in range(len(self.hg.etypes))]))

        for i in tqdm(range(self.hg.num_nodes()), desc='Evaluating...'):
            train_inputs = (
                torch.tensor([i for _ in range(len(self.hg.etypes))])
                    .unsqueeze(1)
                    .to(self.device)
            )  # [i, i]
            train_types = (
                torch.tensor(list(range(len(self.hg.etypes)))).unsqueeze(1).to(self.device)
            )  # [0, 1]
            pairs = torch.cat(
                (train_inputs, train_inputs, train_types), dim=1
            )  # (2, 3)
            (
                train_blocks,
                train_invmap,
                fake_tails,
                train_types,
            ) = self.neighbor_sampler.sample(pairs)

            node_emb = self.model(train_blocks[0].to(self.device))[train_invmap]
            node_emb = node_emb.gather(
                1,
                train_types.to(self.device)
                    .view(-1, 1, 1)
                    .expand(node_emb.shape[0], 1, node_emb.shape[2]),
            )[:, 0]

            for j in range(len(self.hg.etypes)):
                final_model[self.hg.etypes[j]][i] = node_emb[j].detach()
        metric = {}
        score = []
        for etype in self.hg.etypes:
            self.task.val_hg = dgl.edge_type_subgraph(self.orig_val_hg, [etype])
            self.task.test_hg = dgl.edge_type_subgraph(self.orig_test_hg, [etype])

            for split in ['test', 'valid']:
                n_embedding = {self.hg.ntypes[0]: final_model[etype].to(self.device)}
                res = self.task.evaluate(n_embedding=n_embedding, mode=split)
                metric[split] = res
                if split == 'valid':
                    score.append(res.get('roc_auc'))
            self.logger.train_info(etype + self.logger.metric2str(metric))

        avg_score = sum(score) / len(score)
        return avg_score

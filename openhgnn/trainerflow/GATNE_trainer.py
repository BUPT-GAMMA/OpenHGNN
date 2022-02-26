import copy
import dgl
import numpy as np
import torch as th
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, get_nodes_dict
from collections.abc import Mapping
from ..models import build_model


@register_flow("GATNE_trainer")
class GATNE(BaseFlow):
    """Demo flows."""

    def __init__(self, args):
        super(GATNE, self).__init__(args)

        # self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg).to(self.device)
        # self.model = self.model.to(self.device)
        # self.mp2vec_sampler = None
        # self.dataloader = None


        self.args = args
        self.model_name = args.model
        self.device = args.device

        self.task = build_task(args)
        self.hg = self.task.get_graph().to(self.device)

        self.model = build_model(self.model_name).build_model_from_args(self.args, self.hg)
        self.model = self.model.to(self.device)

    def preprocess(self):
        return

    def train(self):
        args = parse_args()
        file_name = args.input
        print(args)

        training_data_by_type = load_training_data(file_name + "/train.txt")
        valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
            file_name + "/valid.txt"
        )
        testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
            file_name + "/test.txt"
        )
        start = time.time()
        average_auc, average_f1, average_pr = self.train_model(training_data_by_type)
        end = time.time()

        print("Overall ROC-AUC:", average_auc)
        print("Overall PR-AUC", average_pr)
        print("Overall F1:", average_f1)
        print("Training Time", end - start)

    def train_model(self, ):
        index2word, vocab, type_nodes = generate_vocab(network_data)

        edge_types = list(network_data.keys())
        num_nodes = len(index2word)
        edge_type_count = len(edge_types)
        epochs = args.epoch
        batch_size = args.batch_size
        embedding_size = args.dimensions
        embedding_u_size = args.edge_dim
        u_num = edge_type_count
        num_sampled = args.negative_samples
        dim_a = args.att_dim
        att_head = 1
        neighbor_samples = args.neighbor_samples
        num_workers = args.workers

        device = torch.device(
            "cuda" if args.gpu is not None and torch.cuda.is_available() else "cpu"
        )

        g = get_graph(network_data, vocab)
        all_walks = []
        for i in range(edge_type_count):
            nodes = torch.LongTensor(type_nodes[i] * args.num_walks)
            traces, types = dgl.sampling.random_walk(
                g, nodes, metapath=[edge_types[i]] * (neighbor_samples - 1)
            )
            all_walks.append(traces)

        train_pairs = generate_pairs(all_walks, args.window_size, num_workers)
        neighbor_sampler = NeighborSampler(g, [neighbor_samples])
        train_dataloader = torch.utils.data.DataLoader(
            train_pairs,
            batch_size=batch_size,
            collate_fn=neighbor_sampler.sample,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        model = DGLGATNE(
            num_nodes, embedding_size, embedding_u_size, edge_types, edge_type_count, dim_a
        )
        nsloss = NSLoss(num_nodes, num_sampled, embedding_size)
        model.to(device)
        nsloss.to(device)

        optimizer = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-3
        )

        best_score = 0
        patience = 0
        for epoch in range(epochs):
            model.train()
            random.shuffle(train_pairs)

            data_iter = tqdm(
                train_dataloader,
                desc="epoch %d" % (epoch),
                total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            )
            avg_loss = 0.0

            for i, (block, head_invmap, tails, block_types) in enumerate(data_iter):
                optimizer.zero_grad()
                # embs: [batch_size, edge_type_count, embedding_size]
                block_types = block_types.to(device)
                embs = model(block[0].to(device))[head_invmap]
                embs = embs.gather(
                    1, block_types.view(-1, 1, 1).expand(embs.shape[0], 1, embs.shape[2])
                )[:, 0]
                loss = nsloss(
                    block[0].dstdata[dgl.NID][head_invmap].to(device),
                    embs,
                    tails.to(device),
                )
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.set_postfix(post_fix)

            model.eval()
            # {'1': {}, '2': {}}
            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(num_nodes):
                train_inputs = (
                    torch.tensor([i for _ in range(edge_type_count)])
                        .unsqueeze(1)
                        .to(device)
                )  # [i, i]
                train_types = (
                    torch.tensor(list(range(edge_type_count))).unsqueeze(1).to(device)
                )  # [0, 1]
                pairs = torch.cat(
                    (train_inputs, train_inputs, train_types), dim=1
                )  # (2, 3)
                (
                    train_blocks,
                    train_invmap,
                    fake_tails,
                    train_types,
                ) = neighbor_sampler.sample(pairs)

                node_emb = model(train_blocks[0].to(device))[train_invmap]
                node_emb = node_emb.gather(
                    1,
                    train_types.to(device)
                        .view(-1, 1, 1)
                        .expand(node_emb.shape[0], 1, node_emb.shape[2]),
                )[:, 0]

                for j in range(edge_type_count):
                    final_model[edge_types[j]][index2word[i]] = (
                        node_emb[j].cpu().detach().numpy()
                    )

            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        final_model[edge_types[i]],
                        valid_true_data_by_edge[edge_types[i]],
                        valid_false_data_by_edge[edge_types[i]],
                        num_workers,
                    )
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        final_model[edge_types[i]],
                        testing_true_data_by_edge[edge_types[i]],
                        testing_false_data_by_edge[edge_types[i]],
                        num_workers,
                    )
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print("valid auc:", np.mean(valid_aucs))
            print("valid pr:", np.mean(valid_prs))
            print("valid f1:", np.mean(valid_f1s))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break
        return average_auc, average_f1, average_pr

    def _mini_train_step(self, ):
        pass

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        pass

    def _full_train_setp(self):
        pass

    def _test_step(self, split=None, logits=None):
        pass
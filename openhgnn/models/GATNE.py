import dgl
from . import register_model, BaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
import dgl.function as fn


@register_model('GATNE-T')
class GATNE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg.num_nodes(), args.dim, args.edge_dim, hg.etypes, len(hg.etypes), args.att_dim)

    def __init__(
            self,
            num_nodes,
            embedding_size,
            embedding_u_size,
            edge_types,
            edge_type_count,
            att_dim,
    ):
        super(GATNE, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_types = edge_types
        self.edge_type_count = edge_type_count
        self.att_dim = att_dim

        self.node_embeddings = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.node_type_embeddings = Parameter(
            torch.FloatTensor(num_nodes, edge_type_count, embedding_u_size)
        )
        self.trans_weights = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size)
        )
        self.trans_weights_s1 = Parameter(
            torch.FloatTensor(edge_type_count, embedding_u_size, att_dim)
        )
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, att_dim, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.node_embeddings.data.uniform_(-1.0, 1.0)
        self.node_type_embeddings.data.uniform_(-1.0, 1.0)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    # embs: [batch_size, embedding_size]
    def forward(self, block):
        input_nodes = block.srcdata[dgl.NID]
        output_nodes = block.dstdata[dgl.NID]
        batch_size = block.number_of_dst_nodes()
        node_embed = self.node_embeddings
        node_type_embed = []

        with block.local_scope():
            for i in range(self.edge_type_count):
                edge_type = self.edge_types[i]
                block.srcdata[edge_type] = self.node_type_embeddings[input_nodes, i]
                block.dstdata[edge_type] = self.node_type_embeddings[output_nodes, i]
                block.update_all(
                    fn.copy_u(edge_type, "m"), fn.sum("m", edge_type), etype=edge_type
                )
                node_type_embed.append(block.dstdata[edge_type])

            node_type_embed = torch.stack(node_type_embed, 1)
            tmp_node_type_embed = node_type_embed.unsqueeze(2).view(
                -1, 1, self.embedding_u_size
            )
            trans_w = (
                self.trans_weights.unsqueeze(0)
                    .repeat(batch_size, 1, 1, 1)
                    .view(-1, self.embedding_u_size, self.embedding_size)
            )
            trans_w_s1 = (
                self.trans_weights_s1.unsqueeze(0)
                    .repeat(batch_size, 1, 1, 1)
                    .view(-1, self.embedding_u_size, self.att_dim)
            )
            trans_w_s2 = (
                self.trans_weights_s2.unsqueeze(0)
                    .repeat(batch_size, 1, 1, 1)
                    .view(-1, self.att_dim, 1)
            )

            attention = (
                F.softmax(
                    torch.matmul(
                        torch.tanh(torch.matmul(tmp_node_type_embed, trans_w_s1)),
                        trans_w_s2,
                    )
                        .squeeze(2)
                        .view(-1, self.edge_type_count),
                    dim=1,
                )
                    .unsqueeze(1)
                    .repeat(1, self.edge_type_count, 1)
            )

            node_type_embed = torch.matmul(attention, node_type_embed).view(
                -1, 1, self.embedding_u_size
            )
            node_embed = node_embed[output_nodes].unsqueeze(1).repeat(
                1, self.edge_type_count, 1
            ) + torch.matmul(node_type_embed, trans_w).view(
                -1, self.edge_type_count, self.embedding_size
            )
            last_node_embed = F.normalize(node_embed, dim=2)

            return last_node_embed  # [batch_size, edge_type_count, embedding_size]


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        # [ (log(i+2) - log(i+1)) / log(num_nodes + 1)]
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n

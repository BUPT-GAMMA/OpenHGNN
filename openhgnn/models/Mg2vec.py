import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from . import BaseModel, register_model


@register_model("Mg2vec")
class Mg2vec(BaseModel):
    r"""
    This is a model mg2vec from `mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via
    Metagraph Embedding<https://ieeexplore.ieee.org/document/9089251>`__

    It contains following parts:

    Achieve the metagraph and metagraph instances by mining the raw graph. Please go to
    `DataMaker-For-Mg2vec<https://github.com/null-xyj/DataMaker-For-Mg2vec>`__ for more details.

    Initialize the embedding for every node and metagraph and adopt an unsupervised method to train the node embeddings
    and metagraph embeddings. In detail, for every node, we keep its embedding close to the metagraph it belongs to and
    far away from the metagraph we get by negative sampling.

    Every node and meta-graph can be represented as an n-dim vector.We define the first-order loss and second-order
    loss.
    First-Order Loss is for single core node in every meta-graph. We compute the dot product of the node embedding and
    the positive meta-graph embedding as the true logit. Then We compute the dot product of the node embedding and
    the sampled negative meta-graph embedding as the neg logit. We use the binary_cross_entropy_with_logits function to
    compute the first-order loss.
    Second-Order Loss consider two core nodes in every meta-graph. First, we cancat the two node's embedding, what is a
    2n-dim vector. Then we use a 2n*n matrix and an n-dim vector to map the 2n-dim vector to an n-dim vector. The map
    function is showed below:
    .. math::
        f(u,v) = RELU([u||v]W + b)
    u and v means the origin embedding of the two nodes, || is the concatenation operator. W is the 2n*n matrix and b is
    the n-dim vector. RELU is the an activation function. f(u,v) means the n-dim vector after transforming.
    Then, the computation of second-order loss is the same as the first-order loss.
    Finally, we use a parameter alpha to balance the first-order loss and second-order loss.
    .. math::
        L=(1-alpha)*L_1 + alpha*L_2

    After we train the node embeddings, we use the embeddings to complete the relation prediction task.
    The relation prediction task is achieved by edge classification task. If two nodes are connected with a relation, we
    see the relation as an edge. Then we can adopt the edge classification to complete relation prediction task.

    Parameters
    ----------
    node_num: int
        the number of core-nodes
    mg_num: int
        the number of meta-graphs
    emb_dimension: int
        the embedding dimension of nodes and meta-graphs
    unigram: float
        the frequency of every meta-graph, for negative sampling
    sample_num: int
        the number of sampled negative meta-graph

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.node_num, args.mg_num, args.emb_dimension, args.unigram, args.sample_num)

    def __init__(self, node_num, mg_num, emb_dimension, unigram, sample_num):
        super(Mg2vec, self).__init__()
        self.node_num = node_num
        self.mg_num = mg_num
        self.mg_unigrams = torch.tensor(unigram, dtype=torch.float64)
        self.sample_num = sample_num
        self.emb_dimension = emb_dimension
        self.n_embedding = nn.Embedding(node_num, emb_dimension, sparse=False)
        self.m_embedding = nn.Embedding(mg_num, emb_dimension, sparse=False)
        self.n_w_t = nn.Parameter(torch.empty([emb_dimension * 2, emb_dimension]), requires_grad=True)
        self.n_b = nn.Parameter(torch.empty(emb_dimension), requires_grad=True)

        init.xavier_normal_(self.n_embedding.weight.data)
        init.xavier_normal_(self.m_embedding.weight.data)
        init.xavier_normal_(self.n_w_t)
        init.constant_(self.n_b, 0)

    def forward(self, train_a, train_b, train_labels, train_freq, train_weight, device):
        batch_size = len(train_a)
        n_embed_a = self.n_embedding(train_a)
        n_embed_b = self.n_embedding(train_b)
        n_embed_con = torch.cat([n_embed_a, n_embed_b], dim=1)
        mask_o1 = torch.eq(train_a, train_b).type(torch.FloatTensor).reshape(batch_size, 1).to(device)
        mask_o2 = torch.not_equal(train_a, train_b).type(torch.FloatTensor).reshape(batch_size, 1).to(device)
        m_embed_pos = self.m_embedding(train_labels)
        neg_sample_id = torch.multinomial(self.mg_unigrams, min(self.mg_num, self.sample_num)).to(device)
        neg_m_embed = self.m_embedding(neg_sample_id)
        n_embed_o1 = n_embed_a * mask_o1
        n_embed_o2 = F.relu(torch.mm(n_embed_con, self.n_w_t) + self.n_b) * mask_o2
        n_embed = torch.add(n_embed_o1, n_embed_o2)
        true_logit = torch.sum((n_embed * m_embed_pos), dim=1, keepdim=True)
        neg_logit = torch.mm(n_embed, neg_m_embed.T)
        logit = torch.cat([true_logit, neg_logit], dim=1)
        labels = torch.cat([torch.ones_like(true_logit), torch.zeros_like(neg_logit)], dim=1)
        xent = torch.sum(F.binary_cross_entropy_with_logits(logit, labels, reduction='none'), dim=1, keepdim=True)
        unsupervised_loss = torch.mean(train_weight * (train_freq * xent))
        return unsupervised_loss

    def normalize_embedding(self):
        norm = torch.sqrt_(torch.sum(torch.square(self.n_embedding.weight.data), dim=1, keepdim=True))
        self.n_embedding.weight.data = self.n_embedding.weight.data / norm

        m_norm = torch.sqrt_(torch.sum(torch.square(self.m_embedding.weight.data), dim=1, keepdim=True))
        self.m_embedding.weight.data = self.m_embedding.weight.data / m_norm

    def save_embedding(self, id2node, file_name):
        self.normalize_embedding()
        embedding = self.n_embedding.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            for nId, node in id2node.items():
                to_write = str(node) + ' ' + ' '.join(map(lambda x: str(x), embedding[nId])) + '\n'
                f.write(to_write)

    def save_embedding_np(self, file_name):
        self.normalize_embedding()
        embedding = self.n_embedding.weight.cpu().data.numpy()
        np.save(file_name, embedding)

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from . import BaseModel, register_model, HeteroEmbedLayer
from ..utils import get_nodes_dict
"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""

@register_model('Metapath2vec')
class HeteroEmbedding(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(get_nodes_dict(hg), args.dim)

    def __init__(self, n_nodes, dim):
        super(HeteroEmbedding, self).__init__()
        self.dim = dim
        self.u_embeddings = HeteroEmbedLayer(n_nodes, self.dim)
        self.n_nodes = n_nodes
        #self.v_embeddings = HeteroEmbedLayer(n_nodes, self.dim)

    def forward(self, graph):
        seed_nodes = graph.ndata[dgl.NID]
        emd = self.u_embeddings.forward()
        h = {}
        for ntype, idx in seed_nodes.items():
            h[ntype] = emd[ntype][idx]
        return h

    def extract_feature(self, *args, **kwargs):
        emd = self.u_embeddings.forward()
        h = {}
        for ntype in self.n_nodes:
            h[ntype] = emd[ntype]
        return h

    def __(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
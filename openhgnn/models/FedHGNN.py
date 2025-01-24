import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from . import BaseModel, register_model
import json


@register_model('FedHGNN')
class FedHGNN(BaseModel):
    
    @classmethod
    def build_model_from_args(cls, args):
        return cls(meta_paths=args.meta_paths,
                   in_size=args.in_size, hidden_size=args.hidden_size, out_size=args.out_size,
                   num_heads=args.num_heads,dropout=args.dropout)
    
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(FedHGNN, self).__init__()
        num_heads = json.loads(num_heads)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):#1
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size) #gatconv
    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)#更新

        return self.predict(h)#self.emb.weight

class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu, allow_zero_in_degree=True))

        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads) #gatconv
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        for i, meta_path in enumerate(self.meta_paths):
            new_g = dgl.metapath_reachable_graph(g, meta_path)#根据指定元路径从图中生成新图
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)#(batch,num_meta_path,dim)#(4025,2,16)


        return self.semantic_attention(semantic_embeddings)          # (N, D * K)

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)     例如：(4025,2,1)->(2,1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1) 加一个（4025，M，1）

        return (beta * z).sum(1)                       # (N, D * K)
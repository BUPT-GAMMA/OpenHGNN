import torch
import torch.nn as nn
from ..utils.utils import scatter
from . import BaseModel, register_model
import dgl
# from dgl import
from scipy.sparse import csr_matrix
import numpy as np


@register_model('RedGNN')
class RedGNN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, loader):
        return cls(args, loader)

    def __init__(self, args, loader):
        super(RedGNN, self).__init__()
        self.device = args.device
        self.hidden_dim = args.hidden_dim
        self.attn_dim = args.attn_dim
        self.n_layer = args.n_layer
        self.loader = loader
        self.n_rel = self.loader.n_rel
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[args.act]
        self.act = act
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(RedGNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(args.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)


    def forward(self, subs, rels, mode='transductive'): # source node, rels
        n = len(subs)

        n_ent = self.loader.n_ent if mode=='transductive' else self.loader.n_ent_ind

        q_sub = torch.LongTensor(subs).to(self.device)
        q_rel = torch.LongTensor(rels).to(self.device)

        h0 = torch.zeros((1, n, self.hidden_dim)).to(self.device) # 1 * n * d
        nodes = torch.cat([torch.arange(n).unsqueeze(1).to(self.device), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).to(self.device)

        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            edges = edges.to(self.device)
            old_nodes_new_idx = old_nodes_new_idx.to(self.device)
            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).to(self.device).index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).to(self.device)
        scores_all[[nodes[:, 0], nodes[:,1]]] = scores
        return scores_all


class RedGNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(RedGNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        r_idx = edges[:,0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs  + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new








import dgl
import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F

from dgl.ops import edge_softmax, segment_softmax
from dgl.nn import HeteroLinear, TypedLinear
from dgl.nn.pytorch.conv import GraphConv
from . import BaseModel, register_model
from ..utils import to_hetero_feat

@register_model('HGAT')
class HGAT(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.num_layers,
                   args.in_dim,
                   args.hidden_dim,
                   args.attn_dim,
                   args.num_classes,
                   hg.ntypes,
                   args.negative_slope)
    
    def __init__(self, num_layers, in_dim, hidden_dim, attn_dim,
                 num_classes, ntypes, negative_slope):
        super(HGAT, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        
        
        self.hgat_layers = nn.ModuleList()
        self.hgat_layers.append(
            TypeAttention(in_dim,
                          attn_dim,
                          ntypes,
                          negative_slope))
        self.hgat_layers.append(
            NodeAttention(in_dim,
                          attn_dim,
                          hidden_dim,
                          negative_slope)
        )
        for l in range(num_layers - 1):
            self.hgat_layers.append(
                TypeAttention(hidden_dim,
                            attn_dim,
                            ntypes,
                            negative_slope))
            self.hgat_layers.append(
                NodeAttention(hidden_dim,
                            attn_dim,
                            hidden_dim,
                            negative_slope)
            )
        
        self.hgat_layers.append(
            TypeAttention(hidden_dim,
                          attn_dim,
                          ntypes,
                          negative_slope))
        self.hgat_layers.append(
            NodeAttention(hidden_dim,
                          attn_dim,
                          num_classes,
                          negative_slope)
        )
        
        
    def forward(self, hg, h_dict):
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for l in range(self.num_layers):
                attention = self.hgat_layers[2 * l](hg, hg.ndata['h'])
                hg.edata['alpha'] = attention
                g = dgl.to_homogeneous(hg, ndata = 'h', edata = ['alpha'])
                h = self.hgat_layers[2 * l + 1](g, g.ndata['h'], g.ndata['_TYPE'], g.ndata['_TYPE'], presorted = True)
                h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
                hg.ndata['h'] = h_dict

        return h_dict

class TypeAttention(nn.Module):
    def __init__(self, in_dim, ntypes, slope):
        super(TypeAttention, self).__init__()
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = in_dim
        self.mu_l = HeteroLinear(attn_vector, in_dim)
        self.mu_r = HeteroLinear(attn_vector, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        
    def forward(self, hg, h_dict):
        h_t = {}
        attention = {}
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            for srctype, etype, dsttype in hg.canonical_etypes:
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                with rel_graph.local_scope():
                    degs = rel_graph.out_degrees().float().clamp(min = 1)
                    norm = torch.pow(degs, -0.5)
                    feat_src = h_dict[srctype]
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    feat_src = feat_src * norm
                    rel_graph.srcdata['h'] = feat_src
                    rel_graph.update_all(Fn.copy_src('h', 'm'), Fn.sum(msg='m', out='h'))
                    rst = rel_graph.dstdata['h']
                    degs = rel_graph.in_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    rst = rst * norm
                    h_t[srctype] = rst
                    h_l = self.mu_l(h_dict)[dsttype]
                    h_r = self.mu_r(h_t)[srctype]
                    edge_attention = F.elu(h_l + h_r)
                    # edge_attention = F.elu(h_l + h_r).unsqueeze(0)
                    rel_graph.ndata['m'] = {dsttype: edge_attention,
                                    srctype: torch.zeros((rel_graph.num_nodes(ntype = srctype),))}
                    # print(rel_graph.ndata)
                    reverse_graph = dgl.reverse(rel_graph)
                    reverse_graph.apply_edges(Fn.copy_src('m', 'alpha'))
                
                hg.edata['alpha'] = {(srctype, etype, dsttype): reverse_graph.edata['alpha']}
                
                # if dsttype not in attention.keys():
                #     attention[dsttype] = edge_attention
                # else:
                #     attention[dsttype] = torch.cat((attention[dsttype], edge_attention))
            attention = edge_softmax(hg, hg.edata['alpha'])
            # for ntype in hg.dsttypes:
            #     attention[ntype] = F.softmax(attention[ntype], dim = 0)

        return attention
    
class NodeAttention(nn.Module):
    def __init__(self, in_dim, out_dim, slope):
        super(NodeAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Mu_l = nn.Linear(in_dim, in_dim)
        self.Mu_r = nn.Linear(in_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(slope)
        
    def forward(self, g, x, ntype, etype, presorted = False):
        with g.local_scope():
            src = g.edges()[0]
            dst = g.edges()[1]
            h_l = self.Mu_l(x)[src]
            h_r = self.Mu_r(x)[dst]
            edge_attention = self.leakyrelu((h_l + h_r) * g.edata['alpha'])
            edge_attention = edge_softmax(g, edge_attention)
            g.edata['alpha'] = edge_attention
            g.srcdata['x'] = x
            g.update_all(Fn.u_mul_e('x', 'alpha', 'm'),
                         Fn.sum('m', 'x'))
            h = g.ndata['x']
        return h
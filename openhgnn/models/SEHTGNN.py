import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.nn import GRU
from torch.nn import LayerNorm
from collections import defaultdict
from . import BaseModel, register_model

class NodePredictor(nn.Module):
    """
    Standard MLP for node prediction.
    """
    def __init__(self, n_inp, n_classes):
        super(NodePredictor, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_inp)
        self.fc2 = nn.Linear(n_inp, n_classes)

    def forward(self, node_feat):
        node_feat = F.relu(self.fc1(node_feat))
        pred = self.fc2(node_feat)
        return pred

class GraphConv(nn.Module):
    """
    LightGCN-style Graph Convolution.
    """
    def __init__(self, norm="both", activation=None):
        super(GraphConv, self).__init__()
        self._norm = norm
        self._activation = activation

    def forward(self, g, feat):
        with g.local_scope():
            feat_src, feat_dst = feat
            if self._norm in ["left", "both"]:
                degs = g.out_degrees().to(feat_src.device).float().clamp(min=1) 
                norm = torch.pow(degs, -0.5) if self._norm == "both" else 1.0 / degs
                feat_src = feat_src * norm.unsqueeze(-1)
            
            g.srcdata["h"] = feat_src
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_neigh'))
            rst = feat_dst + g.dstdata['h_neigh']
            
            if self._norm in ["right", "both"]:
                degs = g.in_degrees().to(feat_dst.device).float().clamp(min=1)
                norm = torch.pow(degs, -0.5) if self._norm == "both" else 1.0 / degs
                rst = rst * norm.unsqueeze(-1)
            
            if self._activation is not None:
                rst = self._activation(rst)
            return rst

class DynamicAtt(nn.Module):
    """
    Dynamic Attention Module using GRU.
    Logic: GRU -> Mean -> Split.
    """
    def __init__(self, n_inp, n_hid, layer=1):
        super(DynamicAtt, self).__init__()
        self.gru = GRU(n_inp, 1, layer, batch_first=True)

    def forward(self, x, h0):
        out, _ = self.gru(x, h0)
        mask = torch.split(out.mean(0), 1, dim=0)
        dict_mask = {f"t{i}": mask[i].squeeze() for i in range(len(mask))}
        return dict_mask

class LinearProj(nn.Module):
    """
    Linear Projection for temporal aggregation.
    """
    def __init__(self, n_hid, timeframe):
        super(LinearProj, self).__init__()
        self.project = nn.Linear(len(timeframe), 1)

    def forward(self, h):
        out = self.project(h).mean(-1)
        return out

class LLM4init(nn.Module):
    """
    LLM-enhanced Initialization.
    """
    def __init__(self, graph, n_hid, LLM_feature, device):
        super(LLM4init, self).__init__()
        self.device = device
        self.LLM_feature = LLM_feature
        self.reltype = []
        
        for stype, etype, dtype in graph.canonical_etypes:
            real_rel_name = etype.rsplit('_', 1)[0] if '_' in etype else etype
            if (stype, real_rel_name, dtype) not in self.reltype:
                self.reltype.append((stype, real_rel_name, dtype))

    def forward(self):
        feature_dict = {k: v.to(self.device) for k, v in self.LLM_feature.items()}
        
        grouped_edges = defaultdict(list)
        
        for stype, reltype, dtype in self.reltype:
            stype_feat = feature_dict[stype]
            dtype_feat = feature_dict[dtype]
            
            inner_product = torch.dot(stype_feat.reshape(-1), dtype_feat.reshape(-1))
            grouped_edges[dtype].append((inner_product, stype, reltype, dtype))

        normalized_inner_products = {}
        for dtype, edges in grouped_edges.items():
            if not edges: continue
            
            inner_products = torch.stack([edge[0] for edge in edges])
            
            inner_products = inner_products - inner_products.max()
            softmax_weights = F.softmax(inner_products, dim=0)
            
            for i, (inner_product, stype, reltype, dtype) in enumerate(edges):
                normalized_inner_products[reltype] = softmax_weights[i]
        
        return normalized_inner_products, feature_dict

class HTGNNLayer(nn.Module):
    def __init__(self, graph, n_inp, n_hid, timeframe, norm):
        super(HTGNNLayer, self).__init__()
        self.timeframe = timeframe
        self.norm = norm
        self.intra_rel_agg = GraphConv(norm='right', activation=nn.ELU())
        
        self.predict = nn.ModuleDict()
        for srctype, etype, dsttype in graph.canonical_etypes:
            real_rel_name = etype.rsplit('_', 1)[0] if '_' in etype else etype
            if real_rel_name not in self.predict:
                self.predict[real_rel_name] = DynamicAtt(n_inp, n_hid)
        
        if norm:
            self.norm_layer = nn.ModuleDict({ntype: LayerNorm(n_hid) for ntype in graph.ntypes})

    def forward(self, graph, node_features, init_attention):
        intra_features = {ttype: {} for ttype in self.timeframe}
        h_mask_collector = defaultdict(list)
        
        for stype, etype, dtype in graph.canonical_etypes:
            if '_' in etype:
                ttype = etype.rsplit('_', 1)[1]
                reltype = etype.rsplit('_', 1)[0]
            else:
                ttype = 't0'; reltype = etype
            
            if ttype in node_features[stype] and ttype in node_features[dtype]:
                dst_feat = self.intra_rel_agg(
                    graph[stype, etype, dtype], 
                    (node_features[stype][ttype], node_features[dtype][ttype])
                )
                intra_features[ttype][(stype, etype, dtype)] = dst_feat
                h_mask_collector[reltype].append(dst_feat)

        h_mask = {}
        for reltype, feats in h_mask_collector.items():
            if not feats: continue
            stacked_feats = torch.stack(feats, dim=1)
            
            if reltype in init_attention:
                h0 = init_attention[reltype].reshape(1, 1, 1).expand(1, stacked_feats.size(0), 1).contiguous()
                h_mask[reltype] = self.predict[reltype](stacked_feats, h0)
        
        inter_features = {ntype: {} for ntype in graph.ntypes}
        for ttype in self.timeframe:
            for ntype in graph.ntypes:
                collector, weights = [], []
                
                if ttype in intra_features:
                    for (stype, etype, dtype), feat in intra_features[ttype].items():
                        if dtype == ntype:
                            reltype = etype.rsplit('_', 1)[0] if '_' in etype else etype
                            if reltype in h_mask and ttype in h_mask[reltype]:
                                collector.append(feat)
                                weights.append(h_mask[reltype][ttype])
                
                if not collector:
                    if ttype in node_features[ntype]: 
                        inter_features[ntype][ttype] = node_features[ntype][ttype]
                    continue
                
                w_norm = F.softmax(torch.stack(weights, dim=0), dim=0)
                out_feat = sum([collector[i] * w_norm[i] for i in range(len(collector))])
                
                if self.norm: 
                    inter_features[ntype][ttype] = self.norm_layer[ntype](out_feat)
                else: 
                    inter_features[ntype][ttype] = out_feat + node_features[ntype][ttype]
                    
        return inter_features

@register_model('SEHTGNN')
class SEHTGNN(BaseModel):
    r"""
    This is the model SE-HTGNN from `Simple and Efficient Heterogeneous Temporal Graph Neural Network 
    <https://arxiv.org/abs/2510.18467v1>`__.

    The model proposes a unified spatial-temporal learning paradigm that integrates temporal modeling into 
    spatial learning via a dynamic attention mechanism, guided by LLM-enhanced prior knowledge.

    1. Heterogeneous Feature Projection:

    .. math::
        H_v^t = W_v \cdot X_v^t + b_v

    2. Simplified Neighbor Aggregation (GCN-style):

    .. math::
        H_{v,r}^t = \sigma(A_r^t H_{\mathcal{N}_r^t(v)}^t)

    3. Dynamic-Attention-based Fusion (via GRU):

    .. math::
        e_{v,r}^t = GRU_r(H_{v,r}^t, e_{v,r}^{t-1})
        
    .. math::
        \alpha_r^t = \frac{\exp(\overline{e}_{v,r}^t)}{\sum_{r' \in \mathcal{R}(v)} \exp(\overline{e}_{v,r'}^t)}

    4. Representation Fusion:

    .. math::
        H_v^t = \sum_{r \in \mathcal{R}(v)} \alpha_r^t \cdot H_{v,r}^t

    5. Linear Temporal Projection:

    .. math::
        Z_v' = Z_v \cdot W + b

    Parameters
    ----------
    graph : DGLGraph
        The input heterogeneous temporal graph snapshots.
    n_inp : int
        The input feature dimension.
    n_hid : int
        The hidden dimension for representations.
    n_layers : int
        The number of GNN layers.
    time_window : int
        The size of the temporal window (number of snapshots).
    norm : bool
        Whether to use LayerNorm or residual connections.
    device : torch.device
        The device for computation.
    dropout : float
        The dropout rate for feature adaptation.
    LLM_feature : dict
        The dictionary containing LLM-encoded semantic features for node types.
    num_classes : int
        The number of output classes or prediction steps.
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        llm_feat = getattr(args, 'semantic_feature', None)
        if llm_feat is None: llm_feat = getattr(args, 'LLM_feature', None)
        
        if llm_feat is not None and isinstance(llm_feat, dict):
            dim = list(llm_feat.values())[0].shape[1]
            print(f"[SEHTGNN] LLM feature dimension: {dim}")
            
        hidden_dim = getattr(args, 'hidden_dim', 32)
        num_layers = getattr(args, 'num_layers', 2)
        time_window = getattr(args, 'time_window', 7)
        num_classes = getattr(args, 'num_classes', 1)
        dropout = getattr(args, 'dropout', 0.3)
        norm = getattr(args, 'norm', True)
        
        return cls(hg, hidden_dim, num_layers, time_window, norm, args.device, dropout, llm_feat, num_classes)

    def __init__(self, graph, n_hid, n_layers, time_window, norm, device, dropout, LLM_feature, num_classes):
        super(SEHTGNN, self).__init__()
        self.timeframe = [f't{i}' for i in range(time_window)]
        self.device = device
        self.n_layers = n_layers
        self.norm = norm
        self.drop = nn.Dropout(dropout)
        
        self.adaption_layer = nn.ModuleDict()
        
        for ntype in graph.ntypes:
            if 't0' in graph.nodes[ntype].data:
                feat_dim = graph.nodes[ntype].data['t0'].shape[1]
            else:
                feat_dim = 1
            self.adaption_layer[ntype] = nn.Linear(feat_dim, n_hid)
        
        if LLM_feature is not None: 
            self.LLM_init = LLM4init(graph, n_hid, LLM_feature, device)
        else:
            print("Initializing with Random Semantics...")
            dummy_feat = {nt: torch.randn(1, 4096).to(device) for nt in graph.ntypes}
            self.LLM_init = LLM4init(graph, n_hid, dummy_feat, device)
            
        self.gnn_layers = nn.ModuleList([
            HTGNNLayer(graph, n_hid, n_hid, self.timeframe, norm) for _ in range(n_layers)
        ])
        
        self.LinearProj = LinearProj(n_hid, self.timeframe)
        self.predictor = NodePredictor(n_hid, num_classes)

    def forward(self, hg, h_dict=None):
        init_attention, _ = self.LLM_init()
        
        spatial_feat = {ntype: {} for ntype in hg.ntypes}
        for ntype in hg.ntypes:
            for ttype in self.timeframe:
                if ttype in hg.nodes[ntype].data:
                    raw_feat = hg.nodes[ntype].data[ttype]
                    if raw_feat.dtype != torch.float32:
                        raw_feat = raw_feat.float()
                    
                    if raw_feat.dim() == 1:
                        raw_feat = raw_feat.unsqueeze(1)
                        
                    spatial_feat[ntype][ttype] = self.adaption_layer[ntype](self.drop(raw_feat))
        
        for layer in self.gnn_layers:
            spatial_feat = layer(hg, spatial_feat, init_attention)

        out_embeddings = {}
        for ntype in hg.ntypes:
            feats_to_stack = []
            for tt in self.timeframe:
                if tt in spatial_feat[ntype]: 
                    feats_to_stack.append(spatial_feat[ntype][tt])
            
            if len(feats_to_stack) == len(self.timeframe):
                t_stacked = torch.stack(feats_to_stack, dim=2)
                hidden_vec = self.LinearProj(t_stacked)
                out_embeddings[ntype] = self.predictor(hidden_vec)
                
        return out_embeddings
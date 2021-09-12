import dgl
import torch.nn as nn
from openhgnn.models import BaseModel, register_model
from openhgnn.utils.utils import extract_metapaths
from space4hgnn.models.skipgnn import stage_dict
from space4hgnn.models.MLP import HGNNPostMP, HGNNPreMP

@register_model('mp_GNN')
class mp_GNN(BaseModel):
    r"""
    General MetaPath GNN model, e.g.: HAN, HPN
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg, **kwargs):
        """
        """
        super(mp_GNN, self).__init__()
        self.out_node_type = args.out_node_type
        self.category = list(self.out_node_type)
        self.meta_paths_dict = {key: [] for key in self.category}
        if args.meta_paths is None:
            for key in self.category:
                self.meta_paths_dict[key].append(extract_metapaths(key, hg.canonical_etypes))
        else:
            for mp in args.meta_paths:
                self.meta_paths_dict[mp[0][0]].append(mp)

        if args.layers_pre_mp - 1 > 0:
            self.pre_mp = HGNNPreMP(args, self.out_node_type, args.layers_pre_mp, args.hidden_dim, args.hidden_dim)

        if args.layers_gnn > 0:
            GNNStage = stage_dict[args.stage_type]
            self.gnn = nn.ModuleDict()
            for key in self.category:
                self.gnn[key] = GNNStage(gnn_type=args.gnn_type,
                                    stage_type=args.stage_type,
                                    dim_in=args.hidden_dim,
                                    dim_out=args.hidden_dim,
                                    num_layers=args.layers_gnn,
                                    skip_every=1,
                                    dropout=args.dropout,
                                    act=args.activation,
                                    has_bn=args.has_bn,
                                    has_l2norm=args.has_l2norm,
                                    macro_func=args.macro_func,
                                    meta_paths=self.meta_paths_dict[key])
        #     d_in = self.mp.dim_out

        gnn_out_dim = self.gnn[self.category[0]].dim_out
        self.post_mp = HGNNPostMP(args, self.out_node_type, args.layers_post_mp, gnn_out_dim, args.out_dim)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, hg, h_dict):
        with hg.local_scope():

            if self._cached_graph is None or self._cached_graph is not hg:
                self._cached_graph = hg
                self._cached_coalesced_graph.clear()
                for key, metapaths in self.meta_paths_dict.items():
                    self._cached_coalesced_graph[key] = {}
                    for meta_path in metapaths:
                        mp_g = dgl.metapath_reachable_graph(hg, meta_path)
                        mp_g = dgl.remove_self_loop(mp_g)
                        mp_g = dgl.add_self_loop(mp_g)
                        self._cached_coalesced_graph[key][meta_path] = mp_g
            out_h = {}
            for key in self.category:
                if hasattr(self, 'pre_mp'):
                    h = self.pre_mp({key: h_dict[key]})
                else:
                    h = {key: h_dict[key]}
                if hasattr(self, 'gnn'):
                    h = self.gnn[key](self._cached_coalesced_graph[key], h[key])
                if hasattr(self, 'post_mp'):
                    out_h[key] = self.post_mp({key: h})[key]
        return out_h

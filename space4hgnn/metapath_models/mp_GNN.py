import dgl
from space4hgnn.models.MLP import GNNPreMP, GNNPostMP
from openhgnn.models import BaseModel, register_model
from openhgnn.utils.utils import extract_metapaths
from space4hgnn.models.skipgnn import stage_dict


@register_model('mp_GNN')
class mp_GNN(BaseModel):
    r"""
    General MetaPath GNN model, e.g.: HAN
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args, hg)

    def __init__(self, args, hg, **kwargs):
        """
        """
        super(mp_GNN, self).__init__()
        self.category = args.category
        if args.meta_paths is None:
            extract_metapaths(self.category, hg.canonical_etypes)
        else:
            self.meta_paths = args.meta_paths
        if args.layers_pre_mp - 1 > 0:
            self.pre_mp = GNNPreMP(args, args.hidden_dim, args.hidden_dim)

        if args.layers_gnn > 0:
            GNNStage = stage_dict[args.stage_type]
            self.gnn = GNNStage(gnn_type=args.gnn_type,
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
                                meta_paths=self.meta_paths)
        #     d_in = self.mp.dim_out


        gnn_out_dim = self.gnn.dim_out
        self.post_mp = GNNPostMP(args, gnn_out_dim, args.out_dim)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, hg, h_dict):
        with hg.local_scope():

            if self._cached_graph is None or self._cached_graph is not hg:
                self._cached_graph = hg
                self._cached_coalesced_graph.clear()
                for meta_path in self.meta_paths:
                    mp_g = dgl.metapath_reachable_graph(hg, meta_path)
                    mp_g = dgl.remove_self_loop(mp_g)
                    mp_g = dgl.add_self_loop(mp_g)
                    self._cached_coalesced_graph[meta_path] = mp_g

            h = h_dict[self.category]
            if hasattr(self, 'pre_mp'):
                h = self.pre_mp(h)
            if hasattr(self, 'gnn'):
                out_h = self.gnn(self._cached_coalesced_graph, h)
            if hasattr(self, 'post_mp'):
                out_h = self.post_mp(out_h)
        return {self.category: out_h}

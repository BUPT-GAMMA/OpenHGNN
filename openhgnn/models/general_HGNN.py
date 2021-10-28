import dgl
from ..layers import SkipConnection
from openhgnn.models import BaseModel, register_model
from ..models.HeteroMLP import HGNNPostMP, HGNNPreMP


stage_dict = {
    'stack': SkipConnection.HGNNStackStage,
    'skipsum': SkipConnection.HGNNSkipStage,
    'skipconcat': SkipConnection.HGNNSkipStage,
}


def HG_transformation(hg, metapaths_dict):
    graph_data = {}
    for key, mp in metapaths_dict.items():
        mp_g = dgl.metapath_reachable_graph(hg, mp)
        n_edge = mp_g.canonical_etypes[0]
        graph_data[(n_edge[0], key, n_edge[2])] = mp_g.edges()
    return dgl.heterograph(graph_data)


@register_model('general_HGNN')
class general_HGNN(BaseModel):
    """
    General heterogeneous GNN model
    """

    @classmethod
    def build_model_from_args(cls, args, hg):
        out_node_type = args.out_node_type
        args.subgraph_extraction = args.subgraph
        if args.subgraph_extraction == 'relation':
            new_hg = hg
            print('relation extraction!')
        elif args.subgraph_extraction == 'metapath':
            if hasattr(args, 'meta_paths_dict'):
                new_hg = HG_transformation(hg, args.meta_paths_dict)
                print('metapath extraction!')
            else:
                raise ValueError('No meta-path is specified!')
        elif args.subgraph_extraction == 'mixed':
            relation_dict = args.meta_paths_dict
            for etype in hg.canonical_etypes:
                relation_dict[etype[1]] = [etype]
            new_hg = HG_transformation(hg, relation_dict)
            print('mixed extraction!')
            pass
        else:
            raise ValueError('subgraph_extraction only supports relation, metapath and mixed')
        return cls(args, new_hg, out_node_type)

    def __init__(self, args, hg, out_node_type, **kwargs):
        """
        """
        super(general_HGNN, self).__init__()
        self.hg = hg
        self.out_node_type = out_node_type
        # the first linear is operated in outside of model (in trainerflow)
        if args.layers_pre_mp - 1 > 0:
            self.pre_mp = HGNNPreMP(args, self.hg.ntypes, args.layers_pre_mp, args.hidden_dim, args.hidden_dim)

        if args.layers_gnn > 0:
            HGNNStage = stage_dict[args.stage_type]
            self.hgnn = HGNNStage(gnn_type=args.gnn_type,
                                  rel_names=self.hg.etypes,
                                  stage_type=args.stage_type,
                                  dim_in=args.hidden_dim,
                                  dim_out=args.hidden_dim,
                                  num_layers=args.layers_gnn,
                                  skip_every=1,
                                  dropout=args.dropout,
                                  act=args.activation,
                                  has_bn=args.has_bn,
                                  has_l2norm=args.has_l2norm,
                                  num_heads=args.num_heads,
                                  macro_func=args.macro_func)
        gnn_out_dim = self.hgnn.dim_out
        self.post_mp = HGNNPostMP(args, self.out_node_type, args.layers_post_mp, gnn_out_dim, args.out_dim)

    def forward(self, hg, h_dict):
        with hg.local_scope():
            hg = self.hg
            h_dict = {key: value for key, value in h_dict.items() if key in hg.ntypes}
            if hasattr(self, 'pre_mp'):
                h_dict = self.pre_mp(h_dict)
            if hasattr(self, 'hgnn'):
                h_dict = self.hgnn(hg, h_dict)
            if hasattr(self, 'post_mp'):
                out_h = {}
                for key, value in h_dict.items():
                    if key in self.out_node_type:
                        out_h[key] = value
                out_h = self.post_mp(out_h)
        return out_h

import torch
import torch.nn as nn   
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes

@register_model('HGOT')
class HGOT(BaseModel):
    @classmethod  
    def build_model_from_args(cls, args, hg):   
        ntypes = set()  
        if hasattr(args, 'target_link'):  
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)  
        elif hasattr(args, 'category'):  
            ntypes.add(args.category)  
        else:  
            raise ValueError  
  
        ntype_meta_paths_dict = {}  
        for ntype in ntypes:  
            ntype_meta_paths_dict[ntype] = {}  
            for meta_path_name, meta_path in args.meta_paths_dict.items():  
                if meta_path[0][0] == ntype:  
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path  
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():  
            if len(meta_paths_dict) == 0:  
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)  
  
        return cls(ntype_meta_paths_dict=ntype_meta_paths_dict,  
                   in_dim=args.hidden_dim,  
                   hidden_dim=args.hidden_dim,  
                   out_dim=args.out_dim,  
                   num_heads=args.num_heads,  
                   dropout=args.dropout,  
                   ot_sigma=getattr(args, 'sigma', 1.0),  
                   ot_rho=getattr(args, 'rho', 0.1))  

    def __init__(self, ntype_meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout, ot_sigma, ot_rho):  
        super(HGOT, self).__init__()  
        self.out_dim = out_dim  
          
        self.feature_transforms = nn.ModuleDict()  
        for ntype in ntype_meta_paths_dict.keys():  
            self.feature_transforms[ntype] = nn.Linear(in_dim, hidden_dim)  
           
        self.backbone_encoders = nn.ModuleDict()  
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():  
            self.backbone_encoders[ntype] = HGOTLayer(  
                meta_paths_dict, hidden_dim, hidden_dim, num_heads[0], dropout  
            )  
          
        self.ot_solver = OTSolver(sigma=ot_sigma, rho=ot_rho)  

    def forward(self, g, h_dict):   
        projected_features = {}  
        for ntype, features in h_dict.items():  
            projected_features[ntype] = self.feature_transforms[ntype](features)  
            
        branch_views = {}  
        for ntype, encoder in self.backbone_encoders.items():  
            if isinstance(g, dict):  
                if ntype not in g:  
                    continue  
                _g = g[ntype]  
                _in_h = projected_features[ntype]  
            else:  
                _g = g  
                _in_h = projected_features  
            branch_views[ntype] = encoder(_g, {ntype: _in_h})  
            
        aggregated_view = self.generate_aggregated_view(g, projected_features)  
           
        ot_loss = self.ot_solver.compute_ot_loss(branch_views, aggregated_view)  
          
        return ot_loss

    def generate_aggregated_view(self, g, h_dict):   
        meta_path_views = list(h_dict.values())  
        meta_path_names = list(h_dict.keys())  
        
        if len(meta_path_views) == 0:  
            return {'aggregated': torch.zeros_like(list(h_dict.values())[0])}  
          
        omega_scores = []  
        for view in meta_path_views:   
            transformed = self.meta_linear(view)    
            activated = torch.tanh(transformed)  
            pooled = activated.mean(dim=0) 
            score = self.query_vector(pooled)  
            omega_scores.append(score)  
        omega_scores = torch.stack(omega_scores)  
        beta_weights = F.softmax(omega_scores, dim=0) 
        h_agg = torch.zeros_like(meta_path_views[0])
        for i, view in enumerate(meta_path_views):  
            h_agg += beta_weights[i] * view  
        A_agg = self.build_aggregated_adjacency(g)  
        
        return {  
            'aggregated': h_agg,  
            'adjacency': A_agg,  
            'meta_path_weights': beta_weights  
        }  

    def build_aggregated_adjacency(self, g):   
        meta_paths = list(self.meta_paths_dict.keys())  
        
        if not meta_paths:  
            return torch.zeros(g.num_nodes(), g.num_nodes())  
         
        A_agg = torch.zeros(g.num_nodes(), g.num_nodes(), device=g.device)  
          
        for meta_path in meta_paths:   
            mp_graph = dgl.metapath_reachable_graph(g, self.meta_paths_dict[meta_path])  
            
            src, dst = mp_graph.edges()  
            A_agg[src, dst] = 1   
        
        return A_agg
    
class OTSolver(nn.Module):  
    """最优传输求解器"""  
    def __init__(self, sigma=1.0, rho=0.1):  
        super(OTSolver, self).__init__()  
        self.sigma = sigma  
        self.rho = rho  
        
      
    def compute_ot_loss(self, branch_views, aggregated_view):  
        # 实现 Gromov-Wasserstein 距离和匹配损失  
        # 这里需要实现 Sinkhorn 算法  
        return torch.tensor(0.0)  
    
class _HGOT(nn.Module):    
    def __init__(self, meta_paths_dict, in_dim, hidden_dim, out_dim, num_heads, dropout):  
        super(_HGOT, self).__init__()  
        self.layers = nn.ModuleList()   
        self.layers.append(HGOTLayer(meta_paths_dict, in_dim, hidden_dim, num_heads[0], dropout))  
        for l in range(1, len(num_heads)):  
            self.layers.append(HGOTLayer(meta_paths_dict, hidden_dim * num_heads[l - 1],  
                                        hidden_dim, num_heads[l], dropout))  
 
        self.output_dim = hidden_dim * num_heads[-1]  
  
    def forward(self, g, h_dict):  
        for gnn in self.layers:  
            h_dict = gnn(g, h_dict)  
 
        return h_dict  
  
    def get_emb(self, g, h_dict):    
        if isinstance(h_dict, dict):  
            first_ntype = list(h_dict.keys())[0]  
            h = h_dict[first_ntype]  
        else:  
            h = h_dict  
              
        for gnn in self.layers:  
            h = gnn(g, {first_ntype: h} if isinstance(h_dict, dict) else h)  
           
        return {first_ntype: h}
    
class HGOTLayer(nn.Module):  
      
    def __init__(self, meta_paths_dict, in_dim, out_dim, layer_num_heads, dropout):  
        super(HGOTLayer, self).__init__()  
        self.meta_paths_dict = meta_paths_dict  
            
        self.mods = nn.ModuleDict({mp: GATConv(in_dim, out_dim, layer_num_heads,  
                                              dropout, dropout, activation=F.elu,  
                                              allow_zero_in_degree=True) for mp in meta_paths_dict})  
           
        self.semantic_attention = SemanticAttention(in_size=out_dim * layer_num_heads)  
        self.use_semantic_attention = False  # HGOT 默认不使用语义融合  
           
        self._cached_graph = None  
        self._cached_coalesced_graph = {}  
  
    def forward(self, g, h, return_individual_views=False):    
        if isinstance(g, dict):  
            if return_individual_views:   
                individual_views = {}  
                for mp_name, mp_g in g.items():  
                    if h.get(mp_name) is not None:  
                        mp_h = h[mp_name][mp_g.srctypes[0]]  
                    else:  
                        mp_h = h[mp_g.srctypes[0]]  
                    individual_views[mp_name] = self.mods[mp_name](mp_g, mp_h).flatten(1)  
                return individual_views  
            else:  
                h = self.model(g, h)  
   
        else:  
            if self._cached_graph is None or self._cached_graph is not g:  
                self._cached_graph = g  
                self._cached_coalesced_graph.clear()  
                for mp, mp_value in self.meta_paths_dict.items():  
                    self._cached_coalesced_graph[mp] = dgl.metapath_reachable_graph(  
                        g, mp_value)  
              
            if return_individual_views:   
                individual_views = {}  
                for mp_name, mp_g in self._cached_coalesced_graph.items():  
                    mp_h = h[mp_g.srctypes[0]]  
                    individual_views[mp_name] = self.mods[mp_name](mp_g, mp_h).flatten(1)  
                return individual_views  
            else:  
                h = self.model(self._cached_coalesced_graph, h)  
  
        return h  
      
    def get_branch_views(self, g, h):  
        return self.forward(g, h, return_individual_views=True)
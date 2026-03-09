import torch
import torch.nn as nn   
import torch.nn.functional as F
import dgl
import ot
from dgl.nn.pytorch import GATConv
from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes
from ot.lp import emd
from ot.gromov import semirelaxed_fused_gromov_wasserstein
from ot.gromov._utils import init_matrix_semirelaxed, tensor_product
from ot.utils import get_backend
from geomloss import SamplesLoss

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
          
        self.meta_linear = nn.Linear(hidden_dim, hidden_dim)  
        self.query_vector = nn.Linear(hidden_dim, 1)  
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
            return {'aggregated': torch.zeros(1, self.out_dim)}  
          
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
        # 从 ntype_meta_paths_dict 中提取所有 meta_path
        meta_paths = []
        for ntype, meta_paths_dict in self.ntype_meta_paths_dict.items():
            meta_paths.extend(meta_paths_dict.values())
        
        if not meta_paths:  
            return torch.zeros(g.num_nodes(), g.num_nodes(), device=g.device)  
         
        A_agg = torch.zeros(g.num_nodes(), g.num_nodes(), device=g.device)  
          
        for meta_path in meta_paths:   
            mp_graph = dgl.metapath_reachable_graph(g, meta_path)  
            
            src, dst = mp_graph.edges()  
            A_agg[src, dst] = 1   
        
        return A_agg

class OTSolver(nn.Module):  
    def __init__(self, sigma=0.5, rho=1.0, num_iter=100, eps=1e-1):
        """
        Args:
        sigma (float): Balance parameter between node feature loss and edge structure loss
        rho (float): Balance parameter between matching loss and structure loss
        num_iter (int): Number of Sinkhorn iterations
        eps (float): Sinkhorn regularization coefficient (entropy regularization)
        """
        super(OTSolver, self).__init__()  
        self.sigma = sigma  
        self.rho = rho
        self.num_iter = num_iter
        self.eps = eps
            
    def compute_ot_loss(self, branch_views, aggregated_view):  
        """
        Args:
            branch_views: Dict containing 'features' and 'adj' for each meta-path view
                e.g., {'ntype1': {'features': Z_p1, 'adj': A_p1}, ...}
            aggregated_view: Dict containing 'features' and 'adj' for the aggregated view
                {'aggregated': h_agg, 'adjacency': A_agg, 'meta_path_weights': beta_weights}
                
        Returns:
            total_loss: Scalar tensor
        """        
        device = list(branch_views.values())[0]['features'].device
        num_views = len(branch_views)
        total_loss = 0.0
        
        Z_agg = aggregated_view['aggregated']
        A_agg = aggregated_view['adjacency']
        
        for ntype, branch in branch_views.items():
            Z_branch = branch['features']
            A_branch = branch['adj']
            
            h1 = ot.unif(Z_branch.shape[0], type_as=Z_branch)
            h2 = ot.unif(Z_agg.shape[0], type_as=Z_agg)
            
            Mp = ot.dist(Z_branch, Z_agg, metric='euclidean')
            Mb = ot.dist(Z_branch, Z_agg, metric='euclidean')
            
            if self.sigma < 1:
                P = semirelaxed_fused_gromov_wasserstein(
                    Mp, A_branch, A_agg, h1, symmetric=True, 
                    alpha=1 - self.sigma, log=False, G0=None
                )
                
                nx = get_backend(h1, A_branch, A_agg)
                constC, hC1, hC2, fC2t = init_matrix_semirelaxed(
                    A_branch, A_agg, h1, loss_fun='square_loss', nx=nx
                )
                
                N1l = Z_branch.shape[0]
                N2l = Z_agg.shape[0]
                OM = torch.ones(N1l, N2l).to(device)
                OM = OM / (N1l * N2l)
                qOneM = nx.sum(OM, 0)
                ones_p = nx.ones(h1.shape[0], type_as=h1)
                marginal_product = nx.outer(ones_p, nx.dot(qOneM, fC2t))
                
                Mp2 = tensor_product(constC + marginal_product, hC1, hC2, P, nx=nx)
                Mp2 = F.normalize(Mp2)
                
                Mp = (self.sigma) * Mp + (1 - self.sigma) * Mp2
                
                B = emd(h1, h2, Mb)
                
                sloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
                loss = sloss(Mp, Mb)
                
                loss = self.rho * loss + torch.linalg.matrix_norm(P - B, ord='fro')
            elif self.sigma == 1:
                sl = SamplesLoss(loss='sinkhorn', p=2, debias=True, blur=0.1 ** (1 / 2), backend='tensorized')
                m = 0 * Mb + 1 * Mp
                sl.potentials = True
                u, v = sl(Z_branch, Z_agg)
                P = torch.exp((u.t() + v - m) * 1 / 0.1)
                
                sl.potentials = True
                u, v = sl(Z_branch, Z_agg)
                B = torch.exp((u.t() + v - m) * 1 / 0.1)
                
                sloss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
                loss = sloss(Mp, Mb)
                
                loss = self.rho * loss + torch.linalg.matrix_norm(P - B, ord='fro')
            
            total_loss += loss
            
        return total_loss / num_views
    
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
        self.use_semantic_attention = False 
           
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
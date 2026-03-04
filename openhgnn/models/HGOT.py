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
    """最优传输求解器，基于论文 HGOT 中的 Gromov-Wasserstein 距离实现"""
    def __init__(self, sigma=0.5, rho=1.0, num_iter=100, eps=1e-1):
        """
        初始化参数
        Args:
            sigma (float): 节点特征与边结构损失的平衡参数 (公式 13 中的 sigma)
            rho (float): 匹配损失与结构损失的平衡参数 (公式 19 中的 rho)
            num_iter (int): Sinkhorn 迭代次数
            eps (float): Sinkhorn 正则化系数 (熵正则化)
        """
        super(OTSolver, self).__init__()  
        self.sigma = sigma  
        self.rho = rho
        self.num_iter = num_iter
        self.eps = eps
        
    def _compute_cost_matrix(self, X, Y):
        """
        计算成本矩阵 C(X_i, Y_j)
        这里使用欧氏距离的平方，论文公式(8)提到可以使用 cosine，但通常GW用L2平方效果稳定
        """
        # 处理 None 或空张量的情况 (例如当只输入邻接矩阵而不输入特征时)
        if X is None and Y is None:
            return torch.tensor(0.0)
        if X is None:
            X = torch.zeros(Y.shape, Y.shape).to(Y.device)
        if Y is None:
            Y = torch.zeros(X.shape, X.shape).to(X.device)
            
        # 计算成对距离矩阵 ||x_i - y_j||^2
        # 使用广播机制
        XX = torch.sum(X**2, dim=1, keepdim=True) # (n, 1)
        YY = torch.sum(Y**2, dim=1, keepdim=True) # (m, 1)
        XY = torch.mm(X, Y.t()) # (n, m)
        C = XX - 2*XY + YY.t()
        return C
    
    def _sinkhorn(self, C, mu, nu):
        """
        使用 Sinkhorn-Knopp 算法求解熵正则化的最优传输问题
        Args:
            C: 成本矩阵 (n, m)
            mu: 源分布 (n,)
            nu: 目标分布 (m,)
        Returns:
            pi: 传输计划矩阵 (n, m)
        """
        # 初始化对偶变量
        psi = torch.zeros_like(nu)
        # 避免除以零
        u = torch.ones_like(mu) / mu.shape
        
        K = torch.exp(-C / self.eps) # 核矩阵
        
        for _ in range(self.num_iter):
            # 更新 phi (对应 mu)
            phi = torch.log(u + 1e-8) - torch.log(torch.matmul(K, torch.exp(psi)) + 1e-8)
            # 更新 psi (对应 nu)
            psi = torch.log(nu + 1e-8) - torch.log(torch.matmul(K.t(), torch.exp(phi)) + 1e-8)
        
        # 计算传输计划 pi
        pi = torch.matmul(torch.exp(phi).unsqueeze(1), torch.exp(psi).unsqueeze(0)) * K
        return pi
    
    def compute_ot_loss(self, branch_views, aggregated_view):  
        """
        计算完整的 OT 损失 (公式 19)
        Args:
            branch_views: List of dicts containing 'features' and 'adj' for each meta-path view
                e.g., [{'features': Z_p1, 'adj': A_p1}, ...]
            aggregated_view: Dict containing 'features' and 'adj' for the aggregated view
                {'features': Z_agg, 'adj': A_agg}
                
        Returns:
            total_loss: Scalar tensor
        """
        device = branch_views['features'].device
        num_views = len(branch_views)
        total_loss = 0.0
        
        # 假设均匀分布
        n_agg = aggregated_view['features'].shape
        mu = torch.ones(n_agg).to(device) / n_agg
        
        for branch in branch_views:
            n_branch = branch['features'].shape
            nu = torch.ones(n_branch).to(device) / n_branch
            
            # ==========================================
            # Step 1: 计算 Graph Space 的最优传输计划 pi_G (公式 15)
            # 需要计算节点成本 F 和 边成本 E (Gromov-Wasserstein)
            # ==========================================
            
            # 节点特征成本矩阵 (公式 8 & 9)
            C_node = self._compute_cost_matrix(branch['features'], aggregated_view['features'])
            
            # 边结构成本矩阵 (公式 11 & 12) - Gromov-Wasserstein 核心
            # 计算 |A_ik - A_jl| 近似
            # 这里简化处理，计算邻接矩阵行/列的差异作为结构成本的近似
            # 更精确的做法是计算四阶张量，但计算量巨大，通常使用线性近似
            if branch['adj'] is not None and aggregated_view['adj'] is not None:
                # 计算度矩阵或邻接矩阵行和的差异
                # 这是一个简化的 GW 成本近似
                deg_branch = torch.sum(branch['adj'], dim=1) # (n,)
                deg_agg = torch.sum(aggregated_view['adj'], dim=1) # (m,)
                C_edge = self._compute_cost_matrix(
                    branch['adj'] @ branch['features'] if branch['features'] is not None else deg_branch.unsqueeze(1),
                    aggregated_view['adj'] @ aggregated_view['features'] if aggregated_view['features'] is not None else deg_agg.unsqueeze(1)
                )
            else:
                C_edge = torch.zeros_like(C_node)
            
            # 融合成本矩阵 (公式 13 & 14)
            # sigma * C_node + (1-sigma) * C_edge
            # 注意：论文公式(14)中的 E 是张量，这里我们用近似矩阵代替
            C_fused = self.sigma * C_node + (1 - self.sigma) * C_edge
            
            # 求解图空间的传输计划 pi_G
            pi_G = self._sinkhorn(C_fused, mu, nu)
            
            # ==========================================
            # Step 2: 计算 Representation Space 的传输计划 pi_Z (公式 16)
            # ==========================================
            
            # 获取表示空间的特征 (由编码器输出)
            Z_branch = branch['features'] # 这里假设输入已经是编码后的 Z
            Z_agg = aggregated_view['features']
            
            # 计算表示空间的成本矩阵 R (公式 16)
            # 通常使用欧氏距离或余弦距离
            C_rep = self._compute_cost_matrix(Z_branch, Z_agg)
            
            # 求解表示空间的传输计划 pi_Z
            pi_Z = self._sinkhorn(C_rep, mu, nu)
            
            # ==========================================
            # Step 3: 计算对齐损失 (公式 17 & 18)
            # ==========================================
            
            # Loss 1: 匹配损失 (Matching Loss) - 对齐传输计划 (公式 17)
            # Frobenius 范数
            L_mat = torch.norm(pi_G - pi_Z, p='fro')
            
            # Loss 2: 结构损失 (Structural Loss) - 校正成本矩阵 (公式 18)
            # 强制表示空间的成本 R 接近图空间的融合成本
            L_str = torch.norm(C_fused - C_rep, p='fro')
            
            # 总损失 (公式 19)
            loss = L_mat + self.rho * L_str
            total_loss += loss
            
        return total_loss / num_views # 平均损失
    
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
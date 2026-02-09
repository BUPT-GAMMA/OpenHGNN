import os
import os.path as osp
import dgl
import torch
import torch as th
import numpy as np
from collections import Counter
from dgl.data.utils import load_graphs, save_graphs, download, extract_archive
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import gensim
    from gensim.models import Word2Vec
except ImportError:
    gensim = None

from . import BaseDataset, register_dataset

def setorderidx(data):
    data = data.copy()
    row, col = data.shape
    cnt = {}
    for i in range(col):
        cnt[i] = Counter(data[:, i])
        k = list(cnt[i].keys())
        k.sort()
        k2i = dict(zip(k, range(len(k))))
        for j in range(row):
            data[j][i] = k2i[data[j][i]]
    data = np.vectorize(int)(data)
    return data

def time_merge(glist, num_nodes_dict=None, link_pre=True):
    """
    合并时间窗口内的异构图特征
    """
    hetero_dict = {}
    for (t, g_s) in enumerate(glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.edges(etype=etype)
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)
            # hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (dst, src)
            
    if num_nodes_dict is None:
        G_feat = dgl.heterograph(hetero_dict)
    else:
        G_feat = dgl.heterograph(hetero_dict, num_nodes_dict)

    for (t, g_s) in enumerate(glist):
        for ntype in G_feat.ntypes:
            if ntype in g_s.ntypes:
                if 'x' in g_s.nodes[ntype].data:
                    feat = g_s.nodes[ntype].data['x']
                elif 'feat' in g_s.nodes[ntype].data:
                    feat = g_s.nodes[ntype].data['feat']
                else:
                    continue 
                
                feat = feat.type(torch.float32)
                
                target_types = ['user', 'item', 'author', 'venue'] # 包含 Aminer/Eco/Yelp 常见 ID 类型
                
                if link_pre and (ntype in target_types):
                     if feat.dim() == 1:
                         feat = feat.unsqueeze(1)
                
                G_feat.nodes[ntype].data[f't{t}'] = feat
                
            else:
                dim = 32
                pass
                
    return G_feat

def remove_edges_unseen_nodes(data, train_nodes):
    """
    剔除不在 train_nodes 集合中的节点所构成的边。
    """
    num_nodes = data.num_nodes()
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.device)
    
    if len(train_nodes) > 0:
        node_mask[list(train_nodes)] = True
        
    u, v = data.edges()
    
    valid_mask = node_mask[u] & node_mask[v]
    valid_eids = torch.nonzero(valid_mask, as_tuple=True)[0]
    
    sub_g = dgl.edge_subgraph(data, valid_eids, relabel_nodes=False)
    return sub_g

def get_author_graph(hetero_g):
    """
    从异构图提取 Co-author 图 (Homo Graph)
    逻辑：A-P * P-A = A-A
    """
    etype = None
    for et in hetero_g.canonical_etypes:
        if 'author' == et[0] and 'paper' == et[2]:
            etype = et
            break
        if 'paper' == et[0] and 'author' == et[2]:
            etype = et
            break
            
    if etype is None:
        return dgl.graph(([], []), num_nodes=hetero_g.num_nodes('author'))

    src, dst = hetero_g.edges(etype=etype)
    
    if etype[0] == 'author':
        row, col = src, dst
        num_row = hetero_g.num_nodes('author')
        num_col = hetero_g.num_nodes('paper')
    else:
        row, col = dst, src 
        num_row = hetero_g.num_nodes('author')
        num_col = hetero_g.num_nodes('paper')

    indices = torch.stack([row, col])
    values = torch.ones(len(row), device=row.device)
    adj = torch.sparse_coo_tensor(indices, values, (num_row, num_col))
    
    # A * A.T
    try:
        co_adj = torch.matmul(adj, adj.t())
        co_adj = co_adj.coalesce()
        indices = co_adj.indices()
        src_co, dst_co = indices[0], indices[1]
    except (RuntimeError, TypeError):
        adj_dense = adj.to_dense()
        co_adj_dense = torch.mm(adj_dense, adj_dense.t())
        indices = torch.nonzero(co_adj_dense, as_tuple=True)
        src_co, dst_co = indices[0], indices[1]

    mask = src_co != dst_co
    src_co = src_co[mask]
    dst_co = dst_co[mask]
    
    g = dgl.graph((src_co, dst_co), num_nodes=num_row)
    return g

def linksplit(data, device, num_nodes):
    u, v = data.edges()
    pos_edge_index = torch.stack([u, v], dim=0)
    
    pos_g = dgl.graph((u, v), num_nodes=num_nodes).to(device)
    
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index.cpu(),
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1),
        method='sparse'
    )
    
    neg_u = neg_edge_index[0].to(device)
    neg_v = neg_edge_index[1].to(device)
    
    neg_g = dgl.graph((neg_u, neg_v), num_nodes=num_nodes).to(device)
    return pos_g, neg_g

def time_select_edge_time(dataset_dict, t):
    """
    从 dataset_dict 中按时间切分 HeteroGraph
    """
    new_edges = {}
    num_nodes_dict = dataset_dict['num_nodes']
    
    for etype, time_tensor in dataset_dict['edge_time'].items():
        mask = (time_tensor == t)
        if mask.sum() == 0:
            src, dst = [], []
        else:
            edge_index = dataset_dict['edge_index'][etype] 
            src = edge_index[0][mask]
            dst = edge_index[1][mask]
        
        new_edges[etype] = (src, dst)
        
    g = dgl.heterograph(new_edges, num_nodes_dict=num_nodes_dict)
    
    for ntype, feat in dataset_dict['node_feat'].items():
        g.nodes[ntype].data['x'] = feat
        
    return g

def construct_htg_covid(glist, idx, time_window):
    sub_glist = glist[idx-time_window:idx]
    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)

    G_feat = dgl.heterograph(hetero_dict)
    
    for (t, g_s) in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            if 'feat' in g_s.nodes[ntype].data:
                G_feat.nodes[ntype].data[f't{t}'] = g_s.nodes[ntype].data['feat']
            elif 'x' in g_s.nodes[ntype].data:
                 G_feat.nodes[ntype].data[f't{t}'] = g_s.nodes[ntype].data['x']

    G_label = glist[idx]
    return G_feat, G_label

def construct_htg_mag(glist, idx, time_window):
    """
    1. 建立 ID 映射 (Global ID dict within window)
    2. 合并大图
    3. 特征填充 (不存在的节点填0)
    """
    sub_glist = glist[idx-time_window:idx]
    ID_dict = {}
    
    for ntype in glist[0].ntypes:
        ID_set = set()
        for g_s in sub_glist:
            if '_ID' in g_s.ndata:
                tmp_set = set(g_s.ndata['_ID'][ntype].tolist())
                ID_set.update(tmp_set)
        ID_dict[ntype] = {ID: idx for idx, ID in enumerate(sorted(list(ID_set)))}

    hetero_dict = {}
    for (t, g_s) in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.in_edges(g_s.nodes(dsttype), etype=etype)
            if '_ID' in g_s.ndata:
                ID_src = g_s.ndata['_ID'][srctype]
                ID_dst = g_s.ndata['_ID'][dsttype]
                new_src = ID_src[src]
                new_dst = ID_dst[dst]
                
                new_new_src = [ID_dict[srctype][e.item()] for e in new_src]
                new_new_dst = [ID_dict[dsttype][e.item()] for e in new_dst]
                
                hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (new_new_src, new_new_dst)
                hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (new_new_dst, new_new_src)
            else:
                hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)
                hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (dst, src)

    G_feat = dgl.heterograph(hetero_dict)

    for (t, g_s) in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            feat_dim = g_s.nodes[ntype].data['feat'].shape[1]
            G_feat.nodes[ntype].data[f't{t}'] = torch.zeros(G_feat.num_nodes(ntype), feat_dim)
            
            if '_ID' in g_s.ndata:
                node_id = g_s.ndata['_ID'][ntype]
                node_feat = g_s.ndata['feat'][ntype]
                for (id, feat) in zip(node_id, node_feat):
                    G_feat.nodes[ntype].data[f't{t}'][ID_dict[ntype][id.item()]] = feat
            else:
                 G_feat.nodes[ntype].data[f't{t}'] = g_s.nodes[ntype].data['feat']

    return G_feat

def generate_APA(graph, device):
    if ('author', 'writes', 'paper') in graph.canonical_etypes:
        etype = ('author', 'writes', 'paper')
    else:
        return None 
        
    AP = graph.adj(etype=etype).to_dense()
    PA = AP.t()
    APA = torch.mm(AP.to(device), PA.to(device)).detach().cpu()
    APA[torch.eye(APA.shape[0]).bool()] = 0.5
    return APA

def construct_htg_label_mag(glist, idx, device):
    APA_cur = generate_APA(glist[idx], device)
    APA_pre = generate_APA(glist[idx-1], device)
    
    if APA_cur is None or APA_pre is None:
        return None, None

    APA_pre = (APA_pre > 0.5).float()
    APA_cur = (APA_cur > 0.5).float()
    
    APA_sub = APA_cur - APA_pre # 新增的合著关系 (Label=1)
    APA_add = APA_cur + APA_pre
    APA_add[torch.eye(APA_add.shape[0]).bool()] = 0.5
    
    indices_true = (APA_sub == 1).nonzero(as_tuple=True)
    indices_false = (APA_add == 0).nonzero(as_tuple=True) # 从未合著过的 (Label=0)
    
    pos_src = indices_true[0]
    pos_dst = indices_true[1]
    
    # 采样比例 10%
    size = int(pos_src.shape[0] * 0.1)
    if size == 0: size = 1
    
    pos_idx = torch.randperm(pos_src.shape[0])[:size]
    pos_src = pos_src[pos_idx]
    pos_dst = pos_dst[pos_idx] 
    
    neg_src = indices_false[0]
    neg_dst = indices_false[1]

    neg_idx = torch.randperm(neg_src.shape[0])[:size]
    neg_src = neg_src[neg_idx]
    neg_dst = neg_dst[neg_idx]
    
    return dgl.graph((pos_src, pos_dst), num_nodes=APA_cur.shape[0]), dgl.graph((neg_src, neg_dst), num_nodes=APA_cur.shape[0])

def slice_graph_by_year(whole_graph, t):
    """
    从整图(whole_graph)中提取 edge_time == t 的子图。
    替代原来的 time_select_edge_time。
    """
    edge_dict = {}
    # 遍历所有边类型
    for etype in whole_graph.canonical_etypes:
        # 获取边的 key，例如 ('paper', 'published', 'venue') -> 'published' 
        # DGL存储数据通常用 edge type name (中间那个)
        # 但如果是多重边，最好用 canonical etype
        
        if 'time' in whole_graph.edges[etype].data:
            time_tensor = whole_graph.edges[etype].data['time']
            mask = (time_tensor == t)
            
            # 如果这一年这类边没有数据，建立空边
            if mask.sum() == 0:
                edge_dict[etype] = ([], [])
            else:
                # 获取源节点和目标节点
                src, dst = whole_graph.edges(etype=etype)
                edge_dict[etype] = (src[mask], dst[mask])
        else:
            # 如果某些边没有时间属性，根据需求决定是否保留。通常Aminer数据里都有。
            edge_dict[etype] = ([], [])

    # 保留节点数量信息
    num_nodes_dict = {ntype: whole_graph.num_nodes(ntype) for ntype in whole_graph.ntypes}
    
    # 构建当前时间步的异构图
    sub_g = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
    
    # 复制节点特征 (Node features 不随时间变化，直接拷贝)
    for ntype in whole_graph.ntypes:
        if 'feat' in whole_graph.nodes[ntype].data:
            sub_g.nodes[ntype].data['feat'] = whole_graph.nodes[ntype].data['feat']
            
    return sub_g

@register_dataset('sehtgnn_aminer')
class SEHTGNN_Aminer_Dataset(BaseDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_Aminer_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_aminer'

        config = kwargs.get('args')

        if config and hasattr(config, 'time_window'):
            self.time_window = config.time_window
        else:
            self.time_window = 5

        # self.time_window = kwargs.get('time_window', 8)
        self.device = kwargs.get('device', 'cpu')
        
        self._url = 'https://s3.your-region.amazonaws.com/your-bucket/aminer_base.bin'
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.raw_dir = osp.join(current_dir, 'data/Aminer')
        self.data_path = osp.join(self.raw_dir, 'aminer_base.bin')
        
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self._g = None
        
        self.download()
        self.load_data()

    def download(self):
        if osp.exists(self.data_path):
            return
        
        print(f"Downloading Aminer base data to {self.data_path}...")
        try:
            if not osp.exists(self.raw_dir):
                os.makedirs(self.raw_dir)
            download(self._url, path=self.data_path) 
        except Exception as e:
            print(f"Download failed: {e}. Please ensure 'aminer_base.bin' is in {self.raw_dir}")

    def load_data(self):
        if not osp.exists(self.data_path):
            # 本地测试如果没有文件，提示报错
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        print(f"Loading Atomic Graphs from {self.data_path}")
        atomic_graphs, _ = load_graphs(self.data_path) 
        
        print("Generating Author Co-occurrence graphs...")
        eval_datas = [get_author_graph(g) for g in atomic_graphs]
        
        # 索引计算
        test_idx = len(atomic_graphs) - 1
        val_idx = len(atomic_graphs) - 2
        train_idx = len(atomic_graphs) - 3
        
        train_nodes_list = [set()]
        
        # Transductive Setting 过滤
        for i in range(train_idx):
            src, dst = eval_datas[i].edges()
            active_nodes = set(torch.cat([src, dst]).unique().tolist())
            current_set = train_nodes_list[-1] | active_nodes
            train_nodes_list.append(current_set)
            
        for i in range(1, train_idx):
            eval_datas[i] = remove_edges_unseen_nodes(eval_datas[i], train_nodes_list[i])
            
        for i in range(train_idx, test_idx + 1):
            eval_datas[i] = remove_edges_unseen_nodes(eval_datas[i], train_nodes_list[train_idx])

        # Link Split
        num_nodes_author = atomic_graphs[0].num_nodes('author')
        eval_datas_split = [linksplit(g, self.device, num_nodes_author) for g in eval_datas]
        
        # 获取节点数量字典，用于 time_merge
        num_nodes_dict = {nt: atomic_graphs[0].num_nodes(nt) for nt in atomic_graphs[0].ntypes}

        print(f"Constructing Train/Val/Test sets with TimeWindow={self.time_window}...")
        
        # Train
        for k in range(self.time_window, train_idx + 1):
            window_graphs = atomic_graphs[k-self.time_window : k]
            
            feat_g = time_merge(window_graphs, num_nodes_dict, link_pre=True).to(self.device)
            label_g = eval_datas_split[k]
            self.train_set.append((feat_g, label_g))
            
        # Val
        if val_idx > self.time_window:
            window_graphs = atomic_graphs[val_idx-self.time_window : val_idx]
            feat_g = time_merge(window_graphs, num_nodes_dict, link_pre=True).to(self.device)
            label_g = eval_datas_split[val_idx]
            self.val_set.append((feat_g, label_g))
            
        # Test
        if test_idx > self.time_window:
            window_graphs = atomic_graphs[test_idx-self.time_window : test_idx]
            feat_g = time_merge(window_graphs, num_nodes_dict, link_pre=True).to(self.device)
            label_g = eval_datas_split[test_idx]
            self.test_set.append((feat_g, label_g))
            
        if len(self.train_set) > 0:
            self._g = self.train_set[0][0]
            
        print(f"Aminer Processed. Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

    def get_split(self):
        return self.train_set, self.val_set, self.test_set, None, None
        
    @property
    def category(self):
        return 'author'
    
    @property
    def num_classes(self):
        return 1

@register_dataset('sehtgnn_covid')
class SEHTGNN_COVID_Dataset(BaseDataset):
    _url = None 

    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_COVID_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_covid'
        self.time_window = kwargs.get('time_window', 7)
        self.device = kwargs.get('device', 'cpu')
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.raw_dir = osp.join(current_dir, 'data/Covid19')
        self.data_path = osp.join(self.raw_dir, 'covid_graphs.bin')
        self.llm_feat_path = osp.join(self.raw_dir, 'LLM_feature_Llama-3-new.pt')
        
        self.train_set = []
        self.val_set = []
        self.test_set = []
        
        # Debug
        if not osp.exists(self.data_path):
            print(f"[Local Test] File not found: {self.data_path}")
            print("Please generate 'aminer_base.bin' and place it here manually.")
            # self.download()
        else:
            print(f"[Local Test] Found local file: {self.data_path}, skipping download.")

        self.load_data()

    def set_args_and_load_feats(self, args):
        self.args = args
        print(f"[Dataset] Receiving args manually. Path check: {self.llm_feat_path}")
        
        if os.path.exists(self.llm_feat_path):
            print(f"[Dataset] Loading LLM features...")
            try:
                llm_feats = torch.load(self.llm_feat_path, weights_only=False)
            except TypeError:
                llm_feats = torch.load(self.llm_feat_path)
            
            clean_feats = {k: v.float() for k, v in llm_feats.items()}
            
            target_ntype = self.category # 'state'
            
            if target_ntype not in clean_feats:
                # 如果没找到 state，尝试用第一个键的内容顶替
                first_key = list(clean_feats.keys())[0]
                print(f"   > [Auto-Fix] Target '{target_ntype}' missing. Mapping from '{first_key}'")
                clean_feats[target_ntype] = clean_feats[first_key]
            else:
                print(f"   > [Success] Loaded feature for key: '{target_ntype}'")

            # self._g 在 load_data 之后就已经有了
            if hasattr(self, '_g') and self._g is not None:
                ref_feat = next(iter(clean_feats.values()))
                feat_dim = ref_feat.shape[1]
                
                for ntype in self._g.ntypes:
                    if ntype not in clean_feats:
                        print(f"   > [Warning] Node type '{ntype}' missing in LLM file.")
                        num_nodes = self._g.num_nodes(ntype)
                        
                        # 补全策略：生成全0特征 (或者随机特征)
                        print(f"     -> Auto-generating Zero features for '{ntype}' (Shape: {num_nodes}x{feat_dim})")
                        clean_feats[ntype] = torch.zeros(num_nodes, feat_dim)
            
            # 挂载到 args
            self.args.semantic_feature = clean_feats
        else:
            print(f"[Warning] LLM feature file not found.")
            self.args.semantic_feature = None

    def download(self):
        if os.path.exists(self.data_path):
            return
        
        if self._url is None:
            # 如果本地没有文件，打印提示
            if not os.path.exists(self.data_path):
                print(f"\n[Warning] Dataset file not found at {self.data_path}")
                print(f"Please place 'covid_graphs.bin' and 'LLM_feature_Llama-3-new.pt' in: {self.raw_dir}\n")
        else:
            path = download(self._url, path=self.raw_dir)
            extract_archive(path, self.raw_dir)

    def load_data(self):
        if not osp.exists(self.data_path):
            return # 如果文件不存在，直接返回
            
        print(f"Processing COVID data from {self.data_path}")
        glist, _ = load_graphs(self.data_path)
        
        # 预处理：确保所有特征都是 float (融合点)
        for g in glist:
            for ntype in g.ntypes:
                for key in g.nodes[ntype].data:
                    data = g.nodes[ntype].data[key]
                    if torch.is_floating_point(data):
                        g.nodes[ntype].data[key] = data.float()

        testlen = 30
        
        for i in range(len(glist)):
            if i >= self.time_window:
                G_feat, G_label = construct_htg_covid(glist, i, self.time_window)
                
                # 获取 Label Tensor
                target_ntype = 'state'
                key = 'feat' if 'feat' in G_label.nodes[target_ntype].data else 'x'
                label_tensor = G_label.nodes[target_ntype].data[key].float()
                
                # 确保维度匹配
                if label_tensor.dim() == 1:
                    label_tensor = label_tensor.unsqueeze(1)

                G_feat = G_feat.to(self.device)
                label_tensor = label_tensor.to(self.device)
                
                # 划分
                if i >= len(glist) - testlen:
                    self.test_set.append((G_feat, label_tensor))
                elif i >= len(glist) - testlen - 30:
                    self.val_set.append((G_feat, label_tensor))
                else:
                    self.train_set.append((G_feat, label_tensor))
                    
        if len(self.train_set) > 0:
            self._g = self.train_set[0][0]
            print(f"COVID Loaded. Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

    def get_split(self):
        return self.train_set, self.val_set, self.test_set
        
    @property
    def category(self):
        return 'state'


@register_dataset('sehtgnn_mag')
class SEHTGNN_MAG_Dataset(BaseDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_MAG_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_mag'
        self.time_window = kwargs.get('time_window', 3)
        self.device = kwargs.get('device', 'cpu')
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.data_path = osp.join(current_dir, 'data/ogbn/ogbn_graphs.bin')
        
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self._g = None
        
        self.load_data()

    def load_data(self):
        if not osp.exists(self.data_path):
            print(f"MAG data not found at {self.data_path}")
            return

        print(f"Processing MAG data from {self.data_path}")
        glist, _ = load_graphs(self.data_path)
        
        
        for i in range(len(glist)):
            if i >= self.time_window:
                G_feat = construct_htg_mag(glist, i, self.time_window)
                G_feat = G_feat.to(self.device)
                
                pos_label, neg_label = construct_htg_label_mag(glist, i, self.device)
                
                if pos_label is None: continue

                if i == len(glist) - 1:
                    self.test_set.append((G_feat, (pos_label, neg_label)))
                elif i == len(glist) - 2:
                    self.val_set.append((G_feat, (pos_label, neg_label)))
                else: 
                    self.train_set.append((G_feat, (pos_label, neg_label)))
                    
        if len(self.train_set) > 0:
            self._g = self.train_set[0][0]
            
        print(f"MAG Loaded. Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

    def get_split(self):
        return self.train_set, self.val_set, self.test_set, None, None
        
    @property
    def category(self):
        return 'author'

    @property
    def num_classes(self):
        return 1
import os
import os.path as osp
import dgl
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from tqdm import tqdm
from collections import Counter
from dgl.data.utils import load_graphs, download, extract_archive
from torch_geometric.utils import negative_sampling

try:
    import gensim
    from gensim.models import Word2Vec
except ImportError:
    gensim = None

from . import BaseDataset, register_dataset

def time_merge(glist, num_nodes_dict=None, link_pre=True):
    """
    合并时间窗口内的异构图特征，并添加反向边
    """
    hetero_dict = {}
    
    for (t, g_s) in enumerate(glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            if g_s.num_edges(etype) == 0:
                continue

            src, dst = g_s.edges(etype=etype)
            
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)
            
            hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (dst, src)
            
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
                
                target_types = ['user', 'item', 'author', 'venue'] 
                
                if link_pre and (ntype in target_types):
                     if feat.dim() == 1:
                         feat = feat.unsqueeze(1)
                
                G_feat.nodes[ntype].data[f't{t}'] = feat
                
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
    sub_glist = glist[idx-time_window:idx]
    global_ids_per_type = {}
    
    for ntype in glist[0].ntypes:
        ids_list = []
        for g_s in sub_glist:
            if '_ID' in g_s.nodes[ntype].data:
                ids_list.append(g_s.nodes[ntype].data['_ID'])
            else:
                ids_list.append(torch.arange(g_s.num_nodes(ntype)))
        
        if ids_list:
            all_ids = torch.cat(ids_list)
            unique_ids, _ = torch.sort(torch.unique(all_ids))
            global_ids_per_type[ntype] = unique_ids
        else:
            global_ids_per_type[ntype] = torch.tensor([])

    hetero_dict = {}
    for t, g_s in enumerate(sub_glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            if g_s.num_edges(etype) == 0: continue
            src, dst = g_s.edges(etype=etype)
            
            if '_ID' in g_s.nodes[srctype].data:
                global_src = g_s.nodes[srctype].data['_ID'][src]
                global_dst = g_s.nodes[dsttype].data['_ID'][dst]
            else:
                global_src, global_dst = src, dst
            
            super_src = torch.searchsorted(global_ids_per_type[srctype], global_src)
            super_dst = torch.searchsorted(global_ids_per_type[dsttype], global_dst)
            
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (super_src, super_dst)
            hetero_dict[(dsttype, f'{etype}_r_t{t}', srctype)] = (super_dst, super_src)

    num_nodes_dict = {nt: len(ids) for nt, ids in global_ids_per_type.items()}
    G_feat = dgl.heterograph(hetero_dict, num_nodes_dict=num_nodes_dict)

    for t, g_s in enumerate(sub_glist):
        for ntype in G_feat.ntypes:
            feat = None
            if 'feat' in g_s.nodes[ntype].data: feat = g_s.nodes[ntype].data['feat']
            elif 'x' in g_s.nodes[ntype].data: feat = g_s.nodes[ntype].data['x']
            if feat is None: continue
            
            feat_dim = feat.shape[1]
            target_feat = torch.zeros((G_feat.num_nodes(ntype), feat_dim), dtype=feat.dtype)
            
            if '_ID' in g_s.nodes[ntype].data:
                global_ids_gs = g_s.nodes[ntype].data['_ID']
            else:
                global_ids_gs = torch.arange(g_s.num_nodes(ntype))
            
            super_indices = torch.searchsorted(global_ids_per_type[ntype], global_ids_gs)
            target_feat[super_indices] = feat
            G_feat.nodes[ntype].data[f't{t}'] = target_feat

    return G_feat

def get_apa_sparse_matrix(g, year_idx):
    target_et = None
    is_reverse = False
    
    for et in g.canonical_etypes:
        if et[0] == 'author' and et[2] == 'paper': target_et = et; break
        if et[0] == 'paper' and et[2] == 'author': target_et = et; is_reverse = True; break
            
    if target_et is None: 
        print(f"[Debug] Year {year_idx}: No 'author-paper' edge found! Available types: {g.canonical_etypes}")
        return None

    src, dst = g.edges(etype=target_et)
    if len(src) == 0:
        return None
        
    src, dst = src.cpu().numpy(), dst.cpu().numpy()
    
    if '_ID' in g.nodes[target_et[0]].data:
        u_global = g.nodes[target_et[0]].data['_ID'][src].numpy()
        v_global = g.nodes[target_et[2]].data['_ID'][dst].numpy()
    else:
        u_global, v_global = src, dst

    if is_reverse: row, col = v_global, u_global
    else:          row, col = u_global, v_global
    
    data = np.ones(len(row), dtype=np.bool_)
    try:
        max_r = row.max() + 1
        max_c = col.max() + 1
        return sp.csr_matrix((data, (row, col)), shape=(max_r, max_c))
    except Exception as e:
        print(f"[Debug] Matrix creation error: {e}")
        return None

def construct_htg_label_mag(glist, idx, device, time_window):
    A_cur_raw = get_apa_sparse_matrix(glist[idx], idx)
    A_pre_raw = get_apa_sparse_matrix(glist[idx-1], idx-1)
    
    if A_cur_raw is None: return None, None
    
    max_r = A_cur_raw.shape[0]
    max_c = A_cur_raw.shape[1]
    if A_pre_raw is not None:
        max_r = max(max_r, A_pre_raw.shape[0])
        max_c = max(max_c, A_pre_raw.shape[1])
        
    def safe_resize(mat, shape):
        if mat.shape == shape: return mat
        mat_coo = mat.tocoo()
        return sp.csr_matrix((mat_coo.data, (mat_coo.row, mat_coo.col)), shape=shape)

    A_cur = safe_resize(A_cur_raw, (max_r, max_c))
    if A_pre_raw is not None:
        A_pre = safe_resize(A_pre_raw, (max_r, max_c))
    else:
        A_pre = None
    
    APA_cur = A_cur.dot(A_cur.T)
    if A_pre is not None:
        APA_pre = A_pre.dot(A_pre.T)
        APA_diff = (APA_cur > 0).astype(np.int8) - (APA_pre > 0).astype(np.int8)
        APA_diff = (APA_diff > 0)
    else:
        APA_diff = (APA_cur > 0)

    APA_diff = sp.triu(APA_diff, k=1)
    rows, cols = APA_diff.nonzero()
    
    if len(rows) == 0: return None, None

    ids_list = []
    sub_glist = glist[idx-time_window:idx]
    
    for g_s in sub_glist:
        if '_ID' in g_s.nodes['author'].data: ids_list.append(g_s.nodes['author'].data['_ID'])
        else: ids_list.append(torch.arange(g_s.num_nodes('author')))
            
    if not ids_list: return None, None
    
    all_ids = torch.cat(ids_list)
    unique_ids, _ = torch.sort(torch.unique(all_ids))
    unique_ids_np = unique_ids.numpy()
    
    mask_src = np.isin(rows, unique_ids_np)
    mask_dst = np.isin(cols, unique_ids_np)
    valid_mask = mask_src & mask_dst
    
    if not np.any(valid_mask): 
        return None, None

    rows = rows[valid_mask]
    cols = cols[valid_mask]

    src_local = torch.from_numpy(np.searchsorted(unique_ids_np, rows)).long()
    dst_local = torch.from_numpy(np.searchsorted(unique_ids_np, cols)).long()
    num_nodes = len(unique_ids_np)

    pos_g = dgl.graph((src_local, dst_local), num_nodes=num_nodes).to(device)
    
    size = max(1, len(src_local))
    neg_src = torch.randint(0, num_nodes, (size,))
    neg_dst = torch.randint(0, num_nodes, (size,))
    neg_g = dgl.graph((neg_src, neg_dst), num_nodes=num_nodes).to(device)
    
    return pos_g, neg_g

# =============================================================================
# Datasets
# =============================================================================

@register_dataset('sehtgnn_ogbn')
class SEHTGNN_OGBN_Dataset(BaseDataset):
    _memory_cache = {}
    _url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/ogbn4SEHTGNN.bin'

    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_OGBN_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_ogbn'
        config = kwargs.get('args')
        self.time_window = config.time_window if config and hasattr(config, 'time_window') else 5
        self.device = kwargs.get('device', 'cpu')
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.raw_dir = osp.join(current_dir, 'data/ogbn')
        self.data_path = osp.join(self.raw_dir, 'ogbn4SEHTGNN.bin')
        
        self.train_set, self.val_set, self.test_set = [], [], []
        self._g = None
        
        cache_key = f"mag_window_{self.time_window}_debug"
        if cache_key in SEHTGNN_OGBN_Dataset._memory_cache:
            print(f"[Dataset] Hit Cache for {cache_key}")
            c = SEHTGNN_OGBN_Dataset._memory_cache[cache_key]
            self.train_set, self.val_set, self.test_set, self._g = c['train'], c['val'], c['test'], c['g']
        else:
            self.download()
            self.load_data()
            if self.train_set: 
                SEHTGNN_OGBN_Dataset._memory_cache[cache_key] = {'train':self.train_set, 'val':self.val_set, 'test':self.test_set, 'g':self._g}
        
    def download(self):
        if not osp.exists(self.data_path):
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"Downloading OGBN data from {self._url}...")
            download(self._url, path=self.data_path)

    def load_data(self):
        if not osp.exists(self.data_path): return
        print(f"Processing MAG from {self.data_path}")
        glist, _ = load_graphs(self.data_path)
        
        has_id = '_ID' in glist[0].nodes['author'].data
        print(f"[Dataset] Has _ID: {has_id}")
        if not has_id:
            print("[Dataset] Auto-generating IDs...")
            for g in glist:
                for ntype in g.ntypes: g.nodes[ntype].data['_ID'] = torch.arange(g.num_nodes(ntype))

        for i in tqdm(range(len(glist)), desc="Processing"):
            if i >= self.time_window:
                G_feat = construct_htg_mag(glist, i, self.time_window).to(self.device)
                
                pos_label, neg_label = construct_htg_label_mag(glist, i, self.device, time_window=self.time_window)
                
                if pos_label is None: 
                    print(f" -> Step {i} Skipped (Label is None)")
                    continue
                
                item = (G_feat, (pos_label, neg_label))
                if i == len(glist)-1: self.test_set.append(item)
                elif i == len(glist)-2: self.val_set.append(item)
                else: self.train_set.append(item)
        
        if self.train_set: 
            self._g = self.train_set[0][0]
            print(f"OGBN-MAG Loaded: {len(self.train_set)}/{len(self.val_set)}/{len(self.test_set)}")
        else:
            print("[Error] OGBN-MAG dataset is empty! See logs above.")

    def get_split(self): return self.train_set, self.val_set, self.test_set, None, None
    @property
    def category(self): return 'author'
    @property
    def num_classes(self): return 1

@register_dataset('sehtgnn_aminer')
class SEHTGNN_Aminer_Dataset(BaseDataset):
    _memory_cache = {}
    _url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/aminer4SEHTGNN.bin'

    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_Aminer_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_aminer'
        config = kwargs.get('args')
        self.time_window = config.time_window if config and hasattr(config, 'time_window') else 5
        self.device = kwargs.get('device', 'cpu')
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.raw_dir = osp.join(current_dir, 'data/Aminer')
        self.data_path = osp.join(self.raw_dir, 'aminer4SEHTGNN.bin')
        
        self.train_set, self.val_set, self.test_set = [], [], []
        self._g = None
        
        cache_key = f"aminer_window_{self.time_window}"
        if cache_key in SEHTGNN_Aminer_Dataset._memory_cache:
            cached = SEHTGNN_Aminer_Dataset._memory_cache[cache_key]
            self.train_set, self.val_set, self.test_set, self._g = cached['train'], cached['val'], cached['test'], cached['g']
        else:
            self.download()
            self.load_data()
            if len(self.train_set) > 0:
                SEHTGNN_Aminer_Dataset._memory_cache[cache_key] = {'train': self.train_set, 'val': self.val_set, 'test': self.test_set, 'g': self._g}

    def download(self):
        if not osp.exists(self.data_path):
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"Downloading Aminer data from {self._url}...")
            download(self._url, path=self.data_path)

    def load_data(self):
        if not osp.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        print(f"Loading Atomic Graphs from {self.data_path}")
        atomic_graphs, _ = load_graphs(self.data_path) 
        
        print("Generating Author Co-occurrence graphs...")
        eval_datas = [get_author_graph(g) for g in atomic_graphs]
        
        test_idx = len(atomic_graphs) - 1
        val_idx = len(atomic_graphs) - 2
        train_idx = len(atomic_graphs) - 3
        
        train_nodes_list = [set()]
        
        for i in range(train_idx):
            src, dst = eval_datas[i].edges()
            active_nodes = set(torch.cat([src, dst]).unique().tolist())
            current_set = train_nodes_list[-1] | active_nodes
            train_nodes_list.append(current_set)
            
        for i in range(1, train_idx):
            eval_datas[i] = remove_edges_unseen_nodes(eval_datas[i], train_nodes_list[i])
            
        for i in range(train_idx, test_idx + 1):
            eval_datas[i] = remove_edges_unseen_nodes(eval_datas[i], train_nodes_list[train_idx])

        num_nodes_author = atomic_graphs[0].num_nodes('author')
        eval_datas_split = [linksplit(g, self.device, num_nodes_author) for g in eval_datas]
        
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


@register_dataset('sehtgnn_yelp')
class SEHTGNN_Yelp_Dataset(BaseDataset):
    _memory_cache = {}
    _url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/yelp4SEHTGNN.bin'

    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_Yelp_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_yelp'
        config = kwargs.get('args')
        self.time_window = config.time_window if config and hasattr(config, 'time_window') else 12
        self.device = kwargs.get('device', 'cpu')
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.raw_dir = osp.join(current_dir, 'data/yelp')
        self.data_path = osp.join(self.raw_dir, 'yelp4SEHTGNN.bin')
        
        self.train_set, self.val_set, self.test_set = [], [], []
        self._g = None
        
        cache_key = f"yelp_window_{self.time_window}"
        if cache_key in SEHTGNN_Yelp_Dataset._memory_cache:
             cached = SEHTGNN_Yelp_Dataset._memory_cache[cache_key]
             self.train_set, self.val_set, self.test_set, self._g = cached['train'], cached['val'], cached['test'], cached['g']
        else:
             self.download()
             self.load_data()
             if len(self.train_set) > 0:
                 SEHTGNN_Yelp_Dataset._memory_cache[cache_key] = {'train': self.train_set, 'val': self.val_set, 'test': self.test_set, 'g': self._g}

    def download(self):
        if not osp.exists(self.data_path):
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"Downloading Yelp data from {self._url}...")
            download(self._url, path=self.data_path)

    def load_data(self):
        if not osp.exists(self.data_path): raise FileNotFoundError(f"File not found: {self.data_path}")
        atomic_graphs, _ = load_graphs(self.data_path)
        self._num_classes = 3

        num_items = atomic_graphs[0].num_nodes('item')
        np.random.seed(0)
        idxs = np.random.permutation(num_items)
        val_num = int(0.1 * num_items)
        test_num = int(0.1 * num_items)
        train_num = num_items - val_num - test_num
        
        train_idx = torch.tensor(idxs[:train_num])
        val_idx = torch.tensor(idxs[train_num:train_num+val_num])
        test_idx = torch.tensor(idxs[train_num+val_num:])
        
        train_mask = torch.zeros(num_items).bool()
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_items).bool()
        val_mask[val_idx] = True
        test_mask = torch.zeros(num_items).bool()
        test_mask[test_idx] = True

        def build_label_graph(g, mask):
            y = g.nodes['item'].data['y'].long()
            lg = dgl.heterograph({('user', 'interact', 'item'): ([], [])}, 
                                 num_nodes_dict={nt: g.num_nodes(nt) for nt in g.ntypes})
            lg.nodes['item'].data['y'] = y
            lg.nodes['item'].data['mask'] = mask
            return lg.to(self.device)

        end_idx = min(len(atomic_graphs), self.time_window)
        window_graphs = atomic_graphs[0:end_idx]
        num_nodes_dict = {nt: atomic_graphs[0].num_nodes(nt) for nt in atomic_graphs[0].ntypes}
        
        feat_g = time_merge(window_graphs, num_nodes_dict, link_pre=False)
        
        for ntype in feat_g.ntypes:
            for key in feat_g.nodes[ntype].data:
                if 't' in key: # t0, t1, ...
                    feat_g.nodes[ntype].data[key] = F.normalize(feat_g.nodes[ntype].data[key], p=2, dim=1)

        feat_g = feat_g.to(self.device)
        
        train_lg = build_label_graph(atomic_graphs[0], train_mask)
        val_lg = build_label_graph(atomic_graphs[0], val_mask)
        test_lg = build_label_graph(atomic_graphs[0], test_mask)
        
        self.train_set = [(feat_g, train_lg)]
        self.val_set = [(feat_g, val_lg)]
        self.test_set = [(feat_g, test_lg)]
        self._g = feat_g
        print("Yelp Processed.")

    def get_split(self):
        return self.train_set, self.val_set, self.test_set
    def get_labels(self):
        return None
    def multi_label(self):
        return False
    @property
    def category(self):
        return 'item'
    @property
    def num_classes(self):
        return self._num_classes


@register_dataset('sehtgnn_covid')
class SEHTGNN_COVID_Dataset(BaseDataset):
    _memory_cache = {}
    _url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/covid4SEHTGNN.bin'
    _llm_url = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/Llama-3-4SEHTGNN.pt'

    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_COVID_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_covid'
        config = kwargs.get('args')
        self.time_window = config.time_window if config and hasattr(config, 'time_window') else 7
        self.device = kwargs.get('device', 'cpu')
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.raw_dir = osp.join(current_dir, 'data/Covid19')
        self.data_path = osp.join(self.raw_dir, 'covid4SEHTGNN.bin')
        self.llm_feat_path = osp.join(self.raw_dir, 'Llama-3-4SEHTGNN.pt')
        
        self.train_set, self.val_set, self.test_set = [], [], []
        
        if config:
            self.set_args_and_load_feats(config)
        
        self.download()
        self.load_data()

    def download(self):
        if not osp.exists(self.data_path):
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"Downloading COVID data from {self._url}...")
            download(self._url, path=self.data_path)
            print(f"Downloading Llama-3 checkpoint for COVID dataset from {self._url}...")
            download(self._llm_url, path=self.llm_feat_path)

    def set_args_and_load_feats(self, args):
        self.args = args
        if hasattr(self, '_llm_loaded') and self._llm_loaded: return

        if os.path.exists(self.llm_feat_path):
            print(f"[Dataset] Loading LLM features from {self.llm_feat_path}...")
            try:
                llm_feats = torch.load(self.llm_feat_path, weights_only=False)
            except TypeError:
                llm_feats = torch.load(self.llm_feat_path)
            
            clean_feats = {}
            for k, v in llm_feats.items():
                feat = v.float()
                if feat.dim() == 1: feat = feat.unsqueeze(0)
                feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                clean_feats[k] = feat
                
            target_ntype = self.category
            if target_ntype not in clean_feats and len(clean_feats) > 0:
                clean_feats[target_ntype] = clean_feats[list(clean_feats.keys())[0]]
            
            self.args.semantic_feature = clean_feats
            self._llm_loaded = True
        else:
            self.args.semantic_feature = None

    def load_data(self):
        if not osp.exists(self.data_path):
            print(f"Data not found: {self.data_path}")
            return
            
        print(f"Processing COVID data from {self.data_path}")
        glist, _ = load_graphs(self.data_path)
        
        print("[Dataset] Loading Data...")
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
                
                target_ntype = 'state'
                key = 'feat' if 'feat' in G_label.nodes[target_ntype].data else 'x'
                
                raw_label = G_label.nodes[target_ntype].data[key].float()
                label_tensor = raw_label
                
                if label_tensor.dim() == 1:
                    label_tensor = label_tensor.unsqueeze(1)

                if i >= len(glist) - testlen:
                    self.test_set.append((G_feat, label_tensor))
                elif i >= len(glist) - testlen - 30:
                    self.val_set.append((G_feat, label_tensor))
                else:
                    self.train_set.append((G_feat, label_tensor))
                    
        if len(self.train_set) > 0:
            self._g = self.train_set[0][0]
            print(f"COVID Loaded (Raw Scale). Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

    def get_split(self): return self.train_set, self.val_set, self.test_set
    def get_labels(self): return None
    def multi_label(self): return False
    @property
    def category(self): return 'state'
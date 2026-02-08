import os
import os.path as osp
import dgl
import torch
import torch as th
import numpy as np
from collections import Counter
from dgl.data.utils import load_graphs, save_graphs, download, extract_archive
from torch_geometric.utils import negative_sampling

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
    逻辑对齐：如果是 author/user/item 等 ID 特征节点，进行 unsqueeze 处理
    """
    hetero_dict = {}
    for (t, g_s) in enumerate(glist):
        for srctype, etype, dsttype in g_s.canonical_etypes:
            src, dst = g_s.edges(etype=etype)
            hetero_dict[(srctype, f'{etype}_t{t}', dsttype)] = (src, dst)
            
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
                    feat = None
                
                if feat is not None:
                    feat = feat.type(torch.float32)
                    if link_pre and feat.dim() == 1:
                         feat = feat.unsqueeze(1)
                    
                    G_feat.nodes[ntype].data[f't{t}'] = feat
    return G_feat

def remove_edges_unseen_nodes(data, train_nodes):
    """
    剔除不在 train_nodes 集合中的节点所构成的边。
    对应下版代码中的同名函数。
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

    # 移除自环
    mask = src_co != dst_co
    src_co = src_co[mask]
    dst_co = dst_co[mask]
    
    g = dgl.graph((src_co, dst_co), num_nodes=num_row)
    return g

# def linksplit(data, device, num_nodes):
#     u, v = data.edges()
    
#     pos_g = dgl.graph((u, v), num_nodes=num_nodes).to(device)
    
#     num_neg = data.num_edges()
#     # 简单的随机负采样
#     neg_u = torch.randint(0, num_nodes, (num_neg,), device=device)
#     neg_v = torch.randint(0, num_nodes, (num_neg,), device=device)
    
#     neg_g = dgl.graph((neg_u, neg_v), num_nodes=num_nodes).to(device)
#     return pos_g, neg_g

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

class AminerProcessor:
    def __init__(self, root_dir, word2vec_size=32):
        self.root_dir = root_dir
        self.word2vec_size = word2vec_size
        self.fnames = ["Database", "Data Mining", "Medical Informatics", "Theory", "Visualization"]
        
    def parse(self, datafile):
        field = os.path.split(datafile)[-1].replace(".txt", "")
        papers = []
        with open(datafile, "r", encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 5: continue
            (venue, title, authors, year, abstract) = parts[:5]
            try:
                year = int(year)
                papers.append((venue, title, authors, year, abstract, field))
            except:
                pass
        return np.array(papers)

    def sen2vec(self, sentences):
        if gensim is None:
            raise ImportError("Please install gensim")
            
        tokenized = [list(gensim.utils.tokenize(a, lower=True)) for a in sentences]
        model = Word2Vec(tokenized, vector_size=self.word2vec_size, min_count=1)
        embs = []
        for s in tokenized:
            if len(s) > 0:
                emb = model.wv[s].mean(axis=0)
            else:
                emb = np.zeros(self.word2vec_size)
            embs.append(emb)
        return np.stack(embs)

    def process(self):
        #  Read Files
        all_papers = []
        for name in self.fnames:
            path = os.path.join(self.root_dir, f"{name}.txt")
            if not os.path.exists(path):
                print(f"Warning: {path} not found.")
                continue
            all_papers.append(self.parse(path))
        
        if not all_papers:
            raise FileNotFoundError(f"No data files found in {self.root_dir}")
            
        papers = np.concatenate(all_papers)
        
        # Build ID Mappings & Metadata
        venues = list(set(papers[:, 0]))
        v2id = {v: i for i, v in enumerate(venues)}
        
        # 建立 Venue -> Field 映射 (用于特征对齐)
        v2field = {}
        for row in papers:
            v2field[row[0]] = row[5] # row[0] is venue, row[5] is field

        # Author
        all_authors = []
        for row in papers:
            all_authors.extend(row[2].split(","))
        
        # Counter -> sort by value(freq) -> keys
        cnt = Counter(all_authors)
        sorted_authors = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
        authors = [x[0] for x in sorted_authors]
        a2id = {a: i for i, a in enumerate(authors)}
        
        # Field
        fields = list(set(papers[:, 5]))
        fields.sort() # sort for consistency
        # f2id = {f: i for i, f in enumerate(fields)}
        
        num_papers = len(papers)
        
        # Build Edges & Times
        pv_src, pv_dst, pv_time = [], [], []
        pa_src, pa_dst, pa_time = [], [], []
        
        for i, row in enumerate(papers):
            pid = i
            vname = row[0]
            anames = row[2].split(",")
            year = int(row[3])
            
            if vname in v2id:
                pv_src.append(pid)
                pv_dst.append(v2id[vname])
                pv_time.append(year)
                
            for a in anames:
                if a in a2id:
                    pa_src.append(pid)
                    pa_dst.append(a2id[a])
                    pa_time.append(year)

        # Features
        
        # --- Paper Features: Abstract + Field (Word2Vec) ---
        print("Generating Paper features (Abstract + Field)...")
        abstracts = papers[:, 4]
        content = [a + " " + f for a, f in zip(abstracts, papers[:, 5])]
        feat_paper = torch.FloatTensor(self.sen2vec(content))
        
        # --- Venue Features: Field Embedding (Not Venue Name) ---
        # data["venue"].x = emb_field[venue_field]
        print("Generating Venue features (Field Embedding)...")
        # 训练 Field 的 Word2Vec
        field_embs = dict(zip(fields, self.sen2vec(fields)))
        
        venue_feats_list = []
        for i in range(len(venues)):
            v_name = venues[i]
            f_name = v2field.get(v_name, fields[0]) # fallback
            venue_feats_list.append(field_embs.get(f_name, np.zeros(self.word2vec_size)))
        feat_venue = torch.FloatTensor(np.stack(venue_feats_list))
        
        # --- Author Features: ID Embedding (Not Word2Vec) ---
        print("Generating Author features (ID)...")
        feat_author = torch.arange(len(authors), dtype=torch.float32)
        
        # Pack into Dict
        min_year = min(min(pv_time), min(pa_time))
        pv_time = torch.LongTensor(pv_time) - min_year
        pa_time = torch.LongTensor(pa_time) - min_year
        
        edge_index = {
            ('paper', 'published', 'venue'): (torch.LongTensor(pv_src), torch.LongTensor(pv_dst)),
            ('paper', 'written', 'author'): (torch.LongTensor(pa_src), torch.LongTensor(pa_dst)),
            ('venue', 'published_by', 'paper'): (torch.LongTensor(pv_dst), torch.LongTensor(pv_src)),
            ('author', 'writes', 'paper'): (torch.LongTensor(pa_dst), torch.LongTensor(pa_src))
        }
        
        edge_time = {
            ('paper', 'published', 'venue'): pv_time,
            ('paper', 'written', 'author'): pa_time,
            ('venue', 'published_by', 'paper'): pv_time,
            ('author', 'writes', 'paper'): pa_time
        }
        
        node_feat = {
            'paper': feat_paper,
            'venue': feat_venue,
            'author': feat_author
        }
        
        num_nodes = {
            'paper': num_papers,
            'venue': len(venues),
            'author': len(authors)
        }
        
        return {
            'edge_index': edge_index,
            'edge_time': edge_time,
            'node_feat': node_feat,
            'num_nodes': num_nodes,
            'years': sorted(list(set(pv_time.numpy())))
        }

@register_dataset('sehtgnn_aminer')
class SEHTGNN_Aminer_Dataset(BaseDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_Aminer_Dataset, self).__init__(*args, **kwargs)
        self.dataset_name = 'sehtgnn_aminer'
        self.time_window = kwargs.get('time_window', 5)
        self.device = kwargs.get('device', 'cpu')
        current_dir = osp.dirname(osp.abspath(__file__))
        self.data_path = osp.join(current_dir, 'data/Aminer')
        # self.data_path = kwargs.get('raw_dir', './openhgnn/dataset/data/Aminer')
        
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self._g = None
        
        self.load_data()

    def load_data(self):
        # Process Raw Data
        print(f"Processing Aminer data from {self.data_path}")
        processor = AminerProcessor(self.data_path, word2vec_size=32)
        dataset_dict = processor.process()
        
        # Split by Time
        years = dataset_dict['years']
        datas = [time_select_edge_time(dataset_dict, t) for t in years]
        
        # Build Label Graphs (Co-author)
        eval_datas = [get_author_graph(g) for g in datas]
        
        # Transductive Setting Filtering
        # Validation 和 Test 只能包含在 Training 阶段见过的节点
        
        test_idx = len(years) - 1
        val_idx = len(years) - 2
        train_idx = len(years) - 3
        
        # 累计训练节点
        train_nodes_list = [set()]
        
        # 填充 train_nodes_list
        for i in range(train_idx):
            # 获取当前时刻图中的活跃节点
            src, dst = eval_datas[i].edges()
            active_nodes = set(torch.cat([src, dst]).unique().tolist())
            
            current_set = train_nodes_list[-1] | active_nodes
            train_nodes_list.append(current_set)
            
        # Remove edges unseen nodes
        # train_nodes_list[i] 存储的是 0 到 i-1 时刻的累计节点
        for i in range(1, train_idx):
            eval_datas[i] = remove_edges_unseen_nodes(eval_datas[i], train_nodes_list[i])
            
        # 过滤 Val 和 Test 数据
        for i in range(train_idx, test_idx + 1):
            eval_datas[i] = remove_edges_unseen_nodes(eval_datas[i], train_nodes_list[-1])

        # Generate Positive/Negative Labels (Link Split)
        eval_datas_split = [linksplit(g, self.device, dataset_dict['num_nodes']['author']) for g in eval_datas]
        
        num_nodes_dict = dataset_dict['num_nodes']
        
        # Train
        for k in range(self.time_window, train_idx + 1):
            # Input: [k-window, k)
            feat_g = time_merge(datas[k-self.time_window : k], num_nodes_dict, link_pre=True).to(self.device)
            # Label: k
            label_g = eval_datas_split[k]
            self.train_set.append((feat_g, label_g))
            
        # Val
        if val_idx > self.time_window:
            feat_g = time_merge(datas[val_idx-self.time_window : val_idx], num_nodes_dict, link_pre=True).to(self.device)
            label_g = eval_datas_split[val_idx]
            self.val_set.append((feat_g, label_g))
            
        # Test
        if test_idx > self.time_window:
            feat_g = time_merge(datas[test_idx-self.time_window : test_idx], num_nodes_dict, link_pre=True).to(self.device)
            label_g = eval_datas_split[test_idx]
            self.test_set.append((feat_g, label_g))
            
        if len(self.train_set) > 0:
            self._g = self.train_set[0][0]
            
        print(f"Aminer Loaded (Aligned). Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}")

    def get_split(self):
        return self.train_set, self.val_set, self.test_set
        
    @property
    def category(self):
        return 'author'
    
    @property
    def num_classes(self):
        return 1


@register_dataset('covid_regression')
class COVIDDataset(BaseDataset):
    _url = None 
    
    def __init__(self, dataset_name, *args, **kwargs):
        super(COVIDDataset, self).__init__(*args, **kwargs)
        
        self.args = kwargs.get('args', None)
        self.dataset_name = 'sehtgnn_covid'
        self.category = 'state'
        
        self.time_window = kwargs.get('time_window', 7)
        self.test_len = 30
        
        current_dir = osp.dirname(osp.abspath(__file__))
        self.raw_dir = osp.join(current_dir, 'data', 'Covid19')
        self.save_path = osp.join(self.raw_dir, 'covid_graphs.bin')
        self.llm_feat_path = osp.join(self.raw_dir, 'LLM_feature_Llama-3-new.pt')
        
        self.train_set = []
        self.val_set = []
        self.test_set = []
        
        self.download()
        
        # 加载 LLM 特征
        if os.path.exists(self.llm_feat_path):
            print(f"[Dataset] Loading LLM features from {self.llm_feat_path}")
            llm_feats = torch.load(self.llm_feat_path)
            if self.args:
                self.args.semantic_feature = {k: v.float() for k, v in llm_feats.items()}
        else:
            if self.args:
                self.args.semantic_feature = None

        self.process()
        print(f"[Dataset Info] Train samples: {len(self.train_set)} | Val: {len(self.val_set)} | Test: {len(self.test_set)}")

    def download(self):
        if os.path.exists(self.save_path):
            return
        if self._url is None:
            # 如果本地没有且没有 URL，提示用户放置文件
            if not os.path.exists(self.save_path):
                print(f"Dataset file not found at {self.save_path}.")
                print("Please place 'covid_graphs.bin' and 'LLM_feature_Llama-3-new.pt' in:", self.raw_dir)
        else:
            path = download(self._url, path=self.raw_dir)
            extract_archive(path, self.raw_dir)

    def process(self):
        if not os.path.exists(self.save_path):
            return

        glist, _ = load_graphs(self.save_path)
        
        # 预处理：转 float
        for g in glist:
            for ntype in g.ntypes:
                for key in g.nodes[ntype].data:
                    data = g.nodes[ntype].data[key]
                    if torch.is_floating_point(data):
                        g.nodes[ntype].data[key] = data.float()
        
        total_days = len(glist)
        num_samples = total_days - self.time_window
        
        all_data = []
        
        sample_g = glist[0]
        num_nodes_dict = {ntype: sample_g.num_nodes(ntype) for ntype in sample_g.ntypes}

        for i in range(num_samples):
            sub_glist = glist[i : i + self.time_window]
            target_g = glist[i + self.time_window]
            
            # 获取 Label
            key = 'feat' if 'feat' in target_g.nodes[self.category].data else 'x'
            label = target_g.nodes[self.category].data[key].float()
            
            if label.dim() == 1:
                label = label.unsqueeze(1)
            
            merged_g = time_merge(sub_glist, num_nodes_dict, link_pre=False)
            
            all_data.append((merged_g, label))
            
        self.test_set = all_data[-self.test_len:]
        self.val_set = all_data[-2*self.test_len : -self.test_len]
        self.train_set = all_data[: -2*self.test_len]
        
        if len(self.train_set) > 0:
            self._g = self.train_set[0][0]

    def get_labels(self):
        return None 

    def get_split(self):
        return self.train_set, self.val_set, self.test_set


@register_dataset('sehtgnn_mag')
class SEHTGNN_MAG_Dataset(BaseDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(SEHTGNN_MAG_Dataset, self).__init__(*args, **kwargs)
        pass
    def get_split(self): return [], [], []
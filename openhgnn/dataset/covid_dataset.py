import os
import dgl
import torch
from dgl.data.utils import load_graphs, download, extract_archive
from . import BaseDataset, register_dataset

@register_dataset('covid_regression')
class COVIDDataset(BaseDataset):
    _url = None 
    
    def __init__(self, args, **kwargs):
        super(COVIDDataset, self).__init__(**kwargs)
        
        self.args = args
        self.dataset_name = 'covid_regression'
        self.category = 'state'
        
        self.time_window = getattr(args, 'time_window', 7)
        self.test_len = 30
        
        self.raw_dir = os.path.join(os.path.dirname(__file__), 'data', 'Covid19')
        self.save_path = os.path.join(self.raw_dir, 'covid_graphs.bin')
        self.llm_feat_path = os.path.join(self.raw_dir, 'LLM_feature_Llama-3-new.pt')
        
        self.download()
        
        if os.path.exists(self.llm_feat_path):
            print(f"[Dataset] Loading LLM features from {self.llm_feat_path}")
            llm_feats = torch.load(self.llm_feat_path)
            self.args.semantic_feature = {k: v.float() for k, v in llm_feats.items()}
        else:
            self.args.semantic_feature = None

        self.process()
        print(f"[Dataset Info] Train samples: {len(self.train_set)} | Val: {len(self.val_set)} | Test: {len(self.test_set)}")

    def download(self):
        if os.path.exists(self.save_path):
            return
        if self._url is None:
            print(f"Dataset file not found at {self.save_path}.")
            print("Please place 'covid_graphs.bin' and 'LLM_feature_Llama-3-new.pt' in:", self.raw_dir)
        else:
            path = download(self._url, path=self.raw_dir)
            extract_archive(path, self.raw_dir)

    def process(self):
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"File not found: {self.save_path}")

        glist, _ = load_graphs(self.save_path)
        
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
            
            key = 'feat' if 'feat' in target_g.nodes[self.category].data else 'x'
            label = target_g.nodes[self.category].data[key].float()
            
            if label.dim() == 1:
                label = label.unsqueeze(1)
            
            merged_g = self._time_merge(sub_glist, num_nodes_dict)
            
            all_data.append((merged_g, label))
            
        self.test_set = all_data[-self.test_len:]
        self.val_set = all_data[-2*self.test_len : -self.test_len]
        self.train_set = all_data[: -2*self.test_len]
        self.g = self.train_set[0][0]

    def _time_merge(self, sub_glist, num_nodes_dict):
        hetero_dict = {}
        
        for t, g_s in enumerate(sub_glist):
            for srctype, etype, dsttype in g_s.canonical_etypes:
                src, dst = g_s.edges(etype=etype)
                
                new_etype = f"{etype}_t{t}"
                hetero_dict[(srctype, new_etype, dsttype)] = (src, dst)
        
        G_feat = dgl.heterograph(hetero_dict, num_nodes_dict=num_nodes_dict)
        
        for t, g_s in enumerate(sub_glist):
            for ntype in G_feat.ntypes:
                feat = None
                ndata = g_s.nodes[ntype].data
                if 'feat' in ndata:
                    feat = ndata['feat']
                elif 'x' in ndata:
                    feat = ndata['x']
                
                if feat is not None:
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(1)
                    G_feat.nodes[ntype].data[f't{t}'] = feat
        
        return G_feat

    def get_labels(self):
        return None 

    def get_split(self):
        return self.train_set, self.val_set, self.test_set
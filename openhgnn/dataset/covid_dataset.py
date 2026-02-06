import os
import dgl
import torch
from dgl.data.utils import load_graphs, download, extract_archive
from . import BaseDataset, register_dataset

@register_dataset('covid_regression')
class COVIDDataset(BaseDataset):
    r"""
    The class *COVIDDataset* is for temporal node regression tasks (e.g., COVID-19 prediction).
    Unlike standard static datasets, this dataset constructs samples using a sliding window mechanism.

    Attributes
    -------------
    time_window : int
        The size of the sliding window (history length).
    category : str
        The target node type to predict (default: 'state').
    train_set : list
        List of (Graph, Label) tuples for training.
    val_set : list
        List of (Graph, Label) tuples for validation.
    test_set : list
        List of (Graph, Label) tuples for testing.
    """
    
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
            self.args.semantic_feature = torch.load(self.llm_feat_path)
        else:
            self.args.semantic_feature = None

        self.process()
        
        print(f"[Dataset Info] Train samples: {len(self.train_set)} | Val: {len(self.val_set)} | Test: {len(self.test_set)}")

    def download(self):
        r"""
        Download dataset from url if not exists.
        """
        if os.path.exists(self.save_path):
            return

        if self._url is None:
            print(f"Dataset file not found at {self.save_path}.")
            print("Please place 'covid_graphs.bin' and 'LLM_feature_Llama-3-new.pt' in:", self.raw_dir)
        else:
            path = download(self._url, path=self.raw_dir)
            extract_archive(path, self.raw_dir)

    def process(self):
        r"""
        Load graph data without ANY normalization.
        """
        if not os.path.exists(self.save_path):
            raise FileNotFoundError(f"File not found: {self.save_path}")

        glist, _ = load_graphs(self.save_path)
        
        total_days = len(glist)
        num_samples = total_days - self.time_window
        
        all_data = []
        for i in range(num_samples):
            sub_glist = glist[i : i + self.time_window]
            
            target_g = glist[i + self.time_window]
            
            key = 'feat' if 'feat' in target_g.nodes[self.category].data else 'x'
            label = target_g.nodes[self.category].data[key]
            
            merged_g = self._time_merge(sub_glist)
            
            all_data.append((merged_g, label))
            
        self.test_set = all_data[-self.test_len:]
        self.val_set = all_data[-2*self.test_len : -self.test_len]
        self.train_set = all_data[: -2*self.test_len]
        
        self.g = self.train_set[0][0]

    def _time_merge(self, sub_glist):
        r"""
        Internal function to merge a list of graphs into one graph with 't0', 't1'... features.
        """
        base_g = sub_glist[-1].clone()
        
        for ntype in base_g.ntypes:
            if 'feat' in base_g.nodes[ntype].data:
                del base_g.nodes[ntype].data['feat']
        
        for t, g_s in enumerate(sub_glist):
            for ntype in base_g.ntypes:
                feat = None
                if 'feat' in g_s.nodes[ntype].data:
                    feat = g_s.nodes[ntype].data['feat']
                elif 'x' in g_s.nodes[ntype].data:
                    feat = g_s.nodes[ntype].data['x']
                
                if feat is not None:
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(1)
                    base_g.nodes[ntype].data[f't{t}'] = feat
                else:
                    pass 
                    
        return base_g

    
    def get_labels(self):
        return None 

    def get_split(self):
        return self.train_set, self.val_set, self.test_set
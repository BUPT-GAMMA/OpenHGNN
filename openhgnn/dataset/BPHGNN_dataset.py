import torch.nn as nn  
import torch  
import dgl
from torch.utils.data import Dataset, DataLoader



class TextDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath, 'r') as f:
            for line in f:
                
                split_line = list(map(int, line.strip().split()))
                self.data.append(split_line)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      
        data = self.data[idx]
        features = torch.tensor(data[:-1], dtype=torch.float32)
        labels = torch.tensor(data[-1], dtype=torch.long)
        return features, labels
    
def load_edges(file_path):
    edge_data = {
        1: [],  
        2: [],   
        3: []   
    }
    
    with open(file_path, 'r') as f:
        for line in f:
           
            split_line = list(map(int, line.strip().split()))
            edge_type, src_id, dst_id, _ = split_line
            edge_data[edge_type].append((src_id, dst_id))
            
    return edge_data

def build_hetero_graph():
   
    edge_data = load_edges('OpenHGNN/openhgnn/dataset/data/test/BPHGNN_dataset/alibaba_small/train.txt')
    
    data_dict = {
        ('node', 'relation1', 'node'): edge_data[1],
        ('node', 'relation2', 'node'): edge_data[2],
        ('node', 'relation3', 'node'): edge_data[3],
    }
    
    hg = dgl.heterograph(data_dict)
    return hg


train_dataset = TextDataset('OpenHGNN/openhgnn/dataset/data/BPHGNN_dataset/train.txt')
valid_dataset = TextDataset('OpenHGNN/openhgnn/dataset/data/BPHGNN_dataset/valid.txt')
test_dataset = TextDataset('OpenHGNN/openhgnn/dataset/data/BPHGNN_dataset/test.txt')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
import torch.nn as nn  
import torch  
import dgl

from . import register_flow  
from ..tasks import build_task  
from ..utils import extract_embed, get_nodes_dict  
from ..models import build_model  
from torch.utils.data import Dataset, DataLoader
from ..models.MHGCN import MHGCN 
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from abc import ABC
  
class TextDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath, 'r') as f:
            for line in f:
                # 假设每行的数据由空格分隔
                split_line = list(map(int, line.strip().split()))
                self.data.append(split_line)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据并转换为张量
        data = self.data[idx]
        features = torch.tensor(data[:-1], dtype=torch.float32)
        labels = torch.tensor(data[-1], dtype=torch.long)
        return features, labels
    
def load_edges(file_path):
    edge_data = {
        1: [],  # 用于存储类型1的边
        2: [],  # 用于存储类型2的边
        3: []   # 用于存储类型3的边
    }
    
    with open(file_path, 'r') as f:
        for line in f:
            # 假设每行的数据由空格分隔
            split_line = list(map(int, line.strip().split()))
            edge_type, src_id, dst_id, _ = split_line
            edge_data[edge_type].append((src_id, dst_id))
            
    return edge_data

def build_hetero_graph():
    # 假设边的文件路径为 'train.txt'
    edge_data = load_edges('OpenHGNN/openhgnn/dataset/data/test/MHGCN_dataset/alibaba_small/train.txt')
    
    # 创建异质图
    data_dict = {
        ('node', 'relation1', 'node'): edge_data[1],
        ('node', 'relation2', 'node'): edge_data[2],
        ('node', 'relation3', 'node'): edge_data[3],
    }
    
    hg = dgl.heterograph(data_dict)
    return hg



@register_flow("MHGCN_trainer")
class MHGCN_trainer(ABC):

    def __init__(self, args):
        super(MHGCN_trainer, self).__init__()

        self.args = args
        self.model_name = args.model
        self.device = args.device
        self.task = args.task
        self.hg=build_hetero_graph()
        self.model = MHGCN(
            in_feats=64,
            out_feats=args.out,
            hidden_size=args.hidden,
            num_edge_types=3,                
            dropout=args.dropout
        )
        self.optimizer = (
            torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay))
        
        

    def preprocess(self):
        pass

    def train(self):
        self.preprocess()

        train_dataset = TextDataset('OpenHGNN/openhgnn/dataset/data/MHGCN_dataset/train.txt')
        valid_dataset = TextDataset('OpenHGNN/openhgnn/dataset/data/MHGCN_dataset/valid.txt')
        test_dataset = TextDataset('OpenHGNN/openhgnn/dataset/data/MHGCN_dataset/test.txt')

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        best_valid_loss = float('inf')
        best_model = None

        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            for features, labels in train_loader:
                # features 和 labels 从数据加载器中提取
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
               
                output = self.model(features=features, hg=self.hg, encode=features)
                
                loss = F.cross_entropy(output, labels)  # 计算损失
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            avg_valid_loss = self.evaluate(valid_loader)
            print(f"Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                best_model = self.model.state_dict()   
        
        if best_model:
            self.model.load_state_dict(best_model)
        
        self.evaluate(test_loader, is_test=True)

    def _full_train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        h_dict = self.model.input_feature()
        loss = self.model(self.hg, h_dict, self.pos)
        loss.backward()
        self.optimizer.step()
        loss = loss.cpu()
        loss = loss.detach().numpy()
        return loss

    def evaluate(self, loader, is_test=False):
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for features, labels in loader:
                features = torch.tensor(features).to(self.device)
                labels = torch.tensor(labels).to(self.device)

                outputs, _, _ = self.model(features, self.hg, self.pos)

                loss = F.cross_entropy(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        avg_loss = total_loss / len(loader)

        if is_test:
            all_labels = torch.tensor(all_labels)
            all_preds = torch.tensor(all_preds)

            macro_f1 = f1_score(all_labels, all_preds, average='macro')
            micro_f1 = f1_score(all_labels, all_preds, average='micro')
            auc_score = roc_auc_score(all_labels, F.softmax(outputs, dim=1), multi_class='ovr')

            print(f"Test Loss: {avg_loss:.4f}")
            print(f"Macro-F1: {macro_f1:.4f}, Micro-F1: {micro_f1:.4f}, AUC: {auc_score:.4f}")
        else:
            return avg_loss

import torch
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping
import copy
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
from torch_geometric.nn import MLP
from torch_geometric.nn.pool import global_add_pool
import torch.nn.functional as F
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from datetime import datetime
import os

blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'

@register_flow("PolyGNN")
class PolyGNN(BaseFlow):
    r"""
    Graph classification flow for the model of PolyGNN
    The task is to classify the graphs.
    """

    def __init__(self, args):
        """

        Attributes
        ------------
        num_classes: int
            The number of classes for category graph type
        mlpmodel: nn.Module
            The linear model for the final classification

        """

        super(PolyGNN, self).__init__(args)
        self.num_classes = self.task.num_classes
        self.criterion=self.task.get_loss_fn()
        self.model = build_model(self.model).build_model_from_args(self.args,self.hg).to(self.device)
        self.mlpmodel=MLP(in_channels=args.h_dim*args.num_interactions, hidden_channels=args.h_dim,out_channels=self.num_classes, num_layers=args.classifier_depth,dropout=args.dropout)
        opt_list=list(self.model.parameters())+list(self.mlpmodel.parameters())
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu==0 else 'cpu')
        self.model.to(self.device)
        self.mlpmodel.to(self.device)
        self.optimizer = torch.optim.Adam( opt_list, lr=args.lr)    # 优化器和调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)   
        self.train_loader, self.val_loader, self.test_loader = self.task.get_split()
        self.lr = args.lr
        self.args.save_dir=os.path.join('./save/',args.dataset,args.model)

    def train(self):
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        best_val_trigger = -1
        old_lr = self.lr
        lr = self.lr
        for epoch in epoch_iter:
            train_loss,train_closs,y_hat, y_true,y_hat_logit=self._full_train_step()

            train_acc = calculate(y_hat,y_true,y_hat_logit)
            # try:util.record({"loss":train_loss,"closs":train_closs,"acc":train_acc},epoch,writer,"Train") 
            # except: pass
            print_1(epoch,'Train',{"loss":train_loss,"closs":train_closs,"acc":train_acc})
            if epoch % self.args.test_per_round == 0:
                val_loss, yhat_val, ytrue_val, yhatlogit_val = self._full_test_step(self.val_loader,self.model,self.mlpmodel)
                test_loss, yhat_test, ytrue_test, yhatlogit_test = self._full_test_step(self.test_loader,self.model,self.mlpmodel)
                val_acc=calculate(yhat_val,ytrue_val,yhatlogit_val)
                # try:util.record({"loss":val_loss,"acc":val_acc},epoch,writer,"Val")
                # except: pass
                print_1(epoch,'Val',{"loss":val_loss,"acc":val_acc},color=blue) 
                test_acc=calculate(yhat_test,ytrue_test,yhatlogit_test)
                # try:util.record({"loss":test_loss,"acc":test_acc},epoch,writer,"Test")            
                # except: pass
                print_1(epoch,'Test',{"loss":test_loss,"acc":test_acc},color=blue)
                val_trigger=val_acc
                if val_trigger > best_val_trigger:
                    best_val_trigger = val_trigger
                    best_model = copy.deepcopy(self.model)
                    best_mlpmodel=copy.deepcopy(self.mlpmodel)
                    best_info=[epoch,val_trigger]
            """ 
            update lr when epoch≥30
            """
            if epoch >= 30: # lr更新
                lr = self.scheduler.optimizer.param_groups[0]['lr']
                if old_lr!=lr:
                    print(red('lr'), epoch, (lr), sep=', ')
                    old_lr=lr
                self.scheduler.step(val_trigger)        

        """
        use best model to get best model result  测试
        """
        val_loss, yhat_val, ytrue_val, yhat_logit_val  = self._full_test_step(self.val_loader,best_model,best_mlpmodel)
        test_loss, yhat_test, ytrue_test, yhat_logit_test= self._full_test_step(self.test_loader,best_model,best_mlpmodel)

        val_acc=calculate(yhat_val,ytrue_val,yhat_logit_val)
        print_1(best_info[0],'BestVal',{"loss":val_loss,"acc":val_acc},color=blue)
        test_acc=calculate(yhat_test,ytrue_test,yhat_logit_test)
        print_1(best_info[0],'BestTest',{"loss":test_loss,"acc":test_acc},color=blue)
                                                                
        """
        save training info and best result  存储
        """

        """
        build dir 
        """
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir,exist_ok=True)
        tensorboard_dir=os.path.join(self.args.save_dir,'log')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir,exist_ok=True)
        model_dir=os.path.join(self.args.save_dir,'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir,exist_ok=True)    
        info_dir=os.path.join(self.args.save_dir,'info')
        if not os.path.exists(info_dir):
            os.makedirs(info_dir,exist_ok=True) 
        timestamp = datetime.now().strftime("%b%d-%H-%M-%S")  # 格式示例: Apr14-21-38-13
        suffix = f"{timestamp}"  # 直接使用单次strftime生成
        result_file=os.path.join(info_dir, suffix)
        with open(result_file, 'w') as f:
            print("Random Seed: ", self.args.man_seed,file=f)
            print(f"acc  val : {val_acc:.3f}, Test : {test_acc:.3f}", file=f)
            print(f"Best info: {best_info}", file=f)
            for i in [[a,getattr(self.args, a)] for a in self.args.__dict__]:
                print(i,sep='\n',file=f)
        to_save_dict={'model':best_model.state_dict(),'mlpmodel':best_mlpmodel.state_dict(),'args':self.args,'labels':ytrue_test,'yhat':yhat_test,'yhat_logit':yhat_logit_test}
        torch.save(to_save_dict, os.path.join(model_dir,suffix+'.pth') )
        print("done")

    def forward(self,data):
        data = data.to(self.device)
        edge_index1=data['vertices', 'inside', 'vertices']['edge_index']
        edge_index2=data['vertices', 'apart', 'vertices']['edge_index']
        combined_edge_index=torch.cat([data['vertices', 'inside', 'vertices']['edge_index'],data['vertices', 'apart', 'vertices']['edge_index']],1)
        num_edge_inside=edge_index1.shape[1]
        edge_weight=torch.rand(combined_edge_index.shape[1]) + 1
        undirected_spanning_edge = build_spanning_tree_edge(combined_edge_index, edge_weight,num_nodes=data.pos.shape[0])
        
        edge_set_1 = set(map(tuple, edge_index2.t().tolist()))
        edge_set_2 = set(map(tuple, undirected_spanning_edge.t().tolist()))

        common_edges = edge_set_1.intersection(edge_set_2)
        common_edges_tensor = torch.tensor(list(common_edges), dtype=torch.long).t().to(self.device)
        spanning_edge=torch.cat([edge_index1,common_edges_tensor],1)
        combined_edge_index=spanning_edge


        x,batch=data.pos, data['vertices'].batch
        label=data.y.long().view(-1)

        num_nodes=x.shape[0]
        edge_index_2rd, num_triplets_real, edx_jk, edx_ij = triplets(combined_edge_index, num_nodes)
        self.optimizer.zero_grad()
        input_feature=torch.zeros([x.shape[0],self.args.h_dim],device=self.device) 
        output=self.model(input_feature,x,[edge_index1,edge_index2], edge_index_2rd,edx_jk, edx_ij,batch,num_edge_inside,self.args.edge_rep)  
        output=torch.cat(output,dim=1)
        if self.args.dataset in ["dbp"]:
            graph_embeddings=global_add_pool(output,batch)
        else:
            graph_embeddings=global_add_pool(output,batch)
        graph_embeddings.clamp_(max=1e6)
        c_loss=contrastive_loss(graph_embeddings,label,margin=1)
        output=self.mlpmodel(graph_embeddings) # 先把SModel得到的embedding结合为graphEmbedding，然后再利用MLP归到类别上

        loss = self.criterion(output, label) 
        loss+=c_loss*self.args.loss_coef
        return loss,c_loss*self.args.loss_coef,output,label
    def _full_train_step(self):
        epochloss=0
        epochcloss=0
        y_hat, y_true,y_hat_logit = [], [], []        
        self.optimizer.zero_grad()
        self.model.train()
        self.mlpmodel.train()
        for i,data in enumerate(self.train_loader):
            loss,c_loss,output,label  = self.forward(data)

            loss.backward()
            self.optimizer.step()
            epochloss+=loss.detach().cpu()
            epochcloss+=c_loss.detach().cpu()
            
            _, pred = output.topk(1, dim=1, largest=True, sorted=True)
            pred,label,output=pred.cpu(),label.cpu(),output.cpu()
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(label.detach().numpy().reshape(-1))
            y_hat_logit+=list(output.detach().numpy())
        return epochloss.item()/len(self.train_loader),epochcloss.item()/len(self.train_loader),y_hat, y_true,y_hat_logit


    def _full_test_step(self,loader,model,mlpmodel):
        y_hat, y_true,y_hat_logit = [], [], []
        loss_total, pred_num = 0, 0
        model.eval()
        mlpmodel.eval()
        with torch.no_grad():
            for data in loader:
                loss,c_loss,output,label  = self.forward(data)
                
                _, pred = output.topk(1, dim=1, largest=True, sorted=True)
                pred,label,output=pred.cpu(),label.cpu(),output.cpu()
                y_hat += list(pred.detach().numpy().reshape(-1))
                y_true += list(label.detach().numpy().reshape(-1))
                y_hat_logit+=list(output.detach().numpy())
                
                pred_num += len(label.reshape(-1, 1))
                loss_total += loss.detach() * len(label.reshape(-1, 1))
        return loss_total/pred_num,y_hat, y_true, y_hat_logit
    


def contrastive_loss(embeddings,labels,margin):
    
    positive_mask = labels.view(-1, 1) == labels.view(1, -1)
    negative_mask = ~positive_mask

    # Calculate the number of positive and negative pairs
    num_positive_pairs = positive_mask.sum() - labels.shape[0] 
    num_negative_pairs = negative_mask.sum()

    # If there are no negative pairs, return a placeholder loss
    if num_negative_pairs==0 or num_positive_pairs== 0:
        print("all pos or neg")
        return torch.tensor(0, dtype=torch.float)
    # Calculate the pairwise Euclidean distances between embeddings
    distances = torch.cdist(embeddings, embeddings)/np.sqrt(embeddings.shape[1])
    
    if num_positive_pairs>num_negative_pairs:
        # Sample an equal number of + pairs 
        positive_indices = torch.nonzero(positive_mask)
        random_positive_indices = torch.randperm(len(positive_indices))[:num_negative_pairs]
        selected_positive_indices = positive_indices[random_positive_indices]

        # Select corresponding negative pairs
        negative_mask.fill_diagonal_(False)
        negative_distances = distances[negative_mask].view(-1, 1)
        positive_distances = distances[selected_positive_indices[:,0],selected_positive_indices[:,1]].view(-1, 1)
    else: # case for most datasets
        # Sample an equal number of - pairs 
        negative_indices = torch.nonzero(negative_mask)
        random_negative_indices = torch.randperm(len(negative_indices))[:num_positive_pairs]
        selected_negative_indices = negative_indices[random_negative_indices]

        # Select corresponding positive pairs
        positive_mask.fill_diagonal_(False)
        positive_distances = distances[positive_mask].view(-1, 1)
        negative_distances = distances[selected_negative_indices[:,0],selected_negative_indices[:,1]].view(-1, 1)

    # Calculate the loss for positive and negative pairs
    loss = (positive_distances - negative_distances + margin).clamp(min=0).mean()
    return loss


def scipy_spanning_tree(edge_index, edge_weight,num_nodes ):
    row, col = edge_index.cpu()
    edge_weight=edge_weight.cpu()
    cgraph = csr_matrix((edge_weight, (row, col)), shape=(num_nodes, num_nodes))
    Tcsr = minimum_spanning_tree(cgraph)
    tree_row, tree_col = Tcsr.nonzero()
    spanning_edges = np.stack([tree_row,tree_col],0)    
    return spanning_edges
    
def build_spanning_tree_edge(edge_index,edge_weight, num_nodes):
    spanning_edges = scipy_spanning_tree(edge_index, edge_weight,num_nodes,)
        
    spanning_edges = torch.tensor(spanning_edges, dtype=torch.long, device=edge_index.device)
    spanning_edges_undirected = torch.cat([spanning_edges,torch.stack([spanning_edges[1],spanning_edges[0]])],1)
    return spanning_edges_undirected




def record(values,epoch,writer,phase="Train"):
    """ tfboard write """
    for key,value in values.items():
        writer.add_scalar(key+"/"+phase,value,epoch)           
def calculate(y_hat,y_true,y_hat_logit):
    """ calculate five metrics using y_hat, y_true, y_hat_logit """
    train_acc=(np.array(y_hat) == np.array(y_true)).sum()/len(y_true) 
    return train_acc


def print_1(epoch,phase,values,color=None):
    """ print epoch info"""
    if color is not None:
        print(color( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))
    else:
        print(( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))

def get_angle(v1, v2):
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1),value=0)
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1),value=0)
    return torch.atan2( torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))
def get_theta(v1, v2):
    # v1 is starting line, right-hand rule to v2, if thumb is up, +, else -
    angle=get_angle(v1, v2)
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1),value=0)
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1),value=0)
    v = torch.cross(v1, v2, dim=1)[...,2]
    flag = torch.sign((v))
    flag[flag==0]=-1 
    return angle*flag   

def triplets(edge_index, num_nodes):
    row, col = edge_index

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=row, col=col, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_col = adj_t[:,row]
    num_triplets = adj_t_col.set_value(None).sum(dim=0).to(torch.long)

    idx_j = row.repeat_interleave(num_triplets) 
    idx_i = col.repeat_interleave(num_triplets) 
    edx_2nd = value.repeat_interleave(num_triplets) 
    idx_k = adj_t_col.t().storage.col() 
    edx_1st = adj_t_col.t().storage.value()
    mask1 = (idx_i == idx_k) & (idx_j != idx_i)  # Remove go back triplets. 
    mask2 = (idx_i == idx_j) & (idx_j != idx_k)  # Remove repeat self loop triplets
    mask3 = (idx_j == idx_k) & (idx_i != idx_k)  # Remove self-loop neighbors 
    mask = ~(mask1 | mask2 | mask3) 
    idx_i, idx_j, idx_k, edx_1st, edx_2nd = idx_i[mask], idx_j[mask], idx_k[mask], edx_1st[mask], edx_2nd[mask]
    
    num_triplets_real = torch.cumsum(num_triplets, dim=0) - torch.cumsum(~mask, dim=0)[torch.cumsum(num_triplets, dim=0)-1]

    return torch.stack([idx_i, idx_j, idx_k]), num_triplets_real.to(torch.long), edx_1st, edx_2nd
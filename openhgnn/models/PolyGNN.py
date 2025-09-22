import dgl
from dgl.nn import HeteroLinear as Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Parameter
from torch_scatter import scatter
from . import BaseModel, register_model
import numpy as np
torch.autograd.set_detect_anomaly(True)
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

@register_model('PolyGNN')
class PolyGNN(BaseModel):
    """
    A graph classification as well as representation learning model for Polygonal Geometries.

    Parameters
    ----------
    in_dim : int
        Input feature size.
    h_dim : int
        Hidden layer size.
    num_interactions : int
        Number of interaction layers,i.e.,number of SPNN layers.
    localdepth : int
        Number of linear layers for geometric embeddings.
    finaldepth : int
        Number of linear layers for each WMLP of SPNN.
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(in_dim=args.in_dim,
                   h_dim=args.h_dim,
                   num_interactions=args.num_interactions,
                   localdepth=args.localdepth,
                   finaldepth=args.finaldepth
                  )
    def __init__(self, in_dim,h_dim,num_interactions,localdepth,finaldepth,share='0',batchnorm="True"):
        super(PolyGNN,self).__init__()
        self.training=True
        self.h_channel = h_dim
        self.input_featuresize=in_dim
        self.localdepth = localdepth
        self.num_interactions=num_interactions
        self.finaldepth=finaldepth
        self.batchnorm = batchnorm        
        self.activation=nn.ReLU()
        self.att = Parameter(torch.ones(4),requires_grad=True)
        num_gaussians=(1,1,1)
        self.mlp_geo = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                self.mlp_geo.append(Linear(in_size={'vertices':sum(num_gaussians)},out_size = h_dim))
            else:
                self.mlp_geo.append(Linear(in_size={'vertices':h_dim},out_size = h_dim))
            if self.batchnorm == "True":
                self.mlp_geo.append(nn.BatchNorm1d(h_dim))
            self.mlp_geo.append(self.activation)            
         
        self.mlp_geo_backup = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                self.mlp_geo_backup.append(Linear(in_size={'vertices':4},out_size = h_dim))
            else:
                self.mlp_geo_backup.append(Linear(in_size={'vertices':h_dim},out_size = h_dim))
            if self.batchnorm == "True":
                self.mlp_geo_backup.append(nn.BatchNorm1d(h_dim))
            self.mlp_geo_backup.append(self.activation)        
        self.translinear=Linear(in_size={'vertices':in_dim+1},out_size =  self.h_channel)
        self.interactions= ModuleList()
        for i in range(self.num_interactions):
            block = SPNN(
                in_ch=self.input_featuresize,
                hidden_channels=self.h_channel,
                activation=self.activation,
                finaldepth=self.finaldepth,
                batchnorm=self.batchnorm,
                num_input_geofeature=self.h_channel
            )
            self.interactions.append(block)
        self.reset_parameters()
    def reset_parameters(self):
        for lin in self.mlp_geo:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for i in (self.interactions):
                i.reset_parameters()

    def single_forward(self, input_feature,coords,edge_index,edge_index_2rd, edx_jk, edx_ij,batch,num_edge_inside,edge_rep,g):
        if edge_rep:
            i, j, k = edge_index_2rd
            edge_index1,edge_index2=  edge_index
            edge_index_all=torch.cat([edge_index1,edge_index2],1)
            distance_ij=(coords[j] - coords[i]).norm(p=2, dim=1)
            distance_jk=(coords[j] - coords[k]).norm(p=2, dim=1)
            theta_ijk = get_angle(coords[j] - coords[i], coords[k] - coords[j])
            geo_encoding_1st=distance_ij[:,None]
            geo_encoding=torch.cat([geo_encoding_1st,distance_jk[:,None],theta_ijk[:,None]],dim=-1)
        else:    
            coords_j = coords[edge_index[0]]
            coords_i = coords[edge_index[1]]
            geo_encoding=torch.cat([coords_j,coords_i],dim=-1)
        if edge_rep:
            for lin in self.mlp_geo:
                if isinstance(lin, Linear):
                    geo_encoding = lin({'vertices': geo_encoding})['vertices'] 
                else:
                    geo_encoding = lin(geo_encoding)
        else:
            for lin in self.mlp_geo_backup:
                if isinstance(lin, Linear):
                    geo_encoding = lin({'vertices': geo_encoding})['vertices']
                else:
                    geo_encoding = lin(geo_encoding)
            geo_encoding = torch.zeros_like(geo_encoding, device=geo_encoding.device, dtype=geo_encoding.dtype)
        node_feature= input_feature
        node_feature_list=[]
        for interaction in self.interactions:
            node_feature =  interaction(node_feature,geo_encoding,edge_index_2rd,edx_jk,edx_ij,num_edge_inside,self.att,g)
            node_feature_list.append(node_feature)
        return node_feature_list
    def forward(self,g,label,args,device):
        src, dst = g.edges(etype=('vertices', 'inside', 'vertices'))
        edge_index1 = torch.stack([src, dst], dim=0)
        src, dst = g.edges(etype=('vertices', 'apart', 'vertices'))
        edge_index2 = torch.stack([src, dst], dim=0)
        combined_edge_index=torch.cat([edge_index1,edge_index2],1)
        num_edge_inside=edge_index1.shape[1]
        edge_weight=torch.rand(combined_edge_index.shape[1]) + 1
        undirected_spanning_edge = build_spanning_tree_edge(combined_edge_index, edge_weight,num_nodes=g.ndata['pos'].shape[0])
        
        edge_set_1 = set(map(tuple, edge_index2.t().tolist()))
        edge_set_2 = set(map(tuple, undirected_spanning_edge.t().tolist()))

        common_edges = edge_set_1.intersection(edge_set_2)
        common_edges_tensor = torch.tensor(list(common_edges), dtype=torch.long).t().to(device)
        spanning_edge=torch.cat([edge_index1,common_edges_tensor],1)
        combined_edge_index=spanning_edge


        x = g.ndata['pos']
        batch_num_nodes = g.batch_num_nodes()
        batch = torch.arange(len(batch_num_nodes), device=device).repeat_interleave(batch_num_nodes)
        label = label.to(device).long().view(-1)
        num_nodes=x.shape[0]
        edge_index_2rd, num_triplets_real, edx_jk, edx_ij = triplets(combined_edge_index, num_nodes)
        input_feature=torch.zeros([x.shape[0],args.h_dim],device=device) 
        output=self.single_forward(input_feature,x,[edge_index1,edge_index2], edge_index_2rd,edx_jk, edx_ij,batch,num_edge_inside,args.edge_rep,g)  
        output=torch.cat(output,dim=1) # feature list concatenate
        g.ndata['h'] = output
        graph_embeddings = dgl.sum_nodes(g, 'h')
        graph_embeddings = torch.clamp(graph_embeddings, max=1e6)
        c_loss=contrastive_loss(graph_embeddings,label,margin=1)
        return c_loss*args.loss_coef,graph_embeddings
    
class SPNN(nn.Module):
    def __init__(
        self,
        in_ch,
        hidden_channels,
        activation=nn.ReLU(),
        finaldepth=3,
        batchnorm="True",
        num_input_geofeature=13
    ):
        super(SPNN, self).__init__()
        self.activation = activation
        self.finaldepth = finaldepth
        self.batchnorm = batchnorm
        self.num_input_geofeature=num_input_geofeature
        self.hidden_channels = hidden_channels
        self.WMLP_list = ModuleList()
        for _ in range(4):
            WMLP = ModuleList()
            for i in range(self.finaldepth + 1):
                if i == 0:
                    WMLP.append(Linear(in_size={'vertices':hidden_channels*3+num_input_geofeature},out_size = hidden_channels))
                else:
                    WMLP.append(Linear(in_size={'vertices':hidden_channels},out_size = hidden_channels))  
                if self.batchnorm == "True":
                    WMLP.append(nn.BatchNorm1d(hidden_channels))
                WMLP.append(self.activation)
            self.WMLP_list.append(WMLP)
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.WMLP_list:
            for lin in mlp:
                if hasattr(lin, 'reset_parameters'):
                    lin.reset_parameters() 
    def forward(self, node_feature,geo_encoding,edge_index_2rd,edx_jk,edx_ij,num_edge_inside,att,g):
        i,j,k = edge_index_2rd
        if node_feature is None:
            concatenated_vector = geo_encoding
        else:
            node_attr_0st = node_feature[i]
            node_attr_1st = node_feature[j]
            node_attr_2 = node_feature[k]
            concatenated_vector = torch.cat(
                [
                    node_attr_0st,
                    node_attr_1st,
                    node_attr_2,
                    geo_encoding,
                ],
                dim=-1,
            )
        x_i = concatenated_vector
        
        edge1_edge1_mask = (edx_ij < num_edge_inside) & (edx_jk < num_edge_inside) 
        edge1_edge2_mask = (edx_ij < num_edge_inside) & (edx_jk >= num_edge_inside)
        edge2_edge1_mask = (edx_ij >= num_edge_inside) & (edx_jk < num_edge_inside)
        edge2_edge2_mask = (edx_ij >= num_edge_inside) & (edx_jk >= num_edge_inside)  
        masks=[edge1_edge1_mask,edge1_edge2_mask,edge2_edge1_mask,edge2_edge2_mask]
        
        x_output = torch.zeros(x_i.shape[0], self.hidden_channels, device=x_i.device)
        
        for index in range(4):
            WMLP = self.WMLP_list[index]
            x = x_i[masks[index]]
            for lin in WMLP:
                if isinstance(lin,Linear):
                    x = lin({'vertices': x})['vertices']
                else:
                    x = lin(x)
            x = F.leaky_relu(x) * att[index]
            x_output[masks[index]] += x
        out_feature = scatter(x_output, i, dim=0, reduce='add')
    
  
        return out_feature
    

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Parameter
from torch_geometric.nn import  Linear
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter
from . import BaseModel, register_model
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
                self.mlp_geo.append(Linear(sum(num_gaussians), h_dim))
            else:
                self.mlp_geo.append(Linear(h_dim, h_dim))
            if self.batchnorm == "True":
                self.mlp_geo.append(nn.BatchNorm1d(h_dim))
            self.mlp_geo.append(self.activation)            
         
        self.mlp_geo_backup = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                self.mlp_geo_backup.append(Linear(4, h_dim))
            else:
                self.mlp_geo_backup.append(Linear(h_dim, h_dim))
            if self.batchnorm == "True":
                self.mlp_geo_backup.append(nn.BatchNorm1d(h_dim))
            self.mlp_geo_backup.append(self.activation)        
        self.translinear=Linear(in_dim+1, self.h_channel)
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
            if isinstance(lin, Linear):
                torch.nn.init.xavier_uniform_(lin.weight)
                lin.bias.data.fill_(0)
        for i in (self.interactions):
                i.reset_parameters()

    def single_forward(self, input_feature,coords,edge_index,edge_index_2rd, edx_jk, edx_ij,batch,num_edge_inside,edge_rep):
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
                geo_encoding=lin(geo_encoding)
        else:
            for lin in self.mlp_geo_backup:
                geo_encoding=lin(geo_encoding)
            geo_encoding=torch.zeros_like(geo_encoding,device=geo_encoding.device,dtype=geo_encoding.dtype)
        node_feature= input_feature
        node_feature_list=[]
        for interaction in self.interactions:
            node_feature =  interaction(node_feature,geo_encoding,edge_index_2rd,edx_jk,edx_ij,num_edge_inside,self.att)
            node_feature_list.append(node_feature)
        return node_feature_list
    def forward(self, input_feature, coords,edge_index,edge_index_2rd, edx_jk, edx_ij,batch,num_edge_inside,edge_rep):
        output=self.single_forward(input_feature,coords,edge_index,edge_index_2rd, edx_jk, edx_ij,batch,num_edge_inside,edge_rep)
        return output
    
class SPNN(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        hidden_channels,
        activation=torch.nn.ReLU(),
        finaldepth=3,
        batchnorm="True",
        num_input_geofeature=13
    ):
        super(SPNN, self).__init__()
        self.activation = activation
        self.finaldepth = finaldepth
        self.batchnorm = batchnorm
        self.num_input_geofeature=num_input_geofeature
        
        self.WMLP_list = ModuleList()
        for _ in range(4):
            WMLP = ModuleList()
            for i in range(self.finaldepth + 1):
                if i == 0:
                    WMLP.append(Linear(hidden_channels*3+num_input_geofeature, hidden_channels))
                else:
                    WMLP.append(Linear(hidden_channels, hidden_channels))  
                if self.batchnorm == "True":
                    WMLP.append(nn.BatchNorm1d(hidden_channels))
                WMLP.append(self.activation)
            self.WMLP_list.append(WMLP)
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.WMLP_list:
            for lin in mlp:
                if isinstance(lin, Linear):
                    torch.nn.init.xavier_uniform_(lin.weight)
                    lin.bias.data.fill_(0)
    def forward(self, node_feature,geo_encoding,edge_index_2rd,edx_jk,edx_ij,num_edge_inside,att):
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
                    node_attr_1st,node_attr_2,
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
        
        x_output=torch.zeros(x_i.shape[0],self.WMLP_list[0][0].weight.shape[0],device=x_i.device)
        for index in range(4):
            WMLP=self.WMLP_list[index]
            x=x_i[masks[index]]
            for lin in WMLP:
                x=lin(x)    
            x = F.leaky_relu(x)*att[index]
            x_output[masks[index]]+=x
    
        out_feature = scatter(x_output, i, dim=0, reduce='add')  
        return out_feature
    
def get_angle(v1, v2):
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1),value=0)
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1),value=0)
    return torch.atan2( torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))
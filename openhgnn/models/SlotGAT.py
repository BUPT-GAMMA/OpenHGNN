import torch.nn as nn
import torch.nn.functional as F
from . import BaseModel, register_model
import torch
import math
from collections.abc import Mapping
from dgl import function as fn
from dgl.nn.pytorch import edge_softmax

@register_model('SlotGAT')
class SlotGAT(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.edge_dim,
                   args.num_etypes,
                   args.in_dim,
                   args.hid_dim,
                   args.num_classes,
                   args.num_layers,
                   args.num_heads,
                   args.feat_drop,
                   args.attn_drop,
                   args.negative_slope,
                   args.residual,
                   args.alpha,
                   args.num_ntype
                   )
    def __init__(self,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 hid_dim,
                 num_classes,
                 num_layers,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 num_ntype,
                 eindexer = None, aggregator="SA",  SAattDim=32):
        super(SlotGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu 
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, hid_dim, bias=True) for in_dim in in_dims])  # 在输入层，对于每种类型的节点创建一个全连接层
        self.num_ntype=num_ntype
        self.num_classes=num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.SAattDim=SAattDim
        self.e_feat = None
        # SlotAttention
        
        last_dim=num_classes
        heads = [num_heads] * num_layers + [1]
            
        self.macroLinear=nn.Linear(last_dim, self.SAattDim, bias=True);nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414);nn.init.normal_(self.macroLinear.bias, std=1.414*math.sqrt(1/(self.macroLinear.bias.flatten().shape[0])))
        self.macroSemanticVec=nn.Parameter(torch.FloatTensor(self.SAattDim,1));nn.init.normal_(self.macroSemanticVec,std=1)
        
        self.last_fc = nn.Parameter(torch.FloatTensor(size=(num_classes*self.num_ntype, num_classes))) ;nn.init.xavier_normal_(self.last_fc, gain=1.414)
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            hid_dim, hid_dim, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer,inputhead=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = hid_dim * num_heads
            self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
                hid_dim* heads[l-1] , hid_dim, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer))
        # output projection
        self.gat_layers.append(slotGATConv(edge_dim, num_etypes,
            hid_dim* heads[-2] , num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha,num_ntype=num_ntype, eindexer=eindexer))
        self.aggregator=aggregator
        self.by_slot=[f"by_slot_{nt}" for nt in range(num_ntype)]
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        
    def forward(self, g,features_list,e_feat, get_out="False"):
        encoded_embeddings=None
        h = []
        for nt_id,(fc, feature) in enumerate(zip(self.fc_list, features_list)):  #节点特征初始化
            nt_ft=fc(feature)
            emsen_ft=torch.zeros([nt_ft.shape[0],nt_ft.shape[1]*self.num_ntype]).to(feature.device)
            emsen_ft[:,nt_ft.shape[1]*nt_id:nt_ft.shape[1]*(nt_id+1)]=nt_ft
            h.append(emsen_ft)   # the id is decided by the node types
        h = torch.cat(h, 0)        #  num_nodes*(num_type*hidden_dim)
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g, h, e_feat,get_out=get_out, res_attn=res_attn)   #num_nodes*num_heads*(num_ntype*hidden_dim)
            h = h.flatten(1)    #num_nodes*(num_heads*num_ntype*hidden_dim)
            encoded_embeddings=h
        # output projection
        logits, _ = self.gat_layers[-1](g, h, e_feat,get_out=get_out, res_attn=None)   #num_nodes*num_heads*num_ntype*hidden_dim

        logits=logits.squeeze(1)
        logits=self.l2byslot(logits)
        logits=logits.view(-1, self.num_ntype,int(logits.shape[1]/self.num_ntype))
        
        if "getSlots" in get_out:
            self.logits=logits.detach()

            
        
        slot_scores=(F.tanh(self.macroLinear(logits))@self.macroSemanticVec).mean(0,keepdim=True)  #num_slots
        self.slot_scores=F.softmax(slot_scores,dim=1)
        logits=(logits*self.slot_scores).sum(1)
        logits=logits.view(-1,1, 1,self.num_classes).flatten(2)


        #average across the heads
        ### logits = [num_nodes *  num_of_heads *num_classes]
        self.logits_mean=logits.flatten().mean()
        logits = logits.mean(1)
        
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits, encoded_embeddings    #hidden_logits

        
    def l2byslot(self,x):
        
        x=x.view(-1, self.num_ntype,int(x.shape[1]/self.num_ntype))
        x=x / (torch.max(torch.norm(x, dim=2, keepdim=True), self.epsilon))
        x=x.flatten(1)
        return x


            
        
        
        
        
        
class slotGATConv(nn.Module):

    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.,
                 num_ntype=None, eindexer=None,inputhead=False, dataRecorder=None):
        super(slotGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = in_feats,in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats) if edge_feats else None
        self.eindexer=eindexer
        self.num_ntype=num_ntype 
        
        self.attentions=None
        self.dataRecorder=dataRecorder

        if isinstance(in_feats, tuple):
            raise NotImplementedError()
        else:
            self.fc = nn.Parameter(torch.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False) if edge_feats else None
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats   *self.num_ntype)))  #源节点上的注意力参数
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats*self.num_ntype)))   # 目标节点上的注意力参数
        self.attn_e = nn.Parameter(torch.FloatTensor(size=(1, num_heads, edge_feats))) if edge_feats else None # 边上的注意力参数
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc =nn.Parameter(torch.FloatTensor(size=(self.num_ntype, self._in_src_feats, out_feats * num_heads)))
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        self.alpha = alpha
        self.inputhead=inputhead
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc, gain=gain)
            
        else:
            raise NotImplementedError()
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self._edge_feats:
            nn.init.xavier_normal_(self.attn_e, gain=gain) 
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        elif isinstance(self.res_fc, Identity):
            pass
        elif isinstance(self.res_fc, nn.Parameter):
            nn.init.xavier_normal_(self.res_fc, gain=gain)
        if self._edge_feats:
            nn.init.xavier_normal_(self.fc_e.weight, gain=gain) 

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat,get_out=[""], res_attn=None):
        #feature transformation first
        h_src = h_dst = self.feat_drop(feat)   #num_nodes*(num_ntype*input_dim)

        if self.inputhead:
            h_src=h_src.view(-1,1,self.num_ntype,self._in_src_feats) # num_nodes*1*(num_ntype*input_dim)
        else:
            h_src=h_src.view(-1,self._num_heads,self.num_ntype,int(self._in_src_feats/self._num_heads))
        h_dst=h_src=h_src.permute(2,0,1,3).flatten(2)  #num_ntype*num_nodes*(in_feat_dim)
        if "getEmb" in get_out:
            self.emb=h_dst.cpu().detach()
        #self.fc with num_ntype*(in_feat_dim)*(out_feats * num_heads)
        feat_dst = torch.bmm(h_src,self.fc)  #num_ntype*num_nodes*(out_feats * num_heads)
        feat_src = feat_dst =feat_dst.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)
        if graph.is_block:
            feat_dst = feat_src[:graph.number_of_dst_nodes()]
        e_feat = self.edge_emb(e_feat) if self._edge_feats else None
        e_feat = self.fc_e(e_feat).view(-1, self._num_heads, self._edge_feats)  if self._edge_feats else None
        ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1) if self._edge_feats else 0  #(-1, self._num_heads, 1) 
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        graph.edata.update({'ee': ee}) if self._edge_feats else None
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e_=graph.edata.pop('e')
        ee=graph.edata.pop('ee') if self._edge_feats else 0
        e=e_+ee
        
        e = self.leaky_relu(e)
        # compute softmax
        a=self.attn_drop(edge_softmax(graph, e))
        if res_attn is not None:
            a=a * (1-self.alpha) + res_attn * self.alpha 
        graph.edata['a'] = a
        # then message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                            fn.sum('m', 'ft'))
                            
        rst = graph.dstdata['ft'] 
        # residual
        if self.res_fc is not None:
            
            if self._in_dst_feats != self._out_feats:
                resval =torch.bmm(h_src,self.res_fc)
                resval =resval.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                    -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)
            else:
                resval = self.res_fc(h_src).view(h_dst.shape[0], -1, self._out_feats*self.num_ntype)  #Identity
            rst = rst + resval
        # bias
        if self.bias:
            rst = rst + self.bias_param
        # activation
        if self.activation:
            rst = self.activation(rst)
        self.attentions=graph.edata.pop('a').detach()
        torch.cuda.empty_cache()
        return rst, self.attentions
        
        
# pylint: disable=W0235
class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x
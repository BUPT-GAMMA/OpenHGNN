from openhgnn.models import HAN, register_model
from openhgnn.models.base_model import BaseModel
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from torch import nn    
import dgl
from dgl import convert
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes

class HeteroAttLayer(nn.Module):
    # nchannel, nhid*nheads[-1], nhid, device, dropout
    def __init__(self, ntype_meta_path,in_dim, att_dim,dropout):
        super(HeteroAttLayer, self).__init__()
        self.ntype_meta_type = ntype_meta_path
        self.nchannel = len(ntype_meta_path)
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.meta_att = nn.Parameter(torch.zeros(size=(len(ntype_meta_path), att_dim)))#2*64
        nn.init.xavier_uniform_(self.meta_att.data, gain=1.414)

        self.linear_block2 = nn.Sequential(nn.Linear(att_dim, att_dim), nn.Tanh())

    def forward(self, hs, nnode):

        hs = torch.cat([self.linear_block2(hs[i]).view(1,nnode,-1) for i in range(self.nchannel)], dim=0)
        meta_att = []
        for i in range(self.nchannel):
            meta_att.append(torch.sum(torch.mm(hs[i], self.meta_att[i].view(-1,1)).squeeze(1)) / nnode)
        meta_att = torch.stack(meta_att, dim=0)
        meta_att = F.softmax(meta_att, dim=0)#(shape=2)
        aggre_hid = []
        aggre_hid=torch.bmm

        # 将 meta_att 和 hs 调整为适合 bmm 的形状
        meta_att_expanded = meta_att.unsqueeze(0).expand(nnode, -1, -1)
        hs_transposed = hs.permute(1, 0, 2)

        # 执行批量矩阵乘法
        aggre_hid = torch.bmm(meta_att_expanded, hs_transposed)

        # 调整形状以匹配 [nnode, self.att_dim]
        aggre_hid = aggre_hid.view(nnode, self.att_dim)

        # for i in range(nnode):
        #     aggre_hid.append(torch.mm(meta_att.view(1,-1), hs[:,i,:]))# 1*2 2*4177*4
        # aggre_hid = torch.stack(aggre_hid, dim=0).view(nnode, self.att_dim)# 4177*4
        return aggre_hid

class NodeAttLayer(nn.Module):
    def __init__(self, meta_paths_dict,nfeat, hidden_dim, nheads, dropout,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #nheads个头
        self.meta_paths_dict=meta_paths_dict
        self.layers=nn.ModuleList()
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.mods = nn.ModuleDict({mp: GATConv(nfeat, hidden_dim, nheads,
                                          dropout, dropout, activation=F.elu,
                                          allow_zero_in_degree=True) for mp in meta_paths_dict})
    def metapath_reachable_graph(self, g, metapath):
        adj = 1
        for etype in metapath:
            adj = adj * g.adj_external(
                etype=etype, scipy_fmt="csr", transpose=False
            )

        adj = (adj != 0).tocsr()
        srctype = g.to_canonical_etype(metapath[0])[0]
        dsttype = g.to_canonical_etype(metapath[-1])[2]
        new_g = convert.heterograph(
            {(srctype, "_E", dsttype): adj.nonzero()},
            {srctype: adj.shape[0], dsttype: adj.shape[0]},
            idtype=g.idtype,
            device=g.device,
        )

        # copy srcnode features
        new_g.nodes[srctype].data.update(g.nodes[srctype].data)
        # copy dstnode features

        return new_g
    def forward(self,g,h_dict):
        # minibatch
        if isinstance(g, dict):
            g_dict=g

        # full batch
        else:
            if self._cached_graph is None or self._cached_graph is not g:
                self._cached_graph = g
                self._cached_coalesced_graph.clear()
                for mp, mp_value in self.meta_paths_dict.items():
                    self._cached_coalesced_graph[mp] = self.metapath_reachable_graph(
                        g, mp_value)

            g_dict=self._cached_coalesced_graph

        # 计算NodeAtt Embedding
        outputs = {}

        for meta_path_name, meta_path in self.meta_paths_dict.items():
            new_g = g_dict[meta_path_name]

            # han minibatch
            if h_dict.get(meta_path_name) is not None:
                h = h_dict[meta_path_name][new_g.srctypes[0]]
            # full batch
            else:
                h = h_dict[new_g.srctypes[0]]

            outputs[meta_path_name]=self.mods[meta_path_name](new_g, h).flatten(1)
            #[1:[131:(4177,512),121:(4177,512)]]
        return outputs

class NodeAttEmb(nn.Module):
    def __init__(self, ntype_meta_paths_dict, nfeat, hidden_dim, nheads, dropout,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # 不同meta_path的NodeAttLayer
        self.NodeAttLayers=nn.ModuleList()
        for ntype,meta_path_dict in ntype_meta_paths_dict:#ntype:1
            self.NodeAttLayers(NodeAttLayer(meta_path_dict,nfeat, hidden_dim, nheads, dropout))
        self.linear = nn.Linear(hidden_dim * nheads[-1], hidden_dim*nheads[-1])

    def forward(self, g, h_dict=None):
        if h_dict==None:
            h_dict=g.ndata['h']
        for gnn in self.NodeAttLayers:
            h_dict = gnn(g, h_dict)
        out_dict = {}
        for ntype, h in h_dict.items():  # only one ntype here
            out_dict[ntype] = self.linear(h_dict[ntype])
        return out_dict

@register_model('HGA')
class HGA(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        ntypes = set()# 进行分类的类别
        if hasattr(args, 'target_link'):
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)
        elif hasattr(args, 'category'):
            ntypes.add(args.category)
        else:
            raise ValueError
        ntype_meta_paths_dict = {}# 每个待分类的节点对应的元路径
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in args.meta_paths_dict.items():
                # 一条元路径
                if meta_path[0][0] == ntype:
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)

        return cls(ntype_meta_paths_dict=ntype_meta_paths_dict,
                   nfeat=args.hidden_dim,
                   hidden_dim=args.hidden_dim,
                   nlabel=args.out_dim,
                   nheads=args.num_heads,
                   dropout=args.dropout)

    def __init__(self,ntype_meta_paths_dict, nfeat,nlabel, hidden_dim,nheads, dropout):
        super().__init__()
        self.out_dim = nlabel
        self.mod_dict = nn.ModuleDict()
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():# 对每个需要分类的节点建立一个模型
            self.mod_dict[ntype] = _HGA(meta_paths_dict, nfeat,nlabel, hidden_dim,nheads, dropout)

    def forward(self, gT,h_dictT,gS=None, h_dictS=None):
        r"""
        Parameters
        -----------
        g : DGLHeteroGraph or dict[str, dict[str, DGLBlock]]
            For full batch, it is a heterogeneous graph. For mini batch, it is a dict from node type to dict from
            mata path name to DGLBlock.
        h_dict : dict[str, Tensor] or dict[str, dict[str, dict[str, Tensor]]]
            The input features. For full batch, it is a dict from node type to node features. For mini batch, it is
            a dict from node type to dict from meta path name to dict from node type to node features.
        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features. Dict from node type to node features.
        """
        if gS is not None:# 训练
            for ntype, hga in self.mod_dict.items():
                if isinstance(gS, dict):
                    # mini batch
                    if ntype not in gS:
                        continue
                    _gS = gS[ntype]
                    _gT = gT[ntype]
                    _in_hS = h_dictS[ntype]
                    _in_hT = h_dictT[ntype]
                else:
                    # full batch
                    _gS = gS
                    _gT = gT
                    _in_hS = h_dictS
                    _in_hT = h_dictT
                homo_outS,homo_outT,clabel_predSs,clabel_predTs,target_probs,clabel_predS = hga(gS=_gS, gS_dict=_in_hS,gT=_gT,gT_dict=_in_hT)
                # for ntype, h in _out_h.items():
                #     out_dict[ntype] = h

            return homo_outS,homo_outT,clabel_predSs,clabel_predTs,target_probs,clabel_predS
        else:# 推理
            for ntype, hga in self.mod_dict.items():
                if isinstance(gT, dict):
                    # mini batch
                    if ntype not in gT:
                        continue
                    _gT = gT[ntype]
                    _in_hT = h_dictT[ntype]
                else:
                    # full batch
                    _gT = gT
                    _in_hT = h_dictT
                target_probs = hga(gT=_gT, gT_dict=_in_hT)

            return target_probs


class _HGA(BaseModel):

    def __init__(self,ntype_meta_paths_dict, nfeat,nlabel, hidden_dim,nheads, dropout):
        super().__init__()
        self.sharedNet=NodeAttLayer(ntype_meta_paths_dict, nfeat, hidden_dim, nheads, dropout)# att_{node}^{\phi}对节点进行注意力机制
        self.linear_block = nn.Sequential(nn.Linear(hidden_dim*nheads, hidden_dim), nn.Tanh())# 分类器

        # self.HeteroAttLayerS = HeteroAttLayer(nchannel, nhid*nheads[-1], nlabel, device, dropout).to(device)
        self.HeteroAttLayerT = HeteroAttLayer(ntype_meta_paths_dict, hidden_dim*nheads, nlabel, dropout)# 语义级别的注意力att_{sem}^{\phi}
        # self.add_module('hetero_att', self.HeteroAttLayer)
        # self.add_module('hetero_attT', self.HeteroAttLayerT)
        self.ntype_meta_paths_dict=ntype_meta_paths_dict
        self.cls_fcs=nn.ModuleList()# 
        #对每个元路径
        for i in ntype_meta_paths_dict:# clf^{\phi} 在源代码中clf^{\phi}是对dst和src公用的，每个元路径对应一个分类器
            self.cls_fcs.append(torch.nn.Linear(hidden_dim, nlabel))


    def forward(self, gT,gT_dict,gS=None,gS_dict=None):  
        if gS is not None: # 训练
            # att_{node}
            homo_outS=self.sharedNet(gS,gS_dict)
            homo_outT=self.sharedNet(gT,gT_dict)

            new_hsS = {i:self.linear_block(homo_outS[i]).view(list(homo_outS.values())[0].shape[0],-1) for i in homo_outS}
            new_hsT = {i:self.linear_block(homo_outT[i]).view(list(homo_outT.values())[0].shape[0],-1) for i in homo_outT}

            clabel_predSs=[]
            clabel_predTs=[]
            for idx,(path_name,meta_path) in enumerate(self.ntype_meta_paths_dict.items()):# clf
                clabel_predSs.append(self.cls_fcs[idx](new_hsS[path_name]))
                clabel_predTs.append(self.cls_fcs[idx](new_hsT[path_name]))

            tworeS = torch.cat([i.unsqueeze(0) for i in clabel_predSs], dim=0)
            clabel_predS = self.HeteroAttLayerT(tworeS,tworeS.shape[1])#att_{sem}+L_{cls}

            twore = torch.cat([i.unsqueeze(0) for i in clabel_predTs], dim=0)
            clabel_predF = self.HeteroAttLayerT(twore,twore.shape[1])#att_{sem}+L_{cls}
            target_probs = F.softmax(clabel_predF, dim=-1)
            target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

            return homo_outS,homo_outT,clabel_predSs,clabel_predTs,target_probs,clabel_predS
        else:# 推理
            homo_outT=self.sharedNet(gT,gT_dict)
            new_hsT = {i:self.linear_block(homo_outT[i]).view(list(homo_outT.values())[0].shape[0],-1) for i in homo_outT}

            clabel_predTs=[]
            for idx,(path_name,meta_path) in enumerate(self.ntype_meta_paths_dict.items()):
                clabel_predTs.append(self.cls_fcs[idx](new_hsT[path_name]))
            # TODO mmd_loss、l1_loss

            twore = torch.cat([i.unsqueeze(0) for i in clabel_predTs], dim=0)
            clabel_predF = self.HeteroAttLayerT(twore,twore.shape[1])#! category
            target_probs = F.softmax(clabel_predF, dim=-1)
            target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

            return target_probs
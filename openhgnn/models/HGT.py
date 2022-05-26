import dgl
import torch.nn as nn

from dgl.nn.pytorch.conv import HGTConv
from . import BaseModel, register_model
from ..utils import to_hetero_feat


@register_model('HGT')
class HGT(BaseModel):
    r"""Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    <https://arxiv.org/abs/2003.01332>`__

    For more details, you may refer to `HGT<https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.conv.HGTConv.html>`__
    
    Parameters
    ----------
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: list
        the list of the number of heads in each layer
    num_etypes: int
        the number of the edge type
    num_ntypes: int
        the number of the node type
    num_layers: int
        the number of layers we used in the computing
    dropout: float
        the feature drop rate
    norm: boolean
        if we need the norm operation
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        
        return cls(args.hidden_dim,
                   args.out_dim,
                   args.num_heads,
                   len(hg.etypes),
                   len(hg.ntypes),
                   args.num_layers,
                   args.dropout,
                   args.norm
                   )
    def __init__(self, in_dim, out_dim, num_heads, num_etypes, num_ntypes, 
                 num_layers, dropout = 0.2, norm = False):
        super(HGT, self).__init__()
        self.num_layers = num_layers
        self.hgt_layers = nn.ModuleList()
        self.hgt_layers.append(
            HGTConv(in_dim,
                    in_dim // num_heads,
                    num_heads,
                    num_ntypes,
                    num_etypes,
                    dropout,
                    norm)
        )
        
        for _ in range(1, num_layers - 1):
            self.hgt_layers.append(
                HGTConv(in_dim,
                        in_dim // num_heads,
                        num_heads,
                        num_ntypes,
                        num_etypes,
                        dropout,
                        norm)
            )
                   
        self.hgt_layers.append(
            HGTConv(in_dim,
                    out_dim,
                    1,
                    num_ntypes,
                    num_etypes,
                    dropout,
                    norm)
        )
        
        
    def forward(self, hg, h_dict):
        """
        The forward part of the HGT.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            g = dgl.to_homogeneous(hg, ndata = 'h')
            h = g.ndata['h']
            for l in range(self.num_layers):
                h = self.hgt_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], presorted = True)
                
        h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
        # hg = dgl.to_heterogeneous(g, hg.ntypes, hg.etypes)
        # h_dict = hg.ndata['h']

        return h_dict
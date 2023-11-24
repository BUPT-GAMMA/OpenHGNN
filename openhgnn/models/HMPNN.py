import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from . import BaseModel, register_model


@register_model("HMPNN")
class HMPNN(BaseModel):
    r"""
    Heterogeneous Message-Passing Neural Network (HMPNN)
    A simple implementation of HMPNN from paper for experimenting in KTN.
    Parameters
    ----------
    in_dim : int
        Input feature size.
    hid_dim : int
        Hidden layer size.
    out_dim : int
        Output feature size.
    etypes : list
        Edge types.
    num_layers : int
        Number of layers.
    device : str
        Device to run the model.
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return HMPNN(
            args.in_dim,
            args.hid_dim,
            args.hid_dim,
            hg.etypes,
            args.num_layers,
            args.device,
        )

    def __init__(self, in_dim, hid_dim, out_dim, etypes, num_layers, device):
        super(HMPNN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.etypes = etypes
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.device = device
        print("initing hmpnn model")
        self.layers.append(HMPNNLayer(in_dim, hid_dim, etypes, activation=F.relu)).to(
            self.device
        )
        for i in range(num_layers - 2):
            self.layers.append(
                HMPNNLayer(hid_dim, hid_dim, etypes, activation=F.relu)
            ).to(self.device)
        self.layers.append(HMPNNLayer(hid_dim, out_dim, etypes, activation=None)).to(
            self.device
        )

    def forward(self, hg, h_dict):
        if hasattr(hg, "ntypes"):
            for layer in self.layers:
                h_dict = layer(hg, h_dict)
        else:
            for layer, block in zip(self.layers, hg):
                block = block.to(self.device)
                h_dict = layer(block, h_dict)
        return h_dict

    def input_feature(self):
        return self.dataset.get_features()


class HMPNNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, etypes, activation=None):
        super(HMPNNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.etypes = etypes
        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(in_feat, out_feat, activation=activation)
                for rel in self.etypes
            },
            aggregate="sum",
        )

    def forward(self, g, h_dict):
        with g.local_scope():
            h_dict = self.conv(g, h_dict)
        return h_dict

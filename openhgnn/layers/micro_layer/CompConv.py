import torch as th
import torch.nn as nn
import dgl.function as fn
from dgl.utils import expand_as_pair
from openhgnn.utils.utils import ccorr
from dgl import DGLError

class CompConv(nn.Module):
    """
    Description
    -----------
    Composition-based convolution was introduced in `Composition-based Multi-Relational Graph Convolutional Networks
    <https://arxiv.org/abs/1911.03082>`__
    and mathematically is defined as follows:

    Parameters
    ----------
    comp_fn : str, one of 'sub', 'mul', 'ccorr'
    """
    def __init__(self,
                 comp_fn,
                 norm='right',
                 linear=False,
                 in_feats=None,
                 out_feats=None,
                 bias=False,
                 activation=None,
                 _allow_zero_in_degree=False):
        super(CompConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._norm = norm

        self.comp_fn = comp_fn
        if self.comp_fn == 'sub':
            self.aggregate = fn.u_sub_e('h', '_edge_weight', out='comp_h')
        elif self.comp_fn == 'mul':
            self.aggregate = fn.u_mul_e('h', '_edge_weight', out='comp_h')
        elif self.comp_fn == 'ccorr':
            self.aggregate = lambda edges: {'comp_h': ccorr(edges.src['h'], edges.data['_edge_weight'])}
        else:
            raise Exception('Only supports sub, mul, and ccorr')

        if linear:
            if in_feats is None or out_feats is None:
                raise DGLError('linear is True, so you must specify the in/out feats')
            else:
                self.Linear = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('Linear', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self._activation = activation
        self._allow_zero_in_degree = _allow_zero_in_degree

    def forward(self, graph, feat, h_e, Linear=None):
        r"""
        Description
        -----------
        Compute Composition-based  convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        Linear : a Linear nn.Module, optional
            Optional external weight tensor.
        h_e : torch.Tensor
            :math:`(1, D_{in})`
            means the edge type feature.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        The h_e is a tensor of size `(1, D_{in})`
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Linear shape: :math:`(\text{in_feats}, \text{out_feats})`.
                """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            graph.edata['_edge_weight'] = h_e.expand(graph.num_edges(), -1)
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            # Compute composition function in 4 steps
            # Step 1: compute composition by edge in the edge direction, and store results in edges.
            graph.apply_edges(self.aggregate)

            # Step 2: use extracted edge direction to compute in and out edges
            if Linear is not None:
                if self.Linear is not None:
                    raise DGLError('External Linear is provided while at the same time the'
                                   ' module has defined its own Linear module. Please'
                                   ' create the module with flag Linear=False.')
            else:
                Linear = self.Linear

            graph.edata['comp_h'] = Linear(graph.edata['comp_h'])
            # Step 3: sum comp results to both src and dst nodes
            graph.update_all(fn.copy_e('comp_h', 'm'), fn.sum('m', 'h'))
            rst = graph.dstdata['h']
            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm
            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

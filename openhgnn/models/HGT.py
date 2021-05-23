import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn
import math

from . import BaseModel, register_model, HeteroEmbedLayer
from ..utils import get_nodes_dict


@register_model('HGT')
class HGT(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        node_dict = {}
        edge_dict = {}
        for ntype in hg.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in hg.etypes:
            edge_dict[etype] = len(edge_dict)
            hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long).to(args.device) * edge_dict[etype]
        in_dim = hg.nodes['paper'].data['feat'].shape[1] + 1

        return cls(in_dim, args.hidden_dim, args.num_heads, len(hg.etypes), len(hg.ntypes), args.n_layers, args.num_heads)

    def __init__(self, in_size, out_size, n_heads, n_etypes, n_ntypes, n_layers, n_classes, dropout=0.2):
        super().__init__()

        self.adapt_ws = TypedLinear(n_ntypes, in_size, out_size)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(HGTConv(out_size, out_size, n_heads, n_etypes, n_ntypes, dropout))
            self.norms.append(TypedLayerNorm(n_ntypes, out_size))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_size, n_classes)

    def forward(self, g, x):
        g = g
        x = x
        h = self.dropout(torch.tanh(self.adapt_ws(x, g.ndata['ntype'])))
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = norm(conv(g, h), g.ndata['ntype'])
        return self.classifier(h)


class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''

    def __init__(self, n_hid, max_len=240):
        super(RelTemporalEncoding, self).__init__()

        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        self.emb = nn.Embedding(max_len, n_hid)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term) / (n_hid ** 0.5)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term) / (n_hid ** 0.5)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class TypedLinear(nn.Module):
    def __init__(self, num_types, in_size, out_size, bias=True):
        super().__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.zeros(num_types, out_size, in_size))
        if bias:
            self.b = nn.Parameter(torch.zeros(num_types, out_size))

        nn.init.xavier_uniform_(self.W)

    def forward(self, x, types):
        """
        x : float[b, ..., in_size]
        types : int64[b]
        """
        return iaddmm(x, types, self.W, self.b) if self.bias else imm(x, types, self.W)


class TypedLayerNorm(nn.Module):
    def __init__(self, num_types, size):
        super().__init__()

        self.W = nn.Parameter(torch.ones(num_types, size))
        self.b = nn.Parameter(torch.zeros(num_types, size))

    def forward(self, x, types):
        """
        x : float[b, ..., size]
        types : int64[b]
        """
        return typed_layer_norm(x, types, self.W, self.b)


class HGTMessage(nn.Module):
    def __init__(self, in_size, out_size, n_heads, n_etypes, n_ntypes):
        super().__init__()

        assert out_size % n_heads == 0
        self.d_k = out_size // n_heads
        self.n_heads = n_heads

        self.rte = RelTemporalEncoding(in_size)
        self.K = TypedLinear(n_ntypes, in_size, out_size)
        self.V = TypedLinear(n_ntypes, in_size, out_size)
        self.W_att = TypedLinear(n_etypes, self.d_k, self.d_k, bias=False)
        self.W_msg = TypedLinear(n_etypes, self.d_k, self.d_k, bias=False)
        self.mu = nn.Parameter(torch.ones(n_etypes, n_heads))

    def forward(self, edges):
        phi = edges.data['etype']  # int64[E]
        tau_s = edges.src['ntype']  # int64[E]
        tau_t = edges.dst['ntype']  # int64[E]
        h_s = edges.src['h']  # float32[E, in_size]
        Q_t = edges.dst['Q']  # float32[E, n_heads, d_k]
        dT = edges.data['dt']  # int64[E]

        # Step 1. compute relative time encoding
        # NOTE: although the figure in the paper writes something \hat{H}^{(l-1)}[s_1] which looks as
        # if it only depends on node s_1, it actually also depends on the target node t because the
        # relative time difference can be different if t differs.  This is confirmed in the original
        # HGT implementation where h_hat_s is computed inside the message function.
        h_hat_s = self.rte(h_s, dT)  # float32[E, in_size]

        # Step 2. compute source keys, and source messages
        # Target queries are computed in HGTConv to save memory
        K_s = self.K(h_hat_s, tau_s).view(-1, self.n_heads, self.d_k)  # float32[E, n_heads, d_k]
        V_s = self.V(h_hat_s, tau_s).view(-1, self.n_heads, self.d_k)  # float32[E, n_heads, d_k]
        M = self.W_msg(V_s, phi)  # float32[E, n_heads, d_k]

        # Step 3. compute attention logits over edges ATT-head.  It will be normalized with edge_softmax
        # later.
        # NOTE: the formulation K^i W_{ATT} Q^i is wrong.  We instead need the attention score computed on
        # the edges.  This is confirmed in the originial HGT implementation where the score is computed
        # with a sum reduction rather than a matrix multiplication.
        att_head_k = self.W_att(K_s, phi)
        att_head = (Q_t * att_head_k).sum(-1)  # float32[E, n_heads]
        att_head = att_head * self.mu[phi] / (self.d_k ** 0.5)  # float32[E, n_heads]

        return {'att_head': att_head, 'M': M}


class HGTConv(nn.Module):
    def __init__(self, in_size, out_size, n_heads, n_etypes, n_ntypes, dropout=0.2):
        super().__init__()

        assert in_size == out_size
        self.out_size = out_size
        self.d_k = out_size // n_heads
        self.n_heads = n_heads

        self.apply_edges = HGTMessage(in_size, out_size, n_heads, n_etypes, n_ntypes)
        self.A = TypedLinear(n_ntypes, out_size, out_size)
        self.Q = TypedLinear(n_ntypes, in_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(n_ntypes))

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h

            # Step 1-3
            g.ndata['Q'] = self.Q(h, g.ndata['ntype']).view(-1, self.n_heads, self.d_k)
            g.apply_edges(self.apply_edges)

            # Step 4. compute attention score and do weighted average of messages
            g.edata['att_score'] = dglnn.edge_softmax(g, g.edata['att_head'])
            g.edata['M'] = g.edata['M'] * g.edata['att_score'][..., None]
            g.update_all(fn.copy_e('M', 'M'), fn.sum('M', 'h_new'))
            h_new = g.ndata['h_new'].view(-1, self.out_size)

            # Step 5. update.
            # NOTE: the official implementation uses learnable gating rather than residual connections to combine
            # the aggregated messages and the node representation itself.
            h_new = self.dropout(self.A(F.gelu(h_new), g.ndata['ntype']))
            alpha = F.sigmoid(self.alpha[g.ndata['ntype']])[:, None]
            h_out = h_new * alpha + h * (1 - alpha)
            return h_out


class IAddMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, types, W, b):
        """
        x : float[B, ..., in_size]
        types : (0..n_types)[B]
        W : float[n_types, out_size, in_size]
        b : float[n_types, out_size]
        return : float[B, ..., out_size] = xW[types] + b[types]
        """
        B = x.shape[0]
        mid_size = x.shape[1:-1]
        mid_dims = len(mid_size)
        in_size = x.shape[-1]
        out_size = W.shape[1]

        y = torch.zeros(B, *mid_size, out_size, device=x.device)

        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            W_t = W[t]
            b_t = b[t]
            x_t = x[t_mask].unsqueeze(-1)
            y_t = (W_t @ x_t).squeeze(-1) + b_t
            y[t_mask] = y_t

        ctx.save_for_backward(x, types, W, b)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, types, W, b = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_W = torch.zeros_like(W)
        grad_b = torch.zeros_like(b)

        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            grad_y_t = grad_y[t_mask]
            grad_x[t_mask] = grad_y_t @ W[t]
            grad_W[t] = grad_y_t.view(-1, grad_y_t.shape[-1]).T @ x[t_mask].view(-1, x.shape[-1])
            grad_b[t] = grad_y_t.view(-1, b.shape[-1]).sum(0)
        return grad_x, None, grad_W, grad_b


def iaddmm(x, types, W, b):
    return IAddMM.apply(x, types, W, b)


class IMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, types, W):
        """
        x : float[B, ..., in_size]
        types : (0..n_types)[B]
        W : float[n_types, out_size, in_size]
        return : float[B, ..., out_size] = xW[types] + b[types]
        """
        B = x.shape[0]
        mid_size = x.shape[1:-1]
        mid_dims = len(mid_size)
        in_size = x.shape[-1]
        out_size = W.shape[1]

        y = torch.zeros(B, *mid_size, out_size, device=x.device)

        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            W_t = W[t]
            x_t = x[t_mask].unsqueeze(-1)
            y_t = (W_t @ x_t).squeeze(-1)
            y[t_mask] = y_t

        ctx.save_for_backward(x, types, W)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, types, W = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        grad_W = torch.zeros_like(W)

        for t in range(types.max().item() + 1):
            t_mask = (types == t)
            if not t_mask.any():
                continue
            grad_y_t = grad_y[t_mask]
            grad_x[t_mask] = grad_y_t @ W[t]
            grad_W[t] = grad_y_t.view(-1, grad_y_t.shape[-1]).T @ x[t_mask].view(-1, x.shape[-1])
        return grad_x, None, grad_W


def imm(x, types, W):
    return IMM.apply(x, types, W)


class TLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, types, w, b):
        ctx.save_for_backward(x, types, w, b)
        x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        w = w[types][(slice(None),) + (None,) * (x.dim() - 2)]
        b = b[types][(slice(None),) + (None,) * (x.dim() - 2)]
        return x * w + b

    @staticmethod
    def backward(ctx, grad_y):
        x, types, w, b = ctx.saved_tensors
        types_expanded = types[(slice(None),) + (None,) * (x.dim() - 1)].expand_as(grad_y)

        delta = x - x.mean(-1, keepdim=True)
        v = (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        x_norm = delta / v
        n = x.shape[-1]

        grad_w = torch.zeros(w.shape[0], *x.shape[1:], device=w.device)
        grad_w.scatter_add_(0, types_expanded, grad_y * x_norm)
        grad_w = torch.einsum('i...j->ij', grad_w)

        grad_b = torch.zeros(b.shape[0], *x.shape[1:], device=b.device)
        grad_b.scatter_add_(0, types_expanded, grad_y)
        grad_b = torch.einsum('i...j->ij', grad_b)

        w_t = w[types][(slice(None),) + (None,) * (x.dim() - 2)].expand_as(grad_y)
        grad_x = grad_y * w_t / v
        grad_x += -1 / n / v * torch.einsum('...ij,...ij->...i', grad_y, w_t)[..., None]
        grad_x += -1 / n / (v ** 3) * delta * torch.einsum('...ij,...ij,...ij->...i', grad_y, delta, w_t)[..., None]

        return grad_x, None, grad_w, grad_b


def typed_layer_norm(x, types, w, b):
    return TLayerNorm.apply(x, types, w, b)
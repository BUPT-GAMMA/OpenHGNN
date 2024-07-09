from functools import partial
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DropEdge
from dgl.nn.pytorch import MetaPath2Vec
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.optim import SparseAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from .base_model import BaseModel


def sce_loss(x, y, gamma=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(gamma)
    loss = loss.mean()
    return loss


class HGMAE(BaseModel):
    r"""
    **Title:** Heterogeneous Graph Masked Autoencoders

    **Authors:** Yijun Tian, Kaiwen Dong, Chunhui Zhang, Chuxu Zhang, Nitesh V. Chawla

    HGMAE was introduced in `[paper] <https://arxiv.org/abs/2208.09957>`_
    and parameters are defined as follows:

    Parameter
    ----------
    metapaths_dict: dict[str, list[etype]]
        Dict from meta path name to meta path.
    category : string
        The category of the nodes to be classificated.
    in_dim : int
        Dim of input feats
    hidden_dim : int
        Dim of encoded embedding.
    num_layers : int
        Number of layers of HAN encoder and decoder.
    num_heads : int
        Number of attention heads of hidden layers in HAN encoder and decoder.
    num_out_heads : int
        Number of attention heads of output projection in HAN encoder and decoder.
    feat_drop : float, optional
        Dropout rate on feature. Default: ``0``
    attn_drop : float, optional
        Dropout rate on attention weight. Default: ``0``
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.

    mp_edge_recon_loss_weight : float
        Trade-off weights for balancing mp_edge_recon_loss. Defaults: ``1.0``
    mp_edge_mask_rate : float
        Metapath-based edge masking rate. Defaults: ``0.6``
    mp_edge_gamma : float
        Scaling factor of mp_edge_recon_loss when using ``sce`` as loss function. Defaults: ``3.0``

    node_mask_rate : str
        Linearly increasing attribute mask rate to sample a subset of nodes, in the format of 'min,delta,max'. Defaults: ``'0.5,0.005,0.8'``
    attr_restore_loss_weight : float
        Trade-off weights for balancing attr_restore_loss. Defaults: ``1.0``
    attr_restore_gamma : float
        Scaling factor of att_restore_loss when using ``sce`` as loss function. Defaults: ``1.0``
    attr_replace_rate : float
        Replacing a percentage of mask tokens by random tokens, with the attr_replace_rate. Defaults: ``0.3``
    attr_unchanged_rate : float
        Leaving a percentage of nodes unchanged by utilizing the origin attribute, with the attr_unchanged_rate. Defaults: ``0.2``
    mp2vec_window_size : int
        In a random walk :attr:`w`, a node :attr:`w[j]` is considered close to a node :attr:`w[i]` if :attr:`i - window_size <= j <= i + window_size`. Defaults: ``3``
    mp2vec_rw_length : int
        The length of each random walk. Defaults: ``10``
    mp2vec_walks_per_node=args.mp2vec_walks_per_node,
        The number of walks to sample for each node. Defaults: ``2``

    mp2vec_negative_size: int
        Number of negative samples to use for each positive sample. Default: ``5``
    mp2vec_batch_size : int
        How many samples per batch to load when training mp2vec_feat. Defaults: ``128``
    mp2vec_train_epoch : int
        The training epochs of MetaPath2Vec model. Default: ``20``
    mp2vec_train_lr : float
        The training learning rate of MetaPath2Vec model. Default: ``0.001``
    mp2vec_feat_dim : int
        The feature dimension of MetaPath2Vec model. Defaults: ``128``
    mp2vec_feat_pred_loss_weight : float
        Trade-off weights for balancing mp2vec_feat_pred_loss. Defaults: ``0.1``
    mp2vec_feat_gamma: flaot
        Scaling factor of mp2vec_feat_pred_loss when using ``sce`` as loss function. Defaults: ``2.0``
    mp2vec_feat_drop: float
        The dropout rate of self.enc_out_to_mp2vec_feat_mapping. Defaults: ``0.2``
    """

    @classmethod
    def build_model_from_args(cls, args, hg, metapaths_dict: dict):
        return cls(
            hg=hg,
            metapaths_dict=metapaths_dict,
            category=args.category,
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_out_heads=args.num_out_heads,
            feat_drop=args.feat_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,

            # Metapath-based Edge Reconstruction
            mp_edge_recon_loss_weight=args.mp_edge_recon_loss_weight,
            mp_edge_mask_rate=args.mp_edge_mask_rate,
            mp_edge_gamma=args.mp_edge_gamma,

            # Type-specific Attribute Restoration
            node_mask_rate=args.node_mask_rate,
            attr_restore_gamma=args.attr_restore_gamma,
            attr_restore_loss_weight=args.attr_restore_loss_weight,
            attr_replace_rate=args.attr_replace_rate,
            attr_unchanged_rate=args.attr_unchanged_rate,

            # Positional Feature Prediction
            mp2vec_negative_size=args.mp2vec_negative_size,
            mp2vec_window_size=args.mp2vec_window_size,
            mp2vec_rw_length=args.mp2vec_rw_length,
            mp2vec_walks_per_node=args.mp2vec_walks_per_node,
            mp2vec_batch_size=args.mp2vec_batch_size,
            mp2vec_train_epoch=args.mp2vec_train_epoch,
            mp2vec_trian_lr=args.mp2vec_train_lr,
            mp2vec_feat_dim=args.mp2vec_feat_dim,
            mp2vec_feat_pred_loss_weight=args.mp2vec_feat_pred_loss_weight,
            mp2vec_feat_gamma=args.mp2vec_feat_gamma,
            mp2vec_feat_drop=args.mp2vec_feat_drop,

        )

    def __init__(self, hg, metapaths_dict, category,
                 in_dim, hidden_dim, num_layers, num_heads, num_out_heads,
                 feat_drop=0, attn_drop=0, negative_slope=0.2, residual=False,
                 mp_edge_recon_loss_weight=1, mp_edge_mask_rate=0.6, mp_edge_gamma=3,
                 attr_restore_loss_weight=1, attr_restore_gamma=1, node_mask_rate='0.5,0.005,0.8',
                 attr_replace_rate=0.3, attr_unchanged_rate=0.2,
                 mp2vec_window_size=3, mp2vec_negative_size=5, mp2vec_rw_length=10, mp2vec_walks_per_node=2,
                 mp2vec_feat_dim=128, mp2vec_feat_drop=0.2,
                 mp2vec_train_epoch=20, mp2vec_batch_size=128, mp2vec_trian_lr=0.001,
                 mp2vec_feat_pred_loss_weight=0.1, mp2vec_feat_gamma=2
                 ):
        super(HGMAE, self).__init__()
        self.metapaths_dict = metapaths_dict
        self.num_metapaths = len(metapaths_dict)
        self.category = category
        self.in_dim = in_dim  # original feat dim
        self.hidden_dim = hidden_dim  # emb dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_out_heads = num_out_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual

        assert self.hidden_dim % self.num_heads == 0
        assert self.hidden_dim % self.num_out_heads == 0

        # The input dimensions of the encoder and decoder are the same
        self.enc_dec_input_dim = self.in_dim

        # num head: encoder
        enc_hidden_dim = self.hidden_dim // self.num_heads
        enc_num_heads = self.num_heads

        # num head: decoder
        dec_hidden_dim = self.hidden_dim // self.num_out_heads
        dec_num_heads = self.num_out_heads

        dec_in_dim = self.hidden_dim

        # NOTE:
        # hidden_dim of HAN and hidden_dim of HGMAE are different,
        # the former one is the hidden_dim insides the HAN model,
        # the latter one is actually the dim of embeddings produced by the encoder insides the HGMAE,
        # The parameter hidden_dim refers specifically to the embedding produced by HGMAE encoder

        # encoder
        # actual output dim of encoder = out_dim * num_out_heads
        #                              = enc_hidden_dim * enc_num_heads
        #                              = hidden_dim (param, that is dim of emb)
        #                              = dec_in_dim
        # emb_dim of encoder = self.hidden_dim (param)
        self.encoder = HAN(
            num_metapaths=self.num_metapaths,
            in_dim=self.in_dim,
            hidden_dim=enc_hidden_dim,
            out_dim=enc_hidden_dim,
            num_layers=self.num_layers,
            num_heads=enc_num_heads,
            num_out_heads=enc_num_heads,
            feat_drop=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=nn.BatchNorm1d,
            activation=nn.PReLU(),
            encoding=True
        )

        # decoder
        self.decoder = HAN(
            num_metapaths=self.num_metapaths,
            in_dim=dec_in_dim,
            hidden_dim=dec_hidden_dim,
            out_dim=self.enc_dec_input_dim,
            num_layers=1,
            num_heads=dec_num_heads,
            num_out_heads=dec_num_heads,
            feat_drop=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            norm=nn.BatchNorm1d,
            activation=nn.PReLU(),
            encoding=False
        )

        self.__cached_gs = None  # cached metapath reachable graphs
        self.__cached_mps = None  # cached metapath adjacency matrices

        # Metapath-based Edge Reconstruction
        self.mp_edge_recon_loss_weight = mp_edge_recon_loss_weight
        self.mp_edge_mask_rate = mp_edge_mask_rate
        self.mp_edge_gamma = mp_edge_gamma
        self.mp_edge_recon_loss = partial(sce_loss, gamma=mp_edge_gamma)
        self.encoder_to_decoder_edge_recon = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # Type-specific Attribute Restoration
        self.attr_restore_gamma = attr_restore_gamma
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.enc_dec_input_dim))  # learnable mask token [M]
        self.encoder_to_decoder_attr_restore = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.attr_restore_loss = partial(sce_loss, gamma=attr_restore_gamma)
        self.attr_restore_loss_weight = attr_restore_loss_weight
        self.node_mask_rate = node_mask_rate

        assert attr_replace_rate + attr_unchanged_rate < 1, "attr_replace_rate + attr_unchanged_rate must " \
                                                            "be smaller than 1 "
        self.attr_unchanged_rate = attr_unchanged_rate
        self.attr_replace_rate = attr_replace_rate

        # Positional Feature Prediction
        self.mp2vec_feat_dim = mp2vec_feat_dim
        self.mp2vec_window_size = mp2vec_window_size
        self.mp2vec_negative_size = mp2vec_negative_size
        self.mp2vec_batch_size = mp2vec_batch_size
        self.mp2vec_train_lr = mp2vec_trian_lr
        self.mp2vec_train_epoch = mp2vec_train_epoch
        self.mp2vec_walks_per_node = mp2vec_walks_per_node
        self.mp2vec_rw_length = mp2vec_rw_length

        self.mp2vec_feat = None
        self.mp2vec_feat_pred_loss_weight = mp2vec_feat_pred_loss_weight
        self.mp2vec_feat_drop = mp2vec_feat_drop
        self.mp2vec_feat_gamma = mp2vec_feat_gamma
        self.mp2vec_feat_pred_loss = partial(sce_loss, gamma=self.mp2vec_feat_gamma)

        self.enc_out_to_mp2vec_feat_mapping = nn.Sequential(
            nn.Linear(dec_in_dim, self.mp2vec_feat_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim),
            nn.PReLU(),
            nn.Dropout(self.mp2vec_feat_drop),
            nn.Linear(self.mp2vec_feat_dim, self.mp2vec_feat_dim)
        )

    def train_mp2vec(self, hg):
        device = hg.device
        num_nodes = hg.num_nodes(self.category)

        # metapath for metapath2vec model
        Mp4Mp2Vec = []
        mp_nodes_seq = []
        for mp_name, mp in self.metapaths_dict.items():
            Mp4Mp2Vec += mp
            assert (mp[0][0] == mp[-1][-1]), "The start node type and the end one in metapath should be the same."

        x = max(self.mp2vec_rw_length // (len(Mp4Mp2Vec) + 1), 1)
        Mp4Mp2Vec *= x
        for mp in Mp4Mp2Vec:
            mp_nodes_seq.append(mp[0])
        mp_nodes_seq.append(mp[-1])
        assert (
                mp_nodes_seq[0] == mp_nodes_seq[-1]
        ), "The start node type and the end one in metapath should be the same."
        print("Metapath for training mp2vec models:", mp_nodes_seq)
        m2v_model = MetaPath2Vec(
            hg, Mp4Mp2Vec, self.mp2vec_window_size, self.mp2vec_feat_dim, self.mp2vec_negative_size
        ).to(device)
        m2v_model.train()
        dataloader = DataLoader(
            list(range(num_nodes)) * self.mp2vec_walks_per_node,
            batch_size=self.mp2vec_batch_size,
            shuffle=True,
            collate_fn=m2v_model.sample,
        )
        optimizer = SparseAdam(m2v_model.parameters(), lr=self.mp2vec_train_lr)
        scheduler = CosineAnnealingLR(optimizer, len(dataloader))
        for _ in tqdm(range(self.mp2vec_train_epoch)):
            for pos_u, pos_v, neg_v in dataloader:
                optimizer.zero_grad()
                loss = m2v_model(pos_u.to(device), pos_v.to(device), neg_v.to(device))
                loss.backward()
                optimizer.step()
                scheduler.step()

        # get the embeddings
        m2v_model.eval()
        nids = torch.LongTensor(m2v_model.local_to_global_nid[self.category]).to(device)
        emb = m2v_model.node_embed(nids)

        del m2v_model, nids, pos_u, pos_v, neg_v
        if device == "cuda":
            torch.cuda.empty_cache()
        return emb.detach()

    def get_mask_rate(self, input_mask_rate, get_min=False, epoch=None):
        try:
            return float(input_mask_rate)
        except ValueError:
            if "," in input_mask_rate:  # 0.6,-0.1,0.4 stepwise increment/decrement
                mask_rate = [float(i) for i in input_mask_rate.split(',')]
                assert len(
                    mask_rate) == 3, "input_mask_rate should be a float number (0-1), or in the format of 'min,delta," \
                                     "max', '0.6,-0.1,0.4', for example "
                start = mask_rate[0]
                step = mask_rate[1]
                end = mask_rate[2]
                if get_min:
                    return min(start, end)
                else:
                    cur_mask_rate = start + epoch * step
                    if cur_mask_rate < min(start, end) or cur_mask_rate > max(start, end):
                        return end
                    return cur_mask_rate
            else:
                raise NotImplementedError(
                    "input_mask_rate should be a float number (0-1), or in the format of 'min,delta,max', '0.6,-0.1,0.4', "
                    "for example")

    def normalize_feat(self, feat):
        rowsum = torch.sum(feat, dim=1).reshape(-1, 1)
        r_inv = torch.pow(rowsum, -1)
        r_inv = torch.where(torch.isinf(r_inv), 0, r_inv)
        return feat * r_inv

    def normalize_adj(self, adj):
        rowsum = torch.sum(adj, dim=1).reshape(-1, 1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), 0, d_inv_sqrt)
        return d_inv_sqrt * adj * d_inv_sqrt.T  # T?

    def get_mps(self, hg: dgl.DGLHeteroGraph):
        # mps: a list of metapath-based adjacency matrices
        if self.__cached_mps is None:
            self.__cached_mps = []
            mps = []
            for mp in self.metapaths_dict.values():
                adj = dgl.metapath_reachable_graph(hg, mp).adjacency_matrix()
                adj = self.normalize_adj(adj.to_dense()).to_sparse()  # torch_sparse
                mps.append(adj)
            self.__cached_mps = mps
        return self.__cached_mps.copy()

    def mps_to_gs(self, mps: list):
        # gs: a list of meta path reachable graphs that only contain topological structures
        # without edge and node features
        if self.__cached_gs is None:
            gs = []
            for mp in mps:
                indices = mp.indices()
                cur_graph = dgl.graph((indices[0], indices[1]))
                cur_graph = dgl.add_self_loop(cur_graph)
                gs.append(cur_graph)
            self.__cached_gs = gs
        return self.__cached_gs.copy()

    def mask_mp_edge_reconstruction(self, mps, feat, epoch):
        masked_gs = self.mps_to_gs(mps)
        cur_mp_edge_mask_rate = self.get_mask_rate(self.mp_edge_mask_rate, epoch=epoch)
        drop_edge = DropEdge(p=cur_mp_edge_mask_rate)
        for i in range(len(masked_gs)):
            masked_gs[i] = drop_edge(masked_gs[i])
            masked_gs[i] = dgl.add_self_loop(masked_gs[i])  # we need to add self loop
        enc_emb, _ = self.encoder(masked_gs, feat)
        emb_mapped = self.encoder_to_decoder_edge_recon(enc_emb)

        feat_recon, att_mp = self.decoder(masked_gs, emb_mapped)
        gs_recon = torch.mm(feat_recon, feat_recon.T)

        # loss = att_mp[0] * self.mp_edge_recon_loss(gs_recon, mps[0].to_dense())
        # for i in range(1, len(mps)):
        #     loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())

        loss = None
        for i in range(len(mps)):
            if loss is None:
                loss = att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())
            else:
                loss += att_mp[i] * self.mp_edge_recon_loss(gs_recon, mps[i].to_dense())

        return loss

    def encoding_mask_noise(self, feat, node_mask_rate=0.3):
        # We first sample a percentage of nodes from target node type ``self.category``, with node_mask_rate.
        # Specifically, we first replace a percentage of mask tokens
        # by random tokens, with the attr_replace_rate. In addition,
        # we select another percentage of nodes with attr_unchanged_rate and
        # leave them unchanged by utilizing the origin attribute xv,

        # mask: set nodes to 0.0
        # replace: replace nodes with random tokens
        # keep: leave nodes unchanged, remaining origin attr xv

        num_nodes = feat.shape[0]
        all_indices = torch.randperm(num_nodes, device=feat.device)

        # random masking
        num_mask_nodes = int(node_mask_rate * num_nodes)
        mask_indices = all_indices[:num_mask_nodes]
        keep_indices = all_indices[num_mask_nodes:]

        num_unchanged_nodes = int(self.attr_unchanged_rate * num_mask_nodes)
        num_noise_nodes = int(self.attr_replace_rate * num_mask_nodes)
        num_real_mask_nodes = num_mask_nodes - num_unchanged_nodes - num_noise_nodes

        perm_mask = torch.randperm(num_mask_nodes, device=feat.device)
        token_nodes = mask_indices[perm_mask[: num_real_mask_nodes]]
        noise_nodes = mask_indices[perm_mask[-num_noise_nodes:]]

        # token_nodes = mask_indices[: num_real_mask_nodes]
        # noise_nodes = mask_indices[-num_noise_nodes:]

        nodes_as_noise = torch.randperm(num_nodes, device=feat.device)[:num_noise_nodes]

        out_feat = feat.clone()
        out_feat[token_nodes] = 0.0
        out_feat[token_nodes] += self.enc_mask_token
        if num_nodes > 0:
            out_feat[noise_nodes] = feat[nodes_as_noise]

        return out_feat, (mask_indices, keep_indices)

    def mask_attr_restoration(self, gs, feat, epoch):
        cur_node_mask_rate = self.get_mask_rate(self.node_mask_rate, epoch=epoch)
        use_feat, (mask_nodes, keep_nodes) = self.encoding_mask_noise(feat, cur_node_mask_rate)
        enc_emb, _ = self.encoder(gs, use_feat)  # H3
        emb_mapped = self.encoder_to_decoder_attr_restore(enc_emb)

        # we apply another mask token[DM] to H3 before sending it into the decoder.
        emb_mapped[mask_nodes] = 0.0
        feat_recon, att_mp = self.decoder(gs, emb_mapped)

        feat_before_mask = feat[mask_nodes]
        feat_after_mask = feat_recon[mask_nodes]

        loss = self.attr_restore_loss(feat_before_mask, feat_after_mask)

        return loss, enc_emb

    def forward(self, hg: dgl.heterograph, h_dict, trained_mp2vec_feat_dict=None, epoch=None):
        assert epoch is not None, "epoch should be a positive integer"
        if trained_mp2vec_feat_dict is None:
            if self.mp2vec_feat is None:
                print("Training MetaPath2Vec feat by given metapaths_dict ")
                self.mp2vec_feat = self.train_mp2vec(hg)
                self.mp2vec_feat = self.normalize_feat(self.mp2vec_feat)
            mp2vec_feat = self.mp2vec_feat
        else:
            mp2vec_feat = trained_mp2vec_feat_dict[self.category]
        mp2vec_feat = mp2vec_feat.to(hg.device)

        feat = h_dict[self.category].to(hg.device)
        feat = self.normalize_feat(feat)
        mps = self.get_mps(hg)
        gs = self.mps_to_gs(mps)

        # TAR
        attr_restore_loss, enc_emb = self.mask_attr_restoration(gs, feat, epoch)
        loss = attr_restore_loss * self.attr_restore_loss_weight

        # MER
        mp_edge_recon_loss = self.mp_edge_recon_loss_weight * self.mask_mp_edge_reconstruction(mps, feat, epoch)
        loss += mp_edge_recon_loss

        # PFP
        mp2vec_feat_pred = self.enc_out_to_mp2vec_feat_mapping(enc_emb)  # H3
        mp2vec_feat_pred_loss = self.mp2vec_feat_pred_loss_weight * self.mp2vec_feat_pred_loss(mp2vec_feat_pred,
                                                                                               mp2vec_feat)
        loss += mp2vec_feat_pred_loss

        return loss

    def get_mp2vec_feat(self):
        return self.mp2vec_feat.detach()

    def get_embeds(self, hg, h_dict):
        feat = h_dict[self.category].to(hg.device)
        mps = self.get_mps(hg)
        gs = self.mps_to_gs(mps)
        emb, _ = self.encoder(gs, feat)
        return emb.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class HAN(nn.Module):
    def __init__(self,
                 num_metapaths,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 num_heads,
                 num_out_heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(HAN, self).__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.han_layers = nn.ModuleList()
        self.activation = activation

        last_activation = activation if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.han_layers.append(HANLayer(num_metapaths,
                                            in_dim, out_dim, num_out_heads,
                                            feat_drop, attn_drop, negative_slope, last_residual, last_activation,
                                            norm=last_norm))
        else:
            # input projection (no residual)
            self.han_layers.append(HANLayer(num_metapaths,
                                            in_dim, hidden_dim, num_heads,
                                            feat_drop, attn_drop, negative_slope, residual, self.activation,
                                            norm=norm,
                                            ))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.han_layers.append(HANLayer(num_metapaths,
                                                hidden_dim * num_heads, hidden_dim, num_heads,
                                                feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                norm=norm))
            # output projection
            self.han_layers.append(HANLayer(num_metapaths,
                                            hidden_dim * num_heads, out_dim, num_out_heads,
                                            feat_drop, attn_drop, negative_slope, last_residual,
                                            activation=last_activation, norm=last_norm))

    def forward(self, gs: list[dgl.DGLGraph], h, return_hidden=False):
        for gnn in self.han_layers:
            h, att_mp = gnn(gs, h)
        return h, att_mp


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        out_emb = (beta * z).sum(1)  # (N, D * K)
        att_mp = beta.mean(0).squeeze()

        return out_emb, att_mp


class HANLayer(nn.Module):
    def __init__(self, num_metapaths, in_dim, out_dim, num_heads,
                 feat_drop, attn_drop, negative_slope, residual, activation, norm):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_metapaths):
            self.gat_layers.append(GATConv_norm(
                in_dim, out_dim, num_heads,
                feat_drop, attn_drop, negative_slope, residual, activation, norm=norm))
        self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads)

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, new_g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))  # flatten because of att heads
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        out, att_mp = self.semantic_attention(semantic_embeddings)  # (N, D * K)

        return out, att_mp


class GATConv_norm(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 norm=None):
        super(GATConv_norm, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        """
            feat: Tensor of shape [num_nodes,feat_dim]
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError('There are 0-in-degree nodes in the graph, '
                                       'output for those nodes will be invalid. '
                                       'This is harmful for some applications, '
                                       'causing silent performance regression. '
                                       'Adding self-loop on the input graph by '
                                       'calling `g = dgl.add_self_loop(g)` will resolve '
                                       'the issue. Setting ``allow_zero_in_degree`` '
                                       'to be `True` when constructing this module will '
                                       'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # e[e == 0] = -1e3
            # e = graph.edata.pop('e')
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval

            rst = rst.flatten(1)

            # batchnorm
            if self.norm is not None:
                rst = self.norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

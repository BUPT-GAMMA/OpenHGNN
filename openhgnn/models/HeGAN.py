import torch
from collections import OrderedDict
import torch.nn as nn
from . import BaseModel, register_model


@register_model('HeGAN')
class HeGAN(BaseModel):
    r"""
    HeGAN was introduced in `Adversarial Learning on Heterogeneous Information Networks <https://dl.acm.org/doi/10.1145/3292500.3330970>`_

    It included a **Discriminator** and a **Generator**. For more details please read docs of both.

    Parameters
    ----------
    emb_size: int
        embedding size
    hg: dgl.heteroGraph
        hetorogeneous graph
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(args.emb_size, hg)

    def __init__(self, emb_size, hg):
        super().__init__()

        self.generator = Generator(emb_size, hg)
        self.discriminator = Discriminator(emb_size, hg)

    def forward(self, *args):
        pass

    # def predict(self, data):
    #     pass

    def extra_loss(self):
        pass


class Generator(nn.Module):
    r"""
     A Discriminator :math:`D` eveluates the connectivity between the pair of nodes :math:`u` and :math:`v` w.r.t. a relation :math:`r`. It is formulated as follow:

    .. math::
        D(\mathbf{e}_v|\mathbf{u},\mathbf{r};\mathbf{\theta}^D) = \frac{1}{1+\exp(-\mathbf{e}_u^{D^T}) \mathbf{M}_r^D \mathbf{e}_v}

    where :math:`e_v \in \mathbb{R}^{d\times 1}` is the input embeddings of the sample :math:`v`,
    :math:`e_u^D \in \mathbb{R}^{d \times 1}` is the learnable embedding of node :math:`u`,
    :math:`M_r^D \in \mathbb{R}^{d \times d}` is a learnable relation matrix for relation :math:`r`.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\mathbf{u}, \mathbf{r}; \mathbf{\theta}^G) = f(\mathbf{W_2}f(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) + \mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is :

    .. math::
        L_G = \mathbb{E}_{\langle u,v\rangle \sim P_G, e'_v \sim G(u,r;\theta^G)} = -\log -D(e'_v|u,r)) +\lambda^G || \theta^G ||_2^2

    where :math:`\theta^G` denote all the learnable parameters in Generator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """
    def __init__(self, emb_size, hg):
        super().__init__()
        self.n_relation = len(hg.etypes)
        self.node_emb_dim = emb_size

        self.nodes_embedding = nn.ParameterDict()
        for nodes_type, nodes_emb in hg.ndata['h'].items():
            self.nodes_embedding[nodes_type] = nn.Parameter(nodes_emb, requires_grad=True)

        self.relation_matrix = nn.ParameterDict()
        for et in hg.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)

        self.fc = nn.Sequential(
            OrderedDict([
                ("w_1", nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim, bias=True)),
                ("a_1", nn.LeakyReLU()),
                ("w_2", nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim)),
                ("a_2", nn.LeakyReLU())
            ])
        )

    def forward(self, gen_hg, dis_node_emb, dis_relation_matrix, noise_emb):
        r"""
        Parameters
        -----------
        gen_hg: dgl.heterograph
            sampled graph for generator.
        dis_node_emb: dict[str: Tensor]
            discriminator node embedding.
        dis_relation_matrix: dict[str: Tensor]
            discriminator relation embedding.
        noise_emb: dict[str: Tensor]
            noise embedding.
        """
        score_list = []
        with gen_hg.local_scope():
            self.assign_node_data(gen_hg, dis_node_emb)
            self.assign_edge_data(gen_hg, dis_relation_matrix)
            self.generate_neighbor_emb(gen_hg, noise_emb)
            for et in gen_hg.canonical_etypes:
                gen_hg.apply_edges(lambda edges: {'s': edges.src['dh'].unsqueeze(1).matmul(edges.data['de']).squeeze()}, etype=et)
                gen_hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['g'])}, etype=et)

                score = torch.sum(gen_hg.edata['score'].pop(et), dim=1)
                score_list.append(score)

        return torch.cat(score_list)

    def get_parameters(self):
        return {k: self.nodes_embedding[k] for k in self.nodes_embedding.keys()}

    def generate_neighbor_emb(self, hg, noise_emb):
        for et in hg.canonical_etypes:
            hg.apply_edges(lambda edges: {'g': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).squeeze()}, etype=et)
            hg.apply_edges(lambda edges: {'g': edges.data['g']+noise_emb[et]}, etype=et)
            hg.apply_edges(lambda edges: {'g': self.fc(edges.data['g'])}, etype=et)

        return {et: hg.edata['g'][et] for et in hg.canonical_etypes}

    def assign_edge_data(self, hg, dis_relation_matrix=None):
        for et in hg.canonical_etypes:
            n = hg.num_edges(et)
            e = self.relation_matrix[et[1]]
            hg.edata['e'] = {et: e.expand(n, -1, -1)}
            if dis_relation_matrix:
                de = dis_relation_matrix[et[1]]
                hg.edata['de'] = {et: de.expand(n, -1, -1)}

    def assign_node_data(self, hg, dis_node_emb=None):
        for nt in hg.ntypes:
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        if dis_node_emb:
            hg.ndata['dh'] = dis_node_emb


class Discriminator(nn.Module):
    r"""
    A generator :math:`G` samples fake node embeddings from a continuous distribution. The distribution is Gaussian distribution:

    .. math::
        \mathcal{N}(\mathbf{e}_u^{G^T} \mathbf{M}_r^G, \mathbf{\sigma}^2 \mathbf{I})

    where :math:`e_u^G \in \mathbb{R}^{d \times 1}` and :math:`M_r^G \in \mathbb{R}^{d \times d}` denote the node embedding of :math:`u \in \mathcal{V}` and the relation matrix of :math:`r \in \mathcal{R}` for the generator.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\mathbf{u}, \mathbf{r}; \mathbf{\theta}^G) = f(\mathbf{W_2}f(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) + \mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is:

    .. math::
        L_1^D = \mathbb{E}_{\langle u,v,r\rangle \sim P_G} = -\log D(e_v^u|u,r))

        L_2^D = \mathbb{E}_{\langle u,v\rangle \sim P_G, r' \sim P_{R'}} = -\log (1-D(e_v^u|u,r')))

        L_3^D = \mathbb{E}_{\langle u,v\rangle \sim P_G, e'_v \sim G(u,r;\theta^G)} = -\log (1-D(e_v'|u,r)))

        L_G = L_1^D + L_2^D + L_2^D + \lambda^D || \theta^D ||_2^2

    where :math:`\theta^D` denote all the learnable parameters in Discriminator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """
    def __init__(self, emb_size, hg):
        super().__init__()
        self.n_relation = len(hg.etypes)
        self.node_emb_dim = emb_size

        self.nodes_embedding = nn.ParameterDict()
        for nodes_type, nodes_emb in hg.ndata['h'].items():
            self.nodes_embedding[nodes_type] = nn.Parameter(nodes_emb, requires_grad=True)

        self.relation_matrix = nn.ParameterDict()
        for et in hg.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)

    def forward(self, pos_hg, neg_hg1, neg_hg2, generate_neighbor_emb):
        r"""
        Parameters
        ----------
        pos_hg:
            sampled postive graph.
        neg_hg1:
            sampled negative graph with wrong relation.
        neg_hg2:
            sampled negative graph wtih wrong node.
        generate_neighbor_emb:
            generator node embeddings.
        """
        self.assign_node_data(pos_hg)
        self.assign_node_data(neg_hg1)
        self.assign_node_data(neg_hg2, generate_neighbor_emb)
        self.assign_edge_data(pos_hg)
        self.assign_edge_data(neg_hg1)
        self.assign_edge_data(neg_hg2)

        pos_score = self.score_pred(pos_hg)
        neg_score1 = self.score_pred(neg_hg1)
        neg_score2 = self.score_pred(neg_hg2)

        return pos_score, neg_score1, neg_score2

    def get_parameters(self):
        r"""
        return discriminator node embeddings and relation embeddings.
        """
        return {k: self.nodes_embedding[k] for k in self.nodes_embedding.keys()}, \
               {k: self.relation_matrix[k] for k in self.relation_matrix.keys()}

    def score_pred(self, hg):
        r"""
        predict the discriminator score for sampled heterogeneous graph.
        """
        score_list = []
        with hg.local_scope():
            for et in hg.canonical_etypes:
                hg.apply_edges(lambda edges: {'s': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).reshape(hg.num_edges(et), 64)}, etype=et)
                if len(hg.edata['f']) == 0:
                    hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.dst['h'])}, etype=et)
                else:
                    hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['f'])}, etype=et)
                score = torch.sum(hg.edata['score'].pop(et), dim=1)
                score_list.append(score)
        return torch.cat(score_list)

    def assign_edge_data(self, hg):
        d = {}
        for et in hg.canonical_etypes:
            e = self.relation_matrix[et[1]]
            n = hg.num_edges(et)
            d[et] = e.expand(n, -1, -1)
        hg.edata['e'] = d

    def assign_node_data(self, hg, generate_neighbor_emb=None):
        for nt in hg.ntypes:
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        if generate_neighbor_emb:
            hg.edata['f'] = generate_neighbor_emb

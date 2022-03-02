import torch as th
from dgl.dataloading.negative_sampler import _BaseNegativeSampler, Uniform
from dgl import backend as F


class NegativeSampler(object):
    def __init__(self, g, k):
        # caches the probability distribution
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.etypes
        }
        self.k = k

    def __call__(self, g, eids_dict):
        result_dict = {}
        for etype, eids in eids_dict.items():
            src, _ = g.find_edges(eids, etype=etype)
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            result_dict[etype] = (src, dst)
        return result_dict


class Multinomial(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.

    For each edge ``(u, v)`` of type ``(srctype, etype, dsttype)``, DGL generates
    :attr:`k` pairs of negative edges ``(u, v')``, where ``v'`` is chosen
    uniformly from all the nodes of type ``dsttype``.  The resulting edges will
    also have type ``(srctype, etype, dsttype)``.

    Parameters
    ----------
    k : int
        The number of negative examples per edge.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
    >>> neg_sampler(g, [0, 1])
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """
    def __init__(self, g, k):
        self.weights = {
            etype: g.in_degrees(etype=etype).float() ** 0.75
            for etype in g.canonical_etype
        }
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        dst = self.weights[canonical_etype].multinomial(len(src), replacement=True)
        return src, dst


class Uniform_exclusive(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.

    For each edge ``(u, v)`` of type ``(srctype, etype, dsttype)``, DGL generates
    :attr:`k` pairs of negative edges ``(u, v')``, where ``v'`` is chosen
    uniformly from all the nodes of type ``dsttype``.  The resulting edges will
    also have type ``(srctype, etype, dsttype)``.

    Parameters
    ----------
    k : int
        The number of negative examples per edge.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)
    >>> neg_sampler(g, [0, 1])
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """
    def __init__(self, k):
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src_list = []
        dst_list = []
        for i in range(len(eids)):
            s = src[i]
            exp = g.successors(s, etype=canonical_etype)
            dst = th.randint(low=0, high=g.number_of_nodes(canonical_etype[2]), size=(self.k,))
            for d in range(len(dst)):
                while dst[d] in exp:
                    dst[d] = th.randint(low=0, high=g.number_of_nodes(canonical_etype[2]), size=(1,))
            s = s.repeat_interleave(self.k)
            src_list.append(s)
            dst_list.append(dst)
        return th.cat(src_list), th.cat(dst_list)
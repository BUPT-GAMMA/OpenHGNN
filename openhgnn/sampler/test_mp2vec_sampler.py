import dgl
from .random_walk_sampler import RandomWalkSampler
import torch

hg = dgl.heterograph({
    ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
    ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
    ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])})


def test_get_center_context_negatives():
    rw_walks = 5
    hetero_sampler = RandomWalkSampler(g=hg, metapath=['follow', 'view', 'viewed-by'] * 3,
                                       rw_walks=rw_walks, window_size=3, neg_size=5)
    for i in range(hg.num_nodes('user') * rw_walks):
        print(hetero_sampler.get_center_context_negatives(i))

    metapath = ['view', 'viewed-by']
    for i, elem in enumerate(metapath):
        if i == 0:
            adj = hg.adj(etype=elem)
        else:
            adj = torch.sparse.mm(adj, hg.adj(etype=elem))
    adj = adj.coalesce()
    g = dgl.graph(data=(adj.indices()[0], adj.indices()[1]))
    g.edata['rw_prob'] = adj.values()
    homo_sampler = RandomWalkSampler(g=g, rw_length=10, rw_walks=rw_walks, window_size=3, neg_size=5)

    for i in range(g.num_nodes() * rw_walks):
        print(homo_sampler.get_center_context_negatives(i))

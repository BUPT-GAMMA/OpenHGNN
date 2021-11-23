import dgl
from .mp2vec_sampler import Metapath2VecSampler

hg = dgl.heterograph({
    ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
    ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
    ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])})


def test_get_center_context_negatives():
    rw_walks = 5
    sampler = Metapath2VecSampler(hg=hg, metapath=['follow', 'view', 'viewed-by'], start_ntype='user', rw_length=3,
                                  rw_walks=rw_walks, window_size=3, neg_size=5)
    for i in range(hg.num_nodes('user') * rw_walks):
        print(sampler.get_center_context_negatives(i))

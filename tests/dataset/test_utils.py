from openhgnn.dataset import generate_random_hg


def test_generate_random_hg():
    num_edges_dict = {
        ('user', 'follows', 'user'): 20000,
        ('user', 'follows', 'topic'): 2000,
        ('user', 'plays', 'game'): 10000,
    }
    num_nodes_dict = {
        'game': 100,
        'topic': 10,
        'user': 1000,
    }
    hg = generate_random_hg(num_nodes_dict=num_nodes_dict, num_edges_dict=num_edges_dict)
    for etype, num in num_edges_dict.items():
        assert hg.num_edges(etype) == num
    for ntype, num in num_nodes_dict.items():
        assert hg.num_nodes(ntype) == num

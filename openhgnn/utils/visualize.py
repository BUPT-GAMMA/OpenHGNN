import numpy as np
from collections import Counter
import torch as th

__all__ = ['plot_degree_dist', 'plot_portion', 'plot_number_metapath']


def plot_portion(g, save_path=None, **kwargs):
    import matplotlib.pyplot as plt
    ntypes = g.ntypes
    num_nodes = []
    for ntype in ntypes:
        print(ntype)
        num_nodes.append(g.num_nodes(ntype))
        plt.style.use('ggplot')
    plt.pie(num_nodes, radius=1, autopct='%.3f%%', labels=ntypes, wedgeprops=dict(width=0.4))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_degree_dist(g, save_path=None, **kwargs):
    import matplotlib.pyplot as plt

    canonical_etypes = g.canonical_etypes
    ntypes = g.ntypes
    x_list = []
    y_list = []
    xscales = []
    yscales = []
    for ntype in ntypes:
        print(ntype)
        degrees = np.array([0] * g.num_nodes(ntype))
        for canonical_etype in canonical_etypes:
            etype = canonical_etype[1]
            if ntype in canonical_etype[0]:  # 出度
                out_degree = g.out_degrees(g.nodes(ntype), etype=etype)
                out_degree = out_degree.numpy()
                degrees += out_degree
            if ntype in canonical_etype[2]:  # 入度
                in_degree = g.in_degrees(g.nodes(ntype), etype=etype)
                in_degree = in_degree.numpy()
                degrees += in_degree
        degree_counts = Counter(degrees)
        x, y = zip(*degree_counts.items())
        xscales.append(max(x))
        yscales.append(max(y))
        x_list.append(x)
        y_list.append(y)

    plt.figure(1)
    plt.style.use('ggplot')
    # prep axes
    plt.xlabel('Degree')
    plt.xscale('log')
    plt.xlim(1, max(xscales))

    plt.ylabel('Number_of_Nodes')
    plt.yscale('log')
    plt.ylim(1, max(yscales))
    # do plot
    for i in range(len(ntypes)):
        plt.scatter(x_list[i], y_list[i], marker='*')
    plt.legend(ntypes)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_number_metapath(g, meta_paths_dict, save_path=None, **kwargs):
    import matplotlib.pyplot as plt

    meta_path_names = []
    meta_path_nums = []
    for meta_path_name, meta_path in meta_paths_dict.items():
        meta_path_names.append(meta_path_name)
        for i, etype in enumerate(meta_path):
            if i == 0:
                adj = g.adj(etype=etype)
            else:
                adj = th.sparse.mm(adj, g.adj(etype=etype))
        meta_path_nums.append(int(th.sparse.sum(adj)))

    plt.figure(1)
    plt.style.use('ggplot')
    # prep axes
    plt.xlabel('Meta-path')
    plt.ylabel('Number_of_Meta-path')
    plt.bar(meta_path_names, meta_path_nums,
            color=['gold', 'yellowgreen', 'lightseagreen', 'cornflowerblue', 'royalblue'],
            width=0.5)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

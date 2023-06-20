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
            if(isinstance(etype,list)):
                _new_type = (etype[0],etype[1],etype[2])
                etype = _new_type
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


def vis_emb(emb, label, save_path=None):
    """
    This is a demo of embedding visualization.
    Arguments:
        `emb` (numpy.ndarray): node embeddings 
        `label` (numpy.ndarray): node labels
        `save_path` (str): figure path to save. 
    
    After training is done, using like following in your trainer:
    ``` python
    from openhgnn.utils import vis_emb
    emb = self.model.get_emb()
    label = self.task.get_labels()
    vis_emb(emb, label, "path_to_save")
    ```
    """

    import matplotlib.pyplot as plt
    from sklearn import manifold

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(emb)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    color_values = label/len(set(label))
    plt.scatter(X_norm[:,0], X_norm[:,1], cmap=plt.get_cmap('jet'), c=color_values)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

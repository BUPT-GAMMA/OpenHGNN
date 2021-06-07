import os
import dgl
import torch as th
from dgl.data.utils import load_graphs, save_graphs


class coarsened_line_graph():

    def __init__(self, rw_len, batch_size, n_dataset, symmetric=True):
        self.rw_len = rw_len
        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.symmetric = symmetric # which means the original graph had inverse edges

        return

    def get_cl_graph(self, hg):
        fname = './openhgnn/output/RSHN/{}_cl_graoh_{}_{}.bin'.format(
            self.n_dataset, self.rw_len, self.batch_size)
        if os.path.exists(fname):
            g, _ = load_graphs(fname)
            return g[0]
        else:
            g = self.build_cl_graph(hg)
            save_graphs(fname, g)
            return g

    def init_cl_graph(self, cl_graph):
        cl_graph = give_one_hot_feats(cl_graph, 'h')

        cl_graph = dgl.remove_self_loop(cl_graph)
        edge_attr = cl_graph.edata['w'].type(th.FloatTensor).to(cl_graph.device)


        row, col = cl_graph.edges()
        for i in range(cl_graph.num_nodes()):
            mask = th.eq(row, i)
            edge_attr[mask] = th.nn.functional.normalize(edge_attr[mask], p=2, dim=0)

        # add_self_loop, set 1 as edge feature
        cl_graph = dgl.add_self_loop(cl_graph)
        edge_attr = th.cat([edge_attr, th.ones(cl_graph.num_nodes(), device=edge_attr.device)], dim=0)
        cl_graph.edata['w'] = edge_attr
        return cl_graph

    def build_cl_graph(self, hg):
        if not hg.is_homogeneous:
            self.num_edge_type = len(hg.etypes)
            g = dgl.to_homogeneous(hg).to('cpu')

        traces = self.random_walks(g)
        edge_batch = self.rw_map_edge_type(g, traces)
        cl_graph = self.edge2graph(edge_batch)
        return cl_graph

    def random_walks(self, g):
        source_nodes = th.randint(0, g.number_of_nodes(), (self.batch_size,))
        traces, _ = dgl.sampling.random_walk(g, source_nodes, length=self.rw_len-1)
        return traces

    def rw_map_edge_type(self, g, traces):
        edge_type = g.edata[dgl.ETYPE].long()
        edge_batch = []
        first_flag = True
        for t in traces:
            u = t[:-1]
            v = t[1:]
            edge_path = edge_type[g.edge_ids(u, v)].unsqueeze(0)
            if first_flag == True:
                edge_batch = edge_path
                first_flag = False
            else:
                edge_batch = th.cat((edge_batch, edge_path), dim=0)
        return edge_batch

    def edge2graph(self, edge_batch):

        u = edge_batch[:, :-1].reshape(-1)
        v = edge_batch[:, 1:].reshape(-1)
        if self.symmetric:
            tmp = u
            u = th.cat((u, v), dim=0)
            v = th.cat((v,tmp), dim=0)

        g = dgl.graph((u, v))
        sg = dgl.to_simple(g, return_counts='w')
        return sg


def give_one_hot_feats(g, ntype='h'):
    # if the nodes are featureless, the input feature is then the node id.
    num_nodes = g.num_nodes()
    #g.ndata[ntype] = th.arange(num_nodes, dtype=th.float32, device=g.device)
    g.ndata[ntype] = th.eye(num_nodes).to(g.device)
    return g
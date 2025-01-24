import dgl
import torch as th
import math
import random


class TransX_Sampler():
    def __init__(self, hg, args):
        random.seed(None)

        self.hg = hg.cpu()
        self.src, self.rel, self.dst = self._hg2triplets(self.hg)
        self.num_edges = self.src.shape[0]
        self.num_nodes = self.hg.num_nodes()

        self.batch_size = args.batch_size
        self.neg_size = args.neg_size
        self.batch_num = math.ceil(self.num_edges / self.batch_size)

        # for filter
        self.filter_set = self._build_filter_set(self.src, self.rel, self.dst)
        self.src, self.rel, self.dst = self._random_select_triplets(self.src, self.rel, self.dst,
                                                                    self.num_edges)  # tensor

        self.cur_idx = 0

    def _hg2triplets(self, hg):
        '''
        Attributes
        -------------
        input:
            hg : dgl.heterograph
                One Node type in hg, various edge types
        output:
            src, rel, dst : th.tensor, th.tensor, th.tensor
                #node, #edge, #node
        '''
        src, rel, dst = [], [], []
        if len(hg.ntypes) > 1 and len(hg.etypes) > 1:
            g = dgl.to_homogeneous(hg)
            src, dst = g.edges()
            rel = g.edata[dgl.ETYPE]
            pass

        else:
            for edge_type in hg.etypes:
                src_nx, dst_nx = hg.all_edges(etype=edge_type)
                rel_nx = th.full_like(src_nx, int(edge_type))
                src.append(src_nx)
                rel.append(rel_nx)
                dst.append(dst_nx)
            src = th.cat(src, dim=-1)
            rel = th.cat(rel, dim=-1)
            dst = th.cat(dst, dim=-1)
        return src, rel, dst

    def _random_select_triplets(self, src, rel, dst, num):
        l = list(range(len(src)))
        idx = random.sample(l, num)
        return src[idx], rel[idx], dst[idx]

    def get_pos(self):
        if self.cur_idx + self.batch_size < self.num_edges:
            edge_idx = th.arange(self.cur_idx, self.cur_idx + self.batch_size, dtype=th.int64)
            shuffle = False
        else:
            edge_idx = th.arange(self.cur_idx, self.num_edges, dtype=th.int64)
            shuffle = True
        self.cur_idx = (self.cur_idx + self.batch_size) % self.num_edges
        self.pos_g = (self.src[edge_idx], self.rel[edge_idx], self.dst[edge_idx])
        if shuffle:
            self._random_select_triplets(self.src, self.rel, self.dst, self.num_edges)
            edge_idx = 0
        return self.pos_g

    def get_neg(self):
        srcs, rels, dsts = self.pos_g
        srcs, rels, dsts = srcs.tolist(), rels.tolist(), dsts.tolist()
        node_list = list(range(self.num_nodes))
        replace_node = random.sample(node_list, self.neg_size)

        new_rel = [x for x in rels for i in range(self.neg_size)]
        new_src = [x for x in srcs for i in range(self.neg_size)]
        new_dst = [x for x in dsts for i in range(self.neg_size)]

        batch_size = len(rels)
        part_size = batch_size * self.neg_size

        rep_h = replace_node * batch_size
        for i in range(part_size):
            src, rel, dst = rep_h[i], new_rel[i], new_dst[i]
            while (src, rel, dst) in self.filter_set:
                rep_h[i] = src = random.randint(0, self.num_nodes - 1)

        rep_t = replace_node * batch_size
        for i in range(part_size):
            src, rel, dst = new_src[i], new_rel[i], rep_t[i]
            while (src, rel, dst) in self.filter_set:
                rep_t[i] = dst = random.randint(0, self.num_nodes - 1)

        new_src = rep_h + new_src
        new_rel = new_rel * 2
        new_dst = new_dst + rep_t

        return th.tensor(new_src), th.tensor(new_rel), th.tensor(new_dst)

    def _build_filter_set(self, srcs, rels, dsts):
        srcs, rels, dsts = srcs.tolist(), rels.tolist(), dsts.tolist()
        st = set()
        for i in range(len(srcs)):
            src, rel, dst = srcs[i], rels[i], dsts[i]
            st.add((src, rel, dst))
        return st

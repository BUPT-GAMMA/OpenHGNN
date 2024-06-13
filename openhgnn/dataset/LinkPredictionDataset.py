import os.path

import dgl
import math
import re
from copy import deepcopy
import numpy as np
import torch as th
import itertools
import random
from random import shuffle, choice
from collections import Counter
from os.path import join as joinpath
from os.path import isfile
from dgl.data.knowledge_graph import load_data
from . import BaseDataset, register_dataset
from . import AcademicDataset, HGBDataset, OHGBDataset, NBF_Dataset
from ..utils import add_reverse_edges
from collections import defaultdict
import os
from scipy.sparse import csr_matrix

__all__ = ['LinkPredictionDataset', 'HGB_LinkPrediction']


@register_dataset('link_prediction')
class LinkPredictionDataset(BaseDataset):
    """
    metric: Accuracy, multi-label f1 or multi-class f1. Default: `accuracy`
    """

    def __init__(self, *args, **kwargs):
        super(LinkPredictionDataset, self).__init__(*args, **kwargs)
        self.target_link = None
        self.target_link_r = None

    def get_split(self, val_ratio=0.1, test_ratio=0.2):
        """
        Get subgraphs for train, valid and test.
        Generally, the original will have train_mask and test_mask in edata, or we will split it automatically.

        If the original graph do not has the train_mask in edata, we default that there is no valid_mask and test_mask.
        So we will split the edges of the original graph into train/valid/test 0.7/0.1/0.2.

        The dataset has not validation_mask, so we split train edges randomly.
        Parameters
        ----------
        val_ratio : int
            The ratio of validation. Default: 0.1
        test_ratio : int
            The ratio of test. Default: 0.2

        Returns
        -------
        train_hg
        """

        val_edge_dict = {}
        test_edge_dict = {}
        out_ntypes = []
        train_graph = self.g
        for i, etype in enumerate(self.target_link):
            num_edges = self.g.num_edges(etype)
            if 'train_mask' not in self.g.edges[etype].data:
                """
                split edges into train/valid/test.
                """
                random_int = th.randperm(num_edges)
                val_index = random_int[:int(num_edges * val_ratio)]
                val_edge = self.g.find_edges(val_index, etype)
                test_index = random_int[int(num_edges * val_ratio):int(num_edges * (test_ratio + val_ratio))]
                test_edge = self.g.find_edges(test_index, etype)

                val_edge_dict[etype] = val_edge
                test_edge_dict[etype] = test_edge
                out_ntypes.append(etype[0])
                out_ntypes.append(etype[2])
                train_graph = dgl.remove_edges(train_graph, th.cat((val_index, test_index)), etype)
                # train_graph = dgl.remove_edges(train_graph, val_index, etype)
                if self.target_link_r is None:
                    pass
                else:
                    reverse_edge = self.target_link_r[i]
                    train_graph = dgl.remove_edges(train_graph, th.arange(train_graph.num_edges(reverse_edge)),
                                                   reverse_edge)
                    edges = train_graph.edges(etype=etype)
                    train_graph = dgl.add_edges(train_graph, edges[1], edges[0], etype=reverse_edge)

            else:
                if 'valid_mask' not in self.g.edges[etype].data:
                    train_idx = self.g.edges[etype].data['train_mask']
                    random_int = th.randperm(int(train_idx.sum( )))
                    val_index = random_int[:int(train_idx.sum( ) * val_ratio)]
                    val_edge = self.g.find_edges(val_index, etype)

                else:
                    val_mask = self.g.edges[etype].data['valid_mask'].squeeze( )
                    val_index = th.nonzero(val_mask).squeeze( )
                    val_edge = self.g.find_edges(val_index, etype)

                test_mask = self.g.edges[etype].data['test_mask'].squeeze( )
                test_index = th.nonzero(test_mask).squeeze( )
                test_edge = self.g.find_edges(test_index, etype)

                val_edge_dict[etype] = val_edge
                test_edge_dict[etype] = test_edge
                out_ntypes.append(etype[0])
                out_ntypes.append(etype[2])
                # self.val_label = train_graph.edges[etype[1]].data['label'][val_index]
                self.test_label = train_graph.edges[etype[1]].data['label'][test_index]
                train_graph = dgl.remove_edges(train_graph, th.cat((val_index, test_index)), etype)

        # train_graph = dgl.remove_edges(train_graph, th.cat((val_index, test_index)), 'item-user')
        self.out_ntypes = set(out_ntypes)
        val_graph = dgl.heterograph(val_edge_dict,
                                    {ntype: self.g.number_of_nodes(ntype) for ntype in set(out_ntypes)})
        test_graph = dgl.heterograph(test_edge_dict,
                                     {ntype: self.g.number_of_nodes(ntype) for ntype in set(out_ntypes)})

        # todo: val/test negative graphs should be created before training rather than
        #  create them dynamically in every evaluation.
        return train_graph, val_graph, test_graph, None, None


@register_dataset('demo_link_prediction')
class Test_LinkPrediction(LinkPredictionDataset):
    def __init__(self, dataset_name):
        super(Test_LinkPrediction, self).__init__( )
        self.g = self.load_HIN('./openhgnn/debug/data.bin')
        self.target_link = 'user-item'
        self.has_feature = False
        self.meta_paths_dict = None
        self.preprocess( )
        # self.generate_negative()

    def preprocess(self):
        test_mask = self.g.edges[self.target_link].data['test_mask']
        index = th.nonzero(test_mask).squeeze( )
        self.test_edge = self.g.find_edges(index, self.target_link)
        self.pos_test_graph = dgl.heterograph({('user', 'user-item', 'item'): self.test_edge},
                                              {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        self.g.remove_edges(index, self.target_link)
        self.g.remove_edges(index, 'item-user')
        self.neg_test_graph, _ = dgl.load_graphs('./openhgnn/debug/neg.bin')
        self.neg_test_graph = self.neg_test_graph[0]
        return

    def generate_negative(self):
        k = 99
        e = self.pos_test_graph.edges( )
        neg_src = []
        neg_dst = []
        for i in range(self.pos_test_graph.number_of_edges( )):
            src = e[0][i]
            exp = self.pos_test_graph.successors(src)
            dst = th.randint(high=self.g.number_of_nodes('item'), size=(k,))
            for d in range(len(dst)):
                while dst[d] in exp:
                    dst[d] = th.randint(high=self.g.number_of_nodes('item'), size=(1,))
            src = src.repeat_interleave(k)
            neg_src.append(src)
            neg_dst.append(dst)
        neg_edge = (th.cat(neg_src), th.cat(neg_dst))
        neg_graph = dgl.heterograph({('user', 'user-item', 'item'): neg_edge},
                                    {ntype: self.g.number_of_nodes(ntype) for ntype in ['user', 'item']})
        dgl.save_graphs('./openhgnn/debug/neg.bin', neg_graph)


@register_dataset('hin_link_prediction')
class HIN_LinkPrediction(LinkPredictionDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(HIN_LinkPrediction, self).__init__(*args, **kwargs)
        self.g = self.load_HIN(dataset_name)

    def load_link_pred(self, path):
        u_list = []
        v_list = []
        label_list = []
        with open(path) as f:
            for i in f.readlines( ):
                u, v, label = i.strip( ).split(', ')
                u_list.append(int(u))
                v_list.append(int(v))
                label_list.append(int(label))
        return u_list, v_list, label_list

    def load_HIN(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == 'academic4HetGNN':
            # which is used in HetGNN
            dataset = AcademicDataset(name='academic4HetGNN', raw_dir='')
            g = dataset[0].long( )

            self.train_batch = self.load_link_pred('./openhgnn/dataset/' + dataset_name + '/a_a_list_train.txt')
            self.test_batch = self.load_link_pred('./openhgnn/dataset/' + dataset_name + '/a_a_list_test.txt')
            self.category = 'author'
        elif dataset_name == 'Book-Crossing':
            g, _ = dgl.load_graphs('./openhgnn/dataset/book_graph.bin')
            g = g[0]
            self.target_link = [('user', 'user-item', 'item')]
            self.node_type = ['user', 'item']
        elif dataset_name == 'amazon4SLICE':
            dataset = AcademicDataset(name='amazon4SLICE', raw_dir='')
            g = dataset[0].long( )
            # self.target_link = [('product', 'product-1-product', 'product'),
            #                     ('product', 'product-2-product', 'product')]
            self.target_link = [('product', 'product-1-product', 'product')]
        elif dataset_name == 'MTWM':
            dataset = AcademicDataset(name='MTWM', raw_dir='')
            g = dataset[0].long( )
            g = add_reverse_edges(g)
            self.target_link = [('user', 'user-buy-spu', 'spu')]
            self.target_link_r = [('spu', 'user-buy-spu-rev', 'user')]
            self.meta_paths_dict = {
                'UPU1': [('user', 'user-buy-poi', 'poi'), ('poi', 'user-buy-poi-rev', 'user')],
                'UPU2': [('user', 'user-click-poi', 'poi'), ('poi', 'user-click-poi-rev', 'user')],
                'USU': [('user', 'user-buy-spu', 'spu'), ('spu', 'user-buy-spu-rev', 'user')],
                'UPSPU1': [('user', 'user-buy-poi', 'poi'), ('poi', 'poi-contain-spu', 'spu'),
                           ('spu', 'poi-contain-spu-rev', 'poi'), ('poi', 'user-buy-poi-rev', 'user')
                           ],
                'UPSPU2': [
                    ('user', 'user-click-poi', 'poi'), ('poi', 'poi-contain-spu', 'spu'),
                    ('spu', 'poi-contain-spu-rev', 'poi'), ('poi', 'user-click-poi-rev', 'user')
                ]
            }
            self.node_type = ['user', 'spu']
        elif dataset_name == 'HGBl-ACM':
            dataset = HGBDataset(name='HGBn-ACM', raw_dir='')
            g = dataset[0].long( )
            self.has_feature = True
            self.target_link = [('paper', 'paper-ref-paper', 'paper')]
            self.node_type = ['author', 'paper', 'subject', 'term']
            self.target_link_r = [('paper', 'paper-cite-paper', 'paper')]
            self.meta_paths_dict = {'PAP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')],
                                    'PSP': [('paper', 'paper-subject', 'subject'),
                                            ('subject', 'subject-paper', 'paper')],
                                    'PcPAP': [('paper', 'paper-cite-paper', 'paper'),
                                              ('paper', 'paper-author', 'author'),
                                              ('author', 'author-paper', 'paper')],
                                    'PcPSP': [('paper', 'paper-cite-paper', 'paper'),
                                              ('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')],
                                    'PrPAP': [('paper', 'paper-ref-paper', 'paper'),
                                              ('paper', 'paper-author', 'author'),
                                              ('author', 'author-paper', 'paper')],
                                    'PrPSP': [('paper', 'paper-ref-paper', 'paper'),
                                              ('paper', 'paper-subject', 'subject'),
                                              ('subject', 'subject-paper', 'paper')]
                                    }
        elif dataset_name == 'HGBl-DBLP':
            dataset = HGBDataset(name='HGBn-DBLP', raw_dir='')
            g = dataset[0].long( )
            self.has_feature = True
            self.target_link = [('author', 'author-paper', 'paper')]
            self.node_type = ['author', 'paper', 'venue', 'term']
            self.target_link_r = [('paper', 'paper-author', 'author')]
            self.meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    'APTPA': [('author', 'author-paper', 'paper'), ('paper', 'paper-term', 'term'),
                                              ('term', 'term-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    'APVPA': [('author', 'author-paper', 'paper'), ('paper', 'paper-venue', 'venue'),
                                              ('venue', 'venue-paper', 'paper'), ('paper', 'paper-author', 'author')],
                                    'PAP': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')],
                                    'PTP': [('paper', 'paper-term', 'term'), ('term', 'term-paper', 'paper')],
                                    'PVP': [('paper', 'paper-venue', 'venue'), ('venue', 'venue-paper', 'paper')],
                                    }

        elif dataset_name == 'HGBl-IMDB':
            dataset = HGBDataset(name='HGBn-IMDB', raw_dir='')
            g = dataset[0].long( )
            self.has_feature = True
            # self.target_link = [('author', 'author-paper', 'paper')]
            # self.node_type = ['author', 'paper', 'subject', 'term']
            # self.target_link_r = [('paper', 'paper-author', 'author')]
            self.target_link = [('actor', 'actor->movie', 'movie')]
            self.node_type = ['actor', 'director', 'keyword', 'movie']
            self.target_link_r = [('movie', 'movie->actor', 'actor')]
            self.meta_paths_dict = {
                'MAM': [('movie', 'movie->actor', 'actor'), ('actor', 'actor->movie', 'movie')],
                'MDM': [('movie', 'movie->director', 'director'), ('director', 'director->movie', 'movie')],
                'MKM': [('movie', 'movie->keyword', 'keyword'), ('keyword', 'keyword->movie', 'movie')],
                # 'DMD': [('director', 'director->movie', 'movie'), ('movie', 'movie->director', 'director')],
                # 'DMAMD': [('director', 'director->movie', 'movie'), ('movie', 'movie->actor', 'actor'),
                #           ('actor', 'actor->movie', 'movie'), ('movie', 'movie->director', 'director')],
                'AMA': [('actor', 'actor->movie', 'movie'), ('movie', 'movie->actor', 'actor')],
                'AMDMA': [('actor', 'actor->movie', 'movie'), ('movie', 'movie->director', 'director'),
                          ('director', 'director->movie', 'movie'), ('movie', 'movie->actor', 'actor')]
            }
        return g

    def get_split(self, val_ratio=0.1, test_ratio=0.2):
        if self.dataset_name == 'academic4HetGNN':
            return None, None, None, None, None
        else:
            return super(HIN_LinkPrediction, self).get_split(val_ratio, test_ratio)


@register_dataset('HGBl_link_prediction')
class HGB_LinkPrediction(LinkPredictionDataset):
    r"""
    The HGB dataset will be used in task *link prediction*.

    Dataset Name :
    HGBn-amazon/HGBn-LastFM/HGBn-PubMed

    So if you want to get more information, refer to
    `HGB datasets <https://github.com/THUDM/HGB>`_

    Attributes
    -----------
    has_feature : bool
        Whether the dataset has feature. Except HGBl-LastFM, others have features.
    target_link : list of tuple[canonical_etypes]
        The etypes of test link. HGBl-amazon has two etypes of test link. other has only one.

    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(HGB_LinkPrediction, self).__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.target_link_r = None
        if dataset_name == 'HGBl-amazon':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long( )
            self.has_feature = False
            self.target_link = [('product', 'product-product-0', 'product'),
                                ('product', 'product-product-1', 'product')]
            self.target_link_r = None
            self.link = [0, 1]
            self.node_type = ["product"]
            self.test_edge_type = {'product-product-0': 0, 'product-product-1': 1}
            self.meta_paths_dict = {
                'P0P': [('product', 'product-product-0', 'product'), ('product', 'product-product-1', 'product')],
                'P1P': [('product', 'product-product-1', 'product'), ('product', 'product-product-0', 'product')]
            }

        elif dataset_name == 'HGBl-LastFM':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long( )
            self.has_feature = False
            self.target_link = [('user', 'user-artist', 'artist')]
            self.node_type = ['user', 'artist', 'tag']
            self.test_edge_type = {'user-artist': 0}
            g = add_reverse_edges(g)
            self.target_link_r = [('artist', 'user-artist-rev', 'user')]
            self.meta_paths_dict = {'UU': [('user', 'user-user', 'user')],
                                    'UAU': [('user', 'user-artist', 'artist'), ('artist', 'user-artist-rev', 'user')],
                                    'UATAU': [('user', 'user-artist', 'artist'), ('artist', 'artist-tag', 'tag'),
                                              ('tag', 'artist-tag-rev', 'artist'),
                                              ('artist', 'user-artist-rev', 'user')],
                                    'AUA': [('artist', 'user-artist-rev', 'user'), ('user', 'user-artist', 'artist')],
                                    'ATA': [('artist', 'artist-tag', 'tag'), ('tag', 'artist-tag-rev', 'artist')]
                                    }

        elif dataset_name == 'HGBl-PubMed':
            dataset = HGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long( )
            self.has_feature = True
            self.target_link = [('1', '1_to_1', '1')]
            self.node_type = ['0', '1', '2', '3']
            self.test_edge_type = {'1_to_1': 2}
            g = add_reverse_edges(g)
            self.target_link_r = [('1', '1_to_1-rev', '1')]
            self.meta_paths_dict = {'101': [('1', '0_to_1-rev', '0'), ('0', '0_to_1', '1')],
                                    '111': [('1', '1_to_1', '1'), ('1', '1_to_1-rev', '1')],
                                    '121': [('1', '2_to_1-rev', '2'), ('2', '2_to_1', '1')],
                                    '131': [('1', '3_to_1-rev', '3'), ('3', '3_to_1', '1')]
                                    }

        self.g = g
        self.shift_dict = self.calculate_node_shift( )

    def load_link_pred(self, path):
        return

    def calculate_node_shift(self):
        node_shift_dict = {}
        count = 0
        for type in self.node_type:
            node_shift_dict[type] = count
            count += self.g.num_nodes(type)
        return node_shift_dict

    def get_split(self):
        r"""
        Get graphs for train, valid or test.

        The dataset has not validation_mask, so we split train edges randomly.
        """
        val_edge_dict = {}
        test_edge_dict = {}
        out_ntypes = []
        train_graph = self.g
        val_ratio = 0.1
        for i, etype in enumerate(self.target_link):
            train_mask = self.g.edges[etype].data['train_mask'].squeeze( )
            train_index = th.nonzero(train_mask).squeeze( )
            random_int = th.randperm(len(train_index))[:int(len(train_index) * val_ratio)]
            val_index = train_index[random_int]
            val_edge = self.g.find_edges(val_index, etype)

            test_mask = self.g.edges[etype].data['test_mask'].squeeze( )
            test_index = th.nonzero(test_mask).squeeze( )
            test_edge = self.g.find_edges(test_index, etype)

            val_edge_dict[etype] = val_edge
            test_edge_dict[etype] = test_edge
            out_ntypes.append(etype[0])
            out_ntypes.append(etype[2])

            train_graph = dgl.remove_edges(train_graph, th.cat((val_index, test_index)), etype)
            if self.target_link_r is None:
                pass
            else:
                train_graph = dgl.remove_edges(train_graph, th.cat((val_index, test_index)), self.target_link_r[i])
        self.out_ntypes = set(out_ntypes)
        val_graph = dgl.heterograph(val_edge_dict,
                                    {ntype: self.g.number_of_nodes(ntype) for ntype in set(out_ntypes)})
        test_graph = dgl.heterograph(test_edge_dict,
                                     {ntype: self.g.number_of_nodes(ntype) for ntype in set(out_ntypes)})

        return train_graph, val_graph, test_graph, None, None

    def save_results(self, hg, score, file_path):
        with hg.local_scope( ):
            src_list = []
            dst_list = []
            edge_type_list = []
            for etype in hg.canonical_etypes:
                edges = hg.edges(etype=etype)
                src_id = edges[0] + self.shift_dict[etype[0]]
                dst_id = edges[1] + self.shift_dict[etype[2]]
                src_list.append(src_id)
                dst_list.append(dst_id)
                edge_type_list.append(th.full((src_id.shape[0],), self.test_edge_type[etype[1]]))

            src_list = th.cat(src_list)
            dst_list = th.cat(dst_list)
            edge_type_list = th.cat(edge_type_list)
            with open(file_path, "w") as f:
                for l, r, edge_type, c in zip(src_list, dst_list, edge_type_list, score):
                    f.write(f"{l}\t{r}\t{edge_type}\t{round(float(c), 4)}\n")


@register_dataset('ohgb_link_prediction')
class OHGB_LinkPrediction(LinkPredictionDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(OHGB_LinkPrediction, self).__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.has_feature = True
        if dataset_name == 'ohgbl-MTWM':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long( )
            self.target_link = [('user', 'user-buy-spu', 'spu')]
            self.target_link_r = [('spu', 'user-buy-spu-rev', 'user')]
            self.node_type = ['user', 'spu']
        elif dataset_name == 'ohgbl-yelp1':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long( )
            self.target_link = [('user', 'user-buy-business', 'business')]
            self.target_link_r = [('business', 'user-buy-business-rev', 'user')]
        elif dataset_name == 'ohgbl-yelp2':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long( )
            self.target_link = [('business', 'described-with', 'phrase')]
            self.target_link_r = [('business', 'described-with-rev', 'phrase')]
        elif dataset_name == 'ohgbl-Freebase':
            dataset = OHGBDataset(name=dataset_name, raw_dir='')
            g = dataset[0].long( )
            self.target_link = [('BOOK', 'BOOK-and-BOOK', 'BOOK')]
            self.target_link_r = [('BOOK', 'BOOK-and-BOOK-rev', 'BOOK')]
        self.g = g

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose( )
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64'), norm.astype('int64')


def comp_deg_norm(g):
    g = g.local_var( )
    in_deg = g.in_degrees(range(g.number_of_nodes( ))).float( ).numpy( )
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

@register_dataset('kg_sub_link_prediction')
class KG_RedDataset(LinkPredictionDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(KG_RedDataset, self).__init__(*args, **kwargs)
        self.trans_dir = os.path.join('openhgnn/dataset/data', dataset_name)
        self.ind_dir = self.trans_dir + '_ind'

        folder = os.path.exists(self.trans_dir)
        if not folder:
            os.makedirs(self.trans_dir)
            url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v1.zip"
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
                myzip.extractall(self.trans_dir)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")

        folder = os.path.exists(self.ind_dir)
        if not folder:
            os.makedirs(self.ind_dir)
            # 下载数据
            url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/fb237_v1_ind.zip"
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
                myzip.extractall(self.ind_dir)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")

        with open(os.path.join(self.trans_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id[entity] = int(eid)

        with open(os.path.join(self.trans_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            id2relation = []
            for line in f:
                relation, rid = line.strip().split()
                self.relation2id[relation] = int(rid)
                id2relation.append(relation)

        with open(os.path.join(self.ind_dir, 'entities.txt')) as f:
            self.entity2id_ind = dict()
            for line in f:
                entity, eid = line.strip().split()
                self.entity2id_ind[entity] = int(eid)

        for i in range(len(self.relation2id)):
            id2relation.append(id2relation[i] + '_inv')
        id2relation.append('idd')
        self.id2relation = id2relation

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)
        self.n_ent_ind = len(self.entity2id_ind)

        self.tra_train = self.read_triples(self.trans_dir, 'train.txt')
        self.tra_valid = self.read_triples(self.trans_dir, 'valid.txt')
        self.tra_test = self.read_triples(self.trans_dir, 'test.txt')
        self.ind_train = self.read_triples(self.ind_dir, 'train.txt', 'inductive')
        self.ind_valid = self.read_triples(self.ind_dir, 'valid.txt', 'inductive')
        self.ind_test = self.read_triples(self.ind_dir, 'test.txt', 'inductive')

        self.val_filters = self.get_filter('valid')
        self.tst_filters = self.get_filter('test')

        for filt in self.val_filters:
            self.val_filters[filt] = list(self.val_filters[filt])
        for filt in self.tst_filters:
            self.tst_filters[filt] = list(self.tst_filters[filt])

        self.tra_KG, self.tra_sub = self.load_graph(self.tra_train)
        self.ind_KG, self.ind_sub = self.load_graph(self.ind_train, 'inductive')

        self.tra_train = np.array(self.tra_valid)
        self.tra_val_qry, self.tra_val_ans = self.load_query(self.tra_test)
        self.ind_val_qry, self.ind_val_ans = self.load_query(self.ind_valid)
        self.ind_tst_qry, self.ind_tst_ans = self.load_query(self.ind_test)
        self.valid_q, self.valid_a = self.tra_val_qry, self.tra_val_ans
        self.test_q, self.test_a = self.ind_val_qry + self.ind_tst_qry, self.ind_val_ans + self.ind_tst_ans

        self.n_train = len(self.tra_train)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)


    def read_triples(self, directory, filename, mode='transductive'):
        triples = []
        with open(os.path.join(directory, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                if mode == 'transductive':
                    h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                else:
                    h, r, t = self.entity2id_ind[h], self.relation2id[r], self.entity2id_ind[t]

                triples.append([h, r, t])
                triples.append([t, r + self.n_rel, h])
        return triples

    def load_graph(self, triples, mode='transductive'):
        n_ent = self.n_ent if mode == 'transductive' else self.n_ent_ind

        KG = np.array(triples)
        idd = np.concatenate([np.expand_dims(np.arange(n_ent), 1), 2 * self.n_rel * np.ones((n_ent, 1)),
                              np.expand_dims(np.arange(n_ent), 1)], 1)
        KG = np.concatenate([KG, idd], 0)

        n_fact = KG.shape[0]

        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])), shape=(n_fact, n_ent))
        return KG, M_sub

    def load_query(self, triples):
        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(lambda: list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)

        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='transductive'):
        # nodes: n_node x 2 with (batch_idx, node_idx)

        if mode == 'transductive':
            KG = self.tra_KG
            M_sub = self.tra_sub
            n_ent = self.n_ent
        else:
            KG = self.ind_KG
            M_sub = self.ind_sub
            n_ent = self.n_ent_ind

        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]],
                                       axis=1)  # (batch_idx, head, rela, tail)
        sampled_edges = th.LongTensor(sampled_edges)
        # index to nodes
        head_nodes, head_index = th.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = th.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = th.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data == 'train':
            return self.tra_train[batch_idx]
        if data == 'valid':
            # print(self.)
            query, answer = np.array(self.valid_q), self.valid_a  # np.array(self.valid_a)
            n_ent = self.n_ent
        if data == 'test':
            query, answer = np.array(self.test_q), self.test_a  # np.array(self.test_a)
            n_ent = self.n_ent_ind

        subs = []
        rels = []
        objs = []

        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self, ):
        rand_idx = np.random.permutation(self.n_train)
        self.tra_train = self.tra_train[rand_idx]

    def get_filter(self, data='valid'):
        filters = defaultdict(lambda: set())
        if data == 'valid':
            for triple in self.tra_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.tra_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        else:
            for triple in self.ind_train:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_valid:
                h, r, t = triple
                filters[(h, r)].add(t)
            for triple in self.ind_test:
                h, r, t = triple
                filters[(h, r)].add(t)
        return filters


@register_dataset('kg_subT_link_prediction')
class KG_RedTDataset(LinkPredictionDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(KG_RedTDataset, self).__init__(*args, **kwargs)
        self.task_dir = os.path.join('openhgnn/dataset/data', dataset_name)
        task_dir = self.task_dir
        folder = os.path.exists(self.task_dir)
        if not folder:
            os.makedirs(self.task_dir)
            url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/family.zip"
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
                myzip.extractall(self.task_dir)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")

        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1

        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel

        self.filters = defaultdict(lambda: set())

        self.fact_triple = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple = self.read_triples('test.txt')

        self.fact_data = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data = self.double_triple(self.test_triple)

        self.load_graph(self.fact_data)
        self.load_test_graph(self.double_triple(self.fact_triple) + self.double_triple(self.train_triple))

        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q, self.test_a = self.load_query(self.test_data)

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])

        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h, r, t])
                self.filters[(h, r)].add(t)
                self.filters[(t, r + self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r + self.n_rel, h])
        return triples + new_triples

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1), 2 * self.n_rel * np.ones((self.n_ent, 1)),
                              np.expand_dims(np.arange(self.n_ent), 1)], 1)

        self.KG = np.concatenate([np.array(triples), idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:, 0])),
                                shape=(self.n_fact, self.n_ent))

    def load_test_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1), 2 * self.n_rel * np.ones((self.n_ent, 1)),
                              np.expand_dims(np.arange(self.n_ent), 1)], 1)

        self.tKG = np.concatenate([np.array(triples), idd], 0)
        self.tn_fact = len(self.tKG)
        self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:, 0])),
                                 shape=(self.tn_fact, self.n_ent))

    def load_query(self, triples):
        triples.sort(key=lambda x: (x[0], x[1]))
        trip_hr = defaultdict(lambda: list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h, r)].append(t)

        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers

    def get_neighbors(self, nodes, mode='train'):
        if mode == 'train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub

        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]],
                                       axis=1)  # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data == 'train':
            return np.array(self.train_data)[batch_idx]
        if data == 'valid':
            query, answer = np.array(self.valid_q), self.valid_a
        if data == 'test':
            query, answer = np.array(self.test_q), self.test_a

        subs = []
        rels = []
        objs = []

        subs = query[batch_idx, 0]
        rels = query[batch_idx, 1]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self, ):
        fact_triple = np.array(self.fact_triple)
        train_triple = np.array(self.train_triple)
        all_triple = np.concatenate([fact_triple, train_triple], axis=0)
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]

        # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
        self.fact_data = self.double_triple(all_triple[:n_all * 3 // 4].tolist())
        self.train_data = np.array(self.double_triple(all_triple[n_all * 3 // 4:].tolist()))
        self.n_train = len(self.train_data)
        self.load_graph(self.fact_data)

@register_dataset('kg_link_prediction')
class KG_LinkPrediction(LinkPredictionDataset):
    """
    From `RGCN <https://arxiv.org/abs/1703.06103>`_, WN18 & FB15k face a data leakage.
    """

    def __init__(self, dataset_name, *args, **kwargs):
        super(KG_LinkPrediction, self).__init__(*args, **kwargs)
        if dataset_name in ['wn18', 'FB15k', 'FB15k-237']:
            dataset = load_data(dataset_name)
            g = dataset[0]
            self.num_rels = dataset.num_rels
            self.num_nodes = dataset.num_nodes

            self.train_hg, self.train_triplets = self._build_hg(g, 'train')
            self.valid_hg, self.valid_triplets = self._build_hg(g, 'valid')
            self.test_hg, self.test_triplets = self._build_hg(g, 'test')

            self.g = self.train_hg
            self.category = '_N'
            self.target_link = self.test_hg.canonical_etypes

    def _build_hg(self, g, mode):
        sub_g = dgl.edge_subgraph(g, g.edata[mode + '_edge_mask'], relabel_nodes=False)
        src, dst = sub_g.edges( )
        etype = sub_g.edata['etype']

        edge_dict = {}
        for i in range(self.num_rels):
            mask = (etype == i)
            edge_name = ('_N', str(i), '_N')
            edge_dict[edge_name] = (src[mask], dst[mask])
        hg = dgl.heterograph(edge_dict, {'_N': self.num_nodes})

        return hg, th.stack((src, etype, dst)).T

    def modify_size(self, eval_percent, dataset_type):
        if dataset_type == 'valid':
            self.valid_triplets = th.tensor(
                random.sample(self.valid_triplets.tolist( ), math.ceil(self.valid_triplets.shape[0] * eval_percent)))
        elif dataset_type == 'test':
            self.test_triplets = th.tensor(
                random.sample(self.test_triplets.tolist( ), math.ceil(self.test_triplets.shape[0] * eval_percent)))

    def get_graph_directed_from_triples(self, triples, format='graph'):
        s = th.LongTensor(triples[:, 0])
        r = th.LongTensor(triples[:, 1])
        o = th.LongTensor(triples[:, 2])
        if format == 'graph':
            edge_dict = {}
            for i in range(self.num_rels):
                mask = (r == i)
                edge_name = (self.category, str(i), self.category)
                edge_dict[edge_name] = (s[mask], o[mask])
            return dgl.heterograph(edge_dict, {self.category: self.num_nodes})

    def get_triples(self, g, mask_mode):
        '''
        :param g:
        :param mask_mode: should be one of 'train_mask', 'val_mask', 'test_mask
        :return:
        '''
        edges = g.edges( )
        etype = g.edata['etype']
        mask = g.edata.pop(mask_mode)
        return th.stack((edges[0][mask], etype[mask], edges[1][mask]))

    def get_all_triplets(self, dataset):
        train_data = th.LongTensor(dataset.train)
        valid_data = th.LongTensor(dataset.valid)
        test_data = th.LongTensor(dataset.test)
        return train_data, valid_data, test_data

    def get_split(self):
        return self.train_hg, self.valid_hg, self.test_hg, None, None

    def split_graph(self, g, mode='train'):
        """

        Parameters
        ----------
        g: DGLGraph
            a homogeneous graph fomat
        mode: str
            split the subgraph according to the mode

        Returns
        -------
        hg: DGLHeterograph
        """
        edges = g.edges( )
        etype = g.edata['etype']
        if mode == 'train':
            mask = g.edata['train_mask']
        elif mode == 'valid':
            mask = g.edata['valid_edge_mask']
        elif mode == 'test':
            mask = g.edata['test_edge_mask']
        hg = self.build_graph((edges[0][mask], edges[1][mask]), etype[mask])
        return hg

    def build_graph(self, edges, etype):
        edge_dict = {}
        for i in range(self.num_rels):
            mask = (etype == i)
            edge_name = (self.category, str(i), self.category)
            edge_dict[edge_name] = (edges[0][mask], edges[1][mask])
        hg = dgl.heterograph(edge_dict, {self.category: self.num_nodes})
        return hg

    def build_g(self, train):
        s = train[:, 0]
        r = train[:, 1]
        o = train[:, 2]
        edge_dict = {}
        for i in range(self.num_rels):
            mask = (r == i)
            edge_name = (self.category, str(i), self.category)
            edge_dict[edge_name] = (th.LongTensor(s[mask]), th.LongTensor(o[mask]))
        hg = dgl.heterograph(edge_dict, {self.category: self.num_nodes})
        return hg

import torch
import struct
import os
import json
import logging
from scipy.sparse import csc_matrix
from scipy.special import softmax
from tqdm import tqdm
import pickle
import scipy.sparse as ssp
import lmdb
import requests
import zipfile
import io
from torch.utils.data import Dataset
import networkx as nx
from ..utils.Grail_utils import *
class SubGraphDataset(Dataset):
    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='',
                 kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly= True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.add_traspose_rels = add_traspose_rels

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            #a = txn.get('max_n_label_sub'.encode())
            #print(a)
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        # logging.info('=====================')
        # logging.info(f"Subgraph size stats: \n Avg size {self.avg_subgraph_size}, \n Min size {self.min_subgraph_size}, \n Max size {self.max_subgraph_size}, \n Std {self.std_subgraph_size}")

        # logging.info('=====================')
        # logging.info(f"Enclosed nodes ratio stats: \n Avg size {self.avg_enc_ratio}, \n Min size {self.min_enc_ratio}, \n Max size {self.max_enc_ratio}, \n Std {self.std_enc_ratio}")

        # logging.info('=====================')
        # logging.info(f"# of pruned nodes stats: \n Avg size {self.avg_num_pruned_nodes}, \n Min size {self.min_num_pruned_nodes}, \n Max size {self.max_num_pruned_nodes}, \n Std {self.std_num_pruned_nodes}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        if not isinstance(self.graph, dgl.DGLGraph):
            subgraph = dgl.graph(self.graph.subgraph(nodes))
        else:
            subgraph = self.graph.subgraph(nodes)
        #subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        #print("请输出： ")
        #print(subgraph)
        #edges_btw_roots = subgraph.edge_id(0, 1, return_array=True)
        #edges_btw_roots = subgraph.edge_ids(0, 1)
        edges_btw_roots = torch.tensor([])
        try:
            edges_btw_roots = subgraph.edge_ids(torch.tensor([0]),torch.tensor([1]))
            # edges_btw_roots = np.array([edges_btw_roots])
        except:
            #print("Error")
            edges_btw_roots = torch.tensor([])
        edges_btw_roots = edges_btw_roots.numpy()
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph = dgl.add_edges(subgraph, 0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)



        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph


@register_dataset('grail_link_prediction')
class Grail_LinkPrediction(LinkPredictionDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super(Grail_LinkPrediction, self).__init__(*args, **kwargs)

        self.args = kwargs['args']
        self.args.db_path = f'./openhgnn/dataset/data/{self.args.dataset}/subgraphs_en_{self.args.enclosing_sub_graph}_neg_{self.args.num_neg_samples_per_link}_hop_{self.args.hop}'

        self.args.train_file = "train"
        self.args.valid_file = "valid"
        self.args.file_paths = {
            'train': './openhgnn/dataset/data/{}/{}.txt'.format(self.args.dataset, self.args.train_file),
            'valid': './openhgnn/dataset/data/{}/{}.txt'.format(self.args.dataset, self.args.valid_file)
        }

        relation2id_path = f'./openhgnn/dataset/data/{self.args.dataset}/relation2id.json'

        self.data_folder = f'./openhgnn/dataset/data/{self.args.dataset}'
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)  # makedirs 创建文件时如果路径不存在会创建这个路径
            url = f'https://github.com/kkteru/grail/blob/master/data/{self.args.dataset}'
            self.download_folder(url,self.data_folder)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")

        if not os.path.exists(self.data_folder+'_ind'):
            os.makedirs(self.data_folder+'_ind')  # makedirs 创建文件时如果路径不存在会创建这个路径
            url = f'https://github.com/kkteru/grail/blob/master/data/{self.args.dataset}_ind'
            self.download_folder(url,self.data_folder+'_ind')
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")

        if not os.path.isdir(self.args.db_path):
            generate_subgraph_datasets(self.args, relation2id_path)


        with open(relation2id_path) as f:
            self.relation2id = json.load(f)
        self.train = SubGraphDataset(self.args.db_path, 'train_pos', 'train_neg', self.args.file_paths,add_traspose_rels=self.args.add_traspose_rels,num_neg_samples_per_link=self.args.num_neg_samples_per_link,use_kge_embeddings=self.args.use_kge_embeddings, dataset=self.args.dataset,kge_model=self.args.kge_model, file_name=self.args.train_file)
        self.valid = SubGraphDataset(self.args.db_path, 'valid_pos', 'valid_neg', self.args.file_paths,
                                    add_traspose_rels=self.args.add_traspose_rels,
                                    num_neg_samples_per_link=self.args.num_neg_samples_per_link,
                                    use_kge_embeddings=self.args.use_kge_embeddings, dataset=self.args.dataset,
                                    kge_model=self.args.kge_model, file_name= self.args.valid_file)

    def download_folder(self,url, save_path):
        response = requests.get(url)
        if response.status_code == 200:
            # 确保保存路径存在
            os.makedirs(save_path, exist_ok=True)

            # 解析响应内容
            content = response.content.decode('utf-8')
            lines = content.splitlines()

            for line in lines:
                # 提取文件名
                file_name = line.split('/')[-1]

                # 构建文件的完整URL
                file_url = url + '/' + file_name

                # 构建文件的保存路径
                file_save_path = os.path.join(save_path, file_name)

                # 下载文件
                self.download_file(file_url, file_save_path)

    def download_file(self,url, save_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)

class kg_sampler( ):
    def __init__(self, ):
        self.sampler = 'uniform'
        return

    def generate_sampled_graph_and_labels(self, triplets, sample_size, split_size,
                                          num_rels, adj_list, degrees,
                                          negative_rate, sampler="uniform"):
        """Get training graph and signals
        First perform edge neighborhood sampling on graph, then perform negative
        sampling to generate negative samples
        """
        # perform edge neighbor sampling
        if self.sampler == "uniform":
            edges = sample_edge_uniform(adj_list, degrees, len(triplets), sample_size)
        elif self.sampler == "neighbor":
            edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
        else:
            raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

        # relabel nodes to have consecutive node ids
        edges = triplets[edges]
        src, rel, dst = edges.transpose( )
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose( )

        # negative sampling
        samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                            negative_rate)

        # further split graph, only half of the edges will be used as graph
        # structure, while the rest half is used as unseen positive samples
        split_size = int(sample_size * split_size)
        graph_split_ids = np.random.choice(np.arange(sample_size),
                                           size=split_size, replace=False)
        src = src[graph_split_ids]
        dst = dst[graph_split_ids]
        rel = rel[graph_split_ids]

        # build DGL graph
        print("# sampled nodes: {}".format(len(uniq_v)))
        print("# sampled edges: {}".format(len(src) * 2))
        g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                                 (src, rel, dst))
        return g, uniq_v, rel, norm, samples, labels


def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhool expansion.
    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    # initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


# --- ExpressGNN ---


# grounded rule stats code
BAD = 0  # sample not valid
FULL_OBSERVERED = 1  # sample valid, but rule contains only observed vars and does not have negation for all atoms
GOOD = 2  # sample valid


@register_dataset('express_gnn')
class ExpressGNNDataset(BaseDataset):
    def __init__(self, dataset_name, *args, **kwargs):
        super( ).__init__(*args, **kwargs)
        self.args = kwargs['args']
        self.PRED_DICT = {}
        self.dataset_name = dataset_name
        self.const_dict = ConstantDict()
        self.batchsize = self.args.batchsize
        self.shuffle_sampling = self.args.shuffle_sampling
        data_root = 'openhgnn'
        data_root = os.path.join(data_root, 'dataset')
        data_root = os.path.join(data_root, 'data')
        data_root = os.path.join(data_root, self.dataset_name)
        ext_rule_path = None
        folder = os.path.exists(data_root)
        print(data_root)
        print('folder', folder)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(data_root)  # makedirs 创建文件时如果路径不存在会创建这个路径
            # 下载数据
            url = f"https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{dataset_name}.zip"
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
                myzip.extractall(data_root)
            print("---  download data  ---")

        else:
            print("---  There is data!  ---")

        # Decide the way dataset will be load, set 1 to load FBWN dataset
        load_method = 0
        # print(dataset_name[0:13])
        if dataset_name[0:13] == 'EXP_FB15k-237':
            load_method = 1
        else:
            load_method = 0
        guss_fb = 'EXP_FB15k' in data_root
        if guss_fb != (load_method == 1):
            print("WARNING: set load_method to 1 if you load Freebase dataset, otherwise 0")

        # FBWN dataset
        if load_method == 1:
            fact_path_ls = [joinpath(data_root, 'facts.txt'),
                            joinpath(data_root, 'train.txt')]
            query_path = joinpath(data_root, 'test.txt')
            pred_path = joinpath(data_root, 'relations.txt')
            const_path = joinpath(data_root, 'entities.txt')
            valid_path = joinpath(data_root, 'valid.txt')

            rule_path = joinpath(data_root, 'cleaned_rules_weight_larger_than_0.9.txt')
            print(rule_path)
            print(os.getcwd())
            # print(fact_path_ls + [query_path, pred_path, const_path, valid_path, rule_path])
            # assert all(map(isfile, fact_path_ls + [query_path, pred_path, const_path, valid_path, rule_path]))

            # assuming only one type
            TYPE_SET.update(['type'])

            # add all const
            for line in iterline(const_path):
                self.const_dict.add_const('type', line)

            # add all pred
            for line in iterline(pred_path):
                self.PRED_DICT[line] = Predicate(line, ['type', 'type'])

            # add all facts
            fact_ls = []
            for fact_path in fact_path_ls:
                for line in iterline(fact_path):
                    parts = line.split('\t')

                    assert len(parts) == 3, print(parts)

                    e1, pred_name, e2 = parts

                    assert self.const_dict.has_const('type', e1) and self.const_dict.has_const('type', e2)
                    assert pred_name in self.PRED_DICT

                    fact_ls.append(Fact(pred_name, [e1, e2], 1))

            # add all validations
            valid_ls = []
            for line in iterline(valid_path):
                parts = line.split('\t')

                assert len(parts) == 3, print(parts)

                e1, pred_name, e2 = parts

                assert self.const_dict.has_const('type', e1) and self.const_dict.has_const('type', e2)
                assert pred_name in self.PRED_DICT

                valid_ls.append(Fact(pred_name, [e1, e2], 1))

            # add all queries
            query_ls = []
            for line in iterline(query_path):
                parts = line.split('\t')

                assert len(parts) == 3, print(parts)

                e1, pred_name, e2 = parts

                assert self.const_dict.has_const('type', e1) and self.const_dict.has_const('type', e2)
                assert pred_name in self.PRED_DICT

                query_ls.append(Fact(pred_name, [e1, e2], 1))

            # add all rules
            rule_ls = []
            strip_items = lambda ls: list(map(lambda x: x.strip( ), ls))
            first_atom_reg = re.compile(r'([\d.]+) (!?)([^(]+)\((.*)\)')
            atom_reg = re.compile(r'(!?)([^(]+)\((.*)\)')
            for line in iterline(rule_path):

                atom_str_ls = strip_items(line.split(' v '))
                assert len(atom_str_ls) > 1, 'rule length must be greater than 1, but get %s' % line

                atom_ls = []
                rule_weight = 0.0
                for i, atom_str in enumerate(atom_str_ls):
                    if i == 0:
                        m = first_atom_reg.match(atom_str)
                        assert m is not None, 'matching atom failed for %s' % atom_str
                        rule_weight = float(m.group(1))
                        neg = m.group(2) == '!'
                        pred_name = m.group(3).strip( )
                        var_name_ls = strip_items(m.group(4).split(','))
                    else:
                        m = atom_reg.match(atom_str)
                        assert m is not None, 'matching atom failed for %s' % atom_str
                        neg = m.group(1) == '!'
                        pred_name = m.group(2).strip( )
                        var_name_ls = strip_items(m.group(3).split(','))

                    atom = Atom(neg, pred_name, var_name_ls, self.PRED_DICT[pred_name].var_types)
                    atom_ls.append(atom)

                rule = Formula(atom_ls, rule_weight)
                rule_ls.append(rule)
        else:
            if dataset_name == 'Cora' or dataset_name == 'kinship':
                data_root = joinpath(data_root, 'S' + str(self.args.load_s))
            elif dataset_name == 'uw_cse':
                if self.args.load_s == 1:
                    data_root = joinpath(data_root, 'ai')
                elif self.args.load_s == 2:
                    data_root = joinpath(data_root, 'graphics')
                elif self.args.load_s == 3:
                    data_root = joinpath(data_root, 'language')
                elif self.args.load_s == 4:
                    data_root = joinpath(data_root, 'systems')
                elif self.args.load_s == 5:
                    data_root = joinpath(data_root, 'theory')
                else:
                    print('Warning: Invalid load_s')
            else:
                print('Warning: Invalid dataset for load_method = 0')
            rpath = joinpath(data_root, 'rules') if ext_rule_path is None else ext_rule_path
            fact_ls, rule_ls, query_ls = self.preprocess_kinship(joinpath(data_root, 'predicates'),
                                                                 joinpath(data_root, 'facts'),
                                                                 rpath,
                                                                 joinpath(data_root, 'queries'))
            valid_ls = []

        self.const_sort_dict = dict(
            [(type_name, sorted(list(self.const_dict[type_name]))) for type_name in self.const_dict.constants.keys( )])

        if load_method == 1:
            self.const2ind = dict([(const, i) for i, const in enumerate(self.const_sort_dict['type'])])

        # linear in size of facts
        self.fact_dict = dict((pred_name, set( )) for pred_name in self.PRED_DICT)
        self.test_fact_dict = dict((pred_name, set( )) for pred_name in self.PRED_DICT)
        self.valid_dict = dict((pred_name, set( )) for pred_name in self.PRED_DICT)

        self.ht_dict = dict((pred_name, [dict( ), dict( )]) for pred_name in self.PRED_DICT)
        self.ht_dict_train = dict((pred_name, [dict( ), dict( )]) for pred_name in self.PRED_DICT)

        def add_ht(pn, c_ls, ht_dict):
            if load_method == 0:
                if c_ls[0] in ht_dict[pn][0]:
                    ht_dict[pn][0][c_ls[0]].add(c_ls[0])
                else:
                    ht_dict[pn][0][c_ls[0]] = {c_ls[0]}
            elif load_method == 1:
                if c_ls[0] in ht_dict[pn][0]:
                    ht_dict[pn][0][c_ls[0]].add(c_ls[1])
                else:
                    ht_dict[pn][0][c_ls[0]] = {c_ls[1]}

                if c_ls[1] in ht_dict[pn][1]:
                    ht_dict[pn][1][c_ls[1]].add(c_ls[0])
                else:
                    ht_dict[pn][1][c_ls[1]] = {c_ls[0]}

        const_cnter = Counter()
        for fact in fact_ls:
            self.fact_dict[fact.pred_name].add((fact.val, tuple(fact.const_ls)))
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict)
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict_train)
            const_cnter.update(fact.const_ls)

        for fact in valid_ls:
            self.valid_dict[fact.pred_name].add((fact.val, tuple(fact.const_ls)))
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict)

        # the sorted list version
        self.fact_dict_2 = dict((pred_name, sorted(list(self.fact_dict[pred_name])))
                                for pred_name in self.fact_dict.keys( ))
        self.valid_dict_2 = dict((pred_name, sorted(list(self.valid_dict[pred_name])))
                                 for pred_name in self.valid_dict.keys( ))

        self.rule_ls = rule_ls

        # pred_atom-key dict
        self.atom_key_dict_ls = []
        for rule in self.rule_ls:
            atom_key_dict = dict( )

            for atom in rule.atom_ls:
                atom_dict = dict((var_name, dict( )) for var_name in atom.var_name_ls)

                for i, var_name in enumerate(atom.var_name_ls):

                    if atom.pred_name not in self.fact_dict:
                        continue

                    for v in self.fact_dict[atom.pred_name]:
                        if v[1][i] not in atom_dict[var_name]:
                            atom_dict[var_name][v[1][i]] = [v]
                        else:
                            atom_dict[var_name][v[1][i]] += [v]

                # happens if predicate occurs more than once in one rule then we merge the set
                if atom.pred_name in atom_key_dict:
                    for k, v in atom_dict.items( ):
                        if k not in atom_key_dict[atom.pred_name]:
                            atom_key_dict[atom.pred_name][k] = v
                else:
                    atom_key_dict[atom.pred_name] = atom_dict

            self.atom_key_dict_ls.append(atom_key_dict)

        self.test_fact_ls = []
        self.valid_fact_ls = []

        for fact in query_ls:
            self.test_fact_ls.append((fact.val, fact.pred_name, tuple(fact.const_ls)))
            self.test_fact_dict[fact.pred_name].add((fact.val, tuple(fact.const_ls)))
            add_ht(fact.pred_name, fact.const_ls, self.ht_dict)

        for fact in valid_ls:
            self.valid_fact_ls.append((fact.val, fact.pred_name, tuple(fact.const_ls)))
        self.num_rules = len(rule_ls)

        self.rule_gens = None
        self.reset( )

    def generate_gnd_pred(self, pred_name):
        """
            return a list of all instantiations of a predicate function, this can be extremely large
        :param pred_name:
            string
        :return:
        """

        assert pred_name in self.PRED_DICT

        pred = self.PRED_DICT[pred_name]
        subs = itertools.product(*[self.const_sort_dict[var_type] for var_type in pred.var_types])

        return [(pred_name, sub) for sub in subs]

    def generate_gnd_rule(self, rule):

        subs = itertools.product(*[self.const_sort_dict[rule.rule_vars[k]] for k in rule.rule_vars.keys( )])
        sub = next(subs, None)

        while sub is not None:

            latent_vars = []
            latent_neg_mask = []
            observed_neg_mask = []

            for atom in rule.atom_ls:
                grounding = tuple(sub[rule.key2ind[var_name]] for var_name in atom.var_name_ls)
                pos_gnding, neg_gnding = (1, grounding), (0, grounding)

                if pos_gnding in self.fact_dict[atom.pred_name]:
                    observed_neg_mask.append(0 if atom.neg else 1)
                elif neg_gnding in self.fact_dict[atom.pred_name]:
                    observed_neg_mask.append(1 if atom.neg else 0)
                else:
                    latent_vars.append((atom.pred_name, grounding))
                    latent_neg_mask.append(1 if atom.neg else 0)

            isfullneg = (sum(latent_neg_mask) == len(latent_neg_mask)) and \
                        (sum(observed_neg_mask) > 0)

            yield latent_vars, [latent_neg_mask, observed_neg_mask], isfullneg

            sub = next(subs, None)

    def get_batch(self, epoch_mode=False, filter_latent=True):
        """
            return the ind-th batch of ground formula and latent variable indicators
        :return:

        Parameters
        ----------
        filter_latent
        epoch_mode
        """

        batch_neg_mask = [[] for _ in range(len(self.rule_ls))]
        batch_latent_var_inds = [[] for _ in range(len(self.rule_ls))]
        observed_rule_cnts = [0.0 for _ in range(len(self.rule_ls))]
        flat_latent_vars = dict( )

        cnt = 0

        inds = list(range(len(self.rule_ls)))

        while cnt < self.batchsize:

            if self.shuffle_sampling:
                shuffle(inds)

            hasdata = False
            for ind in inds:
                latent_vars, neg_mask, isfullneg = next(self.rule_gens[ind], (None, None, None))

                if latent_vars is None:
                    if epoch_mode:
                        continue
                    else:
                        self.rule_gens[ind] = self.generate_gnd_rule(self.rule_ls[ind])
                        latent_vars, neg_mask, isfullneg = next(self.rule_gens[ind])

                if epoch_mode:
                    hasdata = True

                # if rule is fully latent
                if (len(neg_mask[1]) == 0) and filter_latent:
                    continue

                # if rule fully observed
                if len(latent_vars) == 0:
                    observed_rule_cnts[ind] += 0 if isfullneg else 1
                    cnt += 1
                    if cnt >= self.batchsize:
                        break
                    else:
                        continue

                batch_neg_mask[ind].append(neg_mask)

                for latent_var in latent_vars:
                    if latent_var not in flat_latent_vars:
                        flat_latent_vars[latent_var] = len(flat_latent_vars)

                batch_latent_var_inds[ind].append([flat_latent_vars[e] for e in latent_vars])

                cnt += 1

                if cnt >= self.batchsize:
                    break

            if epoch_mode and (hasdata is False):
                break

        flat_list = sorted([(k, v) for k, v in flat_latent_vars.items( )], key=lambda x: x[1])
        flat_list = [e[0] for e in flat_list]

        return batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts

    def _instantiate_pred(self, atom, atom_dict, sub, rule, observed_prob):

        key2ind = rule.key2ind
        rule_vars = rule.rule_vars

        # substitute with observed fact
        if np.random.rand( ) < observed_prob:

            fact_choice_set = None
            for var_name in atom.var_name_ls:
                const = sub[key2ind[var_name]]
                if const is None:
                    choice_set = itertools.chain.from_iterable([v for k, v in atom_dict[var_name].items( )])
                else:
                    if const in atom_dict[var_name]:
                        choice_set = atom_dict[var_name][const]
                    else:
                        choice_set = []

                if fact_choice_set is None:
                    fact_choice_set = set(choice_set)
                else:
                    fact_choice_set = fact_choice_set.intersection(set(choice_set))

                if len(fact_choice_set) == 0:
                    break

            if len(fact_choice_set) == 0:
                for var_name in atom.var_name_ls:
                    if sub[key2ind[var_name]] is None:
                        sub[key2ind[var_name]] = choice(self.const_sort_dict[rule_vars[var_name]])
            else:
                val, const_ls = choice(sorted(list(fact_choice_set)))
                for var_name, const in zip(atom.var_name_ls, const_ls):
                    sub[key2ind[var_name]] = const

        # substitute with random facts
        else:
            for var_name in atom.var_name_ls:
                if sub[key2ind[var_name]] is None:
                    sub[key2ind[var_name]] = choice(self.const_sort_dict[rule_vars[var_name]])

    def _gen_mask(self, rule, sub, closed_world):

        latent_vars = []
        observed_vars = []
        latent_neg_mask = []
        observed_neg_mask = []

        for atom in rule.atom_ls:
            grounding = tuple(sub[rule.key2ind[var_name]] for var_name in atom.var_name_ls)
            pos_gnding, neg_gnding = (1, grounding), (0, grounding)

            if pos_gnding in self.fact_dict[atom.pred_name]:
                observed_vars.append((1, atom.pred_name))
                observed_neg_mask.append(0 if atom.neg else 1)
            elif neg_gnding in self.fact_dict[atom.pred_name]:
                observed_vars.append((0, atom.pred_name))
                observed_neg_mask.append(1 if atom.neg else 0)
            else:
                if closed_world and (len(self.test_fact_dict[atom.pred_name]) == 0):
                    observed_vars.append((0, atom.pred_name))
                    observed_neg_mask.append(1 if atom.neg else 0)
                else:
                    latent_vars.append((atom.pred_name, grounding))
                    latent_neg_mask.append(1 if atom.neg else 0)

        return latent_vars, observed_vars, latent_neg_mask, observed_neg_mask

    def _get_rule_stat(self, observed_vars, latent_vars, observed_neg_mask, filter_latent, filter_observed):

        is_full_latent = len(observed_vars) == 0
        is_full_observed = len(latent_vars) == 0

        if is_full_latent and filter_latent:
            return BAD

        if is_full_observed:

            if filter_observed:
                return BAD

            is_full_neg = sum(observed_neg_mask) == 0

            if is_full_neg:
                return BAD

            else:
                return FULL_OBSERVERED

        # if observed var already yields 1
        if sum(observed_neg_mask) > 0:
            return BAD

        return GOOD

    def _inst_var(self, sub, var2ind, var2type, at, ht_dict, gen_latent):

        if len(at.var_name_ls) != 2:
            raise KeyError

        must_latent = gen_latent

        if must_latent:

            tmp = [sub[var2ind[vn]] for vn in at.var_name_ls]

            for i, subi in enumerate(tmp):
                if subi is None:
                    tmp[i] = random.choice(self.const_sort_dict[var2type[at.var_name_ls[i]]])

            islatent = (tmp[0] not in ht_dict[0]) or (tmp[1] not in ht_dict[0][tmp[0]])
            for i, vn in enumerate(at.var_name_ls):
                sub[var2ind[vn]] = tmp[i]
            return [self.const2ind[subi] for subi in tmp], islatent, islatent or at.neg

        vn0 = at.var_name_ls[0]
        sub0 = sub[var2ind[vn0]]
        vn1 = at.var_name_ls[1]
        sub1 = sub[var2ind[vn1]]

        if sub0 is None:

            if sub1 is None:
                if len(ht_dict[0]) > 0:
                    sub0 = random.choice(tuple(ht_dict[0].keys( )))
                    sub1 = random.choice(tuple(ht_dict[0][sub0]))
                    sub[var2ind[vn0]] = sub0
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.neg

            else:
                if sub1 in ht_dict[1]:
                    sub0 = random.choice(tuple(ht_dict[1][sub1]))
                    sub[var2ind[vn0]] = sub0
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.neg
                else:
                    sub0 = random.choice(self.const_sort_dict[var2type[vn0]])
                    sub[var2ind[vn0]] = sub0
                    return [self.const2ind[sub0], self.const2ind[sub1]], True, True

        else:

            if sub1 is None:
                if sub0 in ht_dict[0]:
                    sub1 = random.choice(tuple(ht_dict[0][sub0]))
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], False, at.neg
                else:
                    sub1 = random.choice(self.const_sort_dict[var2type[vn1]])
                    sub[var2ind[vn1]] = sub1
                    return [self.const2ind[sub0], self.const2ind[sub1]], True, True

            else:
                islatent = (sub0 not in ht_dict[0]) or (sub1 not in ht_dict[0][sub0])
                return [self.const2ind[sub0], self.const2ind[sub1]], islatent, islatent or at.neg

    def get_batch_fast(self, batchsize, observed_prob=0.9):

        prob_decay = 0.5

        for rule in self.rule_ls:

            var2ind = rule.key2ind
            var2type = rule.rule_vars
            samples = [[atom.pred_name, []] for atom in rule.atom_ls]
            neg_mask = [[atom.pred_name, []] for atom in rule.atom_ls]
            latent_mask = [[atom.pred_name, []] for atom in rule.atom_ls]
            obs_var = [[atom.pred_name, []] for atom in rule.atom_ls]

            cnt = 0
            while cnt <= batchsize:

                sub = [None] * len(rule.rule_vars)  # substitutions

                sample_buff = [[] for _ in rule.atom_ls]
                neg_mask_buff = [[] for _ in rule.atom_ls]
                latent_mask_buff = [[] for _ in rule.atom_ls]

                atom_inds = list(range(len(rule.atom_ls)))
                shuffle(atom_inds)
                succ = True
                cur_threshold = observed_prob
                obs_list = []

                for atom_ind in atom_inds:
                    atom = rule.atom_ls[atom_ind]
                    pred_ht_dict = self.ht_dict_train[atom.pred_name]

                    gen_latent = np.random.rand( ) > cur_threshold
                    c_ls, islatent, atom_succ = self._inst_var(sub, var2ind, var2type,
                                                               atom, pred_ht_dict, gen_latent)

                    if not islatent:
                        obs_var[atom_ind][1].append(c_ls)

                    cur_threshold *= prob_decay
                    succ = succ and atom_succ
                    obs_list.append(not islatent)

                    if succ:
                        sample_buff[atom_ind].append(c_ls)
                        latent_mask_buff[atom_ind].append(1 if islatent else 0)
                        neg_mask_buff[atom_ind].append(0 if atom.neg else 1)

                if succ and any(obs_list):
                    for i in range(len(rule.atom_ls)):
                        samples[i][1].extend(sample_buff[i])
                        latent_mask[i][1].extend(latent_mask_buff[i])
                        neg_mask[i][1].extend(neg_mask_buff[i])

                cnt += 1

            yield samples, neg_mask, latent_mask, obs_var

    def get_batch_by_q(self, batchsize, observed_prob=1.0, validation=False):

        samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        cnt = 0

        num_ents = len(self.const2ind)
        ind2const = self.const_sort_dict['type']

        def gen_fake(c1, c2, pn):
            for _ in range(10):
                c1_fake = random.randint(0, num_ents - 1)
                c2_fake = random.randint(0, num_ents - 1)
                if np.random.rand( ) > 0.5:
                    if ind2const[c1_fake] not in self.ht_dict_train[pn][1][ind2const[c2]]:
                        return c1_fake, c2
                else:
                    if ind2const[c2_fake] not in self.ht_dict_train[pn][0][ind2const[c1]]:
                        return c1, c2_fake
            return None, None

        if validation:
            fact_ls = self.valid_fact_ls
        else:
            fact_ls = self.test_fact_ls

        for val, pred_name, consts in fact_ls:

            for rule_i, rule in enumerate(self.rule_ls):

                # find rule with pred_name as head
                if rule.atom_ls[-1].pred_name != pred_name:
                    continue

                samples = samples_by_r[rule_i]
                neg_mask = neg_mask_by_r[rule_i]
                latent_mask = latent_mask_by_r[rule_i]
                obs_var = obs_var_by_r[rule_i]
                neg_var = neg_var_by_r[rule_i]

                var2ind = rule.key2ind
                var2type = rule.rule_vars

                sub = [None] * len(rule.rule_vars)  # substitutions
                vn0, vn1 = rule.atom_ls[-1].var_name_ls
                sub[var2ind[vn0]] = consts[0]
                sub[var2ind[vn1]] = consts[1]

                sample_buff = [[] for _ in rule.atom_ls]
                neg_mask_buff = [[] for _ in rule.atom_ls]
                latent_mask_buff = [[] for _ in rule.atom_ls]

                atom_inds = list(range(len(rule.atom_ls) - 1))
                shuffle(atom_inds)
                succ = True
                obs_list = []

                for atom_ind in atom_inds:
                    atom = rule.atom_ls[atom_ind]
                    pred_ht_dict = self.ht_dict_train[atom.pred_name]

                    gen_latent = np.random.rand( ) > observed_prob
                    c_ls, islatent, atom_succ = self._inst_var(sub, var2ind, var2type,
                                                               atom, pred_ht_dict, gen_latent)

                    assert atom_succ

                    if not islatent:
                        obs_var[atom_ind][1].append(c_ls)
                        c1, c2 = gen_fake(c_ls[0], c_ls[1], atom.pred_name)
                        if c1 is not None:
                            neg_var[atom_ind][1].append([c1, c2])

                    succ = succ and atom_succ
                    obs_list.append(not islatent)

                    sample_buff[atom_ind].append(c_ls)
                    latent_mask_buff[atom_ind].append(1 if islatent else 0)
                    neg_mask_buff[atom_ind].append(0 if atom.neg else 1)

                if succ and any(obs_list):
                    for i in range(len(rule.atom_ls)):
                        samples[i][1].extend(sample_buff[i])
                        latent_mask[i][1].extend(latent_mask_buff[i])
                        neg_mask[i][1].extend(neg_mask_buff[i])

                    samples[-1][1].append([self.const2ind[consts[0]], self.const2ind[consts[1]]])
                    latent_mask[-1][1].append(1)
                    neg_mask[-1][1].append(1)

                    cnt += 1

            if cnt >= batchsize:
                yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r

                samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                cnt = 0

        yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r

    def get_batch_by_q_v2(self, batchsize, observed_prob=1.0):

        samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
        cnt = 0

        num_ents = len(self.const2ind)
        ind2const = self.const_sort_dict['type']

        def gen_fake(c1, c2, pn):
            for _ in range(10):
                c1_fake = random.randint(0, num_ents - 1)
                c2_fake = random.randint(0, num_ents - 1)
                if np.random.rand( ) > 0.5:
                    if ind2const[c1_fake] not in self.ht_dict_train[pn][1][ind2const[c2]]:
                        return c1_fake, c2
                else:
                    if ind2const[c2_fake] not in self.ht_dict_train[pn][0][ind2const[c1]]:
                        return c1, c2_fake
            return None, None

        for val, pred_name, consts in self.test_fact_ls:

            for rule_i, rule in enumerate(self.rule_ls):

                # find rule with pred_name as head
                if rule.atom_ls[-1].pred_name != pred_name:
                    continue

                samples = samples_by_r[rule_i]
                neg_mask = neg_mask_by_r[rule_i]
                latent_mask = latent_mask_by_r[rule_i]

                var2ind = rule.key2ind
                var2type = rule.rule_vars

                sub_ls = [[None for _ in range(len(rule.rule_vars))] for _ in range(2)]  # substitutions

                vn0, vn1 = rule.atom_ls[-1].var_name_ls
                sub_ls[0][var2ind[vn0]] = consts[0]
                sub_ls[0][var2ind[vn1]] = consts[1]

                c1, c2 = gen_fake(self.const2ind[consts[0]], self.const2ind[consts[1]], pred_name)
                if c1 is not None:
                    sub_ls[1][var2ind[vn0]] = ind2const[c1]
                    sub_ls[1][var2ind[vn1]] = ind2const[c2]
                else:
                    sub_ls.pop(1)

                pos_query_succ = False

                for sub_ind, sub in enumerate(sub_ls):

                    sample_buff = [[] for _ in rule.atom_ls]
                    neg_mask_buff = [[] for _ in rule.atom_ls]
                    latent_mask_buff = [[] for _ in rule.atom_ls]

                    atom_inds = list(range(len(rule.atom_ls) - 1))
                    shuffle(atom_inds)
                    succ = True
                    obs_list = []

                    for atom_ind in atom_inds:
                        atom = rule.atom_ls[atom_ind]
                        pred_ht_dict = self.ht_dict_train[atom.pred_name]

                        gen_latent = np.random.rand( ) > observed_prob
                        if sub_ind == 1:
                            gen_latent = np.random.rand( ) > 0.5
                        c_ls, islatent, atom_succ = self._inst_var(sub, var2ind, var2type,
                                                                   atom, pred_ht_dict, gen_latent)

                        assert atom_succ

                        succ = succ and atom_succ
                        obs_list.append(not islatent)

                        sample_buff[atom_ind].append(c_ls)
                        latent_mask_buff[atom_ind].append(1 if islatent else 0)
                        neg_mask_buff[atom_ind].append(0 if atom.neg else 1)

                    if succ:
                        if any(obs_list) or ((sub_ind == 1) and pos_query_succ):

                            for i in range(len(rule.atom_ls)):
                                samples[i][1].extend(sample_buff[i])
                                latent_mask[i][1].extend(latent_mask_buff[i])
                                neg_mask[i][1].extend(neg_mask_buff[i])

                            if sub_ind == 0:
                                samples[-1][1].append([self.const2ind[consts[0]], self.const2ind[consts[1]]])
                                latent_mask[-1][1].append(1)
                                neg_mask[-1][1].append(1)
                                pos_query_succ = True
                                cnt += 1
                            else:
                                samples[-1][1].append([c1, c2])
                                latent_mask[-1][1].append(0)  # sample a negative fact at head
                                neg_mask[-1][1].append(1)

            if cnt >= batchsize:
                yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r

                samples_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                latent_mask_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                obs_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                neg_var_by_r = [[[atom.pred_name, []] for atom in rule.atom_ls] for rule in self.rule_ls]
                cnt = 0

        yield samples_by_r, latent_mask_by_r, neg_mask_by_r, obs_var_by_r, neg_var_by_r

    def get_batch_rnd(self, observed_prob=0.7, filter_latent=True, closed_world=False, filter_observed=False):
        """
            return a batch of gnd formulae by random sampling with controllable bias towards those containing
            observed variables. The overall sampling logic is that:
                1) rnd sample a rule from rule_ls
                2) shuffle the predicates contained in the rule
                3) for each of these predicates, with (observed_prob) it will be instantiated as observed variable, and
                   for (1-observed_prob) if will be simply uniformly instantiated.
                3.1) if observed var, then sample from the knowledge base, which is self.fact_dict, if failed for any
                     reason, go to 3.2)
                3.2) if uniformly sample, then for each logic variable in the predicate, instantiate it with a uniform
                     sample from the corresponding constant dict

        :param observed_prob:
            probability of instantiating a predicate as observed variable
        :param filter_latent:
            filter out ground formula containing only latent vars
        :param closed_world:
            if set True, reduce the sampling space of all predicates not in the test_dict to the set specified in
            fact_dict
        :param filter_observed:
            filter out ground formula containing only observed vars
        :return:

        """

        batch_neg_mask = [[] for _ in range(len(self.rule_ls))]
        batch_latent_var_inds = [[] for _ in range(len(self.rule_ls))]
        batch_observed_vars = [[] for _ in range(len(self.rule_ls))]
        observed_rule_cnts = [0.0 for _ in range(len(self.rule_ls))]
        flat_latent_vars = dict( )

        cnt = 0

        inds = list(range(len(self.rule_ls)))

        while cnt < self.batchsize:

            # randomly sample a formula
            if self.shuffle_sampling:
                shuffle(inds)

            for ind in inds:

                rule = self.rule_ls[ind]
                atom_key_dict = self.atom_key_dict_ls[ind]
                sub = [None] * len(rule.rule_vars)  # substitutions

                # randomly sample an atom from the formula
                atom_inds = list(range(len(rule.atom_ls)))
                shuffle(atom_inds)
                for atom_ind in atom_inds:
                    atom = rule.atom_ls[atom_ind]
                    atom_dict = atom_key_dict[atom.pred_name]

                    # instantiate the predicate
                    self._instantiate_pred(atom, atom_dict, sub, rule, observed_prob)

                    # if variable substitution is complete already then exit
                    if not (None in sub):
                        break

                # generate latent and observed var labels and their negation masks
                latent_vars, observed_vars, \
                    latent_neg_mask, observed_neg_mask = self._gen_mask(rule, sub, closed_world)

                # check sampled ground rule status
                stat_code = self._get_rule_stat(observed_vars, latent_vars, observed_neg_mask,
                                                filter_latent, filter_observed)

                # is a valid sample with only observed vars and does not have negation on all of them
                if stat_code == FULL_OBSERVERED:
                    observed_rule_cnts[ind] += 1

                    cnt += 1

                # is a valid sample
                elif stat_code == GOOD:
                    batch_neg_mask[ind].append([latent_neg_mask, observed_neg_mask])

                    for latent_var in latent_vars:
                        if latent_var not in flat_latent_vars:
                            flat_latent_vars[latent_var] = len(flat_latent_vars)

                    batch_latent_var_inds[ind].append([flat_latent_vars[e] for e in latent_vars])
                    batch_observed_vars[ind].append(observed_vars)

                    cnt += 1

                # not a valid sample
                else:
                    continue

                if cnt >= self.batchsize:
                    break

        flat_list = sorted([(k, v) for k, v in flat_latent_vars.items( )], key=lambda x: x[1])
        flat_list = [e[0] for e in flat_list]

        return batch_neg_mask, flat_list, batch_latent_var_inds, observed_rule_cnts, batch_observed_vars

    def reset(self):
        self.rule_gens = [self.generate_gnd_rule(rule) for rule in self.rule_ls]

    def get_stats(self):

        num_ents = sum([len(v) for k, v in self.const_sort_dict.items( )])
        num_rels = len(self.PRED_DICT)
        num_facts = sum([len(v) for k, v in self.fact_dict.items( )])
        num_queries = len(self.test_fact_ls)

        num_gnd_atom = 0
        for pred_name, pred in self.PRED_DICT.items( ):
            cnt = 1
            for var_type in pred.var_types:
                cnt *= len(self.const_sort_dict[var_type])
            num_gnd_atom += cnt

        num_gnd_rule = 0
        for rule in self.rule_ls:
            cnt = 1
            for var_type in rule.rule_vars.values( ):
                cnt *= len(self.const_sort_dict[var_type])
            num_gnd_rule += cnt

        return num_ents, num_rels, num_facts, num_queries, num_gnd_atom, num_gnd_rule

    def preprocess_kinship(self, ppath, fpath, rpath, qpath):
        """

        :param ppath:
            predicate file path
        :param fpath:
            facts file path
        :param rpath:
            rule file path
        :param qpath:
            query file path

        :return:

        """
        assert all(map(isfile, [ppath, fpath, rpath, qpath]))

        strip_items = lambda ls: list(map(lambda x: x.strip( ), ls))

        pred_reg = re.compile(r'(.*)\((.*)\)')

        with open(ppath) as f:
            for line in f:

                # skip empty lines
                if line.strip( ) == '':
                    continue

                m = pred_reg.match(line.strip( ))
                assert m is not None, 'matching predicate failed for %s' % line

                name, var_types = m.group(1), m.group(2)
                var_types = list(map(lambda x: x.strip( ), var_types.split(',')))

                self.PRED_DICT[name] = Predicate(name, var_types)
                TYPE_SET.update(var_types)

        fact_ls = []
        fact_reg = re.compile(r'(!?)(.*)\((.*)\)')
        with open(fpath) as f:
            for line in f:

                # skip empty lines
                if line.strip( ) == '':
                    continue

                m = fact_reg.match(line.strip( ))
                assert m is not None, 'matching fact failed for %s' % line

                val = 0 if m.group(1) == '!' else 1
                name, consts = m.group(2), m.group(3)
                consts = strip_items(consts.split(','))

                fact_ls.append(Fact(name, consts, val))

                for var_type in self.PRED_DICT[name].var_types:
                    self.const_dict.add_const(var_type, consts.pop(0))

        rule_ls = []
        first_atom_reg = re.compile(r'([\d.]+) (!?)([\w\d]+)\((.*)\)')
        atom_reg = re.compile(r'(!?)([\w\d]+)\((.*)\)')
        with open(rpath) as f:
            for line in f:

                # skip empty lines
                if line.strip( ) == '':
                    continue

                atom_str_ls = strip_items(line.strip( ).split(' v '))
                assert len(atom_str_ls) > 1, 'rule length must be greater than 1, but get %s' % line

                atom_ls = []
                rule_weight = 0.0
                for i, atom_str in enumerate(atom_str_ls):
                    if i == 0:
                        m = first_atom_reg.match(atom_str)
                        assert m is not None, 'matching atom failed for %s' % atom_str
                        rule_weight = float(m.group(1))
                        neg = m.group(2) == '!'
                        pred_name = m.group(3).strip( )
                        var_name_ls = strip_items(m.group(4).split(','))
                    else:
                        m = atom_reg.match(atom_str)
                        assert m is not None, 'matching atom failed for %s' % atom_str
                        neg = m.group(1) == '!'
                        pred_name = m.group(2).strip( )
                        var_name_ls = strip_items(m.group(3).split(','))

                    atom = Atom(neg, pred_name, var_name_ls, self.PRED_DICT[pred_name].var_types)
                    atom_ls.append(atom)

                rule = Formula(atom_ls, rule_weight)
                rule_ls.append(rule)

        query_ls = []
        with open(qpath) as f:
            for line in f:

                # skip empty lines
                if line.strip( ) == '':
                    continue

                m = fact_reg.match(line.strip( ))
                assert m is not None, 'matching fact failed for %s' % line

                val = 0 if m.group(1) == '!' else 1
                name, consts = m.group(2), m.group(3)
                consts = strip_items(consts.split(','))

                query_ls.append(Fact(name, consts, val))

                for var_type in self.PRED_DICT[name].var_types:
                    self.const_dict.add_const(var_type, consts.pop(0))

        return fact_ls, rule_ls, query_ls


TYPE_SET = set( )


def iterline(fpath):
    with open(fpath) as f:

        for line in f:

            line = line.strip( )
            if line == '':
                continue

            yield line


class ConstantDict:

    def __init__(self):
        self.constants = {}

    def add_const(self, const_type, const):
        """

        :param const_type:
            string
        :param const:
            string
        """

        # if const_type not in TYPE_DICT:
        #     TYPE_DICT[const_type] = len(TYPE_DICT)

        if const_type in self.constants:
            self.constants[const_type].add(const)
        else:
            self.constants[const_type] = {const}

    def __getitem__(self, key):
        return self.constants[key]

    def has_const(self, key, const):
        if key in self.constants:
            return const in self[key]
        else:
            return False


class Predicate:

    def __init__(self, name, var_types):
        """

        :param name:
            string
        :param var_types:
            list of strings
        """
        self.name = name
        self.var_types = var_types
        self.num_args = len(var_types)

    def __repr__(self):
        return '%s(%s)' % (self.name, ','.join(self.var_types))


class Fact:
    def __init__(self, pred_name, const_ls, val):
        self.pred_name = pred_name
        self.const_ls = deepcopy(const_ls)
        self.val = val

    def __repr__(self):
        return self.pred_name + '(%s)' % ','.join(self.const_ls)


class Atom:
    def __init__(self, neg, pred_name, var_name_ls, var_type_ls):
        self.neg = neg
        self.pred_name = pred_name
        self.var_name_ls = var_name_ls
        self.var_type_ls = var_type_ls

    def __repr__(self):
        return ('!' if self.neg else '') + self.pred_name + '(%s)' % ','.join(self.var_name_ls)


class Formula:
    """
        only support clause form with disjunction, e.g. !
    """

    def __init__(self, atom_ls, weight):
        self.weight = weight
        self.atom_ls = atom_ls
        self.rule_vars = dict( )

        for atom in self.atom_ls:
            self.rule_vars.update(zip(atom.var_name_ls, atom.var_type_ls))
        self.key2ind = dict(zip(self.rule_vars.keys( ), range(len(self.rule_vars.keys( )))))

    def evaluate(self):
        pass

    def __repr__(self):
        return ' v '.join(list(map(repr, self.atom_ls)))


class ConstantDict:

    def __init__(self):
        self.constants = {}

    def add_const(self, const_type, const):
        """

        :param const_type:
            string
        :param const:
            string
        """

        # if const_type not in TYPE_DICT:
        #     TYPE_DICT[const_type] = len(TYPE_DICT)

        if const_type in self.constants:
            self.constants[const_type].add(const)
        else:
            self.constants[const_type] = {const}

    def __getitem__(self, key):
        return self.constants[key]

    def has_const(self, key, const):
        if key in self.constants:
            return const in self[key]
        else:
            return False

@register_dataset('NBF_link_prediction') 
class NBF_LinkPrediction(LinkPredictionDataset):
    r"""
    The NBF dataset will be used in task *link prediction*.

    """

    def __init__(self, dataset_name ,*args, **kwargs): # dataset_name in ['NBF_WN18RR','NBF_FB15k-237']

        self.dataset = NBF_Dataset(root='./openhgnn/dataset/', name=dataset_name[4:], version="v1")



import os
import requests
import zipfile
import io
@register_dataset('DisenKGAT_link_prediction')
class DisenKGAT_LinkPrediction(LinkPredictionDataset):
    def __init__(self, dataset ,*args, **kwargs): # dataset "DisenKGAT"
        self.logger = kwargs.get("Logger")
        self.args = kwargs.get("args")
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_name = dataset                       
        self.raw_dir = os.path.join(self.current_dir, self.dataset_name ,"raw_dir" ) 
        self.processed_dir = os.path.join(self.current_dir, self.dataset_name ,"processed_dir" ) 

        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir) 
            self.download()
        else:
            print("raw_dir already exists")

    def download(self): 

        url = "https://s3.cn-north-1.amazonaws.com.cn/dgl-data/dataset/openhgnn/{}.zip".format(self.dataset_name)          
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as myzip:
            myzip.extractall(self.raw_dir)       
        print("---  download   finished---")

      


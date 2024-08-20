import os
from dgl.data.utils import download, extract_archive
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, idx2mask
import pickle
import scipy
import numpy as np
import dgl
import torch as th



# get dataset from database
class IMDB4MAGNN_Dataset(DGLDataset):

    def __init__(self, name, args, raw_dir=None, force_reload=False, verbose=True):
        assert name in ['imdb4MAGNN', ]

        self.args = args 
        super(IMDB4MAGNN_Dataset, self).__init__(name=name,
                                        url=None,
                                        raw_dir=None, 
                                        force_reload=force_reload,
                                        verbose=verbose)


    def download(self):

        from gdbi import NodeExportConfig, EdgeExportConfig, Neo4jInterface, NebulaInterface
        node_export_config = [
            NodeExportConfig('A', ['attribute'] ),
            NodeExportConfig('M', ['attribute'], ['label']), 
            NodeExportConfig('D', ['attribute'])  
        ]

        edge_export_config = [
            EdgeExportConfig('A_M', ('A','M')),
            EdgeExportConfig('M_A', ('M','A')),
            EdgeExportConfig('M_D', ('M','D')),
            EdgeExportConfig('D_M', ('D','M'))    
        ]

        # neo4j 
        graph_database = Neo4jInterface()

        # # nebula 
        # graph_database = NebulaInterface()

        graph_address = self.args.graph_address
        user_name = self.args.user_name
        password = self.args.password

        conn = graph_database.GraphDBConnection(graph_address, user_name, password)
        self.graph = graph_database.get_graph(conn, 'imdb4MAGNN', node_export_config, edge_export_config)


        

    def process(self):

        graph = self.graph
        cano_edges = {}
        for edge_type in graph['edge_index_dict'].keys():  # 'A_M'
            src_type = edge_type[0] # A
            dst_type = edge_type[-1]    # M
            edge_type_2 = src_type + '-' + dst_type # A-M
            
            cano_edge_type = (src_type,edge_type_2,dst_type)  # ('A','A-M','M')
            u,v = graph['edge_index_dict'][edge_type][0] ,graph['edge_index_dict'][edge_type][1] 

            cano_edges[cano_edge_type] = (u,v)



        hg = dgl.heterograph(cano_edges)

        for node_type in graph['X_dict'].keys() :   
            hg.nodes[node_type].data['h'] = graph['X_dict'][node_type]  
            if node_type == 'M':
                hg.nodes[node_type].data['labels'] = graph['Y_dict'][node_type]  

        import torch



        num_nodes = 4278
        random_indices = torch.randperm(num_nodes)

        num_train = 400
        num_val = 400
        num_test = 3478

        train_mask = torch.zeros(num_nodes, dtype=torch.int)
        train_mask[random_indices[:num_train]] = 1
        val_mask = torch.zeros(num_nodes, dtype=torch.int)
        val_mask[random_indices[num_train:num_train+num_val]] = 1
        test_mask = torch.zeros(num_nodes, dtype=torch.int)
        test_mask[random_indices[num_train+num_val:]] = 1

        assert torch.sum(train_mask * val_mask) == 0
        assert torch.sum(train_mask * test_mask) == 0
        assert torch.sum(val_mask * test_mask) == 0

        hg.nodes['M'].data['train_mask'] = train_mask
        hg.nodes['M'].data['val_mask'] = val_mask
        hg.nodes['M'].data['test_mask'] = test_mask

        self._g = hg

 


    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph" 
        return self._g

    def __len__(self):
        return 1


    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass






class AcademicDataset(DGLDataset):

    _prefix = 'https://s3.cn-north-1.amazonaws.com.cn/dgl-data/'
    _urls = {
        'academic4HetGNN': 'dataset/academic4HetGNN.zip',
        'acm4GTN': 'dataset/acm4GTN.zip',
        'acm4NSHE': 'dataset/acm4NSHE.zip',
        'acm4NARS': 'dataset/acm4NARS.zip',
        'acm4HeCo': 'dataset/acm4HeCo.zip',
        'imdb4MAGNN': 'dataset/imdb4MAGNN.zip',
        'imdb4GTN': 'dataset/imdb4GTN.zip',
        'DoubanMovie': 'dataset/DoubanMovie.zip',
        'dblp4MAGNN': 'dataset/dblp4MAGNN.zip',
        'yelp4HeGAN': 'dataset/yelp4HeGAN.zip',
        'yelp4rec': 'dataset/yelp4rec.zip',
        'HNE-PubMed': 'dataset/HNE-PubMed.zip',
        'MTWM': 'dataset/MTWM3.zip',
        'amazon4SLICE': 'dataset/amazon4SLICE.zip',
        'amazon': 'https://zhiguli.oss-cn-hangzhou.aliyuncs.com/amazon.zip',
        'yelp4HGSL': 'dataset/yelp4HGSL.zip'
    }

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name in ['acm4GTN', 'acm4NSHE', 'academic4HetGNN', 'imdb4MAGNN', 'imdb4GTN', 'HNE-PubMed', 'MTWM',
                        'DoubanMovie', 'dblp4MAGNN', 'acm4NARS', 'acm4HeCo', 'yelp4rec', 'yelp4HeGAN', 'amazon4SLICE','amazon', 'yelp4HGSL']
        if name == 'yelp4HGSL':
            canonical_etypes = [('b', 'b-s', 's'), ('s', 's-b', 'b'), ('b', 'b-l', 'l'), ('l', 'l-b', 'b'), ('b', 'b-u', 'u'),
                                ('u', 'u-b', 'b')]
            target_ntype = 'b'
            meta_paths_dict = {'bub': [('b', 'b-u', 'u'), ('u', 'u-b', 'b')],
                            'bsb': [('b', 'b-s', 's'), ('s', 's-b', 'b')],
                            'bublb': [('b', 'b-u', 'u'), ('u', 'u-b', 'b'),
                                        ('b', 'b-l', 'l'), ('l', 'l-b', 'b')],
                            'bubsb': [('b', 'b-u', 'u'), ('u', 'u-b', 'b'),
                                        ('b', 'b-s', 's'), ('s', 's-b', 'b')]
                   }
            self._canonical_etypes = canonical_etypes
            self._target_ntype = target_ntype
            self._meta_paths_dict = meta_paths_dict
        self.data_path = './openhgnn/' + self._urls[name]
        self.g_path = './openhgnn/dataset/' + name + '/graph.bin'
        raw_dir = './openhgnn/dataset'
        url = self._prefix + self._urls[name]
        if name == 'amazon':
            url = 'https://zhiguli.oss-cn-hangzhou.aliyuncs.com/amazon.zip'
            self.data_path = './openhgnn/dataset/amazon.zip'
        super(AcademicDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def download(self):
        # download raw data to local disk
        # path to store the file
        if os.path.exists(self.data_path):  # pragma: no cover
           pass
        else:
            file_path = os.path.join(self.raw_dir)
            # download file
            download(self.url, path=file_path)
        extract_archive(self.data_path, os.path.join(self.raw_dir, self.name))

    def process(self):
        # process raw data to graphs, labels, splitting masks
        if self.name == 'yelp4HGSL':
            target_ntype = self._target_ntype
            canonical_etypes = self._canonical_etypes
            
            with open(self.raw_path + '/node_features.pkl', 'rb') as f:
                features = pickle.load(f)
            with open(self.raw_path + '/edges.pkl', 'rb') as f:
                edges = pickle.load(f)
            with open(self.raw_path + '/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
            with open(self.raw_path + '/meta_data.pkl', 'rb') as f:
                meta_data = pickle.load(f)
            if scipy.sparse.issparse(features):
                features = features.todense()
            
            '''Load pretrained mp_embedding'''
            mp_emb_dict = {}
            mp_list = ['bub', 'bsb', 'bublb', 'bubsb']
            for mp in mp_list:
                f_name = self.raw_path + '/' + mp + '_emb.pkl'
                with open(f_name, 'rb') as f:
                    z = pickle.load(f)
                    zero_lines = np.nonzero(np.sum(z, 1) == 0)
                    if len(zero_lines) > 0:
                        # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), mode, zero_lines))
                        z[zero_lines, :] += 1e-8
                    mp_emb_dict[mp] = z
            
            num_nodes = edges['s-b'].shape[0]
            assert len(canonical_etypes) == len(edges)

            ntype_mask = dict()
            ntype_idmap = dict()
            ntypes = set()
            data_dict = {}

            # create dgl graph
            for etype in canonical_etypes:
                ntypes.add(etype[0])
                ntypes.add(etype[2])
            for ntype in ntypes:
                ntype_mask[ntype] = np.zeros(num_nodes, dtype=bool)
                ntype_idmap[ntype] = np.full(num_nodes, -1, dtype=int)
            for etype in canonical_etypes:
                src_nodes = edges[etype[1]].nonzero()[0]
                dst_nodes = edges[etype[1]].nonzero()[1]
                src_ntype = etype[0]
                dst_ntype = etype[2]
                ntype_mask[src_ntype][src_nodes] = True
                ntype_mask[dst_ntype][dst_nodes] = True
            for ntype in ntypes:
                ntype_idx = ntype_mask[ntype].nonzero()[0]
                ntype_idmap[ntype][ntype_idx] = np.arange(ntype_idx.size)
            for etype in canonical_etypes:
                src_nodes = edges[etype[1]].nonzero()[0]
                dst_nodes = edges[etype[1]].nonzero()[1]
                src_ntype = etype[0]
                dst_ntype = etype[2]
                data_dict[etype] = \
                    (th.from_numpy(ntype_idmap[src_ntype][src_nodes]).type(th.int64),
                    th.from_numpy(ntype_idmap[dst_ntype][dst_nodes]).type(th.int64))
            g = dgl.heterograph(data_dict)

            # split and label
            all_label = np.full(g.num_nodes(target_ntype), -1, dtype=int)
            for i, split in enumerate(['train', 'val', 'test']):
                node = np.array(labels[i])[:, 0]
                label = np.array(labels[i])[:, 1]
                all_label[node] = label
                g.nodes[target_ntype].data['{}_mask'.format(split)] = \
                    th.from_numpy(idx2mask(node, g.num_nodes(target_ntype))).type(th.bool)
            g.nodes[target_ntype].data['label'] = th.from_numpy(all_label).type(th.long)

            # node feature
            node_features = th.from_numpy(features).type(th.FloatTensor)
            for ntype in ntypes:
                idx = ntype_mask[ntype].nonzero()[0]
                g.nodes[ntype].data['h'] = node_features[idx]

            for ntype in ntypes:
                idx = ntype_mask[ntype].nonzero()[0]
                for mp in mp_list:
                    tmp_tensor = th.from_numpy(mp_emb_dict[mp][idx])
                    g.nodes[ntype].data[mp] = tmp_tensor

            self._g = g
            self._num_classes = len(th.unique(self._g.nodes[self._target_ntype].data['label']))
            self._in_dim = self._g.ndata['h'][self._target_ntype].shape[1]
        else:
            g, _ = load_graphs(self.g_path)
            self._g = g[0]

    def __getitem__(self, idx):
        # get one example by index
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        # number of data examples
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass
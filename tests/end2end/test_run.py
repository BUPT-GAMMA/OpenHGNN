import pytest
from openhgnn.experiment import Experiment

experiments = {
    'node_classification': {  # 25
        'aifb': ['CompGCN', 'RGCN', 'RSHN', ],
        'acm_han_raw': ['DMGI', 'HAN', 'HPN', ],
        'imdb4GTN': ['GIN', 'RHGNN', ],
        'acm4GTN': ['GTN', 'fastGTN', 'HGSL', 'MHNF', ],
        'dblp4MAGNN': ['HERec', 'Metapath2vec', ],
        'imdb4MAGNN': ['HGNN_AC', 'HGT', 'SimpleHGN', 'HetSANN', 'ieHGCN', 'MAGNN'],
        'acm4HeCo': ['HeCo', ],
        'yelp4HeGAN': ['HeGAN', ],
        'academic4HetGNN': ['HetGNN', ],
        'acm4NARS': ['NARS', ],
        'acm4NSHE': ['NSHE', ],
    },
    'edge_classification': {  # 2
        'dblp4Mg2vec_4': ['Mg2vec', ],
        'dblp4Mg2vec_5': ['Mg2vec', ],
    },
    'link_prediction': {  # 8
        'HGBl-amazon': ['GATNE-T', 'RGCN', ],
        'wn18': ['GIE', ],
        'HGBl-IMDB': ['HDE', ],     # 'HGBl-IMDB' slow to run, alternative datasets ['HGBl-DBLP', 'HGBl-ACM']
        'FB15k': ['TransD', 'TransE', 'TransH', 'TransR', ]
    },
    'recommendation': {  # 1
        'LastFM4KGCN': ['KGCN', ],
    },
    'hypergraph': {  # 1
        'drug': ['DHNE', ]
    },
    'meirec': {  # 1
        'meirec': ['MeiRec', ]
    },
}
# HERec, HGNN_AC, Mg2vec, MeiRec, HDE
# test the failed models until June 8th.
class TestExperiment:

    def setup_class(cls):
        cls.gpu = 3

    @pytest.mark.parametrize("dataset,model",
                             [(dataset, model) for dataset, models in experiments['node_classification'].items() for
                              model in models])
    def test_node_classification(self, dataset, model):
        Experiment(model=model, dataset=dataset, task='node_classification', gpu=self.gpu, epoch=1, max_epoch=1).run()

    @pytest.mark.parametrize("dataset,model",
                             [(dataset, model) for dataset, models in experiments['edge_classification'].items() for
                              model in models])
    def test_edge_classification(self, dataset, model):
        Experiment(model=model, dataset=dataset, task='edge_classification', gpu=self.gpu, epoch=1, max_epoch=1).run()

    @pytest.mark.parametrize("dataset,model",
                             [(dataset, model) for dataset, models in experiments['link_prediction'].items() for model
                              in models])
    def test_link_prediction(self, dataset, model):
        Experiment(model=model, dataset=dataset, task='link_prediction', gpu=self.gpu, epoch=1, max_epoch=1).run()

    @pytest.mark.parametrize("dataset,model",
                             [(dataset, model) for dataset, models in experiments['recommendation'].items() for model in
                              models])
    def test_recommendation(self, dataset, model):
        Experiment(model=model, dataset=dataset, task='recommendation', gpu=self.gpu, epoch=1, max_epoch=1).run()

    @pytest.mark.parametrize("dataset,model",
                             [(dataset, model) for dataset, models in experiments['hypergraph'].items() for model in
                              models])
    def test_hypergraph(self, dataset, model):
        Experiment(model=model, dataset=dataset, task='hypergraph', gpu=self.gpu, epoch=1, max_epoch=1).run()

    @pytest.mark.parametrize("dataset,model",
                             [(dataset, model) for dataset, models in experiments['meirec'].items() for model in
                              models])
    def test_meirec(self, dataset, model):
        Experiment(model=model, dataset=dataset, task='meirec', gpu=self.gpu, epoch=1, max_epoch=1).run()

if __name__ == '__main__':
    pytest.main()



"""
node_classification
1. python main.py -m CompGCN -t node_classification -d aifb -g 0
3. python main.py -m DMGI -t node_classification -d acm_han_raw -g 0 --use_best_config
6. python main.py -m GIN -d imdb4GTN -t node_classification -g 0
7. python main.py -m GTN -t node_classification -d acm4GTN -g 0 --use_best_config
   python main.py -m fastGTN -t node_classification -d acm4GTN -g 0 --use_best_config
8. python main.py -m HAN -t node_classification -d acm_han_raw -g 0
10.python main.py -m HERec -t node_classification -d dblp4MAGNN -g 0                        error?
11.python main.py -m HGNN_AC -t node_classification -d imdb4MAGNN -g 0                      error?
12.python main.py -m HGSL -d acm4GTN -t node_classification -g 0 --use_best_config          error?
13.python main.py -m HGT -t node_classification -d imdb4MAGNN -g 0 --use_best_config
   python main.py -m SimpleHGN -t node_classification -d imdb4MAGNN -g 0 --use_best_config
   python main.py -m HetSANN -t node_classification -d imdb4MAGNN -g 0 --use_best_config
   python main.py -m ieHGCN -t node_classification -d imdb4MAGNN -g 0 --use_best_config
14.python main.py -m HPN -t node_classification -d acm_han_raw -g 0
15.python main.py -m HeCo -d acm4HeCo -t node_classification -g 0 --use_best_config
16.python main.py -m HeGAN -d yelp4HeGAN -t node_classification -g 0
17.python main.py -m HetGNN -t node_classification -d academic4HetGNN -g 0
19.python main.py -m MAGNN -t node_classification -d imdb4MAGNN -g 0
20.python main.py -m MHNF -t node_classification -d acm4GTN -g 0 --use_best_config      error?
23.python main.py -m NARS -t node_classification -d acm4NARS -g 0 --use_best_config
24.python main.py -m NSHE -t node_classification -d acm4NSHE -g 0 --use_best_config
25.python main.py -m RGCN -t node_classification -d aifb -g 0 --use_best_config
26.python main.py -m RHGNN -t node_classification -d imdb4GTN -g 0 --use_best_config
27.python main.py -m RSHN -t node_classification -d aifb -g 0
32.python main.py -m Metapath2vec -t node_classification -d dblp4MAGNN -g 0


edge_classification
22.python main.py -m Mg2vec -t edge_classification -d dblp4Mg2vec_4 -g 0 # meta_graph's size is up to 4 
   python main.py -m Mg2vec -t edge_classification -d dblp4Mg2vec_5 -g 0 # meta_graph's size is up to 5


link_prediction
4. python main.py -m GATNE-T -t link_prediction -d HGBl-amazon -g 0 --use_best_config
5. python main.py -m GIE -t link_prediction -d wn18 -g 0 --use_best_config
9. python main.py -m HDE -t link_prediction -d HGBl-IMDB -g 0 --use_best_config
25.python main.py -m RGCN -t link_prediction -d HGBl-amazon -g 0 --use_best_config
28.python main.py -m TransD -t link_prediction -d FB15k -g 0 --use_best_config
29.python main.py -m TransE -t link_prediction -d FB15k -g 0 --use_best_config
30.python main.py -m TransH -t link_prediction -d FB15k -g 0 --use_best_config
31.python main.py -m TransR -t link_prediction -d FB15k -g 0 --use_best_config

recommendation
18.python main.py -m KGCN -t recommendation -d LastFM4KGCN -g 0 --use_best_config



hypergraph
2. python main.py -m DHNE -d drug -t hypergraph -g 0 --use_best_config


meirec
21.python main.py -m MeiREC -d meirec -t meirec -g 0
"""

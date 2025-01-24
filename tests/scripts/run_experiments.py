import argparse
from openhgnn import Experiment

# script for running batch experiments
# python tests/scripts/run_experiments.py -g 0
# or
# nohup python tests/scripts/run_experiments.py -g 0

# from readme of every model
# python main.py -m CompGCN -t node_classification -d aifb -g 0
# python main.py -m DMGI -t node_classification -d acm_han_raw -g 0 --use_best_config
# python main.py -m GTN -t node_classification -d acm4GTN -g 0 --use_best_config
# python main.py -m fastGTN -t node_classification -d acm4GTN -g 0 --use_best_config
# python main.py -m HAN -t node_classification -d acm_han_raw -g 0
# python main.py -m HeCo -d acm4HeCo -t node_classification -g 0 --use_best_config
# python main.py -m HERec -t node_classification -d dblp4MAGNN -g 0
# python main.py -m HetGNN -t node_classification -d academic4HetGNN -g 0
# python main.py -m HGNN_AC -t node_classification -d imdb4MAGNN -g 0
# python main.py -m HGSL -d acm4GTN -t node_classification -g 0 --use_best_config
# python main.py -m HGT -t node_classification -d imdb4MAGNN -g 0 --use_best_config
# python main.py -m HPN -t node_classification -d acm_han_raw -g 0
# python main.py -m MAGNN -t node_classification -d imdb4MAGNN -g 0
# python main.py -m Metapath2vec -t node_classification -d dblp4MAGNN -g 0
# python main.py -m MHNF -t node_classification -d acm4GTN -g 0 --use_best_config
# python main.py -m NARS -t node_classification -d acm4NARS -g 0 --use_best_config
# python main.py -m NSHE -t node_classification -d acm4NSHE -g 0 --use_best_config
# python main.py -m RGCN -t node_classification -d aifb -g 0 --use_best_config
# python main.py -m RHGNN -t node_classification -d imdb4GTN -g 0 --use_best_config
# python main.py -m RSHN -t node_classification -d aifb -g 0
# python main.py -m SimpleHGN -t node_classification -d imdb4MAGNN -g 0 --use_best_config
# python main.py -m HDE -d HGBl-IMDB -t link_prediction -g 0 --use_best_config
# python main.py -m RGCN -t link_prediction -d HGBl-amazon -g 0 --use_best_config
# python main.py -m KGCN -d LastFM4KGCN -t recommendation -g 0 --use_best_config
import sys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    args = parser.parse_args()

    experiments = {'node_classification': [('CompGCN', 'aifb'), ('DMGI', 'acm_han_raw'), ('GTN', 'acm4GTN'),
                                           ('fastGTN', 'acm4GTN'), ('HAN', 'acm_han_raw'),
                                           ('HeCo', 'acm4HeCo'), ('HERec', 'dblp4MAGNN'), ('HetGNN', 'academic4HetGNN'),
                                           ('HGNN_AC', 'imdb4MAGNN'), ('HGSL', 'acm4GTN'),
                                           ('HGT', 'imdb4MAGNN'), ('HPN', 'acm_han_raw'), ('MAGNN', 'imdb4MAGNN'),
                                           ('Metapath2vec', 'dblp4MAGNN'), ('MHNF', 'acm4GTN'), ('NARS', 'acm4NARS'),
                                           ('RGCN', 'aifb'), ('NSHE', 'acm4NSHE'), ('RHGNN', 'imdb4GTN'),
                                           ('RSHN', 'aifb'), ('SimpleHGN', 'imdb4MAGNN'), ],
                   'link_prediction': [('HDE', 'HGBl-IMDB'), ('RGCN', 'HxGBl-amazon')],
                   'recommendation': [('KGCN', 'LastFM4KGCN')]
                   }

    for experiment in experiments.items():
        task = experiment[0]
        for elem in experiment[1]:
            model = elem[0]
            dataset = elem[1]
            try:
                Experiment(model=model, dataset=dataset, task=task, gpu=args.gpu, epoch=1, max_epoch=1).run()
            except Exception as e:
                print(e)

import argparse
from openhgnn import Experiment
from dataset import MyDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='Metapath2vec', type=str, help='name of models')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--meta-path-key', '-mp', default='APA', type=str, help='name of models')

    args = parser.parse_args()

    ds = MyDataset()

    experiment = Experiment(conf_path='../my_config.ini', max_epoch=1, model=args.model, dataset=ds,
                            task='embedding', meta_path_key=args.meta_path_key, gpu=args.gpu)
    experiment.run()

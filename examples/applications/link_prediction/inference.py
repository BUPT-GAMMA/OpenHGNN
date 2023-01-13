import argparse
from openhgnn import Experiment
from openhgnn.dataset import AsLinkPredictionDataset
from dataset import MyLPDatasetWithPredEdges, target_link, target_link_r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='RGCN', type=str, help='name of models')
    parser.add_argument('--gpu', '-g', default='-1', type=int, help='-1 means cpu')
    parser.add_argument('--mini-batch-flag', action='store_true')

    args = parser.parse_args()

    ds = MyLPDatasetWithPredEdges()

    new_ds = AsLinkPredictionDataset(ds, target_link=target_link, target_link_r=target_link_r,
                                     split_ratio=[0.8, 0.1, 0.1], force_reload=True)

    experiment = Experiment(conf_path='../my_config.ini', max_epoch=0, model=args.model, dataset=new_ds,
                            task='link_prediction', mini_batch_flag=args.mini_batch_flag, gpu=args.gpu,
                            test_flag=False, prediction_flag=True, batch_size=1000, load_from_pretrained=True)

    prediction_res = experiment.run()
    print(prediction_res)

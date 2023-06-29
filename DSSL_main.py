import argparse

from openhgnn.experiment import Experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', default='DSSL_openhgnn', type=str, help='name of models')
    parser.add_argument('--task', '-t', default='DSSL_trainerflow', type=str, help='name of task')
    #  node_classification
    parser.add_argument('--dataset', '-d', default='Cora', type=str, help='name of datasets')

    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--sub_dataset', type=str, default='Penn94')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=int, default=0.001)  # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--directed', action='store_true', help='set to not symmetrize adjacency')
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--display_step', type=int, default=25, help='how often to print')
    parser.add_argument('--train_prop', type=float, default=.48, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.32, help='validation label proportion')
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
    parser.add_argument('--rand_split', type=bool, default=True, help='use random splits')
    parser.add_argument('--embedding_dim', type=int, default=10, help="embedding dim")
    parser.add_argument('--neighbor_max', type=int, default=5, help="neighbor num max")
    parser.add_argument('--cluster_num', type=int, default=6, help="cluster num")
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')            # 默认值是False
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--entropy', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.99)
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--mlp_bool', type=int, default=1, help="embedding with mlp predictor")
    parser.add_argument('--tao', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--mlp_inference_bool', type=int, default=1, help="embedding with mlp predictor")
    parser.add_argument('--neg_alpha', type=int, default=0, help="negative alpha ")
    parser.add_argument('--load_json', type=int, default=0, help="load json")

    args = parser.parse_args()

    print(args)

    experiment = Experiment(model=args.model, dataset=args.dataset, task=args.task, gpu=args.gpu)
    experiment.set_params(sub_dataset=args.sub_dataset, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                          hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout,
                          normalize_features=args.normalize_features, seed=args.seed, display_step=args.display_step,
                          train_prop=args.train_prop, valid_prop=args.valid_prop, batch_size=args.batch_size,
                          rand_split=args.rand_split, embedding_dim=args.embedding_dim, neighbor_max=args.neighbor_max,
                          cluster_num=args.cluster_num, alpha=args.alpha, gamma=args.gamma, entropy=args.entropy,
                          tau=args.tau, encoder=args.encoder, mlp_bool=args.mlp_bool, tao=args.tao,
                          beta=args.beta, mlp_inference_bool=args.mlp_inference_bool, neg_alpha=args.neg_alpha,
                          load_json=args.load_json, no_bn=args.no_bn)

    experiment.run()
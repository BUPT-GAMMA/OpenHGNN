# TODO: This file should be deprecated, it's just for testing.
from openhgnn.models.MAGNN import MAGNN
from openhgnn.sampler.MAGNN_sampler import MAGNN_sampler, collate_fn
from openhgnn.sampler.test_config import CONFIG
import argparse
import warnings
import dgl
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader

def load_hg(args):
    hg_dir = 'openhgnn/dataset/'
    hg,_ = dgl.load_graphs(hg_dir+'{}/graph.bin'.format(args.dataset), [0])
    hg = hg[0]
    return hg

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = argparse.Namespace(**CONFIG)
    hg = load_hg(args)
    model = MAGNN.build_model_from_args(args, hg)
    sampler = MAGNN_sampler(g=hg, n_layers=args.num_layers, category=args.category, metapath_list=model.metapath_list)
    dataloader = DataLoader(dataset=sampler, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            collate_fn=collate_fn, drop_last=False)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(args.device)
    batch_idx = 0

    for epoch in range(args.max_epoch):
        for sub_g, mini_mp_inst, seed_nodes in dataloader:
            model.mini_reset_params(mini_mp_inst)
            sub_g = sub_g.to(args.device)
            pred = model(sub_g)[args.category][seed_nodes[args.category]]
            lbl = sub_g.nodes[args.category].data['labels'][seed_nodes[args.category]]
            loss = F.cross_entropy(pred, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("batch_idx:{}, the batch_size is {}, the loss of this batch is {}".format(
                batch_idx, args.batch_size, loss.item()
            ))
            batch_idx += 1

    print(1)






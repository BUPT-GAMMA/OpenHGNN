from unittest import TestCase
from openhgnn.dataset import AsNodeClassificationDataset
import dgl
from dgl.data import DGLDataset
import torch


class TestAsNodeClassificationDataset(TestCase):
    class MyDataset(DGLDataset):
        def __init__(self):
            super().__init__(name='my-dataset', force_reload=True)

        def process(self):
            data_dict = {
                ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
                ('user', 'follows', 'topic'): (torch.tensor([2, 3]), torch.tensor([1, 2])),
                ('user', 'plays', 'game'): (torch.tensor([5, 11]), torch.tensor([3, 4]))
            }
            g = dgl.heterograph(data_dict)
            label = torch.empty(g.num_nodes(ntype = 'user'), 1)
            for i in range(5):
                label[i] = 1
            g.nodes['user'].data['label'] = label
            g.nodes['user'].data['mask'] = torch.tensor([1,1,1,1,1,0,0,1,1,1,1,1])
            self._g =  g
            
        def __getitem__(self, idx):
            return self._g

        def __len__(self):
            return 1

    ds = MyDataset()
    new_ds = AsNodeClassificationDataset(name='new_test', data=ds, target_ntype='user', 
                                         label_mask_feat_name='mask',
                                         labeled_nodes_split_ratio=[0.8, 0.1, 0.1], force_reload=True)
    
    assert new_ds[0].num_nodes() == ds[0].num_nodes()
    print(new_ds[0])
    print(new_ds.train_idx, new_ds.test_idx, new_ds.val_idx, new_ds.pred_idx)
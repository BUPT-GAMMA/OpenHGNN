from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FOLD = Path(__file__).resolve().parent / "meirec"


class MeiRECDataset(Dataset):

    def __init__(self, phase):
        assert phase in ['train', 'test']
        if phase == 'train':
            self.data_path = FOLD / "train_data.txt"
        else:
            self.data_path = FOLD / "test_data.txt"
        self.load_data(self.data_path)

    def load_data(self, path):
        self.features_list = []
        self.label_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                feature, label = line.split()
                self.features_list.append(feature)
                self.label_list.append(float(label))
        self.data_len = len(self.features_list)
        # print(self.data_len)

    def __getitem__(self, item):
        index = item
        feature_list = []
        feature = self.features_list[index]
        features_split = feature.split(',')

        for item in features_split[:81]:
            feature_list.append(float(item))
        for item in features_split[81:]:
            feature_list.append(int(item))

        return {
            'data': torch.from_numpy(np.array(feature_list)),
            'labels': self.label_list[index]
        }

    def __len__(self):
        return self.data_len


def get_data_loader(type, batch_size=64, num_workers=0):
    dataset = MeiRECDataset(type)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader


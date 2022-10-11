# Read and process the meta.txt

from torch.utils.data import Dataset

import random
import math


class DataReader:
    def __init__(self, filename, block_size):
        self.filename = filename
        self.node_count = 0
        self.mg_count = 0
        self.node_dict = dict()
        self.mg_dict = dict()
        self.mg_num_dict = dict()
        self.node_reverse_dict = dict()
        self.mg_reverse_dict = dict()
        self.unigram = list()

        self.data_index = 0
        self.epoch_end = False

        self.file = open(filename, 'r')
        self.block_size = block_size
        self.train_file = list()

        self.read_train_file()

    def read_train_file(self):
        cnt = 0
        with open(self.filename, 'r') as f:
            for line in f:
                temp = list(line.strip('\n').split())
                if temp[2][1:] not in self.mg_dict:
                    self.mg_dict[temp[2][1:]] = self.mg_count
                    self.mg_num_dict[temp[2][1:]] = float(temp[3][1:])
                    self.mg_count += 1
                else:
                    self.mg_num_dict[temp[2][1:]] += float(temp[3][1:])
                if temp[0] not in self.node_dict:
                    self.node_dict[temp[0]] = self.node_count
                    self.node_count += 1
                if temp[1] not in self.node_dict:
                    self.node_dict[temp[1]] = self.node_count
                    self.node_count += 1
                cnt += 1
                if cnt % 10000000 == 0:
                    print("read lines: ", cnt)
        self.node_reverse_dict = dict(zip(self.node_dict.values(), self.node_dict.keys()))
        self.mg_reverse_dict = dict(zip(self.mg_dict.values(), self.mg_dict.keys()))
        for i in range(self.mg_count):
            self.unigram.append(int(math.ceil(self.mg_num_dict[self.mg_reverse_dict[i]])))
        print("read finish, total lines: ", cnt)
        print("total nodes: ", self.node_count)
        print("total metaGraphs: ", self.mg_count)

    def read_block(self):
        self.train_file = [0] * self.block_size
        instance_count = 0
        for i in range(self.block_size):
            line = self.file.readline()
            if line:
                temp = list(line.strip('\n').split())
                temp[0] = self.node_dict[temp[0]]
                temp[1] = self.node_dict[temp[1]]
                temp[2] = self.mg_dict[temp[2][1:]]
                temp[3] = float(temp[3][1:])
                self.train_file[instance_count] = temp
                instance_count += 1
            else:
                self.file.seek(0, 0)
                self.epoch_end = True
                break
        self.train_file = self.train_file[:instance_count]
        random.shuffle(self.train_file)


class Mg2vecSampler(Dataset):
    def __init__(self, filename, block_size, alpha):
        self.data = DataReader(filename, block_size)
        self.alpha = alpha
        self.data.read_block()

    def __len__(self):
        return len(self.data.train_file)

    def __getitem__(self, index):
        a = int(self.data.train_file[index][0])
        b = int(self.data.train_file[index][1])
        weight = (1 - self.alpha) if a == b else self.alpha
        label = int(self.data.train_file[index][2])
        freq = self.data.train_file[index][3]
        return a, b, label, freq, weight

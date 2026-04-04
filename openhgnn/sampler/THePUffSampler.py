import os
import pickle
import random

class THePUffSampler:
    def __init__(self, node_dict, node_dict_dp, node_index, dataset, max_path_len=5, walk_num_per_node=10):
        self.node_dict = node_dict
        self.node_index = node_index
        self.dataset = dataset
        self.max_path_len = max_path_len
        self.walk_num_per_node = walk_num_per_node
        self.node_dict_dp = node_dict_dp
        dir = os.getcwd()
        self.save_dir = os.path.join(dir, 'data')+'/'+self.dataset+'/'
        self.path = self.save_dir + self.dataset + '_p.pickle'
        self.dp_path = self.save_dir + self.dataset + '_dp_p.pickle'

    def sampler(self):
        degrees = {}
        for (node, edges) in self.node_dict.items():
            degrees[node] = len(edges)
        if not os.path.exists(self.path):
            self.random_walk(self.node_dict, self.node_index, degrees, self.walk_num_per_node, self.max_path_len, self.path)
        degrees_dp = {}
        for (node, edges) in self.node_dict_dp.items():
            degrees_dp[node] = len(edges)
        if not os.path.exists(self.dp_path):
            self.random_walk(self.node_dict_dp, self.node_index, degrees_dp, self.walk_num_per_node, self.max_path_len, self.dp_path)

    def random_walk(self, node_dict, node_index, degrees, num_of_walks, length_of_walks, output_file):
        with open(output_file, 'wb') as f:
            sequences = []
            for node in node_dict.keys():
                for i in range(min(num_of_walks, degrees[node])):
                    v = node
                    sequence = [node_index[node]]
                    if degrees[v] == 0:
                        continue
                    for j in range(length_of_walks):
                        if degrees[v] == 0:
                            break
                        p = 1 / degrees[v]
                        r = random.random()
                        k = int(r / p)
                        v_ = '{}_{}'.format(node_dict[v][k][0], node_dict[v][k][1])
                        sequence.append(node_index[v_])
                        v = v_
                    sequences.append(sequence)
            pickle.dump(sequences, f)
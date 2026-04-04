import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
import torch
import networkx as nx
from gensim.models import Word2Vec
import random
import math

class THePUffDataset():
    _dataset_config = {
        'ml-100k': {
            'edge_types': [('u', 'u:m', 'm'), ('u', 'u:o', 'o'), ('m', 'm:g', 'g')],
            'node_type_map': {'u': 0, 'm': 1, 'g': 2, 'o': 3},
            'node_classes': 4
        },
        'taobao': {
            'edge_types': [('u', 'u:i', 'i'), ('i', 'i:c', 'c')],
            'node_type_map': {'u': 0, 'i': 1, 'c': 2},
            'node_classes': 3
        },
        'dblp_kdd': {
            'edge_types': [('a', 'a:p', 'p'), ('p', 'p:p', 'p'), ('a', 'a:o', 'o'), ('p', 'p:f', 'f')],
            'node_type_map': {'a': 0, 'p': 1, 'o': 2, 'f': 3},
            'node_classes': 4
        }
    }

    def __init__(self, dataset='ml-100k', raw_dir=None, force_reload=False, if_dp=False, node_index=None, node_value=None, node_type=None, node_dict=None, node_dict_dp=None, node_level=None, node_level_dp=None, N=None):
        assert dataset in self._dataset_config.keys(), f"仅支持{list(self._dataset_config.keys())}"
        self.force_reload = force_reload
        self.N = N
        self.if_dp = if_dp
        self.dataset_name = dataset
        self.node_index = node_index
        self.node_dict = node_dict
        self.node_dict_dp = node_dict_dp
        self.node_level = node_level
        self.node_level_dp = node_level_dp
        self.node_value = node_value
        self.node_type = node_type
        self.cfg = self._dataset_config[dataset]
        dir = os.getcwd()
        self.final_raw_dir = raw_dir or os.path.join(dir, 'data')+'/'+self.dataset_name
        os.makedirs(self.final_raw_dir, exist_ok=True)
        self.data_path = os.path.join(self.final_raw_dir, f'{self.dataset_name}.txt')
        self.dp_data_path = os.path.join(self.final_raw_dir, f'{self.dataset_name}_dp.txt')
        self.processed_dir = os.path.join(self.final_raw_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        self.emb_path = os.path.join(self.processed_dir, f'{self.dataset_name}_emb.bin')
        self.dp_emb_path = os.path.join(self.processed_dir, f'{self.dataset_name}_emb_dp.bin')
        self.level_path = os.path.join(self.processed_dir, f'{self.dataset_name}_level')
        self.dp_level_path = os.path.join(self.processed_dir, f'{self.dataset_name}_level_dp')

    def download(self):
        _prefix = 'https://github.com/xinyuu-he/THePUff/blob/main/'
        full_url = _prefix + f'data/{self.dataset_name}/{self.dataset_name}.txt'
        if not os.path.exists(self.data_path):
            print(f"please download '{self.dataset_name}.txt' from '{full_url}' and put it to '{self.final_raw_dir}'")
        full_dp_url = _prefix + f'data/{self.dataset_name}/{self.dataset_name}_dp.txt'
        if not os.path.exists(self.dp_data_path):
            print(f"please download '{self.dataset_name}_dp.txt' from '{full_dp_url}' and put it to '{self.final_raw_dir}'")

    def _preprocess(self, directed=True):
        node_dict = dict()
        node_index = dict()
        node_value = dict()
        node_type = dict()
        original_node_index = dict()
        count = 0
        original_node_count = 0
        min_time_stamp = np.inf
        max_time_stamp = 0
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.split()
                line[4] = int(line[4])
                if line[4] < min_time_stamp:
                    min_time_stamp = line[4]
                if line[4] > max_time_stamp:
                    max_time_stamp = line[4]
        interval = max_time_stamp - min_time_stamp + 1
        time_slice = 1
        with open(self.data_path, 'r') as f:
            for line in f:
                nodes = line.split()
                nodes[4] = int(nodes[4])
                nodes[4] = int((nodes[4] - min_time_stamp) / time_slice)
                if nodes[0] not in original_node_index:
                    original_node_index[nodes[0]] = original_node_count
                    original_node_count += 1
                if nodes[1] not in original_node_index:
                    original_node_index[nodes[1]] = original_node_count
                    original_node_count += 1
                if '{}_{}'.format(nodes[0], nodes[4]) not in node_index:
                    node_value[count] = '{}_{}'.format(nodes[0], nodes[4])
                    node_index['{}_{}'.format(nodes[0], nodes[4])] = count
                    node_dict['{}_{}'.format(nodes[0], nodes[4])] = []
                    node_type[count] = self.cfg['node_type_map'][nodes[2]]
                    count += 1
                if '{}_{}'.format(nodes[1], nodes[4]) not in node_index:
                    node_value[count] = '{}_{}'.format(nodes[1], nodes[4])
                    node_index['{}_{}'.format(nodes[1], nodes[4])] = count
                    node_dict['{}_{}'.format(nodes[1], nodes[4])] = []
                    node_type[count] = self.cfg['node_type_map'][nodes[3]]
                    count += 1
                if (nodes[1], nodes[4]) not in node_dict['{}_{}'.format(nodes[0], nodes[4])]:
                    node_dict['{}_{}'.format(nodes[0], nodes[4])].append((nodes[1], nodes[4]))
                if not directed:
                    if (nodes[0], nodes[4]) not in node_dict['{}_{}'.format(nodes[1], nodes[4])]:
                        node_dict['{}_{}'.format(nodes[1], nodes[4])].append((nodes[0], nodes[4]))
                if max_time_stamp < nodes[4]:
                    max_time_stamp = nodes[4]
        node_level = dict()
        for i in range(len(node_index)):
            node_level[i] = [i]
        for node in node_dict.keys():
            nodes = node.split('_')
            nodes[1] = int(nodes[1])
            for i in range(interval):
                if i != nodes[1] and '{}_{}'.format(nodes[0], i) in node_dict:
                    node_level[node_index[node]].append(node_index['{}_{}'.format(nodes[0], i)])
        return node_dict, node_index, original_node_index, node_value, node_level, node_type

    def _preprocess_dp(self):
        count = len(self.node_value)
        node_dict_dp = dict()
        min_time_stamp = np.inf
        max_time_stamp = 0
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.split()
                line[4] = int(line[4])
                if line[4] < min_time_stamp:
                    min_time_stamp = line[4]
                if line[4] > max_time_stamp:
                    max_time_stamp = line[4]
        interval = max_time_stamp - min_time_stamp + 1
        with open(self.data_path, 'r') as f:
            for line in f:
                nodes = line.split()
                nodes[4] = int(nodes[4])
                if '{}_{}'.format(nodes[0], nodes[4]) not in self.node_index:
                    self.node_value[count] = '{}_{}'.format(nodes[0], nodes[4])
                    self.node_index['{}_{}'.format(nodes[0], nodes[4])] = count
                    self.node_type[count] = self.cfg['node_type_map'][nodes[2]]
                    count += 1
                if '{}_{}'.format(nodes[1], nodes[4]) not in self.node_index:
                    self.node_value[count] = '{}_{}'.format(nodes[1], nodes[4])
                    self.node_index['{}_{}'.format(nodes[1], nodes[4])] = count
                    self.node_type[count] = self.cfg['node_type_map'][nodes[3]]
                    count += 1
                if '{}_{}'.format(nodes[0], nodes[4]) not in node_dict_dp:
                    node_dict_dp['{}_{}'.format(nodes[0], nodes[4])] = []
                if '{}_{}'.format(nodes[1], nodes[4]) not in node_dict_dp:
                    node_dict_dp['{}_{}'.format(nodes[1], nodes[4])] = []
                if (nodes[1], nodes[4]) not in node_dict_dp['{}_{}'.format(nodes[0], nodes[4])]:
                    node_dict_dp['{}_{}'.format(nodes[0], nodes[4])].append((nodes[1], nodes[4]))
        node_level_dp = dict()
        for i in range(len(self.node_index)):
            node_level_dp[i] = [i]
        for node in node_dict_dp.keys():
            nodes = node.split('_')
            nodes[1] = int(nodes[1])
            for i in range(interval):
                if i != nodes[1] and '{}_{}'.format(nodes[0], i) in node_dict_dp:
                    node_level_dp[self.node_index[node]].append(self.node_index['{}_{}'.format(nodes[0], i)])
        return node_dict_dp, self.node_index, self.node_value, node_level_dp, self.node_type

    def get_data(self):
        if not os.path.exists(self.processed_dir + "/node_dict.p") \
            or not os.path.exists(self.processed_dir + "/node_index.p") \
            or not os.path.exists(self.processed_dir + "/original_node_index.p") \
            or not os.path.exists(self.processed_dir + "/node_value.p") \
            or not os.path.exists(self.processed_dir + "/node_level.p") \
            or not os.path.exists(self.processed_dir + "/node_type.p")\
            or self.force_reload:
            node_dict, node_index, original_node_index, node_value, node_level, node_type = self._preprocess()
            with open(self.processed_dir + "/node_dict.p", 'wb') as f:
                pickle.dump(node_dict, f)
            with open(self.processed_dir + "/node_index.p", 'wb') as f:
                pickle.dump(node_index, f)
            with open(self.processed_dir + "/original_node_index.p", 'wb') as f:
                pickle.dump(original_node_index, f)
            with open(self.processed_dir + "/node_value.p", 'wb') as f:
                pickle.dump(node_value, f)
            with open(self.processed_dir + "/node_level.p", 'wb') as f:
                pickle.dump(node_level, f)
            with open(self.processed_dir + "/node_type.p", 'wb') as f:
                pickle.dump(node_type, f)
        else:
            with open(self.processed_dir + "/node_dict.p", 'rb') as f:
                node_dict = pickle.load(f)
            with open(self.processed_dir + "/node_index.p", 'rb') as f:
                node_index = pickle.load(f)
            with open(self.processed_dir + "/original_node_index.p", 'rb') as f:
                original_node_index = pickle.load(f)
            with open(self.processed_dir + "/node_value.p", 'rb') as f:
                node_value = pickle.load(f)
            with open(self.processed_dir + "/node_level.p", 'rb') as f:
                node_level = pickle.load(f)
            with open(self.processed_dir + "/node_type.p", 'rb') as f:
                node_type = pickle.load(f)
        return node_dict, node_index, original_node_index, node_value, node_level, node_type

    def get_dp_data(self):
        if not os.path.exists(self.processed_dir + "/node_dict_dp.p") \
            or not os.path.exists(self.processed_dir + "/node_level_dp.p")\
            or self.force_reload:
            node_dict_dp, node_index, node_value, node_level_dp, node_type = self._preprocess_dp()
            with open(self.processed_dir + "/node_dict_dp.p", 'wb') as f:
                pickle.dump(node_dict_dp, f)
            with open(self.processed_dir + "/node_level_dp.p", 'wb') as f:
                pickle.dump(node_level_dp, f)
        else:
            with open(self.processed_dir + "/node_dict_dp.p", 'rb') as f:
                node_dict_dp = pickle.load(f)
            with open(self.processed_dir + "/node_index.p", 'rb') as f:
                node_index = pickle.load(f)
            with open(self.processed_dir + "/node_value.p", 'rb') as f:
                node_value = pickle.load(f)
            with open(self.processed_dir + "/node_level_dp.p", 'rb') as f:
                node_level_dp = pickle.load(f)
            with open(self.processed_dir + "/node_type.p", 'rb') as f:
                node_type = pickle.load(f)
        return node_dict_dp, node_index, node_value, node_level_dp, node_type

    def get_output(self):
        self.download()
        if self.if_dp:
            out_data = self.get_dp_data()
        else:
            out_data = self.get_data()
        return out_data

    def embedding(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(self.emb_path):
            self.get_emb(self.dataset_name, self.node_dict, self.node_type, self.node_index, self.emb_path)
        if not os.path.exists(self.dp_emb_path):
            self.get_emb(self.dataset_name, self.node_dict_dp, self.node_type, self.node_index, self.dp_emb_path)
        node_embs = KeyedVectors.load_word2vec_format(self.emb_path, binary=True)
        node_embs_dp = KeyedVectors.load_word2vec_format(self.dp_emb_path, binary=True)
        if not os.path.exists(self.level_path):
            self.level_emb(self.level_path, self.node_level, self.node_value, node_embs)
        if not os.path.exists(self.dp_level_path):
            self.level_emb(self.dp_level_path, self.node_level_dp, self.node_value, node_embs_dp)
        real_level_embs, dp_level_embs = self.load_level_embeddings(self.N, self.level_path, self.dp_level_path)
        node_type_classified, node_embs_classified = self.get_classified(self.node_type, node_embs_dp, device)
        node_embs.add_vector(-1, np.zeros(128))
        node_embs_dp.add_vector(-1, np.zeros(128))
        return  real_level_embs, dp_level_embs, node_type_classified, node_embs_classified, node_embs, node_embs_dp

    def get_emb(self, dataset, node_dict, node_type, node_index, output_path):
        if dataset == 'dblp_kdd':
            with open('./data/' + dataset + '/edgelist.txt', 'w') as fl, open('./data/' + dataset + '/edgetype.txt','w') as ft:
                for node0 in node_dict.keys():
                    idx0 = node_index[node0]
                    type0 = node_type[idx0]
                    for (node1, time) in node_dict[node0]:
                        idx1 = node_index['{}_{}'.format(node1, time)]
                        type1 = node_type[idx1]
                        if type0 == 0 and type1 == 1:
                            fl.write('a' + str(idx0) + ' p' + str(idx1) + '\n')
                        elif type0 == 1 and type1 == 1:
                            fl.write('p' + str(idx0) + ' p' + str(idx1) + '\n')
                        elif type0 == 0 and type1 == 2:
                            fl.write('a' + str(idx0) + ' o' + str(idx1) + '\n')
                        elif type0 == 1 and type1 == 3:
                            fl.write('p' + str(idx0) + ' f' + str(idx1) + '\n')
                        else:
                            print(node0, '{}_{}'.format(node1, time), idx0, idx1, type0, type1)
                            assert False, 'unexpected edge type'
                ft.write('a : p' + '\n')
                ft.write('p : p' + '\n')
                ft.write('a : o' + '\n')
                ft.write('p : f' + '\n')
        elif dataset == 'ml-100k':
            with open('./data/' + dataset + '/edgelist.txt', 'w') as fl, open('./data/' + dataset + '/edgetype.txt','w') as ft:
                for node0 in node_dict.keys():
                    idx0 = node_index[node0]
                    type0 = node_type[idx0]
                    for (node1, time) in node_dict[node0]:
                        idx1 = node_index['{}_{}'.format(node1, time)]
                        type1 = node_type[idx1]
                        if type0 == 0 and type1 == 1:
                            fl.write('u' + str(idx0) + ' m' + str(idx1) + '\n')
                        elif type0 == 0 and type1 == 3:
                            fl.write('u' + str(idx0) + ' o' + str(idx1) + '\n')
                        elif type0 == 1 and type1 == 2:
                            fl.write('m' + str(idx0) + ' g' + str(idx1) + '\n')
                        else:
                            print(node0, '{}_{}'.format(node1, time), idx0, idx1, type0, type1)
                            assert False, 'unexpected edge type'
                ft.write('u : m' + '\n')
                ft.write('u : o' + '\n')
                ft.write('m : g' + '\n')
        elif dataset == 'taobao':
            with open('./data/' + dataset + '/edgelist.txt', 'w') as fl, open('./data/' + dataset + '/edgetype.txt','w') as ft:
                for node0 in node_dict.keys():
                    idx0 = node_index[node0]
                    type0 = node_type[idx0]
                    for (node1, time) in node_dict[node0]:
                        idx1 = node_index['{}_{}'.format(node1, time)]
                        type1 = node_type[idx1]
                        if type0 == 0 and type1 == 1:
                            fl.write('u' + str(idx0) + ' i' + str(idx1) + '\n')
                        elif type0 == 1 and type1 == 2:
                            fl.write('i' + str(idx0) + ' c' + str(idx1) + '\n')
                        else:
                            print(node0, '{}_{}'.format(node1, time), idx0, idx1, type0, type1)
                            assert False, 'unexpected edge type'
                ft.write('u : i' + '\n')
                ft.write('i : c' + '\n')
        self.just(input='./data/' + dataset + '/edgelist.txt', output = output_path, node_types = './data/' + dataset + '/edgetype.txt')

    def level_emb(self, path, node_level, node_value, embs):
        role_level_emb = dict()
        for idx in range(len(embs)):
            role_level_emb[node_value[idx]] = embs[idx]
        with open(path, 'w') as f_emb:
            f_emb.write('{} {}\n'.format(len(node_level), len(embs[0])))
            for i in range(len(node_level)):
                sum = role_level_emb[node_value[node_level[i][0]]]
                for j in range(1, len(node_level[i])):
                    sum = sum + role_level_emb[node_value[node_level[i][j]]]
                f_emb.write('{} '.format(i) + ' '.join(map(str, sum / len(node_level[i]))) + '\n')

    def load_level_embeddings(self, N, real_level_path, dp_level_path):
        real_level_emb = np.zeros((N + 1, 128))
        with open(real_level_path, 'r') as f_emb:
            next(f_emb)
            for line in f_emb:
                line = line.split()
                real_level_emb[int(line[0])] = np.array(list(map(float, line[1:])))
        dp_level_emb = np.zeros((N + 1, 128))
        with open(dp_level_path, 'r') as f_emb:
            next(f_emb)
            for line in f_emb:
                line = line.split()
                dp_level_emb[int(line[0])] = np.array(list(map(float, line[1:])))
        return real_level_emb.astype(np.float32), dp_level_emb.astype(np.float32)

    def get_classified(self, node_type, node_embs, device):
        node_type_classfied = dict()
        for i in node_type:
            if node_type[i] not in node_type_classfied:
                node_type_classfied[node_type[i]] = []
            node_type_classfied[node_type[i]].append(i)
        node_embs_classified = dict()
        for i in node_type_classfied.keys():
            temp = []
            for j in node_type_classfied[i]:
                temp.append(torch.tensor(node_embs[j]).unsqueeze(0))
            node_embs_classified[i] = torch.cat(temp, 0).to(device)
        return node_type_classfied, node_embs_classified

    def just(self, input, node_types, output, dimensions=128, walk_length=100, num_walks=10, window_size=10, workers=1, alpha=0.5):
        G = nx.read_edgelist(input)
        heterg_dictionary = self.generate_node_types(node_types)
        if self.dataset_name == 'taobao':
            walk_length = 10
            num_walks = 1
        walks = self.generate_walks(G, num_walks, walk_length, heterg_dictionary, alpha)
        model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=0, workers=workers, batch_words=200)
        model.wv.save_word2vec_format(output, binary=True)

    def generate_node_types(self, node_types):
        heterg_dictionary = {}
        heterogeneous_node_types = open (node_types)
        for line in heterogeneous_node_types:
            node_type = (line.split(":")[0]).strip()
            hete_value = (line.split(":")[1]).strip()
            if node_type in heterg_dictionary.keys():
                heterg_dictionary[node_type].append(hete_value)
            else:
                heterg_dictionary[node_type] = [hete_value]
        return heterg_dictionary

    def generate_walks(self, G, num_walks, walk_length, heterg_dictionary, alpha):
        walks = []
        nodes = list(G.nodes())
        for cnt in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                just_walks = self.dblp_generation(G, walk_length, heterg_dictionary, alpha, start=node)
                walks.append(just_walks)
        return walks

    def dblp_generation(self, G, path_length, heterg_dictionary, alpha, start=None):
        path = []
        path.append(start)
        cnt = 1
        homog_length = 1
        no_next_types = 0
        heterg_probability = 0
        while len(path) < path_length:
            if no_next_types == 1:
                break
            cur = path[-1]
            homog_type = []
            heterg_type = []
            for node_type in heterg_dictionary:
                if cur[0] == node_type:
                    homog_type = node_type
                    heterg_type = heterg_dictionary[node_type]
            heterg_probability = 1 - math.pow(alpha, homog_length)
            r = random.uniform(0, 1)
            next_type_options = []
            if r <= heterg_probability:
                for heterg_type_iterator in heterg_type:
                    next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
                if not next_type_options:
                    next_type_options = [e for e in G[cur] if (e[0] == homog_type)]
            else:
                next_type_options = [e for e in G[cur] if (e[0] == homog_type)]
                if not next_type_options:
                    for heterg_type_iterator in heterg_type:
                        next_type_options.extend([e for e in G[cur] if (e[0] == heterg_type_iterator)])
            if not next_type_options:
                no_next_types = 1
                break
            next_node = random.choice(next_type_options)
            path.append(next_node)
            if next_node[0] == cur[0]:
                homog_length = homog_length + 1
            else:
                homog_length = 1
        return path
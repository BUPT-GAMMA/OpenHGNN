import dgl
import dgl.function as fn
import numpy as np
import array
import torch
import random
import networkx as nx
import os
import pickle
import networkx as nx

from dgl._ffi.base import DGLError
from networkx.exception import NetworkXException
from torch.serialization import save
from networkx.utils import pairwise

"""
Notice:
1. All the ids come from the whole graph
2. This sampler is irrelevant to train/valid/test 
"""
class SLiCESampler(object):
    def __init__(self,g,g_sample,num_walks_per_node,
        beam_width,max_num_edges,walk_type,path_option,save_path) -> None:
        super().__init__()
        self.g=g_sample
        self.g_all=g
        self.g_networkx=dgl.to_networkx(self.g,edge_attrs=['label'])
        self.node_types=dict()
        for node in g.nodes():
            self.node_types[int(node)]=int(g.ndata['_TYPE'][int(node)])
        self.num_walks_per_node=num_walks_per_node
        self.beam_width=beam_width
        self.walk_type=walk_type#pretrain
        self.path_option=path_option
        self.max_num_edges=max_num_edges
        self.last_context=[]
        self.node_context_dict=dict()
        self.edge_context_dict=dict()
        self.save_path=save_path
        self.pretrain_path=os.path.join(save_path,'pretrain')
        self.finetune_path=os.path.join(save_path,'finetune')
        random.seed(10)
    def get_node_subgraph(self,seed_nodes) -> list:
        """
        Params:
        seed_nodes: the tensor of seed nodes for each subgraph
        Return:
        return a list of sampled node subgraph
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.pretrain_path):
            os.makedirs(self.pretrain_path)
        all_contexts=[]
        for source in seed_nodes:
            source=int(source)
            node_contexts=[]
            if source in self.node_context_dict:
                node_contexts=self.node_context_dict[source]
                all_contexts.extend(node_contexts)
                continue
            for _ in range(self.num_walks_per_node):
                if self.walk_type == "bfs":
                    node_context = self.bfs_beam(source, 
                                self.beam_width, self.max_num_edges)
                elif self.walk_type == "dfs":
                    node_context = self.random_chain(source, self.max_num_edges)
                else:
                    print("Unknown context generation strategy, select bfs/dfs")
                # if no subgraph
                if len(node_context)==0:
                    node_context=self.last_context
                node_subgraph=dgl.node_subgraph(self.g,node_context)
                all_contexts.append(node_subgraph)
                node_contexts.append(node_subgraph)
                self.last_context=node_subgraph
            self.node_context_dict[source]=node_contexts
        return all_contexts#list of subgraph
    def get_edge_subgraph(self,seed_edges) -> list:
        """
        Parameters:
        seed_edges: List[(int,int)] a list of edge src and dst
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.finetune_path):
            os.makedirs(self.finetune_path)
        g_networkx=self.g_networkx
        all_context=[]
        for ii,edge in enumerate(seed_edges):#10399 (6682,51)
            src,dst=edge
            if not self.g_all.has_edges_between(src,dst):   #a false edge
                label=0
            else:
                eid=self.g_all.edge_ids(src,dst)
                label=int(self.g_all.edata['label'][eid])
            if (src,dst) in self.edge_context_dict:
                context,pair=self.edge_context_dict[(src,dst)]
                all_context.extend(context)
            else:
                context=self.get_selective_context(g_networkx,src,dst)
                if len(context) == 0:
                    context = [src,dst]   
                this_context=[]
                this_pair=[]
                if isinstance(context[0],list):#context has many walks
                    for each in context:
                        edge_context=dgl.node_subgraph(self.g,each)
                        this_context.append(edge_context)
                        this_pair.append((src,dst,label))
                else:
                    edge_context=dgl.node_subgraph(self.g,context)
                    this_context.append(edge_context)
                    this_pair.append((src,dst,label))
                self.edge_context_dict[(src,dst)]=(this_context,this_pair)
                all_context.extend(this_context)
        return all_context
    
    def get_random_k_nbrs(self,nodeid,exclude_list,k):
        """
        returns randomly sampled 'k' nbrs for a given node as
        list[node_ids]
        if k == -1 :  returns all neighbours
        """
        #get all neighbors
        all_nbrs=[]
        # all_nbrs.extend(dgl.sampling.sample_neighbors(self.g,torch.tensor([nodeid]),-1,
        #                                         edge_dir='out',prob='label').edges(order='eid')[1].tolist())  
        all_nbrs.extend(dgl.sampling.sample_neighbors(self.g,torch.tensor([nodeid]),-1,
                                                edge_dir='out').edges(order='eid')[1].tolist())
        if exclude_list is None or len(exclude_list) == 0:
            all_nbrs = list(set(all_nbrs))
        else:
            all_nbrs = list(set(x for x in all_nbrs if x not in exclude_list))
        if len(all_nbrs)<k:
            # all_nbrs.extend(dgl.sampling.sample_neighbors(self.g,torch.tensor([nodeid]),-1,
            #                                     edge_dir='in',prob='label').edges(order='eid')[0].tolist())
            all_nbrs.extend(dgl.sampling.sample_neighbors(self.g,torch.tensor([nodeid]),-1,
                                                edge_dir='in').edges(order='eid')[0].tolist())                   
        if exclude_list is None or len(exclude_list) == 0:
            all_nbrs = list(set(all_nbrs))
        else:
            all_nbrs = list(set(x for x in all_nbrs if x not in exclude_list))
        if k >= 0 and len(all_nbrs) >= k:
            return random.sample(all_nbrs, k)
        else:
            return all_nbrs
    
    def bfs_beam(self, source, beam_width, max_num_edges):
        """
        Given a source node, a beam width and depth=hops
        return a subgraph around source node .
        The subgraph is retuned as [(source, target, relation)]
        """
        source=int(source)
        subgraph = [source]
        G = self.g
        
        source_nbrs = self.get_random_k_nbrs(source, exclude_list=[], k=beam_width)
        cnt=0
        for x in source_nbrs:
            if cnt>=max_num_edges:
                break
            subgraph.append(x)
            cnt+=1
        for src_nbr in source_nbrs:
            src_nbr_hop2_cands = self.get_random_k_nbrs(
                src_nbr, exclude_list=[source], k=beam_width
            )
            for x in src_nbr_hop2_cands:
                if cnt>=max_num_edges:
                    break
                subgraph.append(x)
                cnt+=1
        # print(len(subgraph))
        if len(subgraph)==0:
            print("node %d has no subgraph"%source)
        if len(subgraph)>max_num_edges:
            print("node %d has subgraph bigger than max_num_edges"%source)
        return subgraph
    def random_chain(self, source, max_num_edges):
        # generates a random walk of length=max_num_edges, without cycles
        # starting from source
        source=int(source)
        path = [source]
        nodes_so_far = [source]
        G = self.g
        #原代码有bug，会导致采样数量不对
        for _ in range(max_num_edges):
            nbrs = self.get_random_k_nbrs(source, exclude_list=nodes_so_far, k=max_num_edges)
            nbr = -1
            while True:
                cursor=-1
                if len(nbrs) != 0:
                    for each in nbrs:
                        if each not in nodes_so_far:
                            nbr=each
                            break
                if nbr==-1:#如果没有邻居或者所有邻居均已被采样
                    source=nodes_so_far[cursor]#退回上一个点，重新采样
                    cursor-=1
                    nbrs=self.get_random_k_nbrs(source, exclude_list=nodes_so_far, k=1)
                else:
                    break
            path.append(nbr)
            nodes_so_far.append(source)
            source = nbr
        return path
    def get_selective_context(
        self, G, source, target, valid_patterns=None
    ):  # FIXME - removed dangerous default list init in valid_patterns
        """
        option = 1) all  2) pattern  3)shortest
        1) Return all paths connecting source and target of length <=max_seq_len
        2) Return only paths that match given pattern set
            #valid_patterns = set(['1_1_1', '2_2_2', '1_2_1', '2_1_2'])
            (1.1: generates paths connecting source and target of length <= max_seq_len
             1.2 filter paths by pattern
        3) Return shortest path between source and target
        4) Return random path between source and target
        """
        beam_width=self.beam_width
        max_seq_len=self.max_num_edges
        option=self.path_option
        paths = []
        if option == "shortest":
            try:
                path_node_list = nx.bidirectional_shortest_path(G, source, target)
                if len(path_node_list)<=max_seq_len:
                    paths.append([source]+[target]+path_node_list[1:-1])
                else:
                    paths.append([source]+[target]+path_node_list[1:max_seq_len-1])
            except NetworkXException:
                return []

        elif option == "all":
            paths = list(nx.all_simple_paths(G, source, target, cutoff=max_seq_len))
            # print("found all paths", source, target, paths)
        elif option=='pattern':
            paths = list(nx.all_simple_paths(G, source, target, cutoff=max_seq_len))
        elif option == "random":
            paths = self.get_random_paths_between_nodes(
                source, target, beam_width, max_seq_len, currpath=[source]
            )
        elif option == "default":
            paths = self.get_context(source, target, beam_width, max_seq_len)
        return paths
    
    def get_random_paths_between_nodes(
        self, source, target, beam_width, max_seq_len, currpath
    ):
        """
        Generates random path between source and target , by selecting
        'beam_width' number of neighbors and upto length >= max_seq_len
        """
        if len(currpath) > max_seq_len or source == target:
            return []
        all_paths = []
        exclude_list = []
        source_nbrs = self.get_random_k_nbrs(source, exclude_list, beam_width)
        # print("selected source nbrs", source_nbrs)
        if target in source_nbrs:
            # print("Found path ending in target", source_nbrs, target)
            path = [source,target]
            all_paths.append(path)
        for n in source_nbrs:
            if n != target and n not in currpath:
                new_source = n
                new_path = list(currpath)
                new_path.append(new_source)
                nbr_paths = self.get_random_paths_between_nodes(
                    new_source, target, beam_width, max_seq_len, new_path
                )
                new_path = new_path[0:-1]
                for p in nbr_paths:
                    p.insert(0, new_source)
                all_paths.extend(nbr_paths)
        return all_paths
    
    def get_context(self, source, target, beam_width, max_seq_len):
        all_contexts = []
        # for i in range(self.num_walks_per_node):
        source_context = self.bfs_beam(source, beam_width)[0:max_seq_len]
        # target_context = self.bfs_beam(target, beam_width)[0:max_seq_len]
        all_contexts.append(source_context)
        # all_contexts.append(target_context)
        return all_contexts
    def generate_false_edges2(self, positive_edge_list, save_path):
        """
        This method takes in as input a list of positive edges of form
        (relation, u, v, "1")
        and tries to generates double number of false edges (not present in graph),
        such that for each edge in positive_edge_list:
            (relation,u, v', "0"), here u -> v' does not exist in self.G
            (relation,u', v, "0"), here u'-> v does not exist in self.G
        returns : random.shuffle(true_edges + false_edges)
        """
        # collect nodes of different types
        print("Generating false edges...")
        g=self.g
        node_type_to_ids = dict()
        for node_id in g.nodes():
            node_id=int(node_id)
            node_type=int(g.ndata['_TYPE'][node_id])
            if node_type in node_type_to_ids:
                node_type_to_ids[node_type].append(node_id)
            else:
                node_type_to_ids[node_type] = [node_id]

        # generate false edges for every positive example
        false_edges = []
        for source, target in positive_edge_list:
            # generate false edges of type (source, relation, false_target) for every
            # (source, relation, target) in positive_edge_list
            source=int(source)
            target=int(target)
            
            target_type = self.node_types[target]
            false_target = random.sample(node_type_to_ids[target_type], 1)[0]
            if not g.has_edges_between(source,false_target):
                false_edges.append((source, false_target))
            # generate false edges of type (false_source, relation, target) for every
            # (source, relation, target) in positive_edge_list
            source_type = self.node_types[source]
            false_source = random.sample(node_type_to_ids[source_type], 1)[0]
            if not g.has_edges_between(false_source,target):
                false_edges.append((false_source, target))

        # generate false edges of type (false_source, relation, target) for every
        final_edges = positive_edge_list + false_edges
        labels=[1]*len(positive_edge_list)+[0]*len(false_edges)
        (res_edge,res_label)=self.shuffle_edge_label(final_edges,labels)
        with open(save_path,'wb') as f:
            pickle.dump((res_edge,res_label),f)
        return (res_edge,res_label)
    def shuffle_edge_label(self,edges,labels):
        edges_tuple=list(zip(edges,labels))
        random.shuffle(edges_tuple)
        (res_edge,res_label)=zip(*edges_tuple)
        
        return (res_edge,res_label)
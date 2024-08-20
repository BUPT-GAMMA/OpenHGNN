import dgl
from ..utils import extract_metapaths
import numpy as np
import dgl.backend as F
from dgl.sampling.randomwalks import random_walk
from dgl.sampling.pinsage import _select_pinsage_neighbors
from dgl import utils
from dgl import convert
import torch as th



class RandomWalkerSampler(object):
    def __init__(
        self,
        G,
        num_traversals,
        termination_prob,
        num_random_walks,
        num_neighbors,
        metapath=None,
        weight_column="weights",
        device='cpu'
    ):
        """
        通过随机游走的范式构建IR和AR对。对于关系对ap，获取a->p->a->a3对关系对。对于apt，获取a->p->t->p->a两对关系对。
        """
        self.G = G
        self.weight_column = weight_column
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals
        self.device=device

        if metapath is None:
            if len(G.ntypes) > 1 or len(G.etypes) > 1:
                raise ValueError(
                    "Metapath must be specified if the graph is homogeneous."
                )
            metapath = [G.canonical_etypes[0]]
        self.start_ntype = G.to_canonical_etype(metapath[0])[0]
        self.end_ntype=G.to_canonical_etype(metapath[(len(metapath)-1)//2])[2]

        self.metapath_hops = len(metapath)
        self.metapath = metapath
        if num_traversals%2==0:
            num_traversals+=1
        self.full_metapath = metapath * (num_traversals//2) + metapath[:len(metapath)//2]
        restart_prob = np.zeros(self.metapath_hops * num_traversals)
        restart_prob[
            self.metapath_hops :: self.metapath_hops
        ] = termination_prob
        restart_prob = F.tensor(restart_prob, dtype=F.float32)
        self.restart_prob = F.copy_to(restart_prob, G.device)

    def __call__(self,seed_nodes):
        seed_nodes = utils.prepare_tensor(self.G, seed_nodes, "seed_nodes")
        self.restart_prob = F.copy_to(self.restart_prob, F.context(seed_nodes))

        seed_nodes = F.repeat(seed_nodes, self.num_random_walks, 0)
        paths, _ = random_walk(
            self.G,
            seed_nodes,
            metapath=self.full_metapath,
            restart_prob=self.restart_prob,
        )
        # random_walk会出现-1，需要改为0，但这样可能会出现大量（0,0）关系对，使模型出现误差
        paths=th.clamp(paths,min=0)
        src = F.reshape(
            paths[:, 0 :: self.metapath_hops], (-1,)
        )
        dst = F.reshape(
            paths[:, self.metapath_hops//2 :: self.metapath_hops], (-1,)
        )

        src, dst, counts = _select_pinsage_neighbors(
            src,
            dst,
            (self.num_random_walks * self.num_traversals),
            self.num_neighbors,
        )
        neighbor_graph = convert.heterograph(
            {(self.start_ntype, "_E1", self.end_ntype): (src, dst),
             (self.end_ntype, "_E2", self.start_ntype): (dst, src)},
            {self.start_ntype: self.G.num_nodes(self.start_ntype),
             self.end_ntype:self.G.num_nodes(self.end_ntype)},
        )
        neighbor_graph.edges['_E1'].data[self.weight_column] = counts
        neighbor_graph.edges['_E2'].data[self.weight_column] = counts


        return neighbor_graph.to(self.device)
        

class RHINESampler(dgl.dataloading.Sampler):
    def __init__(self,g,meta_paths_dict,num_neighbors,device):
        self.g=g
        self.meta_paths_dict=meta_paths_dict
        self.mp_sampler={}
        self.num_neighbors=num_neighbors
        self.device=device
        # 根据元路径采样
        for i in self.meta_paths_dict:
            self.mp_sampler[i]=RandomWalkerSampler(g,
                                                    num_traversals=1,
                                                    termination_prob=0,
                                                    num_random_walks=num_neighbors,
                                                    num_neighbors=num_neighbors,
                                                    metapath=self.meta_paths_dict[i],
                                                    device=self.device)
        

    def sample(self,g,seeds,exclude_eids=None):

        pos_mp_blocks={}
        neg_mp_blocks={}

        pos_input_nodes_dict={}
        neg_input_nodes_dict={}
        # 元路径采样
        for ntype,nid in seeds.items():
            for mp,sampler in self.mp_sampler.items():
                frontier=sampler(nid)
                # frontier=dgl.remove_self_loop(frontier,etype='_E1')
                # frontier=dgl.remove_self_loop(frontier,etype='_E2')

                pos_block=dgl.to_block(frontier).to(self.device)
                pos_mp_blocks[mp]=pos_block
                pos_input_nodes_dict[mp]=pos_block.srcdata[dgl.NID]
                # 进行负采样
                neg_sampler=dgl.sampling.global_uniform_negative_sampling(frontier,self.num_neighbors)
                neg_frontier=dgl.heterograph({
                    (sampler.start_ntype,'_E1',sampler.end_ntype):(neg_sampler[0],neg_sampler[1]),
                    (sampler.end_ntype,'_E2',sampler.start_ntype):(neg_sampler[1],neg_sampler[0])
                })
                # neg_frontier=dgl.remove_self_loop(neg_frontier,etype='_E1')
                # neg_frontier=dgl.remove_self_loop(neg_frontier,etype='_E2')
                neg_block=dgl.to_block(neg_frontier).to(self.device)
                neg_mp_blocks[mp]=neg_block
                neg_input_nodes_dict[mp]=neg_block.srcdata[dgl.NID]
        return pos_input_nodes_dict,neg_input_nodes_dict,pos_mp_blocks,neg_mp_blocks,seeds



class RHINETestSampler(dgl.dataloading.Sampler):
    def __init__(self,category,batch_size) -> None:
        super().__init__()
        self.category=category
        self.batch_size=batch_size

    def sample(self,g,seeds,exclude_eids=None):
        sg=dgl.in_subgraph(g,seeds)
        return sg,seeds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import dgl
import random
import dgl.nn as dglnn
from . import BaseModel, register_model
from .CompGCN import CompGraphConvLayer
import os

def get_norm_id(id_map, some_id):
    #如果不存在，返回一个id最大值
    if some_id not in id_map:
        id_map[some_id] = len(id_map)
    return id_map[some_id]

def norm_graph(node_id_map, edge_id_map, edge_list):
    norm_edge_list = []
    for e in edge_list:
        norm_edge_list.append(
            (
                get_norm_id(node_id_map, e[0]),
                get_norm_id(node_id_map, e[1]),
                get_norm_id(edge_id_map, e[2]),
            )
        )
    return norm_edge_list
class NodeEncoder(torch.nn.Module):
    def __init__(
        self,
        base_embedding_dim,
        num_nodes,
        pretrained_node_embedding_tensor,
        is_pre_trained,
    ):

        super().__init__()
        self.pretrained_node_embedding_tensor = pretrained_node_embedding_tensor
        self.base_embedding_dim = base_embedding_dim

        if not is_pre_trained:
            self.base_embedding_layer = torch.nn.Embedding(
                num_nodes, base_embedding_dim
            )#.cuda()
            self.base_embedding_layer.weight.data.uniform_(-1, 1)
        else:
            self.base_embedding_layer = torch.nn.Embedding.from_pretrained(
                pretrained_node_embedding_tensor
            )#.cuda()

    def forward(self, node_id):
        node_id = torch.LongTensor([int(node_id)])#.cuda()
        x_base = self.base_embedding_layer(node_id)

        return x_base

class GCNGraphEncoder(torch.nn.Module):
    def __init__(
        self,
        G,
        pretrained_node_embedding_tensor,
        is_pre_trained,
        base_embedding_dim,
        max_length,
    ):

        super().__init__()
        self.g = G
        self.base_embedding_dim = base_embedding_dim
        self.max_length = max_length
        self.no_nodes = self.g.num_nodes() #用DGL的表示方式
        self.no_relations = self.g.num_edges()
        # print('check *************', self.no_relations)

        self.node_embedding = NodeEncoder(
            base_embedding_dim,
            self.no_nodes,
            pretrained_node_embedding_tensor,
            is_pre_trained,
        )

        self.special_tokens = {"[PAD]": 0, "[MASK]": 1}
        self.special_embed = torch.nn.Embedding(
            len(self.special_tokens), base_embedding_dim
        )
        self.special_embed.weight.data.uniform_(-1, 1)

    def forward(self, subgraphs_list, masked_nodes):
        num_subgraphs = len(subgraphs_list)

        node_emb = torch.zeros(
            num_subgraphs, self.max_length + 1, self.base_embedding_dim#+1是因为包含
        )

        for ii,subgraph in enumerate(subgraphs_list):
            #node_id_map = batch_id_maps[ii][0]
            #edge_type_map = batch_id_maps[ii][1]
            masked_set = masked_nodes[ii]
            for node in subgraph.nodes():
                node_id=subgraph.ndata[dgl.NID][int(node)]
                if node_id not in masked_set:  # used to ignore the masked nodes
                    node_emb[ii][node] = self.node_embedding(int(node_id))

        # get embeddings for special tokens
        # will be used for masking and padding before bert layer
        special_tokens_embed = {}
        for token in self.special_tokens:
            node_id = Variable(torch.LongTensor([self.special_tokens[token]]))
            tmp_embed = self.special_embed(node_id)
            special_tokens_embed[self.special_tokens[token] + self.no_nodes] = {
                "token": token,
                "embed": tmp_embed,
            }

        return node_emb

def get_attn_pad_mask(subgraph_list, pad_id, max_len):         
    #seq_q and seq_k are both all_nodes, which is list(list(subgraph_nodes))                                                                                                                                                                                                                                                                                                                                                                                                           
    batch_size = len(subgraph_list)
    len_q=max_len
    # print(batch_size, len_q, len_k)
    pad_attn_mask = []
    for itm in subgraph_list:
        tmp_mask = []
        for sub in itm.ndata[dgl.NID]:
            if sub == pad_id:
                tmp_mask.append(True)
            else:
                tmp_mask.append(False)
        if len(tmp_mask)<max_len:
            tmp_mask=tmp_mask+[True]*(max_len-len(tmp_mask))
        pad_attn_mask.append(tmp_mask)
        # print(tmp_mask)
    # print('mask', len(pad_attn_mask), len(pad_attn_mask[0]))
    pad_attn_mask = Variable(torch.ByteTensor(pad_attn_mask)).unsqueeze(1)
    pad_attn_mask = pad_attn_mask#.cuda()

    return pad_attn_mask.expand(batch_size, len_q, len_q)  # batch_size x len_q x len_k


def gelu(x):
    """"Implementation of the gelu activation function by Hugging Face."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k):

        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        # print('mask', attn_mask.size())
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask == True, -1e9)#change dropped softmax value into 
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):

        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k  #dimension of K and Q
        self.d_v = d_v  #dimension of V
        self.d_model = d_model

        self.W_Q = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_K = torch.nn.Linear(d_model, d_k * n_heads)
        self.W_V = torch.nn.Linear(d_model, d_v * n_heads)
        self.scaled_dot_prod_attn = ScaledDotProductAttention(d_k)
        self.wrap = torch.nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layerNorm = torch.nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V, attn_mask=None):
        #This V is not the V matrix of dot attention. 
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)#128(batcch)*4(head)*7(n_nodes)*64(d_k)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_prod_attn(q_s, k_s, v_s, attn_mask=attn_mask)#context is H*A
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        output = self.wrap(context)

        return self.layerNorm(output + residual), attn

#fNN in the paper
class PoswiseFeedForwardNet(torch.nn.Module):
    def __init__(self, d_model, d_ff):

        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):

        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(
            enc_outputs
        )  # enc_outputs: [batch_size x len_q x d_model]

        return enc_outputs, attn


@register_model('SLiCE')
class SLiCE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        # if args.embed_dir:
        #     pretrained_node_embedding_tensor=load_pickle(args.embed_dir)
        
        return cls(G=hg,pretrained_node_embedding_tensor=None,args=args)#to-do: 命令行解析
    def load_pretrained_node2vec(self,filename, base_emb_dim):
        """
        loads embeddings from node2vec style file, where each line is
        nodeid node_embedding
        returns tensor containing node_embeddings
        for graph nodes 0 to n-1
        """
        node_embeddings = torch.empty(self.g.num_nodes(), 100)
        with open(filename, "r") as f:
            header = f.readline()
            emb_dim = int(header.strip().split()[1])
            for line in f:
                arr = line.strip().split()
                graph_node_id = arr[0]
                node_emb = [float(x) for x in arr[1:]]
                vocab_id = int(graph_node_id)
                if vocab_id >= 0:
                    node_embeddings[vocab_id] = torch.tensor(node_emb)
                # print(torch.tensor(node_emb).size())
        out = node_embeddings
        print("node2vec tensor", out.size())
        return out
    #参数来自原论文默认参数
    def __init__(self,
        G,  #G为DGLGraph
        args,
        pretrained_node_embedding_tensor,
        num_layers=6,
        d_model=200,
        d_k=64,
        d_v=64,
        d_ff=200 * 4,
        n_heads=4,
        is_pre_trained=False,
        base_embedding_dim=200,#dimension of base embedding
        max_length=6,#max length of walks
        num_gcn_layers=2,#number of gcn layers before bert
        node_edge_composition_func="mult",#options for node and edge compostion, sub|circ_conv|mult|no_rel
        get_embeddings=False,#indicate if need to get node vectors from BERT encoder output
        fine_tuning_layer=False,):

        super().__init__()
        #initialize
        self.g=G
        self.num_layers = num_layers
        self.d_model = d_model
        self.max_length = max_length
        self.get_embeddings = get_embeddings
        self.node_edge_composition_func = node_edge_composition_func
        self.fine_tuning_layer = fine_tuning_layer
        self.no_nodes = G.num_nodes()
        self.n_pred=args.n_pred
        #pretraining use node2vec if not exist
        if not os.path.exists(args.pretrained_embeddings):
            print("Run Node2vec to obtain pre-trained node embeddings ...")
            walks=[]
            for _ in range(10):
                nodes=list(G.nodes())
                random.shuffle(nodes)
                walk = dgl.sampling.node2vec_random_walk(G, torch.tensor(nodes), 1, 1, walk_length=80-1).tolist()#len=walk_length+1
                walks.extend(walk)
            walks = [list(map(str, walk)) for walk in walks]
            from gensim.models import Word2Vec
            model = Word2Vec(
                walks,
                # size=base_embedding_dim,
                window=10,
                min_count=0,
                sg=1,
                workers=8,
                # iter=1,
            )
            model.wv.save_word2vec_format(args.pretrained_embeddings)

        pretrained_node_embedding_tensor = self.load_pretrained_node2vec(
            args.pretrained_embeddings, base_embedding_dim
        )# (n_nodes*d_model)
        #FIXME 暂时是用随机初始化，pretrain tensor是None
        self.gcn_graph_encoder = GCNGraphEncoder(
            G,
            pretrained_node_embedding_tensor,
            is_pre_trained,
            base_embedding_dim,
            max_length,
        )

        self.layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, d_k, d_v, d_ff, n_heads) for _ in range(num_layers)]
        )#.cuda()
        self.linear = torch.nn.Linear(d_model, d_model)#.cuda()
        self.norm = torch.nn.LayerNorm(d_model)#.cuda()

        # decoder
        self.decoder = torch.nn.Linear(self.d_model, self.no_nodes)#.cuda()
    def set_fine_tuning(self):
        self.fine_tuning_layer = True
    def GCN_MaskGeneration(self,subgraph_sequences):
        n_pred=self.n_pred
        masked_nodes = []#node id masked
        masked_position = []# node index masked
        for subgraph in subgraph_sequences:
            num_nodes = subgraph.num_nodes()
            mask_index = random.sample(range(num_nodes), n_pred)
            subgraph_masked_nodes = []
            subgraph_masked_position = []
            for i in range(num_nodes):
                if i in mask_index:
                    subgraph_masked_nodes.append(subgraph.ndata[dgl.NID][i])
                    subgraph_masked_position.append(i)
            masked_nodes.append(subgraph_masked_nodes)
            masked_position.append(subgraph_masked_position)

        return torch.tensor(masked_nodes), torch.tensor(masked_position)
    def forward(self, subgraph_list):
        #subgraph list is a list of node subgraphs sampled by slice_sampler
        if self.fine_tuning_layer:
            masked_nodes=Variable(torch.LongTensor([[] for ii in range(len(subgraph_list))]))
        else:
            masked_nodes,masked_pos=self.GCN_MaskGeneration(subgraph_list)
        # 将节点embedding和关系的embedding初始化，并采样得到
        # context generation
        node_emb = self.gcn_graph_encoder(subgraph_list, masked_nodes)
        output = node_emb#.cuda()
        enc_self_attn_mask = get_attn_pad_mask(subgraph_list,self.no_nodes,self.max_length+1)
        # contextual translation
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
            try:
                layer_output = torch.cat((layer_output, output.unsqueeze(1)), 1)#output embedding of each layer
            except NameError:  # FIXME - replaced bare except
                layer_output = output.unsqueeze(1)#.cuda()

            if self.fine_tuning_layer:
                try:
                    att_output = torch.cat((att_output, enc_self_attn.unsqueeze(0)), 0)#output attention of each layer
                except NameError:  # FIXME - replaced bare except
                    att_output = enc_self_attn.unsqueeze(0)

        # new added for ablation study
        if self.num_layers == 0:
            layer_output = output.unsqueeze(1)
            att_output = "NA"

        if self.fine_tuning_layer:
            # print(output.size(), layer_output.size(), att_output.size())
            return output, layer_output, att_output
        else:
            masked_pos = masked_pos[:,:,None].expand(
                -1, -1, output.size(-1)
            )  # [batch_size, maxlen, d_model]
            h_masked = torch.gather(
                output, 1, masked_pos#.cuda()
            )  # masking position [batch_size, len, d_model]
            h_masked = self.norm(gelu(self.linear(h_masked)))
            pred_score = self.decoder(h_masked)  # [batch_size, maxlen, n_vocab]
            # print('check====', pred_score.size())
            
            if self.get_embeddings:
                return pred_score, masked_nodes, output
            else:
                return pred_score, masked_nodes

class SLiCEFinetuneLayer(torch.nn.Module):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(d_model=args.d_model,ft_d_ff=args.ft_d_ff,
        ft_layer=args.ft_layer,ft_drop_rate=args.ft_drop_rate,
        ft_input_option=args.ft_input_option, num_layers=args.num_layers)
    def __init__(
        self,
        d_model,
        ft_d_ff,
        ft_layer,
        ft_drop_rate,
        ft_input_option,
        num_layers,
    ):

        super().__init__()
        self.d_model = d_model
        self.ft_layer = ft_layer
        self.ft_input_option = ft_input_option
        self.num_layers = num_layers

        if ft_input_option in ["last", "last4_sum"]:
            cnt_layers = 1
        elif ft_input_option in ["last4_cat"]:
            cnt_layers = 4

        if self.num_layers == 0:
            cnt_layers = 1

        if self.ft_layer == "linear":
            self.ft_decoder = torch.nn.Linear(d_model * cnt_layers, d_model)#.cuda()
        elif self.ft_layer == "ffn":
            self.ffn1 = torch.nn.Linear(d_model * cnt_layers, ft_d_ff)#.cuda()
            print(self.num_layers, cnt_layers, self.ffn1)
            self.dropout = torch.nn.Dropout(ft_drop_rate)#.cuda()
            self.ffn2 = torch.nn.Linear(ft_d_ff, d_model)#.cuda()

    def forward(self, graphbert_layer_output):
        """
        graphbert_output = batch_sz * [CLS, source, target, relation, SEP] *
        [emb_size]
        """
        if self.ft_input_option == "last":
            # use the output from laster layer of graphbert
            graphbert_output = graphbert_layer_output[:, -1, :, :].squeeze(1)
            source_embedding = graphbert_output[:, 0, :].unsqueeze(1)
            destination_embedding = graphbert_output[:, 1, :].unsqueeze(1)
        else:
            # concatenate the output from the last four last four layers
            # add for ablation study
            no_layers = graphbert_layer_output.size(1)
            if no_layers == 1:
                start_layer = 0
            else:
                start_layer = no_layers - 4
            for ii in range(start_layer, no_layers):
                source_embed = graphbert_layer_output[:, ii, 0, :].unsqueeze(1)
                destination_embed = graphbert_layer_output[:, ii, 1, :].unsqueeze(1)
                if self.ft_input_option == "last4_cat":
                    try:
                        source_embedding = torch.cat(
                            (source_embedding, source_embed), 2
                        )
                        destination_embedding = torch.cat(
                            (destination_embedding, destination_embed), 2
                        )
                    except:
                        source_embedding = source_embed
                        destination_embedding = destination_embed
                elif self.ft_input_option == "last4_sum":
                    try:
                        source_embedding = torch.add(source_embedding, 1, source_embed)
                        destination_embedding = torch.add(
                            destination_embedding, 1, destination_embed
                        )
                    except:
                        source_embedding = source_embed
                        destination_embedding = destination_embed
        # print(source_embedding.size(), destination_embedding.size())

        if self.ft_layer == "linear":
            src_embedding = self.ft_decoder(source_embedding)
            dst_embedding = self.ft_decoder(destination_embedding)
        elif self.ft_layer == "ffn":
            src_embedding = torch.relu(self.dropout(self.ffn1(source_embedding)))
            src_embedding = self.ffn2(src_embedding)
            dst_embedding = torch.relu(self.dropout(self.ffn1(destination_embedding)))
            dst_embedding = self.ffn2(dst_embedding)

        dst_embedding = dst_embedding.transpose(1, 2)
        pred_score = torch.bmm(src_embedding, dst_embedding).squeeze(1)
        pred_score = torch.sigmoid(pred_score)
        # print('check+++++', pred_score.size())

        return pred_score, src_embedding, dst_embedding.transpose(1, 2)
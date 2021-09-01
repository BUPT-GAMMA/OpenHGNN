import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import dgl
import dgl.nn as dglnn
from . import BaseModel, register_model
from .CompGCN import CompGraphConvLayer
from ..utils.gcn_transform import GCNTransform
from ..utils.gcn_graph_encoder import GCNGraphEncoder
from ..utils.utils import load_pickle

def gcn_out2bert_input(node_emb, batch_id_maps, input_ids, special_tokens):
    for ii in range(len(input_ids)):
        for jj in range(len(input_ids[ii])):
            if input_ids[ii][jj] in batch_id_maps[ii][0]:
                gcn_id = batch_id_maps[ii][0][input_ids[ii][jj]]
                tmp_embed = node_emb[ii][gcn_id].unsqueeze(0)
            else:
                special_id = input_ids[ii][jj]
                tmp_embed = special_tokens[special_id]["embed"]
            if jj == 0:
                tmp_out = tmp_embed
            else:
                tmp_out = torch.cat((tmp_out, tmp_embed), 0)
        tmp_out = tmp_out.unsqueeze(0)
        if ii == 0:
            out = tmp_out
        else:
            out = torch.cat((out, tmp_out), 0)

    return out

def get_attn_pad_mask(seq_q, seq_k, pad_id):
    batch_size, len_q = len(seq_q), len(seq_q[0])
    batch_size, len_k = len(seq_k), len(seq_k[0])
    # print(batch_size, len_q, len_k)
    pad_attn_mask = []
    for itm in seq_k:
        tmp_mask = []
        for sub in itm:
            if sub == pad_id:
                tmp_mask.append(True)
            else:
                tmp_mask.append(False)
        pad_attn_mask.append(tmp_mask)
        # print(tmp_mask)
    # print('mask', len(pad_attn_mask), len(pad_attn_mask[0]))
    pad_attn_mask = Variable(torch.ByteTensor(pad_attn_mask)).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.cuda()

    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


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
        scores.masked_fill_(attn_mask == True, -1e9)
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
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = self.scaled_dot_prod_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        output = self.wrap(context)

        return self.layerNorm(output + residual), attn


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
        return cls(G=hg,pretrained_node_embedding_tensor=None)#to-do: 命令行解析
    #参数来自原论文默认参数
    def __init__(self,
        G,  #G为DGLGraph
        pretrained_node_embedding_tensor,
        n_layers=6,
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
        gcn_option="no_gcn",#preprocess bert input once or alternate gcn and bert, preprocess|alternate|no_gcn
        get_embeddings=False,#indicate if need to get node vectors from BERT encoder output
        fine_tuning_layer=False,):

        super().__init__()
        #initialize
        self.n_layers = n_layers
        self.d_model = d_model
        self.max_length = max_length
        self.get_embeddings = get_embeddings
        self.node_edge_composition_func = node_edge_composition_func
        self.gcn_option = gcn_option
        self.fine_tuning_layer = fine_tuning_layer
        self.no_nodes = G.num_nodes()

        if self.gcn_option == "preprocess":
            self.num_gcn_layers = num_gcn_layers
        elif self.gcn_option == "alternate":
            assert num_gcn_layers % 2 == 0
            self.num_gcn_layers = int(num_gcn_layers / 2)
        #FIXME 暂时是用随机初始化，pretrain tensor是None
        self.gcn_graph_encoder = GCNGraphEncoder(
            G,
            pretrained_node_embedding_tensor,
            is_pre_trained,
            base_embedding_dim,
            max_length,
        )

        if self.gcn_option in ["preprocess", "alternate"]:
            self.gcn_transform = GCNTransform(
                base_embedding_dim, self.num_gcn_layers, self.node_edge_composition_func
            )

        self.layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, d_k, d_v, d_ff, n_heads) for _ in range(n_layers)]
        ).cuda()
        self.linear = torch.nn.Linear(d_model, d_model).cuda()
        self.norm = torch.nn.LayerNorm(d_model).cuda()

        # decoder
        self.decoder = torch.nn.Linear(self.d_model, self.no_nodes).cuda()
    def set_fine_tuning(self):
        self.fine_tuning_layer = True

    def forward(self, subgraphs_list, all_nodes, masked_pos, masked_nodes):
        # 将节点embedding和关系的embedding初始化，并采样得到
        # context generation
        (
            norm_subgraph_list,
            node_emb,
            relation_emb,
            batch_id_maps,
            special_tokens_embed,#PAD和MASK
        ) = self.gcn_graph_encoder(subgraphs_list, masked_nodes)
        if self.gcn_option == "preprocess":
            # Preprocess bert input with 2 or 4 GCN layers
            node_emb, relation_emb = self.gcn_transform(
                "subgraph_list", norm_subgraph_list, node_emb, relation_emb
            )
            # print("Processed GCN subgraph encoder")
        node_emb = gcn_out2bert_input(
            node_emb, batch_id_maps, all_nodes, special_tokens_embed
        )
        output = node_emb.cuda()
        # print('check1',len(all_nodes[0]))
        enc_self_attn_mask = get_attn_pad_mask(all_nodes, all_nodes, self.no_nodes)
        # contextual translation
        for layer in self.layers:
            if self.gcn_option == "alternate":
                # FIXME is this option working?
                # Preprocess with 1/2 GCN layer before each bert layer
                node_emb, relation_emb = self.gcn_transform(
                    "subgraph_list", norm_subgraph_list, output.cpu(), relation_emb
                )
                node_emb = gcn_out2bert_input(
                    node_emb, batch_id_maps, all_nodes, special_tokens_embed
                )
                output = node_emb.cuda()
            output, enc_self_attn = layer(output, enc_self_attn_mask)
            try:
                layer_output = torch.cat((layer_output, output.unsqueeze(1)), 1)
            except NameError:  # FIXME - replaced bare except
                layer_output = output.unsqueeze(1)

            if self.fine_tuning_layer:
                try:
                    att_output = torch.cat((att_output, enc_self_attn.unsqueeze(0)), 0)
                except NameError:  # FIXME - replaced bare except
                    att_output = enc_self_attn.unsqueeze(0)

        # new added for ablation study
        if self.n_layers == 0:
            layer_output = output.unsqueeze(1)
            att_output = "NA"

        if self.fine_tuning_layer:
            # print(output.size(), layer_output.size(), att_output.size())
            return output, layer_output, att_output
        else:
            masked_pos = masked_pos[:, :, None].expand(
                -1, -1, output.size(-1)
            )  # [batch_size, maxlen, d_model]
            h_masked = torch.gather(
                output, 1, masked_pos.cuda()
            )  # masking position [batch_size, len, d_model]
            h_masked = self.norm(gelu(self.linear(h_masked)))
            pred_score = self.decoder(h_masked)  # [batch_size, maxlen, n_vocab]
            # print('check====', pred_score.size())

            if self.get_embeddings:
                return pred_score, output
            else:
                return pred_score

class FinetuneLayer(torch.nn.Module):
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(attr_graph=hg,d_model=args.d_model,ft_d_ff=args.ft_d_ff,
        ft_layer=args.ft_layer,ft_drop_rate=args.ft_drop_rate,
        ft_input_option=args.ft_input_option,n_layers=args.n_layers)
    def __init__(
        self,
        d_model,
        ft_d_ff,
        ft_layer,
        ft_drop_rate,
        attr_graph,
        ft_input_option,
        n_layers,
    ):

        super().__init__()
        self.d_model = d_model
        self.ft_layer = ft_layer
        self.ft_input_option = ft_input_option
        self.n_layers = n_layers

        if ft_input_option in ["last", "last4_sum"]:
            cnt_layers = 1
        elif ft_input_option in ["last4_cat"]:
            cnt_layers = 4

        if self.n_layers == 0:
            cnt_layers = 1

        if self.ft_layer == "linear":
            self.ft_decoder = torch.nn.Linear(d_model * cnt_layers, d_model).cuda()
        elif self.ft_layer == "ffn":
            self.ffn1 = torch.nn.Linear(d_model * cnt_layers, ft_d_ff).cuda()
            print(self.n_layers, cnt_layers, self.ffn1)
            self.dropout = torch.nn.Dropout(ft_drop_rate).cuda()
            self.ffn2 = torch.nn.Linear(ft_d_ff, d_model).cuda()

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
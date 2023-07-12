import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
from openhgnn.models import BaseModel,register_model
# from torch_geometric.nn import GCNConv
from dgl.nn.pytorch import GraphConv

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.0):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, data):
        x = data
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        return x

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, device='cpu', save_mem=False, use_bn=True):
        super(GCN, self).__init__()
        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            # GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops)
            GraphConv(in_channels, hidden_channels, allow_zero_in_degree=False)
        )

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                # GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem,
                #         add_self_loops=add_self_loops)
                GraphConv(hidden_channels, hidden_channels, allow_zero_in_degree=False)
            )

            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            # GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem,
            #         add_self_loops=add_self_loops)
            GraphConv(hidden_channels, out_channels, allow_zero_in_degree=False)
        )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.g.ndata['feat']
        edges = data.g.edges()
        edges_tensor = torch.cat([edges[0].unsqueeze(0), edges[1].unsqueeze(0)], 0)
        x = x.to(self.device)
        edges_tensor = edges_tensor.to(self.device)
        graph = data.g.to(self.device)
        for i, conv in enumerate(self.convs[:-1]):
            # x = conv(x, edges_tensor)
            x = conv(graph,x)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, edges_tensor)
        x = self.convs[-1](graph,x)
        return x

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss

@register_model('DSSL')
class DSSL(BaseModel):

    @classmethod
    def build_model_from_args(cls, args, hg):
        if args.encoder == 'GCN':
            encoder = GCN(in_channels=args.feature_dim,
                          hidden_channels=args.hidden_channels,
                          out_channels=args.hidden_channels,
                          num_layers=args.num_layers, use_bn=not args.no_bn,
                          dropout=args.dropout,
                          device = args.device).to(args.device)
        return cls(encoder,args.hidden_channels,args.dataset,args.device,args.cluster_num,args.alpha,args.gamma,args.tao,
            args.beta,args.tau)

    def __init__(self,encoder, hidden_channels,dataset, device, cluster_num, alpha,gamma,tao,beta,moving_average_decay=0.0):
        super(DSSL, self).__init__()
        self.dataset = dataset
        self.device = device
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.gamma = gamma
        self.tao = tao
        self.beta = beta
        self.inner_dropout = True
        self.inner_activation = True
        self.online_encoder = encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)

        self.mlp_inference = MLP(hidden_channels, hidden_channels, cluster_num, 2)
        self.mlp_predictor = MLP(hidden_channels, hidden_channels, hidden_channels, 2)
        self.clusters = nn.Parameter(nn.init.normal_(torch.Tensor(hidden_channels, cluster_num)))
        self.mlp_predictor2 = MLP(cluster_num, hidden_channels, hidden_channels, 1)

        self.Embedding_mlp = True
        self.inference_mlp = True

    def reset_parameters(self):
        self.mlp_inference.reset_parameters()
        self.mlp_predictor.reset_parameters()
        self.mlp_predictor2.reset_parameters()
        self.online_encoder.reset_parameters()
        self.target_encoder = copy.deepcopy(self.online_encoder)

    def forward(self, embedding,neighbor_embedding):
        # expand node embedding
        embedding_more = embedding[:, None, :]
        embedding_expand = embedding_more.expand(-1, neighbor_embedding.shape[1], -1)
        # loss
        if self.inference_mlp == True:  # self.inference_mlp = True
            k, k_node, entropy_loss = self.inference_network(embedding_expand, neighbor_embedding)
        # else:
        #     k, k_node, entropy_loss = self.inference_network2(embedding_expand, neighbor_embedding)

        if self.Embedding_mlp == True:  # self.Embedding_mlp = True
            main_loss = self.generative_network(embedding_expand, k, neighbor_embedding)
        # else:
        #     main_loss = self.generative_network2(embedding_expand, k, neighbor_embedding)

        context_loss = self.context_network(embedding, k_node)

        return main_loss, context_loss, entropy_loss, k_node

        # N × K, N × 10 * K, k is N*32
    def inference_network(self, embedding_expand, neighbor_embedding):
        # get k
        cat_embedding = embedding_expand * neighbor_embedding
        # cat_embedding=torch.cat((embedding_expand,neighbor_embedding))
        k = F.softmax(self.mlp_inference(cat_embedding), dim=2)
        k_node = k.mean(dim=1)  # to get P(k|x)
        negative_entropy = k_node * torch.log(k_node + 1e-10)
        # minimize negative entropy
        entropy_loss = negative_entropy.sum(-1).mean()
        return k, k_node, entropy_loss

    def generative_network(self, embedding_expand, k, neighbor_embedding):
        #
        # re-parameterization trick
        gumbel_k = F.gumbel_softmax(k, hard=False)
        central=self.mlp_predictor(embedding_expand)+ self.beta*self.mlp_predictor2(gumbel_k)
        neighbor=neighbor_embedding
        loss= loss_fn(central, neighbor.detach()).mean()

        return loss

    def context_network(self, embedding, k_node):
        # self.clusters.data = F.normalize(self.clusters.data, dim=-1, p=2)
        kprior = torch.matmul(embedding, self.clusters)         # 【1024，6】
        kprior = F.softmax(kprior/self.tao, dim=1)

        context_loss = k_node *torch.log(kprior+1e-10)          # 【1024，6】
        context_loss = - 1.0 * context_loss.sum(-1).mean()
        return context_loss

    def update_moving_average(self):
        #assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,self.target_encoder, self.online_encoder)

    def update_cluster(self, new_center,batch_sum):
        with torch.no_grad():
            out_ids = torch.arange(self.cluster_num).to(self.device)
            out_ids = out_ids.long()
            self.clusters.index_copy_(1, out_ids, new_center)
            self.clusters.data=torch.mul(self.clusters.data.T ,1/ (batch_sum+1)).T


    def extra_loss(self):
        pass

    def get_emb(self):
        pass


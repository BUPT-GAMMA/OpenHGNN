from utils.dgl_graph import load_HIN
from model.NSHE import NSHE
import torch.optim as optim
from utils.trainer import train
from utils.evaluater import evaluate


def OpenHGNN(config):
    #load the graph
    g = load_HIN()
    #select the model
    model = NSHE(g=g, gnn_model="GCN", project_dim=64, emd_dim=64, context_dim=64)
    #model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # train the model
    node_emb = train(model, g, optimizer, args)  # 模型训练
    print("Train finished")
    # evaluate the performance
    model.eval()
    evaluate(node_emb, g.t_info)
    return
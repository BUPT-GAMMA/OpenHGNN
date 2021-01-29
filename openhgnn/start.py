from openhgnn.utils.dgl_graph import load_HIN
from openhgnn.model.NSHE import NSHE

from openhgnn.utils.trainer import train
# from openhgnn.utils.evaluater import evaluate


def OpenHGNN(config):
    #load the graph
    g = load_HIN()
    #select the model
    model = NSHE(g=g, gnn_model="GCN", project_dim=config.dim_size['project'],
                 emd_dim=config.dim_size['emd'], context_dim=config.dim_size['context'])
    #model.cuda()

    # train the model
    node_emb = train(model, g, config)  # 模型训练
    print("Train finished")
    # evaluate the performance
    model.eval()
    evaluate(node_emb, g.t_info)
    return
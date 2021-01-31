from openhgnn.utils.dgl_graph import load_HIN
from openhgnn.model.NSHE import NSHE

from openhgnn.utils.trainer import train
# from openhgnn.utils.evaluater import evaluate
from openhgnn.utils.evaluater import evaluate_acm

def OpenHGNN(config):
    #load the graph
    g = load_HIN().to(config.device)
    #select the model
    model = NSHE(g=g, gnn_model="GCN", project_dim=config.dim_size['project'],
                 emd_dim=config.dim_size['emd'], context_dim=config.dim_size['context']).to(config.device)
    #model.cuda()

    # train the model
    node_emb = train(model, g, config)  # 模型训练
    print("Train finished")
    # evaluate the performance
    model.eval()
    #evaluate_acm(config.seed, node_emb['paper'].detach().numpy(), g.nodes['paper'].data['label'], 3)
    evaluate_acm(config.seed, node_emb['movie'].detach().numpy(), g.nodes['movie'].data['label'], 3)
    return

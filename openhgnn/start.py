from openhgnn.model.NSHE import NSHE
from openhgnn.utils.trainer import run, run_GTN
from openhgnn.utils.evaluater import evaluate
from openhgnn.utils.dgl_graph import load_HIN
from openhgnn.model.GTN_sparse import GTN

def OpenHGNN(config):
    #load the graph
    g = load_HIN(config.dataset).to(config.device)
    # x = g.adj(scipy_fmt='coo',etype='author-paper')
    #select the model
    # model = NSHE(g=g, gnn_model="GCN", project_dim=config.dim_size['project'],
    #              emd_dim=config.dim_size['emd'], context_dim=config.dim_size['context']).to(config.device)
    model = GTN(num_edge=5,
                num_channels=config.num_channels,
                w_in=g.ndata['h']['paper'].shape[1],
                w_out=config.emd_size,
                num_class=3,
                num_layers=config.num_layers)
    #model.cuda()
    # train the model
    node_emb = run_GTN(model, g, config)  # 模型训练
    print("Train finished")
    # evaluate the performance
    evaluate(config.seed, config.dataset, node_emb, g)
    return



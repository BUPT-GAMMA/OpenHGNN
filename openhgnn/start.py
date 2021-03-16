from openhgnn.model.NSHE import NSHE
from openhgnn.utils.trainer import run, run_GTN, run_RSHN, run_RGCN, run_CompGCN, run_HetGNN
from openhgnn.utils.evaluater import evaluate
from openhgnn.utils.dgl_graph import load_HIN, load_KG
import torch.nn.functional as F

def OpenHGNN(config):
    # load the graph(HIN or KG)
    if config.model in ['GTN', 'NSHE', 'HetGNN']:
        hg, category, num_classes = load_HIN(config.dataset)
        config.category = category
        config.num_classes = num_classes
        hg = hg.to(config.device)
    elif config.model in ['RSHN', 'RGCN', 'CompGCN']:
        kg, category, num_classes = load_KG(config.dataset)
        config.category = category
        config.num_classes = num_classes
        kg = kg.to(config.device)

    # select the model
    if config.model == 'GTN':
        if config.sparse_flag == 'True':
            from openhgnn.model.GTN_sparse import GTN
            model = GTN(num_edge=5,
                        num_channels=config.num_channels,
                        w_in=hg.ndata['h']['paper'].shape[1],
                        w_out=config.emd_size,
                        num_class=3,
                        num_layers=config.num_layers)
        else:
            from openhgnn.model.GTN import GTN
            model = GTN(num_edge=5,
                        num_channels=config.num_channels,
                        w_in=hg.ndata['h']['paper'].shape[1],
                        w_out=config.emd_size,
                        num_class=3,
                        num_layers=config.num_layers,
                        norm=None)
        model.to(config.device)
        # train the model
        node_emb = run_GTN(model, hg, config)  # 模型训练
    elif config.model == 'NSHE':
        model = NSHE(g=hg, gnn_model="GCN", project_dim=config.dim_size['project'],
                 emd_dim=config.dim_size['emd'], context_dim=config.dim_size['context']).to(config.device)
        run(model, hg, config)
    elif config.model == 'RSHN':
        from openhgnn.model.RSHN import RSHN
        from openhgnn.utils.dgl_graph import coarsened_line_graph
        cl = coarsened_line_graph(rw_len=config.rw_len, batch_size=config.batch_size, n_dataset=config.dataset, symmetric=True)
        cl_graph = cl.get_cl_graph(kg).to(config.device)
        cl_graph = cl.init_cl_graph(cl_graph)
        model = RSHN(in_feats1=kg.num_nodes(), in_feats2=cl_graph.num_nodes(), dim=config.dim, num_classes=config.num_classes, num_node_layer=config.num_node_layer,
                     num_edge_layer=config.num_edge_layer, dropout=config.dropout).to(config.device)
        run_RSHN(model, kg, cl_graph, config)
    elif config.model == 'RGCN':
        # create model
        from openhgnn.model.RGCN import EntityClassify
        model = EntityClassify(kg.number_of_nodes(),
                               config.n_hidden,
                               config.num_classes,
                               len(kg.canonical_etypes),
                               num_bases=config.n_bases,
                               num_hidden_layers=config.n_layers - 2,
                               dropout=config.dropout,
                               use_self_loop=config.use_self_loop,use_cuda=True).to(config.device)
        run_RGCN(model, kg, config)
    elif config.model == 'CompGCN':
        from openhgnn.model.CompGCN import CompGCN
        n_rels = len(kg.etypes)
        model = CompGCN(in_dim=config.n_hidden,
                            hid_dim=config.n_hidden,
                            out_dim=config.num_classes,
                            n_nodes=kg.number_of_nodes(),
                            n_rels = n_rels,
                            num_layers=config.n_layers,
                            comp_fn=config.comp_fn,
                            dropout=config.dropout,
                            activation=F.relu,
                            batchnorm=True).to(config.device)
        run_CompGCN(model, kg, config)
    elif config.model == 'HetGNN':
        from openhgnn.utils.dgl_graph import hetgnn_graph
        hetg = hetgnn_graph(hg, config.dataset)
        het_graph = hetg.get_hetgnn_graph(config.rw_length, config.rw_walks, config.rwr_prob).to(config.device)
        from openhgnn.model.HetGNN import HetGNN
        het_graph = trans_feature(hg, het_graph)
        model = HetGNN(hg.ntypes, config.dim).to(config.device)
        run_HetGNN(model, het_graph, config)
        pass
    elif config.model == 'Metapath2vec':
        pass
    print("Train finished")
    # evaluate the performance
    # evaluate(config.seed, config.dataset, node_emb, g)
    return
def trans_feature(hg, het_gnn):
    for i in hg.ntypes:
        ndata = hg.nodes[i].data
        for j in ndata:
            het_gnn.nodes[i].data[j] = ndata[j]
    return het_gnn



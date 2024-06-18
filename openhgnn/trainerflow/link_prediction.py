import dgl
import torch as th
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from . import BaseFlow, register_flow
from ..models import build_model
from ..utils import EarlyStopping, add_reverse_edges, get_ntypes_from_canonical_etypes
import warnings
from torch.utils.tensorboard import SummaryWriter


@register_flow("link_prediction")
class LinkPrediction(BaseFlow):
    """
    Link Prediction trainer flows.
    Here is a tutorial teach you how to train a GNN for
    `link prediction <https://docs.dgl.ai/en/latest/tutorials/blitz/4_link_predict.html>_`.

    When training, you will need to remove the edges in the test set from the original graph.
    DGL recommends you to treat the pairs of nodes as another graph, since you can describe a pair of nodes with an edge.
    In link prediction, you will have a positive graph consisting of all the positive examples as edges,
    and a negative graph consisting of all the negative examples.
    The positive graph and the negative graph will contain the same set of nodes as the original graph.
    This makes it easier to pass node features among multiple graphs for computation.
    As you will see later, you can directly feed the node representations computed on the entire graph to the positive
    and the negative graphs for computing pair-wise scores.
    """

    def __init__(self, args):
        """

        Parameters
        ----------
        args

        Attributes
        ------------
        target_link: list
            list of edge types which are target link type to be predicted

        score_fn: str
            score function used in calculating the scores of links, supported function: distmult[Default if not specified] & dot product

        r_embedding: nn. ParameterDict
            In DistMult, the representations of edge types are involving the calculation of score.
            General models do not generate the representations of edge types, so we generate the embeddings of edge types.
            The dimension of embedding is `self.args.hidden_dim`.

        """
        super(LinkPrediction, self).__init__(args)

        self.target_link = self.task.dataset.target_link
        self.args.out_node_type = self.task.get_out_ntype()
        self.train_hg = self.task.get_train()
        if hasattr(self.args, 'flag_add_reverse_edges') \
                or self.args.dataset in ['ohgbl-MTWM', 'ohgbl-yelp1', 'ohgbl-yelp2']:
            self.train_hg = add_reverse_edges(self.train_hg)
        if not hasattr(self.args, 'out_dim'):
            self.args.out_dim = self.args.hidden_dim

        self.model = build_model(self.model).build_model_from_args(self.args, self.train_hg).to(self.device)

        if not hasattr(self.args, 'score_fn'):
            self.args.score_fn = 'distmult'
        if self.args.score_fn == 'distmult':
            """
            In DistMult, the representations of edge types are involving the calculation of score.
            General models do not generate the representations of edge types, so we generate the embeddings of edge types.
            """
            self.r_embedding = nn.ParameterDict({etype[1]: nn.Parameter(th.Tensor(1, self.args.out_dim))
                                                 for etype in self.train_hg.canonical_etypes}).to(self.device)
            for _, para in self.r_embedding.items():
                nn.init.xavier_uniform_(para)
        else:
            self.r_embedding = None

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        if self.args.score_fn == 'distmult':
            self.optimizer.add_param_group({'params': self.r_embedding.parameters()})
        self.patience = args.patience
        self.max_epoch = args.max_epoch

        self.positive_graph = self.train_hg.edge_type_subgraph(self.target_link).to(self.device)
        if self.args.mini_batch_flag:
            if not hasattr(args, 'fanout'):
                warnings.warn("please set fanout when using mini batch training.")
                args.fanout = -1
            if isinstance(args.fanout, list):
                self.fanouts = args.fanout
            else:
                self.fanouts = [args.fanout] * self.args.num_layers
            train_eid_dict = {
                etype: self.train_hg.edges(etype=etype, form='eid')
                for etype in self.target_link}
            sampler = dgl.dataloading.NeighborSampler(self.fanouts)
            negative_sampler = dgl.dataloading.negative_sampler.Uniform(2)
            sampler = dgl.dataloading.as_edge_prediction_sampler(sampler=sampler, negative_sampler=negative_sampler)
            self.dataloader = dgl.dataloading.DataLoader(
                self.train_hg, train_eid_dict, sampler,
                batch_size=self.args.batch_size,
                shuffle=True, device=self.device)
            self.category = self.hg.ntypes[0]
        self.writer = SummaryWriter(f'./openhgnn/output/{self.model_name}/')

        if self.args.mini_batch_flag and self.args.graphbolt:
            self.fanouts = [-1] * self.args.num_layers  
            dataset = gb.OnDiskDataset(self.task.dataset_GB.base_dir).load()

            # base_dir = "/data/zzh/TEST_DIR/TEST_base_dir"
            # dataset = gb.OnDiskDataset(base_dir).load()

            graph = dataset.graph.to(self.device)
            tasks = dataset.tasks
            link_pred_task = tasks[0]
            def create_dataloader(itemset,full_graph,split='train'):
                datapipe = gb.ItemSampler(itemset, batch_size=2048, shuffle=True)
                datapipe = datapipe.copy_to(self.device)
                if split != 'test':
                    datapipe = datapipe.sample_uniform_negative(full_graph, 1) 
                datapipe = datapipe.sample_neighbor(full_graph,  self.fanouts) 
                return gb.DataLoader(datapipe)
                
            self.train_GB_loader = create_dataloader(itemset=link_pred_task.train_set, 
                                                  full_graph=graph,
                                                  split='train')
            self.val_GB_loader = create_dataloader(itemset=link_pred_task.validation_set, 
                                                  full_graph=graph,
                                                  split='valid')
            self.test_GB_loader = create_dataloader(itemset=link_pred_task.test_set, 
                                                  full_graph=graph,
                                                  split='test')
                




    def preprocess(self):
        """
        In link prediction, you will have a positive graph consisting of all the positive examples as edges,
        and a negative graph consisting of all the negative examples.
        The positive graph and the negative graph will contain the same set of nodes as the original graph.
        """
        super(LinkPrediction, self).preprocess()
        # to('cpu') & to('self.device')
        self.train_hg = self.train_hg.to(self.device)

    def train(self):
        if self.args.graphbolt and self.args.mini_batch_flag:
            self.train_GB()
        else:
            self.preprocess()
            stopper = EarlyStopping(self.patience, self._checkpoint)
            for epoch in tqdm(range(self.max_epoch)):
                if self.args.mini_batch_flag:
                    loss = self._mini_train_step()
                else:
                    loss = self._full_train_setp()
                if epoch % self.evaluate_interval == 0:
                    val_metric = self._test_step('valid')
                    self.logger.train_info(
                        f"Epoch: {epoch:03d}, train loss: {loss:.4f}. " + self.logger.metric2str(val_metric))
                    self.writer.add_scalar('train_loss', loss, global_step=epoch)
                    self.writer.add_scalars('val_metric', val_metric['valid'], global_step=epoch)
                    early_stop = stopper.loss_step(val_metric['valid']['loss'], self.model)
                    if early_stop:
                        self.logger.train_info(f'Early Stop!\tEpoch:{epoch:03d}.')
                        break
            stopper.load_model(self.model)
            self.writer.close()
            # Test
            if self.args.test_flag:
                if self.args.dataset in ['HGBl-amazon', 'HGBl-LastFM', 'HGBl-PubMed']:
                    # Test in HGB datasets.
                    self.model.eval()
                    with torch.no_grad():
                        val_metric = self._test_step('valid')
                        self.logger.train_info(self.logger.metric2str(val_metric))
                        h_dict = self.model.input_feature()
                        embedding = self.model(self.train_hg, h_dict)
                        score = th.sigmoid(self.task.ScorePredictor(self.task.test_hg, embedding, self.r_embedding))
                        self.task.dataset.save_results(hg=self.task.test_hg, score=score,
                                                    file_path=self.args.HGB_results_path)
                    return dict(metric=val_metric, epoch=epoch)
                else:
                    test_score = self._test_step(split="test")
                    self.logger.train_info(self.logger.metric2str(test_score))
                    return dict(metric=test_score, epoch=epoch)
            elif self.args.prediction_flag:
                if self.args.mini_batch_flag:
                    prediction_res = self._mini_prediction_step()
                else:
                    prediction_res = self._full_prediction_step()
                return prediction_res

    def construct_negative_graph(self, hg):
        e_dict = {
            etype: hg.edges(etype=etype, form='eid')
            for etype in hg.canonical_etypes}
        neg_srcdst = self.negative_sampler(hg, e_dict)
        neg_pair_graph = dgl.heterograph(neg_srcdst,
                                         {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes})
        return neg_pair_graph

    def _full_train_setp(self):
        self.model.train()
        h_dict = self.model.input_feature()
        embedding = self.model(self.train_hg, h_dict)
        # construct a negative graph according to the positive graph in each training epoch.
        negative_graph = self.task.construct_negative_graph(self.positive_graph)
        loss = self.loss_calculation(self.positive_graph, negative_graph, embedding)
        # negative_graph = self.construct_negative_graph(self.train_hg)
        # loss = self.loss_calculation(self.train_hg, negative_graph, embedding)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self, ):
        self.model.train()
        all_loss = 0
        loader_tqdm = tqdm(self.dataloader, ncols=120)
        for input_nodes, positive_graph, negative_graph, blocks in loader_tqdm:
            positive_graph = positive_graph.to(self.device)
            negative_graph = negative_graph.to(self.device)
            if type(input_nodes) == th.Tensor:
                input_nodes = {self.category: input_nodes}
            input_features = self.model.input_feature.forward_nodes(input_nodes)
            logits = self.model(blocks, input_features)
            loss = self.loss_calculation(positive_graph, negative_graph, logits)
            all_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return all_loss

    def loss_calculation(self, positive_graph, negative_graph, embedding):
        p_score = self.task.ScorePredictor(positive_graph, embedding, self.r_embedding)
        n_score = self.task.ScorePredictor(negative_graph, embedding, self.r_embedding)

        p_label = th.ones(len(p_score), device=self.device)
        n_label = th.zeros(len(n_score), device=self.device)
        loss = F.binary_cross_entropy_with_logits(th.cat((p_score, n_score)), th.cat((p_label, n_label)))
        return loss

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.r_embedding.pow(2))

    def _test_step(self, split=None):
        if self.args.mini_batch_flag:
            return self._mini_test_step(split=split)
        else:
            return self._full_test_step(split=split)

    def _full_test_step(self, split=None):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            return {split: self.task.evaluate(embedding, self.r_embedding, mode=split)}

    def _mini_test_step(self, split=None):
        print('mini test...\n')
        self.model.eval()
        with th.no_grad():
            ntypes = get_ntypes_from_canonical_etypes(self.target_link)
            embedding = self._mini_embedding(model=self.model, fanouts=self.fanouts, g=self.train_hg,
                                             device=self.args.device, dim=self.model.out_dim, ntypes=ntypes,
                                             batch_size=self.args.batch_size)
            return {split: self.task.evaluate(embedding, self.r_embedding, mode=split)}

    def _full_prediction_step(self):
        self.model.eval()
        with th.no_grad():
            h_dict = self.model.input_feature()
            embedding = self.model(self.train_hg, h_dict)
            return self.task.predict(embedding, self.r_embedding)

    def _mini_prediction_step(self):
        self.model.eval()
        with th.no_grad():
            ntypes = get_ntypes_from_canonical_etypes(self.target_link)
            embedding = self._mini_embedding(model=self.model, fanouts=[-1] * self.args.num_layers, g=self.train_hg,
                                             device=self.args.device, dim=self.model.out_dim, ntypes=ntypes,
                                             batch_size=self.args.batch_size)
            return self.task.predict(embedding, self.r_embedding)

    def _mini_embedding(self, model, fanouts, g, device, dim, ntypes, batch_size):
        model.eval()
        with th.no_grad():
            sampler = dgl.dataloading.NeighborSampler(fanouts)
            indices = {ntype: torch.arange(g.num_nodes(ntype)).to(device) for ntype in ntypes}
            embedding = {ntype: torch.zeros(g.num_nodes(ntype), dim).to(device) for ntype in ntypes}
            dataloader = dgl.dataloading.DataLoader(
                g, indices, sampler,
                device=device,
                batch_size=batch_size)
            loader_tqdm = tqdm(dataloader, ncols=120)

            for i, (input_nodes, seeds, blocks) in enumerate(loader_tqdm):
                if not isinstance(input_nodes, dict):
                    input_nodes = {self.category: input_nodes}
                input_emb = model.input_feature.forward_nodes(input_nodes)
                output_emb = model(blocks, input_emb)

                for ntype, idx in seeds.items():
                    embedding[ntype][idx] = output_emb[ntype]
            return embedding



    def train_GB(self):
        self.preprocess()
        stopper = EarlyStopping(self.patience, self._checkpoint)
        for epoch in tqdm(range(self.max_epoch)):
            if self.args.mini_batch_flag:
                train_all_loss = self._mini_train_step_GB()  #   训练.把全部训练集中的边都跑一边
            if epoch % self.evaluate_interval == 0:
                val_metric = self._mini_test_step_GB(split='valid')   #   验证
                self.logger.train_info(
                    f"Epoch: {epoch:03d}, 整个训练集的总LOSS: {train_all_loss:.4f}. " + "       整个验证集的总AUC:" + str(val_metric) )                
                early_stop = stopper.loss_step(val_metric, self.model)
                if early_stop:
                    self.logger.train_info(f'提前停止!\tEpoch:{epoch:03d}.')
                    break
        stopper.load_model(self.model)
        # 测试
        if self.args.test_flag:
            all_test_metric = self._mini_test_step_GB(split = 'test') # 【测试集】上计算metric，即包括正边，也包括随机生成的负边
            self.logger.train_info(  "整个测试集的总AUC： "+  str(all_test_metric) )
            return dict(metric=all_test_metric, epoch=epoch)
            
    def _mini_train_step_GB(self, ):
        if self.args.graphbolt:
            self.model.train()
            all_loss = 0    #   整个train_set的Loss
            for step, data in enumerate(self.train_GB_loader):
                # 从Batch中取出输入节点及其特征，目标边及其标签
                #   目标边
                compacted_seeds = data.compacted_seeds
                #   目标边的标签
                labels = data.labels
                # node_features = {
                #     ntype: data.node_features[(ntype, "feat")]
                #     for ntype in data.blocks[0].srctypes
                # }   #   字典：{节点类型->节点特征}

                # input_nodes是   {节点类型：本次用到的  节点ID（整图）   }   
                #input_nodes、node_features 、input_features一一对应        ，都是第0层源节点的 整图ID以及对应特征
                input_nodes = data.input_nodes 
                blocks = data.blocks

                #   input_features是字典{节点类型：节点特征}.       
                #   input_features是第0层源节点  初始化之后的节点特征，node_features是原始数据集节点特征，而input_features是经过变换之后的初始特征
                input_features = self.model.input_feature.forward_nodes(input_nodes)

                #   字典 {节点类型：节点特征}   ，这里是最后一层目标节点的  更新之后的特征。
                #   这个节点特征  是用到连接预测的边上的源点和目点特征
                node_fts_final = self.model(blocks, input_features)

                edge_score_all_type = []
                label_all_type = []
                for cano_edge_type in list(compacted_seeds.keys()):
                    #   cano_edge_type 可以是 P-0-P
                    edge_src_id = compacted_seeds[cano_edge_type][:,0]  #   边上源点id
                    edge_dst_id = compacted_seeds[cano_edge_type][:,1]  #   边上目点Id

                    src_etype_dst = cano_edge_type.split(':')
                    src_ft = node_fts_final[    src_etype_dst[0]    ]   #   源点类型的全部特征
                    dst_ft = node_fts_final[    src_etype_dst[-1]    ]  #   目点类型的全部特征
                    #   train阶段，计算score只需要得到一个数就可以，不用sigmoid换成概率
                    edge_score_all_type.append( 
                                        th.sum(src_ft[edge_src_id] * dst_ft[edge_dst_id]  * self.r_embedding[ src_etype_dst[1] ] ,
                                        dim=1)     
                                        )
                    label_all_type.append(labels[cano_edge_type]   )

                edge_score_all_type = th.cat(edge_score_all_type)
                label_all_type = th.cat(label_all_type)
                #   单个batch的 LOSS，包括所有规范边类型，正边和负边都包括
                loss =  F.binary_cross_entropy_with_logits(edge_score_all_type,  label_all_type )
                all_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return all_loss/(step + 1)

    def _mini_test_step_GB(self, split):
        if self.args.graphbolt:
            self.model.eval()
            with torch.no_grad():
                #   一个epoch
                # 训练集，测试集，验证集。得到训练集的总LOSS和总ACC。验证集的...，测试集的...
                if split == 'train':
                    loader = self.train_GB_loader
                elif split == 'valid':
                    loader = self.val_GB_loader
                elif split == 'test':
                    loader = self.test_GB_loader
                all_true_labels = []
                all_preds = []
                if split != 'test':
                    all_true_labels = []
                    all_preds = []
                    for step, data in enumerate(loader):
                        # 从Batch中取出输入节点及其特征，目标边及其标签
                        #   目标边
                        compacted_seeds = data.compacted_seeds
                        #   目标边的标签
                        labels = data.labels
                        input_nodes = data.input_nodes  #   {节点类型：节点ID（整图）   }   ,input_nodes和node_fts一一对应
                        blocks = data.blocks
                        #   字典{节点类型：节点特征}.       input_features是第0层源节点  初始化之后的节点特征，node_fts是原始的节点特征，两者一一对应
                        input_features = self.model.input_feature.forward_nodes(input_nodes)
                        #   字典 {节点类型：节点特征}   ，这里是最后一层目标节点的  更新之后的特征。
                        #   这个节点特征  是用到连接预测的边上的源点和目点特征
                        node_fts_final = self.model(blocks, input_features)
                        edge_score_all_type = []
                        label_all_type = []
                        for cano_edge_type in list(compacted_seeds.keys()):
                            #   cano_edge_type 可以是 P-0-P
                            edge_src_id = compacted_seeds[cano_edge_type][:,0]  #   边上源点id
                            edge_dst_id = compacted_seeds[cano_edge_type][:,1]  #   边上目点Id
                            src_etype_dst = cano_edge_type.split(':')
                            src_ft = node_fts_final[src_etype_dst[0]]   #   源点类型的全部特征
                            dst_ft = node_fts_final[src_etype_dst[-1]]  #   目点类型的全部特征
                            #   求metric阶段， 边上score 用sigmoid换成概率
                            edge_score_all_type.append( 
                                                th.sigmoid(
                                                    th.sum(src_ft[edge_src_id] * dst_ft[edge_dst_id]  * self.r_embedding[ src_etype_dst[1] ] ,
                                                dim=1) 
                                                )    
                                                )
                            label_all_type.append(labels[cano_edge_type]   )

                        #   单个Batch的预测值   和 真实标签
                        edge_score_all_type = th.cat(edge_score_all_type)
                        label_all_type = th.cat(label_all_type)
                        #   把每个batch的预测值 和真实值  都拼接起来，构造一个训练集所有batch总的 metric
                        all_true_labels.append(label_all_type.detach().cpu())
                        all_preds.append(edge_score_all_type.detach().cpu())

                    all_true_labels = torch.cat(all_true_labels, dim=0)
                    all_preds = torch.cat(all_preds, dim=0)
                    import sklearn
                    #   split所有batch总的 auc
                    metric = sklearn.metrics.roc_auc_score(all_true_labels,all_preds)

                else:   #   测试集

                    all_preds = []
                    for step, data in enumerate(loader):
                        # 从Batch中取出输入节点及其特征，目标边及其标签
                        #   目标边
                        compacted_seeds = data.compacted_seeds
                        input_nodes = data.input_nodes  #   {节点类型：节点ID（整图）   }   ,input_nodes和node_fts一一对应
                        blocks = data.blocks
                        #   字典{节点类型：节点特征}.       input_features是第0层源节点  初始化之后的节点特征，node_fts是原始的节点特征，两者一一对应
                        input_features = self.model.input_feature.forward_nodes(input_nodes)
                        #   字典 {节点类型：节点特征}   ，这里是最后一层目标节点的  更新之后的特征。
                        #   这个节点特征  是用到连接预测的边上的源点和目点特征
                        node_fts_final = self.model(blocks, input_features)
                        edge_score_all_type = []
                        for cano_edge_type in list(compacted_seeds.keys()):
                            #   cano_edge_type 可以是 P-0-P
                            edge_src_id = compacted_seeds[cano_edge_type][:,0]  #   边上源点id
                            edge_dst_id = compacted_seeds[cano_edge_type][:,1]  #   边上目点Id
                            src_etype_dst = cano_edge_type.split(':')
                            src_ft = node_fts_final[src_etype_dst[0]]   #   源点类型的全部特征
                            dst_ft = node_fts_final[src_etype_dst[-1]]  #   目点类型的全部特征
                            #   求metric阶段， 边上score 用sigmoid  换成概率：预测为1类的概率
                            edge_score_all_type.append( 
                                                th.sigmoid(
                                                    th.sum(src_ft[edge_src_id] * dst_ft[edge_dst_id]  * self.r_embedding[ src_etype_dst[1] ] ,
                                                dim=1) 
                                                )    
                                                )

                        #   单个Batch的预测值   和 真实标签
                        edge_score_all_type = th.cat(edge_score_all_type)
                        #   把每个batch的预测值 和真实值  都拼接起来，构造一个训练集所有batch总的 metric
                        all_preds.append(edge_score_all_type.detach().cpu())

                    #   整个测试集的  预测值，此处已经是  概率
                    all_preds = torch.cat(all_preds, dim=0)
                    all_true_labels = torch.ones_like(all_preds, dtype=torch.int) 
                    all_true_labels[-1] = 0
                    import sklearn
                    #   测试集 所有batch总的 metric
                    metric = sklearn.metrics.roc_auc_score(all_true_labels,all_preds)
                    pred_prob = all_preds.cpu().detach().numpy()   #   sigmoid输出值为  预测为1类的概率
                    pred_label = (pred_prob >= 0.5).astype(int)
                    correct_predictions = (pred_label == all_true_labels.cpu().numpy())
                    accuracy = correct_predictions.mean()
                    print(' 整个测试集上的 ACC是:',accuracy)

            return metric




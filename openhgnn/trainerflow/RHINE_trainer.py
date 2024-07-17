from re import S
from tqdm import tqdm
from openhgnn.models import build_model
from openhgnn.tasks import build_task
from . import register_flow,BaseFlow
import dgl.graphbolt as gb
from openhgnn.sampler import RHINESampler,RHINETestSampler  
from ..utils import EarlyStopping
import dgl
import torch
import torch.nn as nn
from torch.autograd import Variable

@register_flow('rhine_trainer')
class RHINETrainer(BaseFlow):
    def __init__(self,args):
        super(RHINETrainer,self).__init__(args)
        self.ARs=args.ARs=self.task.dataset.ARs
        self.IRs=args.IRs=self.task.dataset.IRs
        self.meta_paths_dict=self.task.dataset.meta_paths_dict
        #建立索引
        self.mp_id={}
        for id,mp in enumerate(self.meta_paths_dict.keys()):
            self.mp_id[mp]=id
        #label
        self.labels = self.task.get_labels().to(self.device)

        args.out_dim=max(self.labels)+1

        self.train_idx=self.task.dataset.train_id
        self.valid_idx=self.task.dataset.valid_id
        self.test_idx=self.task.dataset.test_id
        self.pred_idx=self.task.dataset.pred_id
        self.category=self.task.dataset.category

        self.hg=self.task.get_graph().to(self.device)
        args.total_IRs=self.hg.num_edges()+1

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        # sample pos、neg
        sampler=RHINESampler(g=self.hg,meta_paths_dict=self.meta_paths_dict,
                            num_neighbors=self.args.batch_size,device=self.device)
        if self.train_idx is not None:
            self.train_loader = dgl.dataloading.DataLoader(
                self.hg, {self.category: self.train_idx.to(self.device)}, sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True)
        label_sampler=RHINETestSampler(category=self.category,batch_size=self.args.batch_size)
        if self.valid_idx is not None:
            self.val_loader = dgl.dataloading.DataLoader(
                self.hg, {self.category: self.valid_idx.to(self.device)}, label_sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True)
        if self.args.test_flag:
            self.test_loader = dgl.dataloading.DataLoader(
                self.hg, {self.category: self.test_idx.to(self.device)}, label_sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True)
        if self.args.prediction_flag:
            self.pred_loader = dgl.dataloading.DataLoader(
                self.hg, {self.category: self.pred_idx.to(self.device)}, label_sampler,
                batch_size=self.args.batch_size, device=self.device, shuffle=True)
            

    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            train_loss = self._mini_train_step()
            predict_train_loss=self._mini_train_predictor_step()
            if epoch % self.evaluate_interval == 0:
                modes = ['valid']
                if self.args.test_flag:
                    modes = modes + ['test']
                if hasattr(self, 'val_loader'):
                    metric_dict, losses = self._mini_test_step(modes=modes)
                    # train_score, train_loss = self._mini_test_step(modes='train')
                    # val_score, val_loss = self._mini_test_step(modes='valid')
                    val_loss = losses['valid']
                    self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, predict_train_loss: {predict_train_loss:.4f}, Valid loss: {val_loss:.4f}. "
                                        + self.logger.metric2str(metric_dict))
                    early_stop = stopper.loss_step(val_loss, self.model)
                    if early_stop:
                        self.logger.train_info('Early Stop!\tEpoch:' + str(epoch))
                        break

        stopper.load_model(self.model)
        if self.args.prediction_flag:
            if self.args.mini_batch_flag and hasattr(self, 'val_loader'):
                indices, y_predicts = self._mini_prediction_step()
            else:
                y_predicts = self._full_prediction_step()
                indices = torch.arange(self.hg.num_nodes(self.category))
            return indices, y_predicts
        
    def _mini_train_predictor_step(self):
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(self.test_loader, ncols=120)# 只有测试集有标签，因此用测试集训练predictor
        for i, (sg,seeds) in enumerate(loader_tqdm):
            seeds = seeds[self.category]
            logits = self.model(sg,self.category,mod='test')[seeds]
            lbl = self.labels[seeds].squeeze(1).to(self.device)
            loss = self.loss_fn(logits, lbl)
            loss_all += loss.item()
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / (i + 1)

    def _mini_test_step(self,modes=[ 'valid']):
        self.model.eval()
        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'valid':
                    loader_tqdm = tqdm(self.val_loader, ncols=120)
                elif mode == 'test':
                    loader_tqdm = tqdm(self.test_loader, ncols=120)
                y_trues = []
                y_predicts = []

                for i, (sg,seeds) in enumerate(loader_tqdm):
                    seeds = seeds[self.category]
                    logits = self.model(sg,self.category,mod='test')[seeds]
                    lbl = self.labels[seeds].squeeze(1).to(self.device)
                    loss = self.loss_fn(logits, lbl)
                    loss_all += loss.item()
                    y_trues.append(lbl.detach().cpu())
                    y_predicts.append(logits.detach().cpu())
                loss_all /= (i + 1)
                y_trues = torch.cat(y_trues, dim=0)
                y_predicts = torch.cat(y_predicts, dim=0)
                evaluator = self.task.get_evaluator(name='f1')
                metric_dict[mode] = evaluator(y_trues, y_predicts.argmax(dim=1).to('cpu'))
                loss_dict[mode] = loss
        return metric_dict, loss_dict

    def loss_funton(self,p_score,n_score):
        criterion = nn.MarginRankingLoss(1.0, False).to(self.device)
        y = Variable(torch.full([p_score.shape[0],1],-1)).to(self.device)
        loss = criterion(p_score, n_score, y)
        return loss

    def _mini_train_step(self):
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(self.train_loader, ncols=120)
        for i, (pos_input_nodes_dict,neg_input_nodes_dict,pos_mp_blocks,neg_mp_blocks,seeds) in enumerate(loader_tqdm):

            # 添加边特征
            for mp,block in pos_mp_blocks.items():
                block.edges['_E1'].data['r']=block.edges['_E2'].data['r']=torch.full([block.number_of_edges('_E1'),1],self.mp_id[mp]).to(self.device)
            for mp,block in neg_mp_blocks.items():
                block.edges['_E1'].data['r']=block.edges['_E2'].data['r']=torch.full([block.number_of_edges('_E1'),1],self.mp_id[mp]).to(self.device)

            seeds = seeds[self.category]
            
            pos_scores = self.model(pos_mp_blocks)

            neg_scores = self.model(neg_mp_blocks)
            loss = self.loss_funton(pos_scores, neg_scores)
            loss_all += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_all / (i + 1)



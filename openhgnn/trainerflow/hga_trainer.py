import dgl.data
from openhgnn.models import HAN, build_model
from openhgnn.tasks import build_task
from openhgnn.trainerflow import register_flow
from openhgnn.trainerflow.base_flow import BaseFlow
import dgl
import torch.nn.functional as F
from tqdm import tqdm
from ..utils import EarlyStopping
import torch
import numpy as np
from openhgnn.sampler import HANSampler


@register_flow('hga_trainer')
class HGATrainer(BaseFlow):
    def __init__(self, args):
        dataset=args.dataset.split(',')#将dataset分为两个部分，ACM，DBLP->[ACM,DBLP]
        args.dataset=dataset[0]#修改args再继承 #! 这里用了两个dataset，不好处理，而BaseFlow中要有一个dataset，但后续并不使用这个dataset
        super(HGATrainer,self).__init__(args)
        self.args.category = self.task.dataset.category# 需要进行分类的类别
        self.category = self.args.category
        self.num_classes = self.task.dataset.num_classes# 类别数

        self.args.out_node_type=[self.category]
        # 为两个图分别构建task
        self.taskS=build_task(args)#这里分别建立两个task。这里建立src task
        args.dataset=dataset[1]
        self.taskT=build_task(args)# 建立dst task
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg).to(self.device)

        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        self.max_epoch=self.max_epoch
        self.train_idxS, self.valid_idxS, self.test_idxS = self.taskS.get_split()
        self.train_idxT, self.valid_idxT, self.test_idxT = self.taskT.get_split()
        if self.args.prediction_flag:
            self.pred_idx = self.task.dataset.pred_idx
        if self.args.use_uva:
            self.gS=self.taskS.get_graph()
            self.gT=self.taskT.get_graph()
        else:
            self.gS=self.taskS.get_graph().to(self.device)
            self.gT=self.taskT.get_graph().to(self.device)

        self.S_label = self.taskS.get_labels().to(self.device)
        self.gamma=args.gamma

        self.S_label=self.taskS.get_labels().to(self.device)
        self.T_label=self.taskT.get_labels().to(self.device)

        if self.args.mini_batch_flag:#mini batch
            # 使用HAN的采样器
            samplerS = HANSampler(g=self.gS, seed_ntypes=[self.category], meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=20)
            samplerT = HANSampler(g=self.gT, seed_ntypes=[self.category], meta_paths_dict=self.args.meta_paths_dict,
                                 num_neighbors=20)
            if self.train_idxS is not None:
                self.train_loaderS = dgl.dataloading.DataLoader(
                    self.gS, {self.category: self.train_idxS.to(self.device)}, samplerS,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
                self.train_loaderT = dgl.dataloading.DataLoader(
                    self.gT, {self.category: self.train_idxT.to(self.device)}, samplerT,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.valid_idxS is not None:
                self.val_loader = dgl.dataloading.DataLoader(
                    self.gT, {self.category: self.valid_idxT.to(self.device)}, samplerT,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.test_flag:
                self.test_loader = dgl.dataloading.DataLoader(
                    self.gT, {self.category: self.test_idxT.to(self.device)}, samplerT,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)
            if self.args.prediction_flag:
                self.pred_loader = dgl.dataloading.DataLoader(
                    self.gT, {self.category: self.pred_idxT.to(self.device)}, samplerT,
                    batch_size=self.args.batch_size, device=self.device, shuffle=True)


    def preprocess(self):
        if hasattr(self.args, 'activation'):
            if hasattr(self.args.activation, 'weight'):
                import torch.nn as nn
                act = nn.PReLU()
            else:
                act = self.args.activation
        else:
            act = None
        # useful type selection
        if hasattr(self.args, 'feat'):
            pass
        else:
            # Default 0, nothing to do.
            self.args.feat = 0
        # 为两个图分别构建input_feature
        self.hg=self.gS
        self.feature_preprocess(act)
        self.input_featureS=self.input_feature

        self.hg=self.gT
        self.feature_preprocess(act)
        self.input_featureT=self.input_feature

        self.optimizer.add_param_group({'params': self.input_featureT.parameters()})
        self.optimizer.add_param_group({'params': self.input_featureS.parameters()})
        # for early stop, load the model with input_feature module.
        self.model.add_module('input_featureS', self.input_featureS)
        self.model.add_module('input_featureT', self.input_featureT)
        self.load_from_pretrained()


    def train(self):
        self.preprocess()
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter=range(self.max_epoch)
        for epoch in tqdm(epoch_iter):
            if self.args.mini_batch_flag:
                train_loss = self._mini_train_step()
            else:
                train_loss = self._full_train_step()
            if epoch % self.evaluate_interval == 0:
                modes = ['train', 'valid']
                if self.args.test_flag:
                    modes = modes + ['test']
                if self.args.mini_batch_flag:
                    metric_dict, losses = self._mini_test_step(modes=modes)
                    # train_score, train_loss = self._mini_test_step(modes='train')
                    # val_score, val_loss = self._mini_test_step(modes='valid')
                else:
                    metric_dict, losses = self._full_test_step(modes=modes)
                val_loss = losses['valid']
                self.logger.train_info(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Valid loss: {val_loss:.4f}. "
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

    def _full_train_step(self):
        self.model.train()
        mmd_loss=0
        l1_loss=0
        cls_lossS_MP=0
        hS_dict = self.model.input_featureS()
        hT_dict=self.model.input_featureT()
        homo_outS,homo_outT,clabel_predSs,clabel_predTs,target_probs,clabel_predS = self.model(gS=self.gS, h_dictS=hS_dict,gT=self.gT,h_dictT=hT_dict)
        #TODO mmd_loss:homo_out,S_label，clabel_predT1
        for i in range(len(clabel_predSs)):
            mmd_loss += self.lmmd(list(homo_outS.values())[i], list(homo_outT.values())[i], self.S_label, torch.nn.functional.softmax(clabel_predTs[i], dim=1))
        #TODO l1_loss:clabel_pred
        for i in range(len(clabel_predTs)):
            for j in range(i+1,len(clabel_predTs)):
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(clabel_predTs[j], dim=1)
                                                    - torch.nn.functional.softmax(clabel_predTs[i], dim=1)) )
        #TODO cls_lossS
        for i in range(len(clabel_predSs)):
            cls_lossS_MP += F.nll_loss(F.log_softmax(clabel_predSs[i], dim=1), self.S_label.long())
        cls_lossS = F.nll_loss(F.log_softmax(clabel_predS, dim=1), self.S_label.long())
        cls_lossT = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = cls_lossS + self.gamma * (mmd_loss + l1_loss) + cls_lossT
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _mini_train_step(self,logits=None):
        # return self._full_train_step()
        self.model.train()
        loss_all = 0.0
        loader_tqdm = tqdm(zip(self.train_loaderS,self.train_loaderT), ncols=120)
        for i, ((ntype_mp_name_input_nodes_dictS, seedS,gS),(ntype_mp_name_input_nodes_dictT,seedT,gT)) in enumerate(loader_tqdm):
            if len(seedS[self.category]) == len(seedT[self.category]):#计算mmd时，二者维度需要相同
                mmd_loss=0
                l1_loss=0
                cls_lossS_MP=0
                hS_dict = {}
                mp_name_input_nodes_dictS = ntype_mp_name_input_nodes_dictS[self.category]
                for meta_path_name, input_nodes in mp_name_input_nodes_dictS.items():
                    hS_dict[meta_path_name] = self.model.input_featureS.forward_nodes({self.category: input_nodes})
                hT_dict = {}
                mp_name_input_nodes_dictT = ntype_mp_name_input_nodes_dictT[self.category]
                for meta_path_name, input_nodes in mp_name_input_nodes_dictT.items():
                    hT_dict[meta_path_name] = self.model.input_featureT.forward_nodes({self.category: input_nodes})
                S_label = self.S_label[seedS[self.category]].to(self.device)
                homo_outS,homo_outT,clabel_predSs,clabel_predTs,target_probs,clabel_predS = self.model(gS=gS, h_dictS={self.category:hS_dict},gT=gT,h_dictT={self.category:hT_dict})
                #TODO mmd_loss:homo_out,S_label，clabel_predT1
                for i in range(len(clabel_predSs)):
                    mmd_loss += self.lmmd(list(homo_outS.values())[i], list(homo_outT.values())[i], S_label, torch.nn.functional.softmax(clabel_predTs[i], dim=1))
                #TODO l1_loss:clabel_pred
                for i in range(len(clabel_predTs)):
                    for j in range(i+1,len(clabel_predTs)):
                        l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(clabel_predTs[j], dim=1)
                                                            - torch.nn.functional.softmax(clabel_predTs[i], dim=1)) )
                #TODO cls_lossS
                for i in range(len(clabel_predSs)):
                    cls_lossS_MP += F.nll_loss(F.log_softmax(clabel_predSs[i], dim=1), S_label.long())
                cls_lossS = F.nll_loss(F.log_softmax(clabel_predS, dim=1), S_label.long())
                cls_lossT = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

                loss = cls_lossS + self.gamma * (mmd_loss + l1_loss) + cls_lossT
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return loss_all / (i + 1)

    def _full_test_step(self,modes,logits=None):
        self.model.eval()
        with torch.no_grad():
            h_dict = self.model.input_featureT()
            h_dict = {k: e.to(self.device) for k, e in h_dict.items()}
            logits = logits if logits else self.model(self.gT, h_dict)
            masks = {}
            for mode in modes:
                if mode == "train":
                    masks[mode] = self.train_idxT
                elif mode == "valid":
                    masks[mode] = self.valid_idxT
                elif mode == "test":
                    masks[mode] = self.test_idxT

            metric_dict = {key: self.taskT.evaluate(logits, mode=key) for key in masks}
            loss_dict = {key: self.loss_fn(logits[mask], self.labels[mask]).item() for key, mask in masks.items()}
            return metric_dict, loss_dict

    def _mini_test_step(self,modes):
        self.model.eval()
        with torch.no_grad():
            metric_dict = {}
            loss_dict = {}
            loss_all = 0.0
            for mode in modes:
                if mode == 'train':
                    loader_tqdm = tqdm(self.train_loaderT, ncols=120)
                elif mode == 'valid':
                    loader_tqdm = tqdm(self.val_loader, ncols=120)
                elif mode == 'test':
                    loader_tqdm = tqdm(self.test_loader, ncols=120)
                y_trues = []
                y_predicts = []

                for i, (ntype_mp_name_input_nodes_dict, seeds, ntype_mp_name_block_dict) in enumerate(loader_tqdm):
                    seeds = seeds[self.category]
                    mp_name_input_nodes_dict = ntype_mp_name_input_nodes_dict[self.category]
                    emb_dict = {}
                    for meta_path_name, input_nodes in mp_name_input_nodes_dict.items():
                        emb_dict[meta_path_name] = self.model.input_featureT.forward_nodes({self.category: input_nodes})
                    emb_dict = {self.category: emb_dict}
                    logits = self.model(ntype_mp_name_block_dict, emb_dict)
                    lbl = self.T_label[seeds].to(self.device)
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

    def guassian_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)#(512,512)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))#(512,512,512)
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))#(512,512,512)
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)

    def lmmd(self,source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, type='visual')

        weight_ss = torch.from_numpy(weight_ss).to(self.device)
        weight_tt = torch.from_numpy(weight_tt).to(self.device)
        weight_st = torch.from_numpy(weight_st).to(self.device)

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = torch.Tensor([0]).to(self.device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
        return loss

def convert_to_onehot(sca_label, class_num=4):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, type='visual', batch_size=32, class_num=4):
        # print('s_label',s_label.size())#[128])
        # print('t_label',t_label.size())#([128, 4])
        # exit()
        batch_size = min(s_label.size()[0],t_label.size()[0]) # 4177
        # print('batch_size',batch_size)
        s_sca_label = s_label.cpu().data.numpy().astype('int64')
        # print('s_sca_label',s_sca_label)
        s_vec_label = convert_to_onehot(s_sca_label)[:batch_size]
        # print('s_vec_label',s_vec_label)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        # print('s_sum',s_sum)
        s_sum[s_sum == 0] = 100
        # print('s_sum',s_sum)
        s_vec_label = s_vec_label / s_sum
        # print('s_vec_label',s_vec_label)

        # print('t_label',t_label)
        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        ###t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()[:batch_size,:]
        # print('t_vec_label',t_vec_label)
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        # print('t_sum',t_sum)
        t_vec_label = t_vec_label / t_sum
        # print('t_vec_label',t_vec_label)

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))
        # print('weight_ss',weight_ss)
        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        # print('set_s',set_s)
        # print('set_t',set_t)
        count = 0
        for i in range(class_num):#4
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                # print('s_tvec',s_tvec)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                # print('t_tvec',t_tvec)
                ss = np.dot(s_tvec, s_tvec.T)
                # print('ss',ss)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1
        # print('weight_st',weight_st)
        length = count  # len( set_s ) * len( set_t )
        # print('length',length)
        if length != 0:#####算出来是这个类和那个类，就只拉近这两个类的距离。
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        # print('weight_st',weight_st)
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')
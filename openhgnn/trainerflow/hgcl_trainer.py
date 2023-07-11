import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dataloader
import torch.optim as optim
import numpy as np
import datetime
import torch.utils.data as data
from openhgnn.trainerflow.base_flow import BaseFlow
from openhgnn.trainerflow import register_flow
from openhgnn.models import build_model
from ..tasks import build_task
from openhgnn.models.HGCL import HGCL

saveDefault = False

@register_flow('hgcltrainer')
class HGCLtrainer(BaseFlow):
    def __init__(self, args):
        super(HGCLtrainer, self).__init__(args)
        self.args = args

        self.task = build_task(args)
        self.hg = self.task.dataset.g
        self.userNum = self.hg.number_of_nodes('user')
        self.itemNum = self.hg.number_of_nodes('item')

        self.model = build_model(self.model).build_model_from_args(args=self.args, hg=self.hg).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=self.args.lr)

        trainMat = self.hg.adj_external(etype=('user', 'interact_train', 'item'), scipy_fmt='coo')
        testMat = self.hg.adj_external(etype=('user', 'interact_test', 'item'), scipy_fmt='coo')
        train_u, train_v, train_r = trainMat.row, trainMat.col, trainMat.data
        assert np.sum(train_r == 0) == 0
        test_u, test_v = testMat.row, testMat.col
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
        test_data = np.hstack((test_u.reshape(-1, 1), test_v.reshape(-1, 1))).tolist()
        train_dataset = BPRData(train_data, self.itemNum, trainMat, 1, True)
        test_dataset = BPRData(test_data, self.itemNum, trainMat, 0, False)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True,
                                                  num_workers=0)
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=1024 * 1000, shuffle=False, num_workers=0)
        self.train_losses = []
        self.test_hr = []
        self.test_ndcg = []

    def predictModel(self, user, pos_i, neg_j, isTest=False):
        if isTest:
            pred_pos = t.sum(user * pos_i, dim=1)
            return pred_pos
        else:
            pred_pos = t.sum(user * pos_i, dim=1)
            pred_neg = t.sum(user * neg_j, dim=1)
            return pred_pos, pred_neg

    # Contrastive Learning
    def ssl_loss(self, data1, data2, index):
        index = t.unique(index)
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        pos_score = t.sum(t.mul(norm_embeddings1, norm_embeddings2), dim=1)
        all_score = t.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score = t.exp(pos_score / self.args.ssl_temp)
        all_score = t.sum(t.exp(all_score / self.args.ssl_temp), dim=1)
        ssl_loss = (-t.sum(t.log(pos_score / ((all_score)))) / (len(index)))
        return ssl_loss

    # Model train
    def _mini_train_step(self):
        epoch_loss = 0
        self.train_loader.dataset.ng_sample()
        step_num = 0  # count batch num
        for user, item_i, item_j in self.train_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            item_j = item_j.long().cuda()
            step_num += 1
            self.istrain = True
            itemindex = t.unique(t.cat((item_i, item_j)))
            userindex = t.unique(user)
            self.userEmbed, self.itemEmbed, self.ui_userEmbedall, self.ui_itemEmbedall, self.ui_userEmbed, self.ui_itemEmbed, metaregloss = self.model(
                self.istrain, userindex, itemindex, norm=1)

            # Contrastive Learning of collaborative relations
            ssl_loss_user = self.ssl_loss(self.ui_userEmbed, self.userEmbed, user)
            ssl_loss_item = self.ssl_loss(self.ui_itemEmbed, self.itemEmbed, item_i)
            ssl_loss = self.args.ssl_ureg * ssl_loss_user + self.args.ssl_ireg * ssl_loss_item

            # prediction
            pred_pos, pred_neg = self.predictModel(self.ui_userEmbedall[user], self.ui_itemEmbedall[item_i],
                                                   self.ui_itemEmbedall[item_j])
            bpr_loss = - nn.LogSigmoid()(pred_pos - pred_neg).sum()
            epoch_loss += bpr_loss.item()
            regLoss = (t.norm(self.ui_userEmbedall[user]) ** 2 + t.norm(self.ui_itemEmbedall[item_i]) ** 2 + t.norm(
                self.ui_itemEmbedall[item_j]) ** 2)
            loss = ((bpr_loss + regLoss * self.args.reg) / self.args.batch) + ssl_loss * self.args.ssl_beta + metaregloss * self.args.metareg

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
        return epoch_loss

    def test(self):
        HR = []
        NDCG = []

        with t.no_grad():
            uid = np.arange(0, self.userNum)
            iid = np.arange(0, self.itemNum)
            self.istrain = False
            _, _, self.ui_userEmbed, self.ui_itemEmbed, _, _, _ = self.model(self.istrain, uid, iid, norm=1)
            for test_u, test_i in self.test_loader:
                test_u = test_u.long().cuda()
                test_i = test_i.long().cuda()
                pred = self.predictModel(self.ui_userEmbed[test_u], self.ui_itemEmbed[test_i], None, isTest=True)
                batch = int(test_u.cpu().numpy().size / 100)
                for i in range(batch):
                    batch_socres = pred[i * 100:(i + 1) * 100].view(-1)
                    _, indices = t.topk(batch_socres, self.args.topk)
                    tmp_item_i = test_i[i * 100:(i + 1) * 100]
                    recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
                    gt_item = tmp_item_i[0].item()
                    HR.append(self.hit(gt_item, recommends))
                    NDCG.append(self.ndcg(gt_item, recommends))
        return np.mean(HR), np.mean(NDCG)


    def hit(self, gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    def ndcg(self, gt_item, pred_items):
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index + 2))
        return 0

    def log(self, msg, save=None, oneline=False):
        global logmsg
        global saveDefault
        time = datetime.datetime.now()
        tem = '%s: %s' % (time, msg)
        if save != None:
            if save:
                logmsg += tem + '\n'
        elif saveDefault:
            logmsg += tem + '\n'
        if oneline:
            print(tem, end='\r')
        else:
            print(tem)

    def _full_train_setp(self):
        pass

    def _test_step(self, split=None, logits=None):
        pass

    def train(self):
        # self.prepareModel()
        self.curEpoch = 0
        best_hr = -1
        best_ndcg = -1
        best_epoch = -1
        HR_lis = []
        for e in range(self.args.epochs + 1):
            self.curEpoch = e
            # train
            self.log("**************************************************************")
            epoch_loss = self._mini_train_step()
            self.train_losses.append(epoch_loss)
            self.log("epoch %d/%d, epoch_loss=%.2f" % (e, self.args.epochs, epoch_loss))

            # test
            HR, NDCG = self.test()  #
            self.test_hr.append(HR)
            self.test_ndcg.append(NDCG)
            self.log("epoch %d/%d, HR@10=%.4f, NDCG@10=%.4f" % (e, self.args.epochs, HR, NDCG))
            # self.adjust_learning_rate()
            if HR > best_hr:
                best_hr, best_ndcg, best_epoch = HR, NDCG, e

            HR_lis.append(HR)

        print("*****************************")
        self.log("best epoch = %d, HR= %.4f, NDCG=%.4f" % (best_epoch, best_hr, best_ndcg))
        print("*****************************")
        print(self.args)

class BPRData(data.Dataset):
    def __init__(self, data,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.data = np.array(data)
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'
        tmp_trainMat = self.train_mat.todok()
        length = self.data.shape[0]
        self.neg_data = np.random.randint(low=0, high=self.num_item, size=length)

        for i in range(length):
            uid = self.data[i][0]
            iid = self.neg_data[i]
            if (uid, iid) in tmp_trainMat:
                while (uid, iid) in tmp_trainMat:
                    iid = np.random.randint(low=0, high=self.num_item)
                self.neg_data[i] = iid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.data[idx][0]
        item_i = self.data[idx][1]
        if self.is_training:
            neg_data = self.neg_data
            item_j = neg_data[idx]
            return user, item_i, item_j
        else:
            return user, item_i

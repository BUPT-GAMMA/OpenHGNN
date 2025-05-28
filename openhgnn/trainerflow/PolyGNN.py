import torch
from tqdm import tqdm
from ..models import build_model
from . import BaseFlow, register_flow
from ..utils import EarlyStopping
import copy
import numpy as np
torch.autograd.set_detect_anomaly(True)
from torch_geometric.nn import MLP
from datetime import datetime
import os

blue = lambda x: '\033[94m' + x + '\033[0m'
red = lambda x: '\033[31m' + x + '\033[0m'
green = lambda x: '\033[32m' + x + '\033[0m'
yellow = lambda x: '\033[33m' + x + '\033[0m'
greenline = lambda x: '\033[42m' + x + '\033[0m'
yellowline = lambda x: '\033[43m' + x + '\033[0m'

@register_flow("PolyGNN")
class PolyGNN(BaseFlow):
    r"""
    Graph classification flow for the model of PolyGNN
    The task is to classify the graphs.
    """

    def __init__(self, args):
        """

        Attributes
        ------------
        num_classes: int
            The number of classes for category graph type
        mlpmodel: nn.Module
            The linear model for the final classification

        """

        super(PolyGNN, self).__init__(args)
        self.num_classes = self.task.num_classes
        self.criterion=self.task.get_loss_fn()
        self.model = build_model(self.model).build_model_from_args(self.args,self.hg).to(self.device)
        self.mlpmodel=MLP(in_channels=args.h_dim*args.num_interactions, 
                          hidden_channels=args.h_dim,
                          out_channels=self.num_classes, 
                          num_layers=args.classifier_depth,
                          dropout=args.dropout
                          )
        opt_list=list(self.model.parameters())+list(self.mlpmodel.parameters())
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu == 0 else 'cpu')
        self.model.to(self.device)
        self.mlpmodel.to(self.device)
        self.optimizer = torch.optim.Adam( opt_list, lr=args.lr)    # 优化器和调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)   
        self.train_loader, self.val_loader, self.test_loader = self.task.get_split()
        self.lr = args.lr
        self.args.save_dir=os.path.join('./save/',args.dataset,args.model)

    def train(self):
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        epoch_iter = tqdm(range(self.max_epoch))
        best_val_trigger = -1
        old_lr = self.lr
        lr = self.lr
        for epoch in epoch_iter:
            train_loss,train_closs,y_hat, y_true,y_hat_logit=self._full_train_step()
            train_acc = calculate(y_hat,y_true,y_hat_logit)
            # try:util.record({"loss":train_loss,"closs":train_closs,"acc":train_acc},epoch,writer,"Train") 
            # except: pass
            print_1(epoch,'Train',{"loss":train_loss,"closs":train_closs,"acc":train_acc})
            if epoch % self.args.test_per_round == 0:
                val_loss, yhat_val, ytrue_val, yhatlogit_val = self._full_test_step(self.val_loader,self.model,self.mlpmodel)
                test_loss, yhat_test, ytrue_test, yhatlogit_test = self._full_test_step(self.test_loader,self.model,self.mlpmodel)
                val_acc=calculate(yhat_val,ytrue_val,yhatlogit_val)
                # try:util.record({"loss":val_loss,"acc":val_acc},epoch,writer,"Val")
                # except: pass
                print_1(epoch,'Val',{"loss":val_loss,"acc":val_acc},color=blue) 
                test_acc=calculate(yhat_test,ytrue_test,yhatlogit_test)
                # try:util.record({"loss":test_loss,"acc":test_acc},epoch,writer,"Test")            
                # except: pass
                print_1(epoch,'Test',{"loss":test_loss,"acc":test_acc},color=blue)
                val_trigger=val_acc
                if val_trigger > best_val_trigger:
                    best_val_trigger = val_trigger
                    best_model = copy.deepcopy(self.model)
                    best_mlpmodel=copy.deepcopy(self.mlpmodel)
                    best_info=[epoch,val_trigger]
            """ 
            update lr when epoch≥30
            """
            if epoch >= 30: # lr更新
                lr = self.scheduler.optimizer.param_groups[0]['lr']
                if old_lr!=lr:
                    print(red('lr'), epoch, (lr), sep=', ')
                    old_lr=lr
                self.scheduler.step(val_trigger)        

        """
        use best model to get best model result  测试
        """
        val_loss, yhat_val, ytrue_val, yhat_logit_val  = self._full_test_step(self.val_loader,best_model,best_mlpmodel)
        test_loss, yhat_test, ytrue_test, yhat_logit_test= self._full_test_step(self.test_loader,best_model,best_mlpmodel)

        val_acc=calculate(yhat_val,ytrue_val,yhat_logit_val)
        print_1(best_info[0],'BestVal',{"loss":val_loss,"acc":val_acc},color=blue)
        test_acc=calculate(yhat_test,ytrue_test,yhat_logit_test)
        print_1(best_info[0],'BestTest',{"loss":test_loss,"acc":test_acc},color=blue)
                                                                
        """
        save training info and best result  存储
        """

        """
        build dir 
        """
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir,exist_ok=True)
        tensorboard_dir=os.path.join(self.args.save_dir,'log')
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir,exist_ok=True)
        model_dir=os.path.join(self.args.save_dir,'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir,exist_ok=True)    
        info_dir=os.path.join(self.args.save_dir,'info')
        if not os.path.exists(info_dir):
            os.makedirs(info_dir,exist_ok=True) 
        timestamp = datetime.now().strftime("%b%d-%H-%M-%S")  # 格式示例: Apr14-21-38-13
        suffix = f"{timestamp}"  # 直接使用单次strftime生成
        result_file=os.path.join(info_dir, suffix)
        with open(result_file, 'w') as f:
            print("Random Seed: ", self.args.man_seed,file=f)
            print(f"acc  val : {val_acc:.3f}, Test : {test_acc:.3f}", file=f)
            print(f"Best info: {best_info}", file=f)
            for i in [[a,getattr(self.args, a)] for a in self.args.__dict__]:
                print(i,sep='\n',file=f)
        to_save_dict={'model':best_model.state_dict(),'mlpmodel':best_mlpmodel.state_dict(),'args':self.args,'labels':ytrue_test,'yhat':yhat_test,'yhat_logit':yhat_logit_test}
        torch.save(to_save_dict, os.path.join(model_dir,suffix+'.pth') )
        print("done")


    def _full_train_step(self):
        epochloss = torch.tensor(0.0, device=self.device)
        epochcloss = torch.tensor(0.0, device=self.device)
        y_hat, y_true,y_hat_logit = [], [], []        
        self.optimizer.zero_grad()
        self.model.train()
        self.mlpmodel.train()
        for g,label in self.train_loader:
            g = g.to(self.device)
            label = label.to(self.device).long()
            self.optimizer.zero_grad()
            c_loss,graph_embeddings  = self.model(g,label,self.args,device=self.device)
            output=self.mlpmodel(graph_embeddings)
            loss = self.criterion(output, label)
            loss+=c_loss*self.args.loss_coef
            loss.backward()
            self.optimizer.step()
            epochloss+=loss.detach()
            epochcloss+=c_loss.detach()
            
            _, pred = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)
            pred, label, output = pred.cpu(), label.cpu(), output.cpu() 
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(label.detach().numpy().reshape(-1))
            y_hat_logit+=list(output.detach().cpu().numpy())
        return epochloss.cpu().item()/len(self.train_loader),epochcloss.cpu().item()/len(self.train_loader),y_hat, y_true,y_hat_logit


    def _full_test_step(self,loader,model,mlpmodel):
        y_hat, y_true,y_hat_logit = [], [], []
        loss_total, pred_num = 0, 0
        model.eval()
        mlpmodel.eval()
        with torch.no_grad():
            for g,label in loader:
                g = g.to(self.device)
                label = label.to(self.device).long()
                self.optimizer.zero_grad()

                c_loss,graph_embeddings  = self.model(g,label,self.args,device=self.device)
                output=self.mlpmodel(graph_embeddings)
                loss = self.criterion(output, label)
                loss+=c_loss*self.args.loss_coef
                
                _, pred = output.detach().cpu().topk(1, dim=1, largest=True, sorted=True)
                pred, label, output = pred.cpu(), label.cpu(), output.cpu() 
                y_hat += list(pred.detach().numpy().reshape(-1))
                y_true += list(label.detach().cpu().numpy().reshape(-1))
                y_hat_logit+=list(output.detach().cpu().numpy())
                
                pred_num += len(label.reshape(-1, 1))
                loss_total += loss.detach() * len(label.reshape(-1, 1))
        return loss_total/pred_num,y_hat, y_true, y_hat_logit
    


def record(values,epoch,writer,phase="Train"):
    """ tfboard write """
    for key,value in values.items():
        writer.add_scalar(key+"/"+phase,value,epoch)           
def calculate(y_hat,y_true,y_hat_logit):
    """ calculate five metrics using y_hat, y_true, y_hat_logit """
    train_acc=(np.array(y_hat) == np.array(y_true)).sum()/len(y_true) 
    return train_acc


def print_1(epoch,phase,values,color=None):
    """ print epoch info"""
    if color is not None:
        print(color( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))
    else:
        print(( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))


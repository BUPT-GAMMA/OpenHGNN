import numpy as np
import torch

from . import BaseFlow,register_flow
from ..models import build_model
from ..tasks import build_task


@register_flow('QGRL_trainer')
class QGRL_trainer(BaseFlow):
    def __init__(self, args):
        BaseFlow.__init__(self, args)
        self.args = args

        self.model_name = args.model
        self.device = args.device
        self.task = build_task(args)

        self.hg = self.task.get_hg()

        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.model = self.model.to(self.device)
        self.patience = args.patience
        self.max_epoch = args.max_epoch

    def train(self):
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []
        for epoch in range(10):
            self.model.pretrain()
            acc, nmi, ari, f1 = self.model.run()
            acc_list.append(acc)
            nmi_list.append(nmi)
            ari_list.append(ari)
            f1_list.append(f1)

        print("\n")
        acc_list = np.array(acc_list)
        nmi_list = np.array(nmi_list)
        ari_list = np.array(ari_list)
        f1_list = np.array(f1_list)

        print(acc_list.mean(), "±", acc_list.std())
        print(nmi_list.mean(), "±", nmi_list.std())
        print(ari_list.mean(), "±", ari_list.std())
        print(f1_list.mean(), "±", f1_list.std())
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from ..layers.THePUffLayer import TriLevelMultiHeadedAttention
from ..layers.THePUffLayer import FeedForward
from ..layers.THePUffLayer import LayerNorm
from ..layers.THePUffLayer import Encoder
from ..layers.THePUffLayer import EncoderLayer
from copy import deepcopy
import numpy as np

class THePUffGenerator(nn.Module):
    def __init__(self, node_type, N, node_embs, device, args):
        super(THePUffGenerator, self).__init__()
        self.node_type = node_type
        self.N = N
        self.batch_size = args.batch_size
        self.hidden_units = args.hidden_units
        self.noise_dim = args.noise_dim
        self.max_path_len = args.max_path_len
        self.W_down_generator_size = args.W_down_generator_size
        self.num_G_layer = args.num_G_layer
        self.node_classes = 3 if args.dataset_name == 'taobao' else 4
        self.node_embs = node_embs
        self.device = device
        self.node_emb_size = 128
        self.type_0 = torch.tensor(node_type[0], dtype=torch.long).to(self.device)
        self.type_1 = torch.tensor(node_type[1], dtype=torch.long).to(self.device)
        self.type_2 = torch.tensor(node_type[2], dtype=torch.long).to(self.device)
        self.lin_node_0 = nn.Linear(self.hidden_units, self.node_emb_size).to(self.device)
        self.lin_node_1 = nn.Linear(self.hidden_units, self.node_emb_size).to(self.device)
        self.lin_node_2 = nn.Linear(self.hidden_units, self.node_emb_size).to(self.device)
        if self.node_classes == 5:
            self.type_3 = torch.tensor(node_type[3], dtype=torch.long).to(self.device)
            self.lin_node_3 = nn.Linear(self.hidden_units, self.node_emb_size).to(self.device)
        self.W_down_generator_type = nn.Linear(self.node_classes + 1, self.W_down_generator_size).to(self.device)
        self.W_down_generator_node = nn.Linear(self.N + 1, self.W_down_generator_size).to(self.device)
        self.lstm = nn.LSTM(self.hidden_units, self.hidden_units, self.num_G_layer).to(self.device)
        self.init_lin_1 = nn.Linear(self.noise_dim, self.hidden_units).to(self.device)
        self.init_lin_2_h = nn.Linear(self.hidden_units, self.hidden_units).to(self.device)
        self.init_lin_2_c = nn.Linear(self.hidden_units, self.hidden_units).to(self.device)
        self.lin_node_type = nn.Linear(self.hidden_units, self.node_classes + 1).to(self.device)

    def reverse_sampling(self, dist):
        dist_weight = torch.exp(-dist)
        return torch.multinomial(dist_weight, 1)[0]

    def forward(self, z):
        outputs_type, outputs_node, init_c, init_h = [], [], [], []
        outputs_idx = []
        z=z.to(self.device)
        for _ in range(self.num_G_layer):
            intermediate = torch.tanh(self.init_lin_1(z))
            init_c.append(torch.tanh(self.init_lin_2_c(intermediate)))
            init_h.append(torch.tanh(self.init_lin_2_h(intermediate)))
        inputs = torch.zeros((self.batch_size, self.hidden_units),device=self.device)
        hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        for i in range(self.max_path_len):
            out, hidden = self.lstm(inputs.unsqueeze(0), hidden)
            output_bef = self.lin_node_type(out.squeeze(0))
            output_type = F.gumbel_softmax(output_bef, dim=1, tau=3, hard=True)
            temp_node = []
            for j, x in enumerate(torch.argmax(output_type, dim=1)):
                if x == 0:
                    temp_output_node = self.lin_node_0(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_0[candidate])
                elif x == 1:
                    temp_output_node = self.lin_node_1(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_1[candidate])
                elif x == 2:
                    temp_output_node = self.lin_node_2(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_2[candidate])
                elif x == 3 and self.node_classes > 4:
                    temp_output_node = self.lin_node_3(out.squeeze(0)[j])
                    dist = torch.norm(self.node_embs[x.item()] - temp_output_node, dim=1, p=None)
                    dist = dist.topk(int(dist.shape[0] // 1), largest=False)
                    candidate = dist.indices[self.reverse_sampling(dist.values)]
                    temp_node.append(self.type_3[candidate])
                else:  # x == 4:
                    temp_node.append(torch.tensor(self.N).to(self.device))
            outputs_idx.append(list(map(int, temp_node)))
            temp_node = torch.stack(temp_node)
            output_node = F.one_hot(temp_node.to(int), self.N + 1).float()
            outputs_type.append(output_type)
            outputs_node.append(output_node)
            inputs = self.W_down_generator_type(output_type) + self.W_down_generator_node(output_node)
        outputs_type = torch.stack(outputs_type, dim=1).to(self.device)
        outputs_node = torch.stack(outputs_node, dim=1).to(self.device)
        return {
            'type_seq': outputs_type,
            'node_seq': outputs_node,
            'idx': outputs_idx
        }

class THePUffDiscriminator(nn.Module):
    def __init__(self, args, device):
        super(THePUffDiscriminator, self).__init__()
        self.args = args
        self.device = device
        h, N, dropout = args.h, args.N, args.dropout
        d_model, d_ff = args.d_model, args.d_ff
        self.node_classes = 3 if args.dataset_name == 'taobao' else 4
        attn = TriLevelMultiHeadedAttention(h, d_model, self.node_classes + 1).to(self.device)
        ff = FeedForward(d_model, d_ff, dropout).to(self.device)
        self.norm = LayerNorm(d_model).to(self.device)
        self.encoder = Encoder(EncoderLayer(args.d_model, deepcopy(attn), deepcopy(ff), dropout), N).to(self.device)
        self.fc = nn.Linear(d_model, 2).to(self.device)

    def forward(self, x_1, x_2, x_3):
        x_1 = x_1.to(self.device)
        x_2 = x_2.to(self.device)
        x_3 = x_3.to(self.device)
        encoded_sents = self.encoder(x_1, x_2, x_3)
        final_feature_map = encoded_sents[:, -1, :]
        final_out = self.fc(final_feature_map)
        prob=F.softmax(final_out, dim=-1)
        return prob

class THePUffModel():
    def __init__(self, device, args, N, node_embs_classified, node_type_classified, dataloader, real_level_embs, node_embs, dp_level_embs, node_embs_dp, dp_dataloader):
        self.device = device
        self.node_embs = node_embs
        self.real_level_embs = real_level_embs
        self.node_embs_dp = node_embs_dp
        self.dp_level_embs = dp_level_embs
        self.args = args
        self.dataloader = dataloader
        self.dp_dataloader = dp_dataloader
        self.N = N
        self.node_embs_classified = node_embs_classified
        self.node_type_classified = node_type_classified

    def init_model(self):
        self.generator = THePUffGenerator(self.node_type_classified, self.N, self.node_embs_classified, self.device, self.args).to(self.device)
        self.discriminator1 = THePUffDiscriminator(self.args, self.device).to(self.device)
        self.discriminator2 = THePUffDiscriminator(self.args, self.device).to(self.device)
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.args.lr_gen)
        self.optimizer_D1 = torch.optim.SGD(self.discriminator1.parameters(), lr=self.args.lr_dis1, momentum=0.7)
        self.optimizer_D2 = torch.optim.SGD(self.discriminator2.parameters(), lr=self.args.lr_dis2, momentum=0.7)
        return self.generator, self.discriminator1, self.discriminator2, self.optimizer_G, self.optimizer_D1, self.optimizer_D2

    def pretrain(self):
        for epoch in range(self.args.n_epochs_pre):
            for i, (real_type, real_embs, real_node_embs) in enumerate(self.dataloader):
                initial_noise = self.make_noise((self.args.batch_size, self.args.noise_dim), self.args.noise_type).to(self.device)
                self.discriminator2.train()
                self.optimizer_D2.zero_grad()
                fake_result = self.generator(initial_noise)
                fake_type = fake_result['type_seq'].detach()
                fake_node = fake_result['node_seq'].detach()
                fake_idxs = fake_result['idx']
                fake_embs = []
                for idxs in fake_idxs:
                    fake_embs.append(self.node_embs[idxs])
                fake_embs = torch.from_numpy(np.stack(fake_embs, axis=0)).to(self.device)
                fake_node_embs = []
                for idxs in fake_idxs:
                    fake_node_embs.append(self.real_level_embs[idxs])
                fake_node_embs = torch.from_numpy(np.stack(fake_node_embs, axis=0)).to(self.device)
                loss_D2 = torch.mean(self.discriminator2(fake_embs.transpose(0, 1), fake_node_embs.transpose(0, 1), fake_type)[:, 0]) - torch.mean(self.discriminator2(real_embs.to(self.device), real_node_embs.to(self.device), real_type.to(self.device))[:, 0])
                loss_D2.backward()
                self.optimizer_D2.step()
                print("[step 1/3] [Epoch %d/%d] [Batch %d/%d]"% (epoch + 1, self.args.n_epochs_pre, i + 1, len(self.dataloader)))
        dir = os.getcwd()
        save_dir = os.path.join(dir, "data")+"/"+self.args.dataset_name + "/"
        torch.save(self.discriminator2.state_dict(), save_dir + self.args.dataset_name + '_d2.pt')

    def train(self):
        for epoch in range(self.args.n_epochs):
            for i, (dp_type, dp_embs, dp_node_embs) in enumerate(self.dp_dataloader):
                initial_noise = self.make_noise((self.args.batch_size, self.args.noise_dim), self.args.noise_type).to(self.device)
                self.discriminator1.train()
                self.optimizer_D1.zero_grad()
                fake_result = self.generator(initial_noise)
                fake_type = fake_result['type_seq'].detach()
                fake_node = fake_result['node_seq'].detach()
                fake_idxs = fake_result['idx']
                fake_embs = []
                for idxs in fake_idxs:
                    fake_embs.append(self.node_embs_dp[idxs])
                fake_embs = torch.from_numpy(np.stack(fake_embs, axis=0)).to(self.device)
                fake_node_embs = []
                for idxs in fake_idxs:
                    fake_node_embs.append(self.dp_level_embs[idxs])
                fake_node_embs = torch.from_numpy(np.stack(fake_node_embs, axis=0)).to(self.device)
                loss_D1 = torch.mean(self.discriminator1(fake_embs.transpose(0, 1), fake_node_embs.transpose(0, 1), fake_type)[:, 0]) - torch.mean(self.discriminator1(dp_embs.to(self.device), dp_node_embs.to(self.device), dp_type.to(self.device))[:, 0])
                loss_D1.backward()
                self.optimizer_D1.step()
                print("[step 2/3] [Epoch %d/%d] [Batch %d/%d]"% (epoch + 1, self.args.n_epochs, i + 1, len(self.dp_dataloader)))
                if i % self.args.n_critic == 0:
                    self.generator.train()
                    self.optimizer_G.zero_grad()
                    fake_result = self.generator(initial_noise)
                    syn_type = fake_result['type_seq'].detach()
                    syn_node = fake_result['node_seq'].detach()
                    syn_idxs = fake_result['idx']
                    syn_embs = []
                    for idxs in syn_idxs:
                        syn_embs.append(self.node_embs[idxs])
                    syn_embs_dp = []
                    for idxs in syn_idxs:
                        syn_embs_dp.append(self.node_embs_dp[idxs])
                    syn_embs = torch.from_numpy(np.stack(syn_embs, axis=0)).to(self.device)
                    syn_embs_dp = torch.from_numpy(np.stack(syn_embs_dp, axis=0)).to(self.device)
                    syn_node_embs = []
                    for idxs in syn_idxs:
                        syn_node_embs.append(self.real_level_embs[idxs])
                    syn_node_embs_dp = []
                    for idxs in syn_idxs:
                        syn_node_embs_dp.append(self.dp_level_embs[idxs])
                    syn_node_embs = torch.from_numpy(np.stack(syn_node_embs, axis=0)).to(self.device)
                    syn_node_embs_dp = torch.from_numpy(np.stack(syn_node_embs_dp, axis=0)).to(self.device)
                    loss_G1 = - torch.mean(self.discriminator1(syn_embs_dp.transpose(0, 1), syn_node_embs_dp.transpose(0, 1), syn_type)[:, 0])
                    loss_G2 = - torch.mean(self.discriminator2(syn_embs.transpose(0, 1), syn_node_embs.transpose(0, 1), syn_type)[:, 0])
                    loss_G = loss_G1 + loss_G2
                    loss_G.backward()
                    self.optimizer_G.step()
        dir = os.getcwd()
        save_dir = os.path.join(dir, "data")+"/" + self.args.dataset_name + "/"
        torch.save(self.discriminator1.state_dict(), save_dir + self.args.dataset_name + '_d1.pt')
        torch.save(self.generator.state_dict(), save_dir + self.args.dataset_name + '_g.pt')

    def make_noise(self,shape, type='Gaussian', device='cpu'):
        if type == "Gaussian":
            noise = torch.randn(shape, device=device)
        elif type == 'Uniform':
            noise = torch.rand(shape, device=device).uniform_(-1, 1)
        else:
            raise ValueError(f"ERROR: Noise type {type} not supported (only Gaussian/Uniform)")
        return noise
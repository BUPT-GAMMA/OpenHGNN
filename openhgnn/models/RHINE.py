from unittest.mock import Base
import torch.nn as nn
from openhgnn.models import BaseModel
from openhgnn.models import register_model
import torch

@register_model('RHINE')
class RHINE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args, hg):
        total_nodes=hg.num_nodes()
        total_IRs=args.total_IRs
        ARs=args.ARs
        IRs=args.IRs
        score_dim=args.batch_size
        device=args.device
        return cls(total_nodes,total_IRs,ARs,IRs,args.emb_dim,score_dim,device,args.hidden_dim,args.out_dim)
    
    def __init__(self,total_nodes,total_IRs,ARs,IRs,emb_dim,score_dim,device,hid_dim=100,out_dim=4):
        super(RHINE, self).__init__()
        # 对AR和IR分别准备Layer
        self.DisLayer=nn.ModuleDict()
        self.device=device
        
        self.ent_embeddings = nn.Embedding(total_nodes, emb_dim)
        self.rel_embeddings = nn.Embedding(total_IRs, emb_dim)

        for AR in ARs:
            self.DisLayer[AR]=ARLayer(self.ent_embeddings,self.rel_embeddings,score_dim,device)
        for IR in IRs:
            self.DisLayer[IR]=IRLayer(self.ent_embeddings,score_dim,device)

        self.predictor=nn.ModuleList([
            nn.Linear(emb_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        ])


    def forward(self,hg_dict,category=None,mod='train'):
        if mod=='train':
            scores={}
            for mp,hg in hg_dict.items():
                # 通过meta-path的类型，找到对应的Layer
                score=self.DisLayer[mp](hg)
                # 通过Layer进行embedding
                scores[mp]=score
            scores=torch.stack(list(scores.values()))
            return scores
        else:
            assert category is not None
            h=self.ent_embeddings(hg_dict.nodes[category].data['h'])
            for layer in self.predictor:
                h=layer(h)
            return h.squeeze(1)

class ARLayer(nn.Module):
    def __init__(self,ent_emb,rel_emb,score_dim,device):
        super(ARLayer, self).__init__()
        self.ent_embeddings=ent_emb
        self.rel_embeddings=rel_emb
        self.score_dim=score_dim
        self.device=device

    
    def trans_dist(self,edges):
        return {'ar_dis_score':torch.sum(edges._src_data['h_emb']+edges.data['r_emb']-edges._dst_data['t_emb'],1)}

    def forward(self,hg):
        with hg.local_scope():
            for n in hg.ntypes:
                hg.srcnodes[n].data['h_emb']=hg.dstnodes[n].data['h_emb']=self.ent_embeddings(hg.nodes[n].data['_ID'])
                hg.srcnodes[n].data['t_emb']=hg.dstnodes[n].data['t_emb']=self.ent_embeddings(hg.nodes[n].data['_ID'])
            for e in hg.etypes:
                hg.edges[e].data['r_emb']=self.rel_embeddings(hg.edges[e].data['_ID'])

            scores=[]
            for rel in hg.etypes:
                hg.apply_edges(self.trans_dist,etype=rel)
                score=hg.edges[rel].data['ar_dis_score']
                if score.shape[0]<self.score_dim:
                    score=torch.cat([score,torch.zeros(self.score_dim-score.shape[0]).to(self.device)])
                scores.append(score)

            return torch.cat(scores)

class IRLayer(nn.Module):
    def __init__(self,ent_emb,score_dim,device):
        super(IRLayer, self).__init__()
        self.ent_embeddings=ent_emb
        self.score_dim=score_dim    
        self.device=device

    def eur_dist(self,edges):
        return {'ir_dis_score':torch.sum(torch.pow(edges.src['h_emb']-edges.dst['t_emb'],2),1)}
    
    def forward(self,hg):
        
        with hg.local_scope():
            for n in hg.ntypes:
                hg.srcnodes[n].data['h_emb']=hg.dstnodes[n].data['h_emb']=self.ent_embeddings(hg.nodes[n].data['_ID'])
                hg.srcnodes[n].data['t_emb']=hg.dstnodes[n].data['t_emb']=self.ent_embeddings(hg.nodes[n].data['_ID'])
            scores=[]
            for rel in hg.etypes:
                hg.apply_edges(self.eur_dist,etype=rel)
                score=hg.edges[rel].data['ir_dis_score']
                if score.shape[0]<self.score_dim:
                    score=torch.cat([score,torch.zeros(self.score_dim-score.shape[0]).to(self.device)])
                scores.append(score)
            
            return torch.cat(scores)


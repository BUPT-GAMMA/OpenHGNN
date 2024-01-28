from ..models import build_model
from . import BaseFlow, register_flow
from ..tasks import build_task
from ..utils import extract_embed, EarlyStopping
import os
import sys
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
import logging
from ..models import NBF
separator = ">" * 30
line = "-" * 30

def train_and_validate(args, model, train_data, valid_data, filtered_data=None):
    if args.num_epoch == 0:
        return

    world_size = get_world_size()
    rank = get_rank()  

    train_triplets = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, args.batch_size, sampler=sampler)

    optimizer = (
        torch.optim.Adam(model.parameters(), lr=args.lr))

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device])
    else:
        parallel_model = model

    step = math.ceil(args.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, args.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(args.num_epoch, i + step)):
            if get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in train_loader:
                batch = NBF.negative_sampling(train_data, batch, args.num_negative,
                                                strict=args.strict_negative)
                pred = parallel_model(train_data, batch)#forward
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if args.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / args.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / args.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if get_rank() == 0 and batch_id % args.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(args.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        
        
        result = test(args, model, valid_data, filtered_data=filtered_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=args.device)
    model.load_state_dict(state["model"])
    synchronize()

@torch.no_grad()
def test(args, model, test_data, filtered_data=None):
    world_size = get_world_size()
    rank = get_rank()

    test_triplets = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_triplets, world_size, rank)
    test_loader = torch_data.DataLoader(test_triplets, args.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    for batch in test_loader:
        t_batch, h_batch = NBF.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = NBF.strict_negative_mask(test_data, batch)
        else:
            t_mask, h_mask = NBF.strict_negative_mask(filtered_data, batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        t_ranking = NBF.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = NBF.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=args.device)
    all_size[rank] = len(ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=args.device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=args.device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)

    if rank == 0:
        for metric in args.metric:
            if metric == "mr":
                score = all_ranking.float().mean()
            elif metric == "mrr":
                score = (1 / all_ranking.float()).mean()
            elif metric.startswith("hits@"):
                values = metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (all_ranking - 1).float() / all_num_negative
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (all_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
    mrr = (1 / all_ranking.float()).mean()

    return mrr

logger = logging.getLogger(__file__)

def get_rank(): # get random seed
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1

def synchronize():
    if get_world_size() > 1:
        dist.barrier()




@register_flow("NBF_trainer")
class NBF_trainer(BaseFlow):

    """
    NGF_trainer

    """

    def __init__(self, args):  # args == self.config
        args.task = args.model +"_" +args.task # task: NBF_link_prediction
        self.args = args  
        self.model_name = args.model
        self.device = args.device
        self.hg = None

        self.task = build_task(args)
        self.dataset = self.task.dataset.dataset  #  train_data,valid_data,test_data


        # Build the model. 
        self.args.num_relation = self.dataset.num_relations
        
        self.model = args.model
        self.model = build_model(self.model).build_model_from_args(self.args, self.hg)



    def train(self):
       
        filtered_data = None
        self.model = self.model.to(self.device)

        train_data, valid_data, test_data = self.dataset.train_data , self.dataset.valid_data , self.dataset.test_data

        
        train_data.edge_index = train_data.edge_index.to(self.device)
        train_data.edge_type = train_data.edge_type.to(self.device)
        train_data.target_edge_index = train_data.target_edge_index.to(self.device)
        train_data.target_edge_type = train_data.target_edge_type.to(self.device)

        valid_data.edge_index = valid_data.edge_index.to(self.device)
        valid_data.edge_type = valid_data.edge_type.to(self.device)
        valid_data.target_edge_index = valid_data.target_edge_index.to(self.device)
        valid_data.target_edge_type = valid_data.target_edge_type.to(self.device)


        test_data.edge_index = test_data.edge_index.to(self.device)
        test_data.edge_type = test_data.edge_type.to(self.device)
        test_data.target_edge_index = test_data.target_edge_index.to(self.device)
        test_data.target_edge_type = test_data.target_edge_type.to(self.device)

        train_and_validate(self.args, self.model, train_data, valid_data, filtered_data=filtered_data)

        if get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")

        test(self.args, self.model, valid_data, filtered_data=filtered_data)

        if get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on test")

        test(self.args, self.model, test_data, filtered_data=filtered_data)








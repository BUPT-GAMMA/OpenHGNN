

import torch.nn as nn
from openhgnn.models import BaseModel, register_model
import torch


@register_model("DHNE")
class DHNE(BaseModel):
    r"""
        **Title:** Structural Deep Embedding for Hyper-Networks

        **Authors:** Ke Tu, Peng Cui, Xiao Wang, Fei Wang, Wenwu Zhu

        DHNE was introduced in `[paper] <https://arxiv.org/abs/1711.10146>`_
        and parameters are defined as follows:

        Parameters
        ----------
        nums_type : list
            the type of nodes
        dim_features : array
            the embedding dimension of nodes
        embedding_sizes : int
            the embedding dimension size
        hidden_size : int
            The hidden full connected layer size
        device : int
            the device DHNE working on
        """
    @classmethod
    def build_model_from_args(cls, args):
        return cls(dim_features=args.dim_features,
                   embedding_sizes=args.embedding_sizes,
                   hidden_size=args.hidden_size,
                   nums_type=args.nums_type,
                   device = args.device
                   )

    def __init__(self, nums_type, dim_features, embedding_sizes, hidden_size, device):
        super().__init__()
        self.dim_features = dim_features
        self.embedding_sizes = embedding_sizes
        self.hidden_size = hidden_size
        self.nums_type = nums_type
        self.device = device

        # auto-encoder
        self.encodeds = [
            nn.Linear(sum([self.nums_type[j] for j in range(3) if j != i]), self.embedding_sizes[i]) for i in range(3)]
        self.decodeds = [
            nn.Linear(self.embedding_sizes[i], sum([self.nums_type[j] for j in range(3) if j != i])) for i in range(3)]
        self.hidden_layer = nn.Linear(
            sum(self.embedding_sizes), self.hidden_size)

        self.ouput_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids):
        """
        The forward part of the DHNE.

        Parameters
        ----------
        input_ids :
            the input block of this batch

        Returns
        -------
        tensor
            The logits after DHNE training.
        """
        encodeds = []
        decodeds = []
        for i in range(3):
            encoded = torch.tanh(self.encodeds[i].to(self.device)(input_ids[i].to(self.device)))
            encodeds.append(encoded)
            decodeds.append(torch.sigmoid(self.decodeds[i].to(self.device)(encoded)))
        merged = torch.concat(encodeds, axis=1)
        hidden = self.hidden_layer(merged)
        hidden = torch.tanh(hidden)
        output = self.ouput_layer(hidden)
        return decodeds+[output]

    def embedding_lookup(self, index, sparse_input=False):
        return [self.embeddings[i][index[:, i], :] for i in range(3)]

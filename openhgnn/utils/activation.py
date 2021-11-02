import torch.nn as nn

act_dict = {
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'sigmoid': nn.Sigmoid(),
    'lrelu': nn.LeakyReLU(negative_slope=0.5),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'prelu': nn.PReLU(),
    'selu': nn.SELU(),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5),
}

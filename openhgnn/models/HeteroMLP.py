from ..layers.HeteroLinear import HeteroMLPLayer
from ..layers.GeneralGNNLayer import MultiLinearLayer


def HGNNPreMP(args, node_types, num_pre_mp, in_dim, hidden_dim):
    """
    HGNNPreMP, dimension is in_dim, hidden_dim, hidden_dim ...
    Note:
    Final layer has activation.

    Parameters
    ----------
    args
    node_types
    num_pre_mp
    in_dim
    hidden_dim

    Returns
    -------

    """
    if num_pre_mp > 0:
        linear_dict = {}
        for ntype in node_types:
            linear_dict[ntype] = [in_dim]
            for _ in range(num_pre_mp):
                linear_dict[ntype].append(hidden_dim)
        return HeteroMLPLayer(linear_dict, act=args.activation, dropout=args.dropout,
                          has_l2norm=args.has_l2norm, has_bn=args.has_bn, final_act=True)


def HGNNPostMP(args, node_types, num_post_mp, hidden_dim, out_dim):
    """
    HGNNPostMLP, hidden_dim, hidden_dim, ..., out_dim
    Final layer has no activation.

    Parameters
    ----------
    args
    node_types
    num_post_mp
    hidden_dim
    out_dim

    Returns
    -------

    """
    if num_post_mp > 0:
        linear_dict = {}
        for ntype in node_types:
            linear_dict[ntype] = [hidden_dim]
            for _ in range(num_post_mp-1):
                linear_dict[ntype].append(hidden_dim)
            linear_dict[ntype].append(out_dim)
        return HeteroMLPLayer(linear_dict, act=args.activation, dropout=args.dropout,
                          has_l2norm=args.has_l2norm, has_bn=args.has_bn, final_act=False)


# def GNNPreMP(args, in_dim, hidden_dim):
#     linear_list = [in_dim] + args.layers_pre_mp * [hidden_dim]
#     return MultiLinearLayer(linear_list, dropout=args.dropout, act=args.activation, has_bn=args.has_bn,
#                  has_l2norm=args.has_l2norm)
#
#
# def GNNPostMP(args, hidden_dim, out_dim):
#     linear_list = args.layers_pre_mp * [hidden_dim] + [out_dim]
#     return MultiLinearLayer(linear_list, dropout=args.dropout, act=args.activation, has_bn=args.has_bn,
#                  has_l2norm=args.has_l2norm)
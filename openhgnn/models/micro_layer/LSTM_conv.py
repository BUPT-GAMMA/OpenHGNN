import torch as th
import torch.nn as nn
import dgl.function as fn


class LSTMConv(nn.Module):
    '''
    Aggregate the neighbors with LSTM
    '''

    def __init__(self, dim):
        super(LSTMConv, self).__init__()
        self.lstm = nn.LSTM(dim, int(dim / 2), 1, batch_first=True, bidirectional=True)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The LSTM module is using xavier initialization method for its weights.
        """
        self.lstm.reset_parameters()

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m']  # (B, L, D)
        batch_size = m.shape[0]
        all_state, last_state = self.lstm(m)
        return {'neigh': th.mean(all_state, 1)}

    def forward(self, g, inputs):
        with g.local_scope():
            if isinstance(inputs, tuple) or g.is_block:
                if isinstance(inputs, tuple):
                    src_inputs, dst_inputs = inputs
                else:
                    src_inputs = inputs
                    # dead code dst_inputs will not be used
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
                g.srcdata['h'] = src_inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            else:
                g.srcdata['h'] = inputs
                g.update_all(fn.copy_u('h', 'm'), self._lstm_reducer)
                h_neigh = g.dstdata['neigh']
            return h_neigh

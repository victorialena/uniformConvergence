import dgl
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.utils import Sequential as dglSequential

class GraphRegression(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=8):
        super(GraphRegression, self).__init__()
        self.conv = dglSequential(dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu),
                                  dglnn.GraphConv(hidden_dim, hidden_dim))
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, g, h):
        h = F.relu(self.conv(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return self.readout(hg)

    def _get_norm(self, _norm=2): 
        """
        _norm specifies which norm to use, e.g., 'fro' for Frobenius norm or 2 for L2-norm.
        TODO: should we filter for just weight (currently adding bias norm as well.)
        """
        with torch.no_grad():
            return torch.stack([torch.norm(v, _norm) for v in self.conv.parameters()]).sum() +\
        torch.stack([torch.norm(v, _norm) for v in self.readout.parameters()]).sum()

    def _depth(self):
        return len(self.conv)
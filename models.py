import dgl
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.utils import Sequential as dglSequential

class GraphRegression(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=8, norm=None):
        super(GraphRegression, self).__init__()
        self.conv = dglSequential(dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu),
                                  dglnn.GraphConv(hidden_dim, hidden_dim))
        self.readout = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.norm = norm
        if norm:
            self.normalize(norm)

    def forward(self, g, h):
        h = F.relu(self.conv(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            return self.readout(hg)

    def _check_norm(self, norm, throw=False):
        msg = "This is not the same norm as has been used during __init__()"
        if self.norm and (self.norm != norm):
            if throw:
                assert False, msg
            print("UserWarning:" + msg)

    def _get_layer_norms(self, _norm):
        "_norm specifies which norm to use, e.g., 'fro' for Frobenius norm or 2 for L2-norm."
        with torch.no_grad():
            return torch.stack([W.norm(_norm) for name, W in self.conv.named_parameters() if 'weight' in name] + \
                [W.norm(_norm) for name, W in self.readout.named_parameters() if 'weight' in name])

    def normalize(self, _norm):
        self._check_norm(_norm, throw=True)
        for layer in [*self.conv, *self.readout]:
            for name, W in layer.named_parameters():
                setattr(layer, name, torch.nn.Parameter(F.normalize(W, p=_norm, dim=None)))

    def get_norm(self, _norm):
        self._check_norm(_norm)
        return self._get_layer_norms(_norm).prod()

    def depth(self):
        return len(self.conv)+len(self.readout)

class NodeClassification(nn.Module):
    def __init__(self, num_classes, norm=None, hidden_dim=8):
        super(NodeClassification, self).__init__()
        self.conv = dglSequential(dglnn.GraphConv(num_classes, hidden_dim))
        self.readout = nn.Sequential(nn.Linear(hidden_dim, num_classes)) #, nn.Softmax(dim=1))

        self.norm = norm
        if norm:
            self.normalize(norm)

    def forward(self, g, h):
        with g.local_scope():
            h = F.relu(self.conv(g.add_self_loop(), h))
        return self.readout(h)

    def _check_norm(self, norm, throw=False):
        msg = "This is not the same norm as has been used during __init__()"
        if self.norm and (self.norm != norm):
            if throw:
                assert False, msg
            print("UserWarning:" + msg)

    def _get_layer_norms(self, _norm):
        "_norm specifies which norm to use, e.g., 'fro' for Frobenius norm or 2 for L2-norm."
        with torch.no_grad():
            return torch.stack([W.norm(_norm) for name, W in self.conv.named_parameters() if 'weight' in name] + \
                [W.norm(_norm) for name, W in self.readout.named_parameters() if 'weight' in name])

    def normalize(self, _norm):
        self._check_norm(_norm, throw=True)
        for layer in [*self.conv, *self.readout]:
            for name, W in layer.named_parameters():
                setattr(layer, name, torch.nn.Parameter(F.normalize(W, p=_norm, dim=None)))

    def get_norm(self, _norm):
        self._check_norm(_norm)
        return self._get_layer_norms(_norm).prod()

    def depth(self):
        return len(self.conv)+len(self.readout)

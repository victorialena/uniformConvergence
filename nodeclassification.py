import argparse

import dgl
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import NodeClassification

import pdb
import random


NUM_BLOCKS = 3
NUM_TEST_PER_GRAPH = 5


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='conv', required=False)
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--num_samples', type=int, default=10000, required=False)
    parser.add_argument('--num_test_samples', type=int, default=50, required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-3, required=False)
    parser.add_argument('--epochs', type=int, default=10, required=False)
    parser.add_argument('--seed', type=int, default=42, required=False)
    parser.add_argument('--norm', type=int, default=1, required=False)
    return parser.parse_args()


def sbm(block_size, cov, self_loop=False, directed=False):
    """ (Symmetric) Stochastic Block Model
    https://docs.dgl.ai/generated/dgl.data.SBMMixtureDataset.html?highlight=sbm#dgl.data.SBMMixtureDataset
    
    n_blocks : int
        Number of blocks.
    cov : float
        Matrix with diag terms denoting probability for intra-community edge (within a group, e.g., p), 
        and off diagonal terms the probability for inter-community edge (between two groups, e.g., q).

    Returns dgl.Graph
    """
    n = block_size.sum()
    probs = torch.repeat_interleave(cov, block_size, dim=1)
    probs = torch.repeat_interleave(probs, block_size, dim=0)
    if directed:
        adj = torch.rand(n,n) < probs
        _from, _to = torch.where(adj & ~torch.eye(n, dtype=bool))        
    else:
        adj = (torch.rand(n,n) < probs).triu(1)
        _from, _to = torch.where(adj+adj.t())
    
    if self_loop:
        return dgl.graph((_from, _to), num_nodes=n).add_self_loop()
    return dgl.graph((_from, _to), num_nodes=n)


def sbm_helper(n_blocks, cov, self_loop=False, directed=False):
    block_size = torch.randint(low=5, high=10, size=(n_blocks,))
    g = sbm(block_size, cov, self_loop=False, directed=False)
    g.ndata['y'] = torch.repeat_interleave(torch.arange(n_blocks), block_size).unsqueeze(-1)
    return g


def sample_cov(n, p=0.7, q=0.1, dev=0.01, eps=0.05):
    cov = torch.normal(mean=q*torch.ones(n,n) + (p-q)*torch.eye(n), std=dev).clamp(min=eps, max=1-eps).round(decimals=3)
    return cov.triu() + cov.triu(1).t()


def synthetic_dataset(n, prior, add_self_loop=False):
    labels = [prior(NUM_BLOCKS) for _ in range(n)]
    graphs = [sbm_helper(NUM_BLOCKS, cov, self_loop=add_self_loop) for cov in labels]
    return graphs, labels


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, n, prior):
        self.n = n
        self.G, self.labels = synthetic_dataset(self.n, prior)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.G[index], self.labels[index]


def draw(g, random=True):
    if random:
        colors = np.array(["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(NUM_BLOCKS)])
        colors = colors[g.ndata['y'].squeeze().tolist()]
    else:
        colors = "bgrcmykw"
        colors = [colors[i] for i in g.ndata['y'].squeeze().tolist()]
        
    nx.draw(g.to_networkx(), node_color=colors, arrows=False, with_labels=True)


def get_mask(bg, num=NUM_TEST_PER_GRAPH):
    return torch.sort(torch.stack([torch.randperm(n)[:num] for n in bg.batch_num_nodes()]), dim=1)[0]


def get_labels(bg):
    return F.one_hot(bg.ndata['y'].squeeze()).to(torch.float32)


def h_star(g, mask, cov, num=NUM_TEST_PER_GRAPH, logits=True):
    H = get_labels(g)
    out = torch.stack([H[g.successors(nid)].mm(cov[i//num]).log().sum(0) for i, nid in enumerate(mask)])
    if logits: return out
    return out.softmax(1)


def get_normalized_features(bg, num=5):
    mask = get_mask(bg, num=num)
    
    # get prior
    H = get_labels(bg)
    n = torch.cat([torch.tensor([0]), torch.cumsum(bg.batch_num_nodes(), dim=0)])
    prior = torch.stack([H[s:t].mean(0) for s,t in zip(n[:-1], n[1:])])

    prior = prior.repeat_interleave(num, dim=0)
    mask = (mask+n[:-1].unsqueeze(-1)).flatten()
    H[mask] = prior
    
    return H, mask

def node_classification(args):
    dataset = Dataset(args.num_samples, sample_cov)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    model = NodeClassification()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train(True)
    for _ in range(args.epochs):
        for batched_graph, _ in dataloader:
            labels = get_labels(batched_graph)
            opt.zero_grad()
            
            feats, mask = get_normalized_features(batched_graph)
            logits = model(batched_graph, feats)
            loss = loss_fn(logits[mask], labels[mask])
            loss.backward()
            opt.step()
    
    return model


def eval_excess_risk(model, args):
    graphs, cov = synthetic_dataset(args.num_test_samples, sample_cov)

    model.train(False)
    with torch.no_grad():
        batched_graph = dgl.batch(graphs)
        labels = get_labels(batched_graph)
        feats, mask = get_normalized_features(batched_graph)

        logits = model(batched_graph, feats)
        L_hat = F.cross_entropy(logits[mask], labels[mask])
        Lstar = F.cross_entropy(h_star(batched_graph, mask, cov), labels[mask])

    return L_hat - Lstar


def get_feature_norm(args):
    if args._norm == 1:
        return 1
    elif args._norm == 2: # = max singual value < norm 'fro'
        return np.sqrt(args.batch_size)
    elif args._norm == 'fro':
        return np.sqrt(args.batch_size)
    raise NotImplementedError("This norm is not implemented.")


def estimate_rademacher(model, args):
    B = get_feature_norm(args) #1 if args.norm == 1 else # features are normalized
    M = model._get_norm(args.norm)
    d = model._depth()
    return B*M*(np.sqrt(2*np.log(2)*d) + 1) / np.sqrt(args.num_samples)
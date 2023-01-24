import argparse

import dgl
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from csv import DictWriter
from models import NodeClassification

import pdb
import random


NUM_BLOCKS = 3
NUM_TEST_PER_GRAPH = 5

field_names = ['num_samples', 'epoch', 'excess_risk', 'rademacher', 'M', 'norm', 'batch_size', 'learning_rate']


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='conv')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--frac_test_samples', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--norm', default=2)
    parser.add_argument('--normalize', type=bool, default=False)
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
    block_size = torch.randint(low=5, high=25, size=(n_blocks,))
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


def h_star(g, mask, cov, prior, num=NUM_TEST_PER_GRAPH, logits=True):
    H = get_labels(g)
    out = torch.stack([H[g.successors(nid)].mm(cov[i//num]).log().sum(0) + prior[i].log() for i, nid in enumerate(mask)])
    if logits: return out
    return out.softmax(1)


def get_normalized_features(bg, num=NUM_TEST_PER_GRAPH):
    mask = get_mask(bg, num=num)
    
    # get prior
    H = get_labels(bg)
    n = torch.cat([torch.tensor([0]), torch.cumsum(bg.batch_num_nodes(), dim=0)])
    prior = torch.stack([H[s:t].mean(0) for s,t in zip(n[:-1], n[1:])])

    prior = prior.repeat_interleave(num, dim=0)
    mask = (mask+n[:-1].unsqueeze(-1)).flatten()
    H[mask] = prior
    
    return H, mask


def eval_excess_risk(model, args):
    graphs, cov = synthetic_dataset(int(args.frac_test_samples * args.num_samples), sample_cov)

    model.train(False)
    with torch.no_grad():
        batched_graph = dgl.batch(graphs)
        labels = get_labels(batched_graph)
        feats, mask = get_normalized_features(batched_graph)

        logits = model(batched_graph, feats)
        L_hat = F.cross_entropy(logits[mask], labels[mask])
        Lstar = F.cross_entropy(h_star(batched_graph, mask, cov, feats[mask]), labels[mask])

    return L_hat - Lstar


def get_feature_norm(args):
    if args.norm == 1:
        return 1
    elif args.norm == 2: # = max singual value < norm 'fro'
        return np.sqrt(args.batch_size)
    elif args.norm == 'fro':
        return np.sqrt(args.batch_size)
    raise NotImplementedError("This norm is not implemented.")


def estimate_rademacher(model, args):
    B = get_feature_norm(args)
    M = model.get_norm(args.norm)
    d = model.depth()
    return B*M*(np.sqrt(2*np.log(2)*d) + 1) / np.sqrt(args.num_samples)


def node_classification(args):
    dataset = Dataset(args.num_samples, sample_cov)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    model = NodeClassification(num_classes=NUM_BLOCKS, norm=args.norm if args.normalize else None)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train(True)
    for ep in range(args.epochs):
        for batched_graph, _ in dataloader:
            labels = get_labels(batched_graph)
            opt.zero_grad()
            
            feats, mask = get_normalized_features(batched_graph)
            logits = model(batched_graph, feats)
            loss = loss_fn(logits[mask], labels[mask])
            loss.backward()
            opt.step()
            if args.normalize:
                model.normalize(args.norm)

        if ep in [10, 20]:
            print("---Ep:", ep)
            excess_risk = eval_excess_risk(model, args)
            rademacher = estimate_rademacher(model, args)
            print("Excess risk:", excess_risk, "| Rademacher:", rademacher, \
                "(with M=", model.get_norm(args.norm), "=", model._get_layer_norms(args.norm),")\n")

            dict = {'num_samples': args.num_samples,
                    'epoch': ep,
                    'excess_risk': excess_risk.item(),
                    'rademacher': rademacher.item(),
                    'M': model.get_norm(args.norm).item(),
                    'norm': args.norm,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate}
            with open('node_classification.csv', 'a') as file:
                dictwriter_object = DictWriter(file, fieldnames=field_names)
                dictwriter_object.writerow(dict)
                file.close()
                    
    return model


if __name__ == "__main__":
    args = argparser()
    print("---------------------\n", "bs:", args.batch_size, "| lr:", args.learning_rate)

    torch.manual_seed(args.seed)
    print("training model...")
    model = node_classification(args)
    print("done. evaluating exces risk...")
    excess_risk = eval_excess_risk(model, args)
    rademacher = estimate_rademacher(model, args)

    print("Excess risk:", excess_risk, "| Rademacher:", rademacher, \
        "(with M=", model.get_norm(args.norm), "=", model._get_layer_norms(args.norm),")\n")

    dict = {'num_samples': args.num_samples,
            'epoch': args.epochs,
            'excess_risk': excess_risk.item(),
            'rademacher': rademacher.item(),
            'M': model.get_norm(args.norm).item(),
            'norm': args.norm,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate}
    with open('node_classification.csv', 'a') as file:
        dictwriter_object = DictWriter(file, fieldnames=field_names)
        dictwriter_object.writerow(dict)
        file.close()
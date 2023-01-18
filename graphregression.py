import argparse

import dgl
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GraphRegression
import pdb


MIN_NUM_NODES = 10
MAX_NUM_NODES = 100


def synthetic_dataset(n, prior, add_self_loop=True):
    labels = prior(n)
    sizes = np.random.randint(MIN_NUM_NODES, MAX_NUM_NODES, n)
    if add_self_loop:
        graphs = [dgl.from_networkx(nx.erdos_renyi_graph(sz, p)).add_self_loop() for sz, p in zip(sizes, labels)]
        return graphs, labels
    
    graphs = [dgl.from_networkx(nx.erdos_renyi_graph(sz, p)) for sz, p in zip(sizes, labels)]
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


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='conv', required=False)
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--num_samples', type=int, default=10000, required=False)
    parser.add_argument('--num_test_samples', type=int, default=50, required=False)
    parser.add_argument('--epochs', type=int, default=10, required=False)
    parser.add_argument('--seed', type=int, default=42, required=False)
    return parser.parse_args()


def h_star(g):
    if g.batch_size > 1:
        return (g.batch_num_edges() / torch.pow(g.batch_num_nodes(), 2)).unsqueeze(1)
    return g.num_edges() / torch.pow(g.num_nodes(), 2)


def get_normalized_features(bg):
    return 1./np.repeat(bg.batch_num_nodes(), bg.batch_num_nodes(), axis=0).unsqueeze(1)


def get_prior():
    return lambda x: torch.rand(x).round(decimals=3).clamp(min=0.05) # todo: try a clipped gaussian


def eval_excess_risk(model, args):
    prior = get_prior()
    graphs, labels = synthetic_dataset(args.num_test_samples, prior)
    labels = labels.unsqueeze(-1)

    model.train(False)
    with torch.no_grad():
        batched_graph = dgl.batch(graphs)
        feats = get_normalized_features(batched_graph)
        y_hat = model(batched_graph, feats)
        L_hat = F.mse_loss(y_hat, labels)
        Lstar = F.mse_loss(h_star(batched_graph), labels)

    return L_hat - Lstar


def estimate_rademacher(model, args):
    B = 1 # features are normalized
    M = model._get_norm()
    d = model._depth()
    return B*M*(np.sqrt(2*np.log(2)*d) + 1) / np.sqrt(args.num_samples)


def graph_regression(args):
    prior = get_prior()

    model = GraphRegression()
    dataset = Dataset(args.num_samples, prior)
    dataloader = dgl.dataloading.GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    model.train(True)
    for _ in range(args.epochs):
        for batched_graph, labels in dataloader:
            opt.zero_grad()
            
            feats = get_normalized_features(batched_graph)
            y_hat = model(batched_graph, feats)
            loss = loss_fn(y_hat, labels.unsqueeze(-1)) # loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
    
    return model


if __name__ == "__main__":
    args = argparser()

    torch.manual_seed(args.seed)
    print("training model...")
    model = graph_regression(args)
    print("done. evaluating exces risk...")
    excess_risk = eval_excess_risk(model, args)
    rademacher = estimate_rademacher(model, args)

    print("Excess risk:", excess_risk, "| Rademacher:", rademacher, " (with M=", model._get_norm(), ")\n")
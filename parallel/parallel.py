"""Parallel ensembling.
Run from the root directory of this repo as python -m parallel.parallel [--OPTIONS]
"""
from argparse import ArgumentParser

import torch
import einops

from scripts.run_attribution import method_to_path, save_graph
from scripts.run_evaluation import load_graph_from_file

def normalize(graph):
    """normalize the scores of all incoming edges to a node such that they sum to 1.
    This makes sense if one of the methods requires greedy decoding
    (such as information flow routes)

    Args:
        graph (Graph): The input graph
    Returns:
        Graph: the new graph
    """
    graph.scores = graph.scores / einops.reduce(
        graph.scores, 'forward backward -> 1 backward', reduction='sum'
    )
    return graph

def parallel_ensembling_inner(args, list_of_graphs):
    """Take pre-computed edge scores (as graphs) and produce a new graph with ensembled edge scores
    Arguments:
        args (Namespace): from the argument parser. For reduction and normalize_incoming.
        list_of_graphs (list[Graph]): the graphs
    Returns:
        Graph
    """
    if args.normalize_incoming:
        for i,graph in enumerate(list_of_graphs):
            list_of_graphs[i] = normalize(graph)
    new_graph = list_of_graphs[0]
    stacked_scores = torch.stack([graph.scores for graph in list_of_graphs])
    reduction=args.reduction
    if reduction=='weighted':
        assert len(args.weights)==len(list_of_graphs)
        weights=torch.tensor(args.weights)
        stacked_scores *= weights[:,None]
        reduction="mean"
    new_scores = einops.reduce(
        stacked_scores,
        "method ... -> ...",
        reduction
    )
    new_graph.scores = new_scores
    return new_graph


def parallel_ensembling(args):
    """Load the edge scores pre-computed for the given circuit detection methods,
    compute ensemble scores (means),
    and save the result.

    Args:
        args (Namespace):
            The arguments given by argparse, such as model name, task and list of methods
    """
    n_methods = len(args.methods)
    ablations = expand(args.ablations, n_methods)
    levels = expand(args.levels, n_methods)
    list_of_graphs = []
    for i, method in enumerate(args.methods):
        path = method_to_path(
            method=method, model_name=args.model_name, task=args.task, circuit_dir=args.circuit_dir,
            ablation=ablations[i], level=levels[i],
        )
        list_of_graphs.append(load_graph_from_file(path))
    new_graph = parallel_ensembling_inner(
        args, list_of_graphs
    )
    save_graph(
        new_graph,
        args.circuit_dir,
        args.reduction+"-"+"-".join(args.methods),
        "-".join(args.ablations),
        "-".join(args.levels),
        args.task,
        args.model_name,
    )

def expand(singleton:list, n:int):
    """Expand a list of 1 to n elements. If list already has n elements, do nothing.

    Args:
        singleton (list): list of either 1 or n elements
        n (int): number of elements to expand the list to

    Raises:
        ValueError: if singleton does not have either 1 or n entries

    Returns:
        list: expanded list of n elements
    """
    if len(singleton)==1:
        return singleton*n
    if len(singleton)==n:
        return singleton
    raise ValueError(
        f"The list {singleton} should have either 1 or {n} entries, but has {len(singleton)}"
    )

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--normalize_incoming", action="store_true",
        help="""
        if one of the methods (such as information flow routes) requires greedy decoding,
        this method adapts all the scores to it
        """
    )
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--task", default="ioi")
    parser.add_argument("--circuit_dir", default="circuits")
    parser.add_argument(
        "--reduction", default="mean", choices=["mean", "min", "max", "weighted"]
    )#TODO median?
    parser.add_argument(
        "--weights", nargs='+', type=float,
        help="""The weight to apply to each method, in order.
        Only applies if reduction==weighted, ignored otherwise."""
    )
    parser.add_argument("--methods", nargs='+')
    parser.add_argument("--ablations", nargs='+', default=["patching"])
    parser.add_argument("--levels", nargs='+', default=["edge"])
    args = parser.parse_args()
    if "node" in args.levels:#TODO
        raise NotImplementedError("Node-level ensembling not implemented yet")
    parallel_ensembling(args)

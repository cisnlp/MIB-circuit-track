"""
Visualize circuit.

Code given by *Finding Transformer Circuits with Edge Pruning*  (Adithya Bhaskar et al. 2024).
<https://github.com/princeton-nlp/Edge-Pruning>

MIT License

Copyright (c) 2024 Adithya Bhaskar, Alexander Wettig, Dan Friedman, and Danqi Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import logging
from typing import Callable, List, Optional, Set, Tuple, Union

import graphviz


def get_constant_color(color: str = "aquamarine1") -> Callable:
    """
    Get constant color.

    Args:
        color (str, optional): Color to obtain for element. Defaults to "aquamarine1".

    Returns:
        Callable: Constant color mapping.
    """
    return lambda x: color


def get_circuit_colors_v1(
    embeds_color: Optional[str] = "azure",  # azure2
    mlp_color: Optional[str] = "cadetblue2",  # cornflowerblue
    q_color: Optional[str] = "plum1",  # orchid3
    k_color: Optional[str] = "lightpink",  # chocolate1
    v_color: Optional[str] = "khaki1",  # gold1
    o_color: Optional[str] = "darkslategray3",  # aquamarine1
    resid_post_color: Optional[str] = "azure",  # azure2
) -> Callable[[str], str]:
    """
    Get color mapping for circuit elements.

    Args:
        embeds_color (str, optional): Color to obtain for embeds. Defaults to "azure".
        mlp_color (str, optional): Color to obtain for MLPs. Defaults to "cadetblue2".
        q_color (str, optional): Color to obtain for Q elements. Defaults to "plum1".
        k_color (str, optional): Color to obtain for K elements. Defaults to "lightpink".
        v_color (str, optional): Color to obtain for V elements. Defaults to "khaki1".
        o_color (str, optional): Color to obtain for Output elements. Defaults to "darkslategray3".
        resid_post_color (str, optional): Color to obtain for residual post. Defaults to "azure".

    Returns:
        Callable[[str], str]: Color mapping function.
    """

    def decide_color(node_name):
        if "embed" in node_name:
            return embeds_color
        elif node_name == "resid_post":
            return resid_post_color
        elif node_name.startswith("m"):
            return mlp_color
        elif node_name.endswith(".q"):
            return q_color
        elif node_name.endswith(".k"):
            return k_color
        elif node_name.endswith(".v"):
            return v_color
        else:
            return o_color

    return decide_color


def get_circuit_colors(
    tok_embeds_color: Optional[str] = "grey",  # azure2
    pos_embeds_color: Optional[str] = "lightsteelblue",  # azure2
    mlp_color: Optional[str] = "cadetblue2",  # cornflowerblue
    q_color: Optional[str] = "plum1",  # orchid3
    k_color: Optional[str] = "lightpink",  # chocolate1
    v_color: Optional[str] = "khaki1",  # gold1
    o_color: Optional[str] = "darkslategray3",  # aquamarine1
    resid_post_color: Optional[str] = "azure",  # azure2
) -> Callable[[str], str]:
    """
    Get color mapping for circuit elements.

    Args:
        tok_embeds_color (str, optional): Color to obtain for token embeddings. Defaults to "grey".
        pos_embeds_color (str, optional): Color to obtain for positional embeddings. Defaults to "lightsteelblue".
        mlp_color (str, optional): Color to obtain for MLPs. Defaults to "cadetblue2".
        q_color (str, optional): Color to obtain for Q elements. Defaults to "plum1".
        k_color (str, optional): Color to obtain for K elements. Defaults to "lightpink".
        v_color (str, optional): Color to obtain for V elements. Defaults to "khaki1".
        o_color (str, optional): Color to obtain for Output elements. Defaults to "darkslategray3".
        resid_post_color (str, optional): Color to obtain for residual post. Defaults to "azure".

    Returns:
        Callable[[str], str]: Color mapping function.
    """

    def decide_color(node_name):
        node_name = node_name.lower()
        if "embed" in node_name:
            if "pos" in node_name:
                return pos_embeds_color
            else:
                return tok_embeds_color
        elif node_name == "output":
            return resid_post_color
        elif node_name.startswith("m"):
            return mlp_color
        elif node_name.endswith(".q"):
            return q_color
        elif node_name.endswith(".k"):
            return k_color
        elif node_name.endswith(".v"):
            return v_color
        else:
            return o_color

    return decide_color


def sanitize_edges(edges: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Sanitize and modify a list of circuit edges by ensuring specific connectivity
    and removing redundant or unconnected nodes.

    This function performs the following operations:
    1. Adds edges from nodes with "a" prefix that are not q, k, or v, to their
       respective ".q", ".k", and ".v" nodes.
    2. Iteratively removes edges where the destination node is not a source node
       elsewhere, excluding the "resid_post" node.
    3. Removes q, k, v edges if they have no incoming edges and their corresponding
       q -> o, k -> o, v -> o edges are not needed.

    Args:
        edges (List[Tuple[str, str]]): A list of tuples representing edges in the format
            (source_node, destination_node).

    Returns:
        List[Tuple[str, str]] of tuples: A sanitized list of edges with unnecessary edges removed.
    """
    # First, add all q,k,v -> o edges
    new_edges_ = set()
    for edge in edges:
        if edge[0][0] == "a" and edge[0][-1] not in ["q", "k", "v"]:
            new_edges_.add(edge[0])
    for to in new_edges_:
        for suffix in [".q", ".k", ".v"]:
            from_ = to + suffix
            edges.append((from_, to))
    while True:
        orig_len = len(edges)
        # Find all nodes that are destinations but not sources
        froms = set()
        tos = set()
        for edge in edges:
            froms.add(edge[0])
            if edge[1] != "resid_post":
                tos.add(edge[1])
        banned_tos = tos.difference(froms)
        edges = [e for e in edges if e[1] not in banned_tos]

        # Find qkv nodes that have no incoming edges, and remove the q -> o edge for them
        qkv_nodes = set()
        for edge in edges:
            if edge[1].endswith(".q"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".k"):
                qkv_nodes.add(edge[1])
            elif edge[1].endswith(".v"):
                qkv_nodes.add(edge[1])

        edges = [
            e
            for e in edges
            if not (
                (e[0].endswith(".q") and e[0] not in qkv_nodes)
                or (e[0].endswith(".k") and e[0] not in qkv_nodes)
                or (e[0].endswith(".v") and e[0] not in qkv_nodes)
            )
        ]
        if orig_len == len(edges):
            break

    return edges


def rename(name: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Rename a node name into a human-readable format.

    Args:
        name (Union[str, List[str]]): The node name(s) to be renamed.

    Returns:
        Union[str, List[str]]: The renamed node name(s).
    """
    if isinstance(name, list):
        return [rename(n) for n in name]  # type: ignore
    if "embeds" in name:
        return "Embeddings"
    if name == "resid_post":
        return "Output"
    if name.startswith("m"):
        layer = int(name[1:])
        return f"MLP {layer}"
    if name.endswith(".q"):
        parts = name.split(".")
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        return f"Head {layer}.{head}.Q"
    if name.endswith(".k"):
        parts = name.split(".")
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        return f"Head {layer}.{head}.K"
    if name.endswith(".v"):
        parts = name.split(".")
        layer = int(parts[0][1:])
        head = int(parts[1][1:])
        return f"Head {layer}.{head}.V"
    parts = name.split(".")
    assert len(parts) == 2, f"Invalid node name {name}"
    layer = int(parts[0][1:])
    head = int(parts[1][1:])
    return f"Head {layer}.{head}.O"


def draw_graph(
    nodes: List[str] | Set[str],
    constant_nodes: List[str],
    edges: List[Tuple[str, str]],
    coloring_fn: Callable,
    constant_edge_color: str = "gray66",
) -> graphviz.Digraph:
    """
    Draw a graph with nodes and edges colored according to the given coloring function.

    Args:
        nodes (List[str] | Set[str]): The node names.
        constant_nodes (List[str]): The names of nodes that are "constant" and should not be colored.
        edges (List[Tuple[str, str]]): The edges of the graph.
        coloring_fn (Callable): A function that takes a node name and returns a color string.
        constant_edge_color (str, optional): The color to use for edges that connect to constant nodes. Defaults to "gray66".

    Returns:
        graphviz.Digraph: The drawn graph.
    """
    kwargs = {
        "graph_attr": {
            "nodesep": "0.02",
            "ranksep": "0.02",
            "ratio": "1:6",
        },
        "node_attr": {
            "shape": "box",
            "style": "rounded,filled",
        },
    }

    g = graphviz.Digraph(**kwargs)
    for node in nodes:
        g.node(node, color="black", fillcolor=coloring_fn(node))

    for edge in edges:
        g.edge(edge[0], edge[1], color=coloring_fn(edge[0]))

    for node in nodes:
        for cin in constant_nodes:
            if node not in constant_nodes:
                g.edge(cin, node, color=constant_edge_color)

    return g


def vis_circuit(
    edges: List[Tuple[str, str]],
    out_path: str,
    clean_edges: Optional[bool] = True,
    constant_nodes: Optional[List[str]] = None,
) -> None:
    """
    Visualize the given edges and nodes in a graph.

    Args:
        edges (List[Tuple[str, str]]): List of edges in the graph. Each edge is a tuple of two node names.
        out_path (str): The output path to save the graph to.
        clean_edges (Optional[bool], optional): Whether to remove edges that have both source and
            target nodes in the constant_nodes list. Defaults to True.
        constant_nodes (Optional[List[str]], optional): List of node names that are "constant" and
            should not be colored. Defaults to None.

    """
    if clean_edges:
        edges = sanitize_edges(edges)

    coloring_fn = get_circuit_colors()
    constant_edge_color = "gray66"

    if constant_nodes is None:
        constant_nodes = []

    nodes = set(constant_nodes + [x for y in edges for x in y])

    g = draw_graph(
        nodes=nodes,
        constant_nodes=constant_nodes,
        edges=edges,
        coloring_fn=coloring_fn,
        constant_edge_color=constant_edge_color,
    )
    g.render(out_path)
    logging.info(f"Plot for circuit saved at: {out_path}")


# graph_plot.py
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


def layered_positions_lr(nodes: List[str], edges: List[Tuple[str, str]], start: str = "router") -> Dict[str, Tuple[float, float]]:
    """
    Compute left-to-right (layered) positions using BFS distance from `start`.
    Nodes at distance d go in x=d; y is spread within each layer.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # BFS distances from start determine x-layer
    dist = {start: 0}
    q = deque([start])

    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    # Put unreachable nodes into last layer
    max_layer = max(dist.values()) if dist else 0
    for n in G.nodes():
        dist.setdefault(n, max_layer + 1)

    # Group nodes by layer
    layers = defaultdict(list)
    for n, d in dist.items():
        layers[d].append(n)

    # Assign positions
    pos: Dict[str, Tuple[float, float]] = {}
    for layer in sorted(layers.keys()):
        ns = sorted(layers[layer])
        if len(ns) == 1:
            ys = [0.0]
        else:
            # spread vertically centered around 0
            step = 1.4 / (len(ns) - 1)
            ys = [0.7 - i * step for i in range(len(ns))]
        for n, y in zip(ns, ys):
            pos[n] = (float(layer), float(y))

    return pos



def plot_langgraph_lr(nodes, edges, edge_labels=None, title="LangGraph Workflow (Left → Right)", path=None):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    start = "router" if "router" in nodes else (nodes[0] if nodes else "router")
    pos = layered_positions_lr(nodes, edges, start=start)

    path = path or []
    path_set = set(path)
    path_edges = set(zip(path, path[1:]))  # ✅ edges along the executed path

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_title(title)

    # ✅ Base nodes
    base_nodes = [n for n in G.nodes() if n not in path_set]
    nx.draw_networkx_nodes(
        G, pos, nodelist=base_nodes,
        node_size=2400, node_color="#E6F2FF",
        edgecolors="#1F77B4", linewidths=1.6, ax=ax
    )

    # ✅ Highlight nodes
    hi_nodes = [n for n in G.nodes() if n in path_set]
    nx.draw_networkx_nodes(
        G, pos, nodelist=hi_nodes,
        node_size=2600, node_color="#FFF2CC",
        edgecolors="#FF9900", linewidths=2.4, ax=ax
    )

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # ✅ Base edges
    base_edges = [e for e in G.edges() if e not in path_edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=base_edges,
        arrowstyle="->", arrowsize=18,
        width=1.6, edge_color="#444", alpha=0.6, ax=ax
    )

    # ✅ Highlight edges
    hi_edges = [e for e in G.edges() if e in path_edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=hi_edges,
        arrowstyle="->", arrowsize=22,
        width=3.2, edge_color="#FF9900", ax=ax
    )

    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)

    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def default_workflow_metadata():
    """
    Convenience helper returning nodes/edges/labels for your current lean workflow.
    Update here when you add/remove nodes.
    """
    nodes = ["router", "general", "news", "clarify", "END"]
    edges = [
        ("router", "news"),
        ("router", "general"),
        ("router", "clarify"),
        ("news", "END"),
        ("general", "END"),
        ("clarify", "END"),
    ]
    edge_labels = {
        ("router", "news"): "intent=news",
        ("router", "general"): "intent=general",
        ("router", "clarify"): "intent=ambiguous",
    }
    return nodes, edges, edge_labels


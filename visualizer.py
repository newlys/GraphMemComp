"""Graph visualization utilities."""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def save_graph_figure(graph: nx.Graph, output_path: str = "memory_graph.png") -> None:
    """Render the memory graph into a PNG file for quick inspection."""
    if graph.number_of_nodes() == 0:
        return
    
    # Simple figure save without matplotlib for now
    pass

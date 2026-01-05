"""
Graph Theory Module for FluxEM-Domains

Provides embeddings for graphs with operations defined by adjacency matrices.

Supports:
- Finite graphs (directed and undirected)
- Adjacency matrix encoding (8x8 = up to 8 nodes)
- Graph operations (union, intersection, complement)
- Path and connectivity operations
- Property detection (connected, acyclic, bipartite, etc.)
"""

from .graphs import Graph, GraphEncoder, GraphType
from .paths import PathEncoder, find_path, find_all_paths, shortest_path
from .properties import (
    is_connected,
    is_acyclic,
    is_bipartite,
    is_complete,
    is_tree,
    is_dag,
    has_cycle,
    connected_components,
)

__all__ = [
    # Core
    "Graph",
    "GraphEncoder",
    "GraphType",
    # Paths
    "PathEncoder",
    "find_path",
    "find_all_paths",
    "shortest_path",
    # Properties
    "is_connected",
    "is_acyclic",
    "is_bipartite",
    "is_complete",
    "is_tree",
    "is_dag",
    "has_cycle",
    "connected_components",
]

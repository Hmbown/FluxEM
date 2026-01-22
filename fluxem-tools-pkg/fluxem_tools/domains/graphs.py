"""Graphs domain - connectivity, paths, properties.

This module provides deterministic graph computations.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def build_adjacency(nodes: List[int], edges: List[Tuple[int, int]], directed: bool = False) -> Dict[int, Set[int]]:
    """Build adjacency list from nodes and edges."""
    adj = {n: set() for n in nodes}
    for u, v in edges:
        if u in adj:
            adj[u].add(v)
        if not directed and v in adj:
            adj[v].add(u)
    return adj


def shortest_path_bfs(nodes: List[int], edges: List[Tuple[int, int]], start: int, target: int) -> Tuple[List[int], int]:
    """Find shortest path using BFS.

    Args:
        nodes: List of node IDs
        edges: List of (u, v) edge tuples
        start: Start node
        target: Target node

    Returns:
        Tuple of (path, distance) or ([], -1) if no path
    """
    adj = build_adjacency(nodes, edges)

    if start not in adj or target not in adj:
        return [], -1

    if start == target:
        return [start], 0

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        for neighbor in adj.get(node, []):
            if neighbor == target:
                return path + [neighbor], len(path)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return [], -1


def is_connected(nodes: List[int], edges: List[Tuple[int, int]]) -> bool:
    """Check if an undirected graph is connected."""
    if not nodes:
        return True
    if len(nodes) == 1:
        return True

    adj = build_adjacency(nodes, edges)
    visited = set()
    stack = [nodes[0]]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(adj.get(node, []))

    return len(visited) == len(nodes)


def has_cycle(nodes: List[int], edges: List[Tuple[int, int]]) -> bool:
    """Check if an undirected graph has a cycle using DFS."""
    adj = build_adjacency(nodes, edges)
    visited = set()

    def dfs(node: int, parent: int) -> bool:
        visited.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    for node in nodes:
        if node not in visited:
            if dfs(node, -1):
                return True
    return False


def is_tree(nodes: List[int], edges: List[Tuple[int, int]]) -> bool:
    """Check if graph is a tree (connected and acyclic)."""
    if len(edges) != len(nodes) - 1:
        return False
    return is_connected(nodes, edges) and not has_cycle(nodes, edges)


def is_bipartite(nodes: List[int], edges: List[Tuple[int, int]]) -> bool:
    """Check if graph is bipartite using BFS coloring."""
    if not nodes:
        return True

    adj = build_adjacency(nodes, edges)
    color = {}

    for start in nodes:
        if start in color:
            continue

        queue = deque([start])
        color[start] = 0

        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    return False
    return True


def node_count(nodes: List[int], edges: List[Tuple[int, int]]) -> int:
    """Count nodes in graph."""
    return len(nodes)


def edge_count(nodes: List[int], edges: List[Tuple[int, int]]) -> int:
    """Count edges in graph."""
    return len(edges)


def graph_properties(nodes: List[int], edges: List[Tuple[int, int]]) -> Dict[str, bool]:
    """Analyze multiple graph properties at once."""
    return {
        "connected": is_connected(nodes, edges),
        "acyclic": not has_cycle(nodes, edges),
        "tree": is_tree(nodes, edges),
        "bipartite": is_bipartite(nodes, edges),
    }


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_graph(args) -> Tuple[List[int], List[Tuple[int, int]]]:
    if isinstance(args, dict):
        nodes = args.get("nodes", [])
        edges = args.get("edges", [])

        # Convert nodes to list of ints
        if isinstance(nodes, (list, tuple)):
            nodes = [int(n) for n in nodes]

        # Convert edges to list of tuples
        if isinstance(edges, (list, tuple)):
            edges = [(int(e[0]), int(e[1])) for e in edges]

        return nodes, edges

    raise ValueError(f"Cannot parse graph: {args}")


def _parse_graph_with_endpoints(args) -> Tuple[List[int], List[Tuple[int, int]], int, int]:
    nodes, edges = _parse_graph(args)
    start = int(args.get("start", args.get("source", nodes[0] if nodes else 0)))
    target = int(args.get("target", args.get("end", nodes[-1] if nodes else 0)))
    return nodes, edges, start, target


def _parse_graph_property(args, prop: str):
    nodes, edges = _parse_graph(args)
    props = graph_properties(nodes, edges)
    return props.get(prop, False)


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register graph tools in the registry."""

    registry.register(ToolSpec(
        name="graphs_shortest_path",
        function=lambda args: shortest_path_bfs(*_parse_graph_with_endpoints(args)),
        description="Finds shortest path between nodes in a graph using BFS.",
        parameters={
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "integer"}, "description": "List of node IDs"},
                "edges": {"type": "array", "items": {"type": "array"}, "description": "List of [u, v] edges"},
                "start": {"type": "integer", "description": "Start node"},
                "target": {"type": "integer", "description": "Target node"},
            },
            "required": ["nodes", "edges", "start", "target"]
        },
        returns="Tuple of (path, distance) or ([], -1) if no path",
        examples=[
            {"input": {"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]], "start": 0, "target": 2}, "output": [[0, 1, 2], 2]},
        ],
        domain="graphs",
        tags=["path", "bfs", "shortest"],
    ))

    registry.register(ToolSpec(
        name="graphs_properties",
        function=lambda args: graph_properties(*_parse_graph(args)),
        description="Analyzes graph properties: connected, acyclic, tree, bipartite.",
        parameters={
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "integer"}, "description": "List of node IDs"},
                "edges": {"type": "array", "items": {"type": "array"}, "description": "List of [u, v] edges"},
            },
            "required": ["nodes", "edges"]
        },
        returns="Dict of boolean properties",
        examples=[
            {"input": {"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]}, "output": {"connected": True, "acyclic": True, "tree": True, "bipartite": True}},
        ],
        domain="graphs",
        tags=["properties", "analysis"],
    ))

    registry.register(ToolSpec(
        name="graphs_node_count",
        function=lambda args: node_count(*_parse_graph(args)),
        description="Returns the number of nodes in a graph.",
        parameters={
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "integer"}, "description": "List of node IDs"},
                "edges": {"type": "array", "items": {"type": "array"}, "description": "List of edges"},
            },
            "required": ["nodes"]
        },
        returns="Node count as integer",
        examples=[
            {"input": {"nodes": [0, 1, 2, 3, 4], "edges": []}, "output": 5},
        ],
        domain="graphs",
        tags=["count", "nodes"],
    ))

    registry.register(ToolSpec(
        name="graphs_is_connected",
        function=lambda args: _parse_graph_property(args, "connected"),
        description="Checks if a graph is connected.",
        parameters={
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "integer"}, "description": "List of node IDs"},
                "edges": {"type": "array", "items": {"type": "array"}, "description": "List of edges"},
            },
            "required": ["nodes", "edges"]
        },
        returns="Boolean",
        examples=[
            {"input": {"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]}, "output": True},
        ],
        domain="graphs",
        tags=["connected", "connectivity"],
    ))

    registry.register(ToolSpec(
        name="graphs_is_tree",
        function=lambda args: _parse_graph_property(args, "tree"),
        description="Checks if a graph is a tree (connected and acyclic).",
        parameters={
            "type": "object",
            "properties": {
                "nodes": {"type": "array", "items": {"type": "integer"}, "description": "List of node IDs"},
                "edges": {"type": "array", "items": {"type": "array"}, "description": "List of edges"},
            },
            "required": ["nodes", "edges"]
        },
        returns="Boolean",
        examples=[
            {"input": {"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]}, "output": True},
        ],
        domain="graphs",
        tags=["tree", "acyclic"],
    ))

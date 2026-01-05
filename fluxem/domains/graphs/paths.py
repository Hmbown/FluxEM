"""
Path Operations for Graph Embeddings

Provides path-finding operations on graph embeddings.
These operations work directly on adjacency matrix representations.
"""

from typing import Any, List, Optional, Set, Tuple
from ...backend import get_backend

from .graphs import Graph, GraphEncoder


class PathEncoder:
    """
    Encoder for paths in graphs.

    A path is encoded as a sequence of nodes with adjacency verification.
    """

    def __init__(self, graph_encoder: Optional[GraphEncoder] = None):
        """Initialize with an optional graph encoder."""
        self.graph_encoder = graph_encoder or GraphEncoder()

    def encode_path(self, path: List[int], max_length: int = 8) -> Any:
        """
        Encode a path as an embedding.

        Path is stored as a sequence of node indices.
        """
        backend = get_backend()
        embedding = backend.zeros((max_length * 2,))

        # Store path length
        embedding = backend.at_add(embedding, 0, float(len(path)))

        # Store node sequence
        for i, node in enumerate(path[:max_length]):
            embedding = backend.at_add(embedding, 1 + i, float(node))

        return embedding

    def decode_path(self, embedding: Any) -> List[int]:
        """Decode a path embedding back to a node sequence."""
        length = int(float(embedding[0]) + 0.5)
        path = []
        for i in range(length):
            node = int(float(embedding[1 + i]) + 0.5)
            path.append(node)
        return path

    def is_valid_path(self, graph_embedding: Any, path: List[int]) -> bool:
        """
        Check if a path is valid in the given graph.

        Verifies each consecutive pair has an edge.
        """
        if len(path) < 2:
            return True

        for i in range(len(path) - 1):
            if not self.graph_encoder.has_edge(graph_embedding, path[i], path[i + 1]):
                return False
        return True

    def path_length(self, path: List[int]) -> int:
        """Return the number of edges in the path."""
        return max(0, len(path) - 1)


def find_path(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """
    Find any path from start to end using BFS.

    Returns None if no path exists.

    Uses breadth-first search for correctness.
    """
    if start not in graph.nodes or end not in graph.nodes:
        return None

    if start == end:
        return [start]

    visited = {start}
    queue = [(start, [start])]

    while queue:
        node, path = queue.pop(0)

        for neighbor in graph.neighbors(node):
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def find_all_paths(
    graph: Graph,
    start: int,
    end: int,
    max_length: Optional[int] = None
) -> List[List[int]]:
    """
    Find all simple paths from start to end.

    Args:
        graph: The graph to search
        start: Starting node
        end: Ending node
        max_length: Maximum path length (edges). None for no limit.

    Returns list of paths (each path is a list of nodes).

    Uses DFS with backtracking to enumerate all paths.
    """
    if start not in graph.nodes or end not in graph.nodes:
        return []

    if max_length is None:
        max_length = graph.num_nodes

    all_paths = []

    def dfs(node: int, path: List[int], visited: Set[int]):
        if node == end:
            all_paths.append(path.copy())
            return

        if len(path) > max_length:
            return

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    dfs(start, [start], {start})
    return all_paths


def shortest_path(graph: Graph, start: int, end: int) -> Optional[List[int]]:
    """
    Find the shortest path from start to end using BFS.

    Returns None if no path exists.

    BFS guarantees shortest path in unweighted graphs.
    """
    # BFS naturally finds shortest path
    return find_path(graph, start, end)


def path_exists(graph: Graph, start: int, end: int) -> bool:
    """
    Check if any path exists from start to end.

    Uses reachability analysis.
    """
    return find_path(graph, start, end) is not None


def reachable_nodes(graph: Graph, start: int) -> Set[int]:
    """
    Find all nodes reachable from start.

    BFS traversal from start node.
    """
    if start not in graph.nodes:
        return set()

    visited = {start}
    queue = [start]

    while queue:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


def distance(graph: Graph, start: int, end: int) -> int:
    """
    Find the shortest distance (number of edges) from start to end.

    Returns -1 if no path exists.

    BFS distance computation.
    """
    if start not in graph.nodes or end not in graph.nodes:
        return -1

    if start == end:
        return 0

    visited = {start}
    queue = [(start, 0)]

    while queue:
        node, dist = queue.pop(0)

        for neighbor in graph.neighbors(node):
            if neighbor == end:
                return dist + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return -1


def all_pairs_shortest_paths(graph: Graph) -> dict:
    """
    Compute shortest path distances between all pairs of nodes.

    Returns dict mapping (src, tgt) -> distance (-1 if unreachable).

    Floyd-Warshall algorithm.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]

    # Self-loops have distance 0
    for i in range(n):
        dist[i][i] = 0

    # Direct edges have distance 1
    for src, tgt in graph.edges:
        i = node_to_idx[src]
        j = node_to_idx[tgt]
        dist[i][j] = 1

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Convert to dict
    result = {}
    for i, src in enumerate(nodes):
        for j, tgt in enumerate(nodes):
            d = dist[i][j]
            result[(src, tgt)] = -1 if d == INF else int(d)

    return result


def diameter(graph: Graph) -> int:
    """
    Compute the diameter of the graph (longest shortest path).

    Returns -1 if graph is disconnected.

    Maximum of all-pairs shortest paths.
    """
    if graph.num_nodes == 0:
        return 0

    distances = all_pairs_shortest_paths(graph)

    max_dist = 0
    for (src, tgt), d in distances.items():
        if d == -1:
            return -1  # Disconnected
        max_dist = max(max_dist, d)

    return max_dist


def eccentricity(graph: Graph, node: int) -> int:
    """
    Compute the eccentricity of a node (max distance to any other node).

    Returns -1 if some node is unreachable.

    Maximum BFS distance from node.
    """
    if node not in graph.nodes:
        return -1

    max_dist = 0
    for other in graph.nodes:
        d = distance(graph, node, other)
        if d == -1:
            return -1
        max_dist = max(max_dist, d)

    return max_dist


def center(graph: Graph) -> Set[int]:
    """
    Find the center of the graph (nodes with minimum eccentricity).

    Nodes that minimize maximum distance to all other nodes.
    """
    if graph.num_nodes == 0:
        return set()

    eccentricities = {}
    for node in graph.nodes:
        e = eccentricity(graph, node)
        if e == -1:
            return set()  # Disconnected
        eccentricities[node] = e

    min_ecc = min(eccentricities.values())
    return {n for n, e in eccentricities.items() if e == min_ecc}


def has_cycle(graph: Graph) -> bool:
    """
    Check if the graph contains a cycle.

    DFS-based cycle detection.
    """
    if graph.directed:
        # For directed graphs, use color-based DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in graph.nodes}

        def dfs(node: int) -> bool:
            color[node] = GRAY
            for neighbor in graph.neighbors(node):
                if color[neighbor] == GRAY:
                    return True  # Back edge = cycle
                if color[neighbor] == WHITE:
                    if dfs(neighbor):
                        return True
            color[node] = BLACK
            return False

        for node in graph.nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False
    else:
        # For undirected graphs, check if |E| >= |V|
        return graph.num_edges >= graph.num_nodes


def find_cycle(graph: Graph) -> Optional[List[int]]:
    """
    Find a cycle in the graph if one exists.

    Returns the cycle as a list of nodes, or None if acyclic.

    DFS-based cycle finding.
    """
    if not has_cycle(graph):
        return None

    if graph.directed:
        # DFS with parent tracking
        parent = {}
        in_stack = set()

        def dfs(node: int, path: List[int]) -> Optional[List[int]]:
            in_stack.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor in in_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
                if neighbor not in parent:
                    parent[neighbor] = node
                    result = dfs(neighbor, path + [neighbor])
                    if result:
                        return result
            in_stack.remove(node)
            return None

        for start in graph.nodes:
            if start not in parent:
                parent[start] = None
                result = dfs(start, [start])
                if result:
                    return result
    else:
        # For undirected, find any cycle using DFS
        parent = {}

        def dfs(node: int, prev: Optional[int]) -> Optional[List[int]]:
            for neighbor in graph.neighbors(node):
                if neighbor == prev:
                    continue
                if neighbor in parent:
                    # Found cycle - reconstruct
                    cycle = [neighbor]
                    curr = node
                    while curr != neighbor:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.append(neighbor)
                    return cycle
                parent[neighbor] = node
                result = dfs(neighbor, node)
                if result:
                    return result
            return None

        for start in graph.nodes:
            if start not in parent:
                parent[start] = start
                result = dfs(start, None)
                if result:
                    return result

    return None

"""
Core Graph Structures and Encoder for FluxEM-Domains

Provides Graph data structure and GraphEncoder for graph embeddings.
Operations on embeddings follow adjacency-matrix definitions.

Embedding Layout (128 dimensions):
  - dims 0-7:   Domain tag (graph_directed or graph_undirected)
  - dims 8-71:  Adjacency matrix (64 bits for 8x8 matrix)
  - dims 72-79: Graph type encoding
  - dims 80:    Node count
  - dims 81:    Edge count
  - dims 82-89: Property flags (connected, acyclic, bipartite, complete, tree, dag, eulerian, hamiltonian)
  - dims 90-95: Reserved for semantic features
  - dims 96-127: Cross-domain features
"""

from enum import Enum, auto
from typing import Any, Set, Tuple, Optional, List, FrozenSet
from ...backend import get_backend

from ...core.base import DOMAIN_TAGS, EMBEDDING_DIM


class GraphType(Enum):
    """Types of graphs supported."""
    DIRECTED = auto()      # Directed graph (edges have direction)
    UNDIRECTED = auto()    # Undirected graph (edges are bidirectional)
    DAG = auto()           # Directed acyclic graph
    TREE = auto()          # Connected acyclic undirected graph
    COMPLETE = auto()      # Every pair of nodes connected
    BIPARTITE = auto()     # Nodes can be split into two disjoint sets
    CYCLE = auto()         # Single cycle graph
    PATH = auto()          # Single path graph


class Graph:
    """
    Finite graph with up to 8 nodes.

    Nodes are represented as integers 0-7.
    Edges are (source, target) tuples.
    """

    MAX_NODES = 8

    def __init__(
        self,
        nodes: Optional[Set[int]] = None,
        edges: Optional[Set[Tuple[int, int]]] = None,
        directed: bool = False,
    ):
        """
        Create a graph.

        Args:
            nodes: Set of node indices (0-7)
            edges: Set of (source, target) edge tuples
            directed: If True, edges have direction; if False, edges are bidirectional
        """
        self.nodes: Set[int] = set(nodes) if nodes else set()
        self.edges: Set[Tuple[int, int]] = set()
        self.directed = directed

        # Validate nodes
        for n in self.nodes:
            if not (0 <= n < self.MAX_NODES):
                raise ValueError(f"Node {n} out of range [0, {self.MAX_NODES})")

        # Add edges (validating and normalizing)
        if edges:
            for src, tgt in edges:
                self.add_edge(src, tgt)

    def add_node(self, node: int) -> None:
        """Add a node to the graph."""
        if not (0 <= node < self.MAX_NODES):
            raise ValueError(f"Node {node} out of range [0, {self.MAX_NODES})")
        self.nodes.add(node)

    def add_edge(self, src: int, tgt: int) -> None:
        """Add an edge to the graph. Also adds nodes if not present."""
        if not (0 <= src < self.MAX_NODES):
            raise ValueError(f"Source node {src} out of range")
        if not (0 <= tgt < self.MAX_NODES):
            raise ValueError(f"Target node {tgt} out of range")

        # Add nodes
        self.nodes.add(src)
        self.nodes.add(tgt)

        # Add edge
        self.edges.add((src, tgt))

        # For undirected graphs, also add reverse edge
        if not self.directed:
            self.edges.add((tgt, src))

    def remove_edge(self, src: int, tgt: int) -> None:
        """Remove an edge from the graph."""
        self.edges.discard((src, tgt))
        if not self.directed:
            self.edges.discard((tgt, src))

    def remove_node(self, node: int) -> None:
        """Remove a node and all its incident edges."""
        self.nodes.discard(node)
        # Remove all edges involving this node
        self.edges = {(s, t) for s, t in self.edges if s != node and t != node}

    def has_edge(self, src: int, tgt: int) -> bool:
        """Check if edge exists."""
        return (src, tgt) in self.edges

    def neighbors(self, node: int) -> Set[int]:
        """Get all neighbors of a node (outgoing for directed graphs)."""
        return {tgt for src, tgt in self.edges if src == node}

    def predecessors(self, node: int) -> Set[int]:
        """Get all predecessors of a node (incoming edges)."""
        return {src for src, tgt in self.edges if tgt == node}

    def degree(self, node: int) -> int:
        """Get the degree of a node (out-degree for directed graphs)."""
        return len(self.neighbors(node))

    def in_degree(self, node: int) -> int:
        """Get the in-degree of a node."""
        return len(self.predecessors(node))

    def out_degree(self, node: int) -> int:
        """Get the out-degree of a node (same as degree)."""
        return self.degree(node)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges (counting each direction once for undirected)."""
        if self.directed:
            return len(self.edges)
        else:
            # For undirected, each edge is stored twice, but self-loops once
            self_loops = sum(1 for s, t in self.edges if s == t)
            return (len(self.edges) - self_loops) // 2 + self_loops

    def adjacency_matrix(self) -> List[List[int]]:
        """Return the adjacency matrix as a 2D list."""
        matrix = [[0] * self.MAX_NODES for _ in range(self.MAX_NODES)]
        for src, tgt in self.edges:
            matrix[src][tgt] = 1
        return matrix

    def copy(self) -> "Graph":
        """Create a copy of this graph."""
        return Graph(
            nodes=self.nodes.copy(),
            edges=self.edges.copy() if self.directed else {(s, t) for s, t in self.edges if s <= t},
            directed=self.directed,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Graph):
            return False
        return (
            self.nodes == other.nodes
            and self.edges == other.edges
            and self.directed == other.directed
        )

    def __repr__(self) -> str:
        edge_str = ", ".join(f"({s},{t})" for s, t in sorted(self.edges))
        return f"Graph(nodes={sorted(self.nodes)}, edges=[{edge_str}], directed={self.directed})"


class GraphEncoder:
    """
    Encoder for finite graphs.

    Encodes graphs as 128-dimensional embeddings where:
    - Adjacency matrix is stored as a 64-bit bitmap (8x8 matrix)
    - Graph operations map to bitwise operations on the adjacency matrix
    """

    # Embedding layout - absolute positions
    ADJACENCY_START = 8       # dims 8-71: 64-bit adjacency matrix
    ADJACENCY_SIZE = 64
    TYPE_START = 72           # dims 72-79: graph type
    NODE_COUNT_POS = 80
    EDGE_COUNT_POS = 81
    PROPERTY_FLAGS_START = 82 # dims 82-89: property flags

    def __init__(self):
        """Initialize the graph encoder."""
        pass

    def encode(self, graph: Graph) -> Any:
        """
        Encode a graph as a 128-dimensional embedding.

        The adjacency matrix is encoded as a bitmap where position (i*8 + j)
        represents the edge from node i to node j.
        """
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        # Set domain tag
        tag_key = "graph_directed" if graph.directed else "graph_undirected"
        if tag_key in DOMAIN_TAGS:
            embedding = backend.at_add(embedding, slice(0, 8), DOMAIN_TAGS[tag_key])

        # Encode adjacency matrix as bitmap
        for src, tgt in graph.edges:
            bit_pos = src * 8 + tgt
            if bit_pos < self.ADJACENCY_SIZE:
                embedding = backend.at_add(embedding, self.ADJACENCY_START + bit_pos, 1.0)

        # Encode graph type
        type_vec = self._encode_graph_type(graph)
        embedding = backend.at_add(embedding, slice(self.TYPE_START, self.TYPE_START + 8), type_vec)

        # Encode counts
        embedding = backend.at_add(embedding, self.NODE_COUNT_POS, float(graph.num_nodes))
        embedding = backend.at_add(embedding, self.EDGE_COUNT_POS, float(graph.num_edges))

        # Encode property flags
        props = self._compute_properties(graph)
        for i, prop in enumerate(props):
            if prop and i < 8:
                embedding = backend.at_add(embedding, self.PROPERTY_FLAGS_START + i, 1.0)

        return embedding

    def decode(self, embedding: Any) -> Graph:
        """
        Decode an embedding back to a Graph.

        Reconstructs the adjacency matrix from the bitmap.
        """
        backend = get_backend()
        # Determine if directed from domain tag
        tag = embedding[:8]
        directed_tag = DOMAIN_TAGS.get("graph_directed", backend.zeros(8))
        undirected_tag = DOMAIN_TAGS.get("graph_undirected", backend.zeros(8))

        directed_dist = float(backend.sum((tag - directed_tag) ** 2))
        undirected_dist = float(backend.sum((tag - undirected_tag) ** 2))
        directed = directed_dist < undirected_dist

        # Decode adjacency matrix
        nodes = set()
        edges = set()

        for i in range(8):
            for j in range(8):
                bit_pos = i * 8 + j
                if float(embedding[self.ADJACENCY_START + bit_pos]) > 0.5:
                    nodes.add(i)
                    nodes.add(j)
                    edges.add((i, j))

        # For undirected graphs, normalize edges
        if not directed:
            normalized_edges = set()
            for s, t in edges:
                if s <= t:
                    normalized_edges.add((s, t))
                else:
                    normalized_edges.add((t, s))
            edges = normalized_edges

        return Graph(nodes=nodes, edges=edges, directed=directed)

    def _encode_graph_type(self, graph: Graph) -> Any:
        """Encode the graph type as an 8-bit vector."""
        backend = get_backend()
        type_vec = backend.zeros((8,))

        # Bit 0: directed
        if graph.directed:
            type_vec = backend.at_add(type_vec, 0, 1.0)

        # Other type bits can be set based on detected properties
        return type_vec

    def _compute_properties(self, graph: Graph) -> List[bool]:
        """
        Compute graph properties.

        Returns list of 8 boolean flags:
        [connected, acyclic, bipartite, complete, tree, dag, eulerian, hamiltonian]
        """
        props = [False] * 8

        if graph.num_nodes == 0:
            return props

        # Connected
        props[0] = self._is_connected(graph)

        # Acyclic
        props[1] = self._is_acyclic(graph)

        # Bipartite
        props[2] = self._is_bipartite(graph)

        # Complete
        props[3] = self._is_complete(graph)

        # Tree (connected + acyclic + undirected)
        props[4] = props[0] and props[1] and not graph.directed

        # DAG (directed + acyclic)
        props[5] = graph.directed and props[1]

        # Eulerian (all vertices even degree, connected)
        props[6] = self._is_eulerian(graph)

        # Hamiltonian (expensive to compute, skip for now)
        props[7] = False

        return props

    def _is_connected(self, graph: Graph) -> bool:
        """Check if graph is connected using BFS."""
        if graph.num_nodes == 0:
            return True

        start = next(iter(graph.nodes))
        visited = {start}
        queue = [start]

        while queue:
            node = queue.pop(0)
            # For undirected, check both directions
            neighbors = graph.neighbors(node)
            if not graph.directed:
                neighbors = neighbors | graph.predecessors(node)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == graph.num_nodes

    def _is_acyclic(self, graph: Graph) -> bool:
        """Check if graph is acyclic."""
        if graph.directed:
            # Topological sort approach for directed graphs
            in_degree = {n: 0 for n in graph.nodes}
            for src, tgt in graph.edges:
                in_degree[tgt] = in_degree.get(tgt, 0) + 1

            queue = [n for n in graph.nodes if in_degree[n] == 0]
            count = 0

            while queue:
                node = queue.pop(0)
                count += 1
                for neighbor in graph.neighbors(node):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            return count == graph.num_nodes
        else:
            # For undirected: acyclic iff |E| = |V| - 1 and connected
            # Or |E| < |V| (forest)
            return graph.num_edges <= graph.num_nodes - 1

    def _is_bipartite(self, graph: Graph) -> bool:
        """Check if graph is bipartite using 2-coloring."""
        if graph.num_nodes == 0:
            return True

        color = {}

        for start in graph.nodes:
            if start in color:
                continue

            color[start] = 0
            queue = [start]

            while queue:
                node = queue.pop(0)
                current_color = color[node]

                neighbors = graph.neighbors(node)
                if not graph.directed:
                    neighbors = neighbors | graph.predecessors(node)

                for neighbor in neighbors:
                    if neighbor not in color:
                        color[neighbor] = 1 - current_color
                        queue.append(neighbor)
                    elif color[neighbor] == current_color:
                        return False

        return True

    def _is_complete(self, graph: Graph) -> bool:
        """Check if graph is complete (all pairs connected)."""
        n = graph.num_nodes
        if n <= 1:
            return True

        expected_edges = n * (n - 1)
        if not graph.directed:
            expected_edges //= 2

        return graph.num_edges == expected_edges

    def _is_eulerian(self, graph: Graph) -> bool:
        """Check if graph has an Eulerian circuit."""
        if not self._is_connected(graph):
            return False

        if graph.directed:
            # For directed: in-degree = out-degree for all vertices
            for node in graph.nodes:
                if graph.in_degree(node) != graph.out_degree(node):
                    return False
            return True
        else:
            # For undirected: all vertices have even degree
            for node in graph.nodes:
                if graph.degree(node) % 2 != 0:
                    return False
            return True

    # Graph operations

    def union(self, emb1: Any, emb2: Any) -> Any:
        """
        Graph union: combine edges from both graphs.
        """
        backend = get_backend()
        result = mx.zeros_like(emb1)

        # Union of adjacency matrices (OR operation)
        adj1 = emb1[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]
        adj2 = emb2[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]
        union_adj = backend.maximum(adj1, adj2)

        result = backend.at_add(result, slice(self.ADJACENCY_START, self.ADJACENCY_START + self.ADJACENCY_SIZE), union_adj)

        # Keep domain tag from first graph
        result = backend.at_add(result, slice(0, 8), emb1[:8])

        # Recompute counts from union
        edge_count = float(backend.sum(union_adj > 0.5))
        node_count = self._count_nodes_from_adjacency(union_adj)
        result = backend.at_add(result, self.NODE_COUNT_POS, float(node_count))
        result = backend.at_add(result, self.EDGE_COUNT_POS, edge_count)

        return result

    def intersection(self, emb1: Any, emb2: Any) -> Any:
        """
        Graph intersection: edges present in both graphs.
        """
        backend = get_backend()
        result = mx.zeros_like(emb1)

        # Intersection of adjacency matrices (AND operation)
        adj1 = emb1[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]
        adj2 = emb2[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]
        inter_adj = backend.minimum(adj1, adj2)

        result = backend.at_add(result, slice(self.ADJACENCY_START, self.ADJACENCY_START + self.ADJACENCY_SIZE), inter_adj)

        # Keep domain tag from first graph
        result = backend.at_add(result, slice(0, 8), emb1[:8])

        # Recompute counts
        edge_count = float(backend.sum(inter_adj > 0.5))
        node_count = self._count_nodes_from_adjacency(inter_adj)
        result = backend.at_add(result, self.NODE_COUNT_POS, float(node_count))
        result = backend.at_add(result, self.EDGE_COUNT_POS, edge_count)

        return result

    def complement(self, embedding: Any) -> Any:
        """
        Graph complement: invert all edges.
        """
        backend = get_backend()
        result = mx.zeros_like(embedding)

        # Get adjacency and invert (1 - x for binary)
        adj = embedding[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]

        # Create mask for valid node pairs (based on current nodes)
        node_count = int(float(embedding[self.NODE_COUNT_POS]) + 0.5)
        mask = backend.zeros((self.ADJACENCY_SIZE,))
        for i in range(node_count):
            for j in range(node_count):
                if i != j:  # No self-loops in complement
                    mask = backend.at_add(mask, i * 8 + j, 1.0)

        # Complement within the node set
        comp_adj = mask * (1.0 - adj)

        result = backend.at_add(result, slice(self.ADJACENCY_START, self.ADJACENCY_START + self.ADJACENCY_SIZE), comp_adj)
        result = backend.at_add(result, slice(0, 8), embedding[:8])
        result = backend.at_add(result, self.NODE_COUNT_POS, float(node_count))

        edge_count = float(backend.sum(comp_adj > 0.5))
        result = backend.at_add(result, self.EDGE_COUNT_POS, edge_count)

        return result

    def subgraph(self, embedding: Any, nodes: Set[int]) -> Any:
        """
        Extract induced subgraph on given nodes.
        """
        backend = get_backend()
        result = mx.zeros_like(embedding)

        adj = embedding[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]

        # Create mask for edges within the node subset
        mask = backend.zeros((self.ADJACENCY_SIZE,))
        for i in nodes:
            for j in nodes:
                mask = backend.at_add(mask, i * 8 + j, 1.0)

        sub_adj = adj * mask

        result = backend.at_add(result, slice(self.ADJACENCY_START, self.ADJACENCY_START + self.ADJACENCY_SIZE), sub_adj)
        result = backend.at_add(result, slice(0, 8), embedding[:8])
        result = backend.at_add(result, self.NODE_COUNT_POS, float(len(nodes)))

        edge_count = float(backend.sum(sub_adj > 0.5))
        result = backend.at_add(result, self.EDGE_COUNT_POS, edge_count)

        return result

    def transpose(self, embedding: Any) -> Any:
        """
        Transpose graph (reverse all edge directions).
        """
        backend = get_backend()
        result = mx.zeros_like(embedding)

        adj = embedding[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]

        # Transpose: swap position (i,j) with (j,i)
        trans_adj = mx.zeros_like(adj)
        for i in range(8):
            for j in range(8):
                old_pos = i * 8 + j
                new_pos = j * 8 + i
                trans_adj = backend.at_add(trans_adj, new_pos, adj[old_pos])

        result = backend.at_add(result, slice(self.ADJACENCY_START, self.ADJACENCY_START + self.ADJACENCY_SIZE), trans_adj)
        result = backend.at_add(result, slice(0, 8), embedding[:8])
        result = backend.at_add(result, self.NODE_COUNT_POS, embedding[self.NODE_COUNT_POS])
        result = backend.at_add(result, self.EDGE_COUNT_POS, embedding[self.EDGE_COUNT_POS])

        return result

    def _count_nodes_from_adjacency(self, adj: Any) -> int:
        """Count unique nodes from adjacency matrix."""
        nodes = set()
        for i in range(8):
            for j in range(8):
                if float(adj[i * 8 + j]) > 0.5:
                    nodes.add(i)
                    nodes.add(j)
        return len(nodes)

    # Predicates

    def has_edge(self, embedding: Any, src: int, tgt: int) -> bool:
        """Check if edge exists in the embedded graph."""
        if not (0 <= src < 8 and 0 <= tgt < 8):
            return False
        bit_pos = src * 8 + tgt
        return float(embedding[self.ADJACENCY_START + bit_pos]) > 0.5

    def get_node_count(self, embedding: Any) -> int:
        """Get the number of nodes."""
        return int(float(embedding[self.NODE_COUNT_POS]) + 0.5)

    def get_edge_count(self, embedding: Any) -> int:
        """Get the number of edges."""
        return int(float(embedding[self.EDGE_COUNT_POS]) + 0.5)

    def is_connected(self, embedding: Any) -> bool:
        """Check if the graph is connected."""
        return float(embedding[self.PROPERTY_FLAGS_START + 0]) > 0.5

    def is_acyclic(self, embedding: Any) -> bool:
        """Check if the graph is acyclic."""
        return float(embedding[self.PROPERTY_FLAGS_START + 1]) > 0.5

    def is_bipartite(self, embedding: Any) -> bool:
        """Check if the graph is bipartite."""
        return float(embedding[self.PROPERTY_FLAGS_START + 2]) > 0.5

    def is_complete(self, embedding: Any) -> bool:
        """Check if the graph is complete."""
        return float(embedding[self.PROPERTY_FLAGS_START + 3]) > 0.5

    def is_tree(self, embedding: Any) -> bool:
        """Check if the graph is a tree."""
        return float(embedding[self.PROPERTY_FLAGS_START + 4]) > 0.5

    def is_dag(self, embedding: Any) -> bool:
        """Check if the graph is a DAG."""
        return float(embedding[self.PROPERTY_FLAGS_START + 5]) > 0.5

    def equals(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if two graph embeddings represent the same graph.
        """
        backend = get_backend()
        # Check node and edge counts first
        if self.get_node_count(emb1) != self.get_node_count(emb2):
            return False
        if self.get_edge_count(emb1) != self.get_edge_count(emb2):
            return False

        adj1 = emb1[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]
        adj2 = emb2[self.ADJACENCY_START:self.ADJACENCY_START + self.ADJACENCY_SIZE]

        # Binary comparison
        diff = backend.abs(adj1 - adj2)
        return float(mx.max(diff)) < 0.5

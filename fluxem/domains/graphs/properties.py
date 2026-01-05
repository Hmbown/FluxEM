"""
Graph Property Detection for FluxEM-Domains

Provides graph property detection using standard algorithms.
"""

from typing import Set, List, Tuple, Optional
from .graphs import Graph


def is_connected(graph: Graph) -> bool:
    """
    Check if the graph is connected.

    For directed graphs, checks weak connectivity (ignoring edge direction).
    For undirected graphs, checks standard connectivity.

    BFS reachability from any node.
    """
    if graph.num_nodes == 0:
        return True

    start = next(iter(graph.nodes))
    visited = {start}
    queue = [start]

    while queue:
        node = queue.pop(0)
        # Check both directions for connectivity
        neighbors = graph.neighbors(node) | graph.predecessors(node)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == graph.num_nodes


def is_strongly_connected(graph: Graph) -> bool:
    """
    Check if directed graph is strongly connected.

    Every node is reachable from every other node following edge directions.

    BFS from each node.
    """
    if not graph.directed:
        return is_connected(graph)

    if graph.num_nodes == 0:
        return True

    # Check reachability from first node
    start = next(iter(graph.nodes))

    # Forward reachability
    forward = {start}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph.neighbors(node):
            if neighbor not in forward:
                forward.add(neighbor)
                queue.append(neighbor)

    if len(forward) != graph.num_nodes:
        return False

    # Backward reachability (using reverse edges)
    backward = {start}
    queue = [start]
    while queue:
        node = queue.pop(0)
        for neighbor in graph.predecessors(node):
            if neighbor not in backward:
                backward.add(neighbor)
                queue.append(neighbor)

    return len(backward) == graph.num_nodes


def connected_components(graph: Graph) -> List[Set[int]]:
    """
    Find all connected components of the graph.

    For directed graphs, finds weakly connected components.

    BFS from each unvisited node.
    """
    visited = set()
    components = []

    for start in graph.nodes:
        if start in visited:
            continue

        component = {start}
        queue = [start]

        while queue:
            node = queue.pop(0)
            neighbors = graph.neighbors(node) | graph.predecessors(node)
            for neighbor in neighbors:
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)

        visited.update(component)
        components.append(component)

    return components


def strongly_connected_components(graph: Graph) -> List[Set[int]]:
    """
    Find all strongly connected components of a directed graph.

    Uses Kosaraju's algorithm.

    Two-pass DFS algorithm.
    """
    if not graph.directed:
        return connected_components(graph)

    # First pass: get finish order
    visited = set()
    finish_order = []

    def dfs1(node: int):
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs1(neighbor)
        finish_order.append(node)

    for node in graph.nodes:
        if node not in visited:
            dfs1(node)

    # Second pass: DFS on reverse graph in reverse finish order
    visited.clear()
    components = []

    def dfs2(node: int, component: Set[int]):
        visited.add(node)
        component.add(node)
        for neighbor in graph.predecessors(node):
            if neighbor not in visited:
                dfs2(neighbor, component)

    for node in reversed(finish_order):
        if node not in visited:
            component = set()
            dfs2(node, component)
            components.append(component)

    return components


def is_acyclic(graph: Graph) -> bool:
    """
    Check if the graph is acyclic.

    Topological sort for directed, edge count for undirected.
    """
    if graph.directed:
        # Kahn's algorithm for cycle detection
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
        # Undirected acyclic: |E| <= |V| - 1
        return graph.num_edges <= graph.num_nodes - 1


def has_cycle(graph: Graph) -> bool:
    """
    Check if the graph contains a cycle.

    Negation of is_acyclic.
    """
    return not is_acyclic(graph)


def is_dag(graph: Graph) -> bool:
    """
    Check if the graph is a Directed Acyclic Graph.

    Directed and acyclic.
    """
    return graph.directed and is_acyclic(graph)


def is_tree(graph: Graph) -> bool:
    """
    Check if the graph is a tree.

    A tree is connected, acyclic, and undirected with |E| = |V| - 1.

    Structural verification.
    """
    if graph.directed:
        return False

    if graph.num_nodes == 0:
        return True

    return (
        is_connected(graph)
        and graph.num_edges == graph.num_nodes - 1
    )


def is_forest(graph: Graph) -> bool:
    """
    Check if the graph is a forest (collection of trees).

    Undirected and acyclic.
    """
    if graph.directed:
        return False

    return is_acyclic(graph)


def is_bipartite(graph: Graph) -> Tuple[bool, Optional[Tuple[Set[int], Set[int]]]]:
    """
    Check if the graph is bipartite and return the partition if so.

    Returns (is_bipartite, (set1, set2)) or (False, None).

    2-coloring algorithm.
    """
    if graph.num_nodes == 0:
        return True, (set(), set())

    color = {}
    set1, set2 = set(), set()

    for start in graph.nodes:
        if start in color:
            continue

        color[start] = 0
        set1.add(start)
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
                    if color[neighbor] == 0:
                        set1.add(neighbor)
                    else:
                        set2.add(neighbor)
                    queue.append(neighbor)
                elif color[neighbor] == current_color:
                    return False, None

    return True, (set1, set2)


def is_complete(graph: Graph) -> bool:
    """
    Check if the graph is complete (all pairs connected).

    Edge count verification.
    """
    n = graph.num_nodes
    if n <= 1:
        return True

    expected = n * (n - 1)
    if not graph.directed:
        expected //= 2

    return graph.num_edges == expected


def is_regular(graph: Graph, k: Optional[int] = None) -> Tuple[bool, Optional[int]]:
    """
    Check if the graph is regular (all nodes have same degree).

    Returns (is_regular, degree) or (False, None).

    Degree sequence verification.
    """
    if graph.num_nodes == 0:
        return True, None

    degrees = [graph.degree(n) for n in graph.nodes]
    first_degree = degrees[0]

    if all(d == first_degree for d in degrees):
        if k is None or k == first_degree:
            return True, first_degree

    return False, None


def is_planar(graph: Graph) -> bool:
    """
    Check if the graph is planar using Kuratowski's theorem approximation.

    For small graphs (≤ 8 nodes), uses edge count heuristic.
    Euler's formula: |E| <= 3|V| - 6 for planar graphs with |V| >= 3.

    Rejection check: If |E| > 3|V| - 6, definitely not planar.
    Approximate for acceptance: May give false positives.
    """
    n = graph.num_nodes
    e = graph.num_edges

    if n <= 4:
        return True

    # Necessary condition for planarity
    if e > 3 * n - 6:
        return False

    # For small graphs, this is usually sufficient
    return True


def is_eulerian(graph: Graph) -> bool:
    """
    Check if the graph has an Eulerian circuit.

    Degree parity verification.
    """
    if not is_connected(graph):
        return False

    if graph.directed:
        for node in graph.nodes:
            if graph.in_degree(node) != graph.out_degree(node):
                return False
        return True
    else:
        for node in graph.nodes:
            if graph.degree(node) % 2 != 0:
                return False
        return True


def is_semi_eulerian(graph: Graph) -> bool:
    """
    Check if the graph has an Eulerian path (but not circuit).

    Exactly 2 odd-degree vertices (undirected) or
           exactly one source and one sink differ by 1 (directed).
    """
    if not is_connected(graph):
        return False

    if graph.directed:
        sources = 0  # out > in
        sinks = 0    # in > out

        for node in graph.nodes:
            diff = graph.out_degree(node) - graph.in_degree(node)
            if diff == 1:
                sources += 1
            elif diff == -1:
                sinks += 1
            elif diff != 0:
                return False

        return sources == 1 and sinks == 1
    else:
        odd_count = sum(1 for n in graph.nodes if graph.degree(n) % 2 != 0)
        return odd_count == 2


def degree_sequence(graph: Graph) -> List[int]:
    """
    Get the degree sequence of the graph (sorted descending).

    List of all degrees.
    """
    degrees = [graph.degree(n) for n in graph.nodes]
    return sorted(degrees, reverse=True)


def density(graph: Graph) -> float:
    """
    Calculate the density of the graph.

    Density = |E| / (|V| * (|V| - 1)) for directed
    Density = 2|E| / (|V| * (|V| - 1)) for undirected

    Ratio calculation.
    """
    n = graph.num_nodes
    if n <= 1:
        return 0.0

    max_edges = n * (n - 1)
    if not graph.directed:
        max_edges //= 2

    return graph.num_edges / max_edges if max_edges > 0 else 0.0


def topological_sort(graph: Graph) -> Optional[List[int]]:
    """
    Return a topological ordering of nodes if the graph is a DAG.

    Returns None if the graph has a cycle.

    Kahn's algorithm.
    """
    if not graph.directed:
        return None

    in_degree = {n: 0 for n in graph.nodes}
    for src, tgt in graph.edges:
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    queue = [n for n in graph.nodes if in_degree[n] == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in graph.neighbors(node):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != graph.num_nodes:
        return None  # Cycle exists

    return result


def chromatic_number_upper_bound(graph: Graph) -> int:
    """
    Get an upper bound on the chromatic number using greedy coloring.

    Provides valid coloring (upper bound is tight for many graphs).
    """
    if graph.num_nodes == 0:
        return 0

    colors = {}
    max_color = 0

    for node in sorted(graph.nodes):
        neighbor_colors = set()
        neighbors = graph.neighbors(node)
        if not graph.directed:
            neighbors = neighbors | graph.predecessors(node)

        for neighbor in neighbors:
            if neighbor in colors:
                neighbor_colors.add(colors[neighbor])

        # Find smallest available color
        color = 0
        while color in neighbor_colors:
            color += 1

        colors[node] = color
        max_color = max(max_color, color)

    return max_color + 1


def clique_number_lower_bound(graph: Graph) -> int:
    """
    Get a lower bound on the clique number using greedy approach.

    Finds a valid clique.
    """
    if graph.num_nodes == 0:
        return 0

    # Greedy clique: keep adding nodes that are connected to all current clique members
    best_clique = set()

    for start in graph.nodes:
        clique = {start}
        candidates = set(graph.neighbors(start))
        if not graph.directed:
            candidates = candidates | graph.predecessors(start)

        for node in sorted(candidates):
            if all(graph.has_edge(node, c) or graph.has_edge(c, node) for c in clique):
                clique.add(node)

        if len(clique) > len(best_clique):
            best_clique = clique

    return len(best_clique)


def is_subgraph(sub: Graph, main: Graph) -> bool:
    """
    Check if sub is a subgraph of main.

    All nodes and edges of sub exist in main.
    """
    if not sub.nodes.issubset(main.nodes):
        return False

    for edge in sub.edges:
        if edge not in main.edges:
            return False

    return True


def is_isomorphic(g1: Graph, g2: Graph) -> bool:
    """
    Check if two graphs are isomorphic (for small graphs).

    Uses degree sequence as quick filter, then brute-force for small graphs.

    Rejection check: Different invariants means not isomorphic.
    Small-graph check: Complete enumeration.
    """
    if g1.num_nodes != g2.num_nodes:
        return False
    if g1.num_edges != g2.num_edges:
        return False
    if g1.directed != g2.directed:
        return False

    # Check degree sequences
    if degree_sequence(g1) != degree_sequence(g2):
        return False

    # For very small graphs, try all permutations
    if g1.num_nodes <= 8:
        import itertools

        nodes1 = sorted(g1.nodes)
        nodes2 = sorted(g2.nodes)

        for perm in itertools.permutations(nodes2):
            mapping = dict(zip(nodes1, perm))
            # Check if this mapping preserves edges
            match = True
            for s, t in g1.edges:
                if (mapping[s], mapping[t]) not in g2.edges:
                    match = False
                    break
            if match:
                return True

        return False

    # For larger graphs, we can only say "maybe"
    return True  # Degree sequence match, likely isomorphic

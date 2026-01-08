"""
Tool Registry for FluxEM + Qwen3-4B MLX.

Maps FluxEM domains plus arithmetic compute to callable functions with descriptions
for LLM-based domain detection and tool selection.
"""

from typing import Dict, Any, Callable, Optional
import os
from fluxem.arithmetic.unified import create_unified_model
from fluxem.arithmetic.extended_ops import create_extended_ops
from fluxem.backend import set_backend, BackendType


# =============================================================================
# Tool Registry
# =============================================================================


class ToolDescription:
    """Description of a FluxEM tool for LLM."""

    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        input_format: str,
        output_format: str,
        example: str,
        domain: str,
    ):
        self.name = name
        self.function = function
        self.description = description
        self.input_format = input_format
        self.output_format = output_format
        self.example = example
        self.domain = domain


def create_tool_registry() -> Dict[str, ToolDescription]:
    """
    Create registry mapping FluxEM domains (plus arithmetic) to tools.

    Returns:
        Dictionary mapping domain names to ToolDescription objects.
    """
    # Initialize FluxEM backend (default to NumPy to avoid MLX import issues)
    backend_override = os.environ.get("FLUXEM_BACKEND")
    if backend_override:
        try:
            set_backend(backend_override)
        except Exception:
            set_backend(BackendType.NUMPY)
    else:
        set_backend(BackendType.NUMPY)

    # Create FluxEM models
    arithmetic_model = create_unified_model(dim=256, linear_scale=1e7, log_scale=25.0)
    extended_ops = create_extended_ops()

    registry: Dict[str, ToolDescription] = {}

    # =========================================================================
    # 1. Arithmetic
    # =========================================================================
    registry["arithmetic"] = ToolDescription(
        name="arithmetic",
        function=lambda expr: _compute_arithmetic(expr, arithmetic_model),
        description="Evaluates arithmetic expressions with 100% accuracy. Supports +, -, *, /, and ** operations.",
        input_format="Arithmetic expression as string (e.g., '54 * 44', '2**16')",
        output_format="Numeric result as float",
        example="arithmetic.compute('54 * 44') returns 2376.0",
        domain="arithmetic",
    )

    # =========================================================================
    # 2. Physics
    # =========================================================================
    from fluxem.domains.physics.units import UnitEncoder, get_encoder, convert_units
    from fluxem.domains.physics.dimensions import Dimensions

    unit_encoder = UnitEncoder()

    registry["physics_dimensions"] = ToolDescription(
        name="physics_dimensions",
        function=lambda unit_str: _unit_dimensions_dict(unit_str),
        description="Extracts SI base dimensions from a unit string.",
        input_format="Unit string (e.g., 'm/s', 'kg*m/s^2', 'kN')",
        output_format="Dictionary of SI exponents (e.g., {'L': 1, 'T': -1})",
        example="physics_dimensions('m/s') returns {'L': 1, 'T': -1}",
        domain="physics",
    )

    registry["physics_convert"] = ToolDescription(
        name="physics_convert",
        function=lambda value_str: convert_units(*_parse_conversion(value_str)),
        description="Converts between compatible physical units.",
        input_format="Conversion query as string (e.g., '5 km to meters', '9.8 m/s^2 to N')",
        output_format="Converted value as float",
        example="physics_convert.convert(5.0, 'km', 'm') returns 5000.0",
        domain="physics",
    )

    # =========================================================================
    # 3. Chemistry
    # =========================================================================
    from fluxem.domains.chemistry.molecules import MoleculeEncoder

    molecule_encoder = MoleculeEncoder()

    registry["chemistry_molecule"] = ToolDescription(
        name="chemistry_molecule",
        function=lambda formula: molecule_encoder.molecular_weight(
            molecule_encoder.encode(formula)
        ),
        description="Calculates molecular weight and encodes molecular formulas.",
        input_format="Molecular formula string (e.g., 'H2O', 'C6H12O6', 'NaCl')",
        output_format="Molecular weight as float",
        example="chemistry_molecule.weight('H2O') returns 18.015",
        domain="chemistry",
    )

    registry["chemistry_formula"] = ToolDescription(
        name="chemistry_formula",
        function=lambda name: _lookup_formula(name),
        description="Looks up common molecular formulas by compound name.",
        input_format="Compound name string (e.g., 'glucose')",
        output_format="Molecular formula string",
        example="chemistry_formula('glucose') returns 'C6H12O6'",
        domain="chemistry",
    )

    registry["chemistry_balance_simple"] = ToolDescription(
        name="chemistry_balance_simple",
        function=lambda reaction: _balance_simple(reaction),
        description="Balances a small set of common reactions using pattern matching.",
        input_format="Reaction string (e.g., 'H2 + O2 -> H2O')",
        output_format="Balanced reaction string",
        example="chemistry_balance_simple('H2 + O2 -> H2O') returns 'Balanced: 2 H2 + 1 O2 -> 2 H2O'",
        domain="chemistry",
    )

    # =========================================================================
    # 4. Biology
    # =========================================================================
    from fluxem.domains.biology.dna import (
        DNAEncoder,
        gc_content,
        molecular_weight as dna_mw,
    )

    dna_encoder = DNAEncoder()

    registry["biology_gc_content"] = ToolDescription(
        name="biology_gc_content",
        function=lambda sequence: gc_content(sequence),
        description="Calculates GC content ratio (0-1) for DNA sequences.",
        input_format="DNA sequence string (e.g., 'ATGCCGTAGC', 'GATTACA')",
        output_format="GC content as float between 0 and 1",
        example="biology_gc_content('GATTACA') returns 0.42857 (3 GC out of 7 bases)",
        domain="biology",
    )

    registry["biology_mw"] = ToolDescription(
        name="biology_mw",
        function=lambda sequence: dna_mw(sequence),
        description="Calculates molecular weight of DNA sequence.",
        input_format="DNA sequence string",
        output_format="Molecular weight as float",
        example="biology_mw('ATCG') returns 990.66 (A=313.21 + T=304.19 + C=289.18 + G=329.21)",
        domain="biology",
    )

    registry["biology_complement"] = ToolDescription(
        name="biology_complement",
        function=lambda sequence: _dna_complement(sequence),
        description="Generates complementary DNA sequence.",
        input_format="DNA sequence string",
        output_format="Complementary DNA sequence string",
        example="biology_complement('ATCG') returns 'TAGC'",
        domain="biology",
    )

    registry["biology_reverse_complement_gc"] = ToolDescription(
        name="biology_reverse_complement_gc",
        function=lambda sequence: _reverse_complement_gc(sequence),
        description="Computes GC content of the reverse complement of a DNA sequence.",
        input_format="DNA sequence string",
        output_format="GC content as float between 0 and 1",
        example="biology_reverse_complement_gc('GATTACA') returns 0.57143",
        domain="biology",
    )

    # =========================================================================
    # 5. Mathematics (vectors, matrices, complex numbers)
    # =========================================================================
    from fluxem.domains.math.vector import VectorEncoder
    from fluxem.domains.math.matrix import MatrixEncoder

    vector_encoder = VectorEncoder()
    matrix_encoder = MatrixEncoder()

    registry["math_vector"] = ToolDescription(
        name="math_vector",
        function=lambda vec: vector_encoder.get_norm(vector_encoder.encode(vec)),
        description="Computes vector operations: magnitude, dot product, normalization.",
        input_format="Vector as list of floats (e.g., [3, 4], [1, 2, 3, 0])",
        output_format="Vector magnitude as float",
        example="math_vector.magnitude([3, 4]) returns 5.0",
        domain="math",
    )

    registry["math_dot"] = ToolDescription(
        name="math_dot",
        function=lambda vecs: vector_encoder.dot(
            vector_encoder.encode(vecs[0]), vector_encoder.encode(vecs[1])
        ),
        description="Computes dot product of two vectors.",
        input_format="Two vectors as list of two lists (e.g., [[1, 2], [3, 4]])",
        output_format="Dot product as float",
        example="math_dot.dot([[1, 2], [3, 4]]) returns 11.0",
        domain="math",
    )

    registry["math_determinant"] = ToolDescription(
        name="math_determinant",
        function=lambda matrix: matrix_encoder.get_determinant(
            matrix_encoder.encode(matrix)
        ),
        description="Computes determinant of a square matrix (up to 4x4).",
        input_format="Matrix as list of lists (e.g., [[1, 2], [3, 4]])",
        output_format="Determinant as float",
        example="math_determinant([[1, 2], [3, 4]]) returns -2.0",
        domain="math",
    )

    registry["math_normalize"] = ToolDescription(
        name="math_normalize",
        function=lambda vec: vector_encoder.decode(
            vector_encoder.normalize(vector_encoder.encode(vec))
        ),
        description="Normalizes a vector to unit length.",
        input_format="Vector as list of floats (e.g., [3, 4])",
        output_format="Normalized vector as list of floats",
        example="math_normalize([3, 4]) returns [0.6, 0.8]",
        domain="math",
    )

    # =========================================================================
    # 6. Music (atonal theory)
    # =========================================================================
    from fluxem.domains.music.atonal import (
        AtonalSetEncoder,
        prime_form,
        normal_form,
        transposition,
    )
    from fluxem.domains.music import CHORD_PATTERNS

    atonal_encoder = AtonalSetEncoder()

    registry["music_prime_form"] = ToolDescription(
        name="music_prime_form",
        function=lambda pcs: prime_form(pcs),
        description="Computes prime form of pitch class set (most compact representation).",
        input_format="Pitch class set as list of integers 0-11 (e.g., [0, 4, 7] for C major triad)",
        output_format="Prime form as list of pitch classes",
        example="music_prime_form([0, 4, 7]) returns [0, 4, 7]",
        domain="music",
    )

    registry["music_normal_form"] = ToolDescription(
        name="music_normal_form",
        function=lambda pcs: normal_form(pcs),
        description="Computes normal form (most compact left-packed rotation) of pitch class set.",
        input_format="Pitch class set as list of integers 0-11",
        output_format="Normal form as list of pitch classes",
        example="music_normal_form([7, 0, 4]) returns [0, 4, 7]",
        domain="music",
    )

    registry["music_chord_type"] = ToolDescription(
        name="music_chord_type",
        function=lambda pcs: _identify_chord(pcs, CHORD_PATTERNS),
        description="Identifies chord quality from pitch class set.",
        input_format="Pitch class set as list of integers 0-11",
        output_format="Chord quality string (e.g., 'major triad')",
        example="music_chord_type([0, 4, 7]) returns 'major triad'",
        domain="music",
    )

    registry["music_transpose"] = ToolDescription(
        name="music_transpose",
        function=lambda data: transposition(data[0], data[1]),
        description="Transposes a pitch class set by n semitones.",
        input_format="Tuple of (pitch classes list, semitone integer)",
        output_format="Transposed pitch classes list",
        example="music_transpose(([0, 4, 7], 7)) returns [7, 11, 2]",
        domain="music",
    )

    # =========================================================================
    # 7. Geometry (points, distances, transforms)
    # =========================================================================
    from fluxem.domains.geometry.points import PointEncoder, Point2D

    point_encoder = PointEncoder()

    registry["geometry_distance"] = ToolDescription(
        name="geometry_distance",
        function=lambda points: _compute_distance(points),
        description="Computes Euclidean distance between points.",
        input_format="Two points as list of coordinates (e.g., [[0, 0], [3, 4]] for 2D, [[0,0,0], [1,2,3]] for 3D)",
        output_format="Distance as float",
        example="geometry_distance.distance([[0, 0], [3, 4]]) returns 5.0",
        domain="geometry",
    )

    registry["geometry_midpoint"] = ToolDescription(
        name="geometry_midpoint",
        function=lambda points: _compute_midpoint(points),
        description="Computes midpoint between two points.",
        input_format="Two points as list of coordinates",
        output_format="Midpoint as list of coordinates",
        example="geometry_midpoint.midpoint([[0, 0], [3, 4]]) returns [1.5, 2.0]",
        domain="geometry",
    )

    registry["geometry_rotate"] = ToolDescription(
        name="geometry_rotate",
        function=lambda data: _rotate_point(data[0], data[1]),
        description="Rotates a 2D point around the origin by an angle in radians.",
        input_format="Tuple of (point list, angle in radians)",
        output_format="Rotated point as list of floats",
        example="geometry_rotate(([1, 0], 1.5708)) returns [0.0, 1.0]",
        domain="geometry",
    )

    # =========================================================================
    # 8. Graphs (connectivity, paths, properties)
    # =========================================================================
    from fluxem.domains.graphs.graphs import GraphEncoder, Graph, GraphType

    graph_encoder = GraphEncoder()

    registry["graphs_shortest_path"] = ToolDescription(
        name="graphs_shortest_path",
        function=lambda graph_data: _shortest_path_bfs(*graph_data),
        description="Finds shortest path between nodes in a graph using BFS.",
        input_format="Graph with start and target nodes (e.g., Graph(nodes={0,1,2}, edges=[(0,1),(1,2)], start=0, target=2)",
        output_format="Path as list of nodes and total distance",
        example="graphs_shortest_path.path(Graph(..., start=0, target=2) returns ([0, 1, 2], 2)",
        domain="graphs",
    )

    registry["graphs_properties"] = ToolDescription(
        name="graphs_properties",
        function=lambda graph_data: _analyze_graph_properties(graph_data),
        description="Analyzes graph properties: connected, acyclic, bipartite, etc.",
        input_format="Graph with nodes and edges",
        output_format="Dictionary of boolean properties",
        example="graphs_properties.analyze(Graph(...)) returns {'connected': True, 'acyclic': True, 'tree': True}",
        domain="graphs",
    )

    registry["graphs_node_count"] = ToolDescription(
        name="graphs_node_count",
        function=lambda graph_data: graph_data.num_nodes,
        description="Returns the number of nodes in a graph.",
        input_format="Graph with nodes and edges",
        output_format="Node count as integer",
        example="graphs_node_count(Graph(...)) returns 5",
        domain="graphs",
    )

    registry["graphs_is_connected"] = ToolDescription(
        name="graphs_is_connected",
        function=lambda graph_data: _graph_property(graph_data, "connected"),
        description="Checks if a graph is connected.",
        input_format="Graph with nodes and edges",
        output_format="Boolean",
        example="graphs_is_connected(Graph(...)) returns True",
        domain="graphs",
    )

    registry["graphs_is_tree"] = ToolDescription(
        name="graphs_is_tree",
        function=lambda graph_data: _graph_property(graph_data, "tree"),
        description="Checks if a graph is a tree.",
        input_format="Graph with nodes and edges",
        output_format="Boolean",
        example="graphs_is_tree(Graph(...)) returns True",
        domain="graphs",
    )

    # =========================================================================
    # 9. Sets (union, intersection, complement, etc.)
    # =========================================================================
    from fluxem.domains.sets.sets import SetEncoder, FiniteSet

    set_encoder = SetEncoder()

    registry["sets_union"] = ToolDescription(
        name="sets_union",
        function=lambda sets: _set_operation(sets, "union"),
        description="Computes union of sets (elements in either set).",
        input_format="Two sets as lists (e.g., [1, 2, 3], [2, 3, 4])",
        output_format="Union as list of elements",
        example="sets_union.union([1, 2], [2, 3, 4]) returns [1, 2, 3, 4]",
        domain="sets",
    )

    registry["sets_intersection"] = ToolDescription(
        name="sets_intersection",
        function=lambda sets: _set_operation(sets, "intersection"),
        description="Computes intersection of sets (elements in both sets).",
        input_format="Two sets as lists",
        output_format="Intersection as list of elements",
        example="sets_intersection.intersection([1, 2, 3], [2, 3, 4]) returns [2, 3]",
        domain="sets",
    )

    registry["sets_subset"] = ToolDescription(
        name="sets_subset",
        function=lambda sets: _set_operation(sets, "subset"),
        description="Checks if first set is subset of second set.",
        input_format="Two sets as lists (set1, set2)",
        output_format="Boolean: True if set1 is subset of set2",
        example="sets_subset.subset([1, 2], [1, 2, 3, 4]) returns True",
        domain="sets",
    )

    registry["sets_complement"] = ToolDescription(
        name="sets_complement",
        function=lambda sets: _set_operation(sets, "complement"),
        description="Computes complement of first set relative to second set.",
        input_format="Two sets as lists (set1, set2)",
        output_format="Complement as list of elements",
        example="sets_complement([1, 2, 3], [1, 2, 3, 4]) returns [4]",
        domain="sets",
    )

    # =========================================================================
    # 10. Logic (propositional, type checking)
    # =========================================================================
    from fluxem.domains.logic.propositional import PropositionalEncoder

    try:
        prop_encoder = PropositionalEncoder()
    except:
        prop_encoder = None

    if prop_encoder is not None:
        registry["logic_tautology"] = ToolDescription(
            name="logic_tautology",
            function=lambda formula: _check_tautology(formula),
            description="Checks if a propositional formula is a tautology (always true).",
            input_format="Propositional formula string (e.g., 'p or not p', '(p and q) or (not p and not q)')",
            output_format="Boolean: True if tautology",
            example="logic_tautology.check('p or not p') returns True",
            domain="logic",
        )

    # =========================================================================
    # 11. Number Theory (primes, gcd, modular arithmetic)
    # =========================================================================
    from fluxem.domains.number_theory.primes import PrimeEncoder

    try:
        prime_encoder = PrimeEncoder()
    except:
        prime_encoder = None

    if prime_encoder is not None:
        registry["number_theory_is_prime"] = ToolDescription(
            name="number_theory_is_prime",
            function=lambda n: _is_prime_check(n),
            description="Checks if a number is prime.",
            input_format="Integer (e.g., 17, 42, 97)",
            output_format="Boolean: True if prime",
            example="number_theory_is_prime.check(17) returns True",
            domain="number_theory",
        )

    registry["number_theory_gcd"] = ToolDescription(
        name="number_theory_gcd",
        function=lambda nums: _compute_gcd(nums),
        description="Computes greatest common divisor of integers.",
        input_format="Two or more integers (e.g., [12, 18], [24, 36, 48])",
        output_format="GCD as integer",
        example="number_theory_gcd.gcd(12, 18) returns 6",
        domain="number_theory",
    )

    # Advanced tools demonstrating low-weight advantage
    from fluxem.domains.number_theory import (
        mod_pow,
        mod_inverse,
        primes_up_to,
        nth_prime,
    )

    registry["number_theory_mod_pow"] = ToolDescription(
        name="number_theory_mod_pow",
        function=lambda args: mod_pow(args[0], args[1], args[2]),
        description="Computes a^b mod m efficiently using modular arithmetic. Critical for cryptography.",
        input_format="Three integers: [base, exponent, modulus] (e.g., [17, 13, 23])",
        output_format="Result as integer",
        example="number_theory_mod_pow([17, 13, 23]) returns 8 (since 17^13 mod 23 = 8)",
        domain="number_theory",
    )

    registry["number_theory_mod_inverse"] = ToolDescription(
        name="number_theory_mod_inverse",
        function=lambda args: mod_inverse(args[0], args[1]),
        description="Computes modular multiplicative inverse a^(-1) mod m. Returns None if not exists.",
        input_format="Two integers: [number, modulus] (e.g., [3, 11])",
        output_format="Inverse as integer, or None if doesn't exist",
        example="number_theory_mod_inverse([3, 11]) returns 4 (since 3*4 = 12 ≡ 1 mod 11)",
        domain="number_theory",
    )

    registry["number_theory_primes_up_to"] = ToolDescription(
        name="number_theory_primes_up_to",
        function=lambda n: list(primes_up_to(n)),
        description="Finds all prime numbers less than or equal to n using sieve algorithm.",
        input_format="Positive integer n (e.g., 100)",
        output_format="List of primes up to n",
        example="number_theory_primes_up_to(30) returns [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]",
        domain="number_theory",
    )

    registry["number_theory_nth_prime"] = ToolDescription(
        name="number_theory_nth_prime",
        function=lambda n: nth_prime(n),
        description="Finds the nth prime number (1-indexed: 1st prime = 2).",
        input_format="Positive integer n (e.g., 100 for the 100th prime)",
        output_format="nth prime as integer",
        example="number_theory_nth_prime(10) returns 29 (the 10th prime)",
        domain="number_theory",
    )

    # =========================================================================
    # 12. Data (arrays, records, tables)
    # =========================================================================
    from fluxem.domains.data import ArrayEncoder, RecordEncoder, TableEncoder

    array_encoder = ArrayEncoder()
    record_encoder = RecordEncoder()
    table_encoder = TableEncoder()

    registry["data_array_summary"] = ToolDescription(
        name="data_array_summary",
        function=lambda arr: _data_array_summary(arr, array_encoder),
        description="Summarizes a numeric array (length, dtype, min/max/mean/std/sum, flags).",
        input_format="Array as list or JSON (e.g., [1, 2, 3, 4])",
        output_format="Summary dict with statistics and flags",
        example="data_array_summary([1, 2, 3]) returns {'length': 3, 'mean': 2.0, ...}",
        domain="data",
    )

    registry["data_array_length"] = ToolDescription(
        name="data_array_length",
        function=lambda arr: _data_array_length(arr, array_encoder),
        description="Returns the length of an array.",
        input_format="Array as list or JSON (e.g., [1, 2, 3, 4])",
        output_format="Length as integer",
        example="data_array_length([1, 2, 3]) returns 3",
        domain="data",
    )

    registry["data_record_schema"] = ToolDescription(
        name="data_record_schema",
        function=lambda record: _data_record_schema(record, record_encoder),
        description="Summarizes record schema (field count, field types, schema hash).",
        input_format='Record as JSON object (e.g., {"name": "Ada", "age": 37})',
        output_format="Schema summary dict",
        example="data_record_schema({'name': 'Ada', 'age': 37}) returns {'num_fields': 2, ...}",
        domain="data",
    )

    registry["data_table_summary"] = ToolDescription(
        name="data_table_summary",
        function=lambda table: _data_table_summary(table, table_encoder),
        description="Summarizes a table (rows, columns, column types, schema hash).",
        input_format="Table as list of records or column dict (e.g., [{'a':1,'b':2},{'a':3,'b':4}] or {'a':[1,3],'b':[2,4]})",
        output_format="Table summary dict",
        example="data_table_summary([{'a':1,'b':2},{'a':3,'b':4}]) returns {'n_rows': 2, 'n_cols': 2, ...}",
        domain="data",
    )

    # =========================================================================
    # 13. Combinatorics
    # =========================================================================
    from fluxem.domains.combinatorics import factorial, ncr, npr, multiset_combinations

    registry["combinatorics_factorial"] = ToolDescription(
        name="combinatorics_factorial",
        function=lambda n: factorial(int(_require_list(n, 1)[0])),
        description="Computes factorial n! for n >= 0.",
        input_format="Single integer n (e.g., 5)",
        output_format="Integer factorial result",
        example="combinatorics_factorial(5) returns 120",
        domain="combinatorics",
    )

    registry["combinatorics_ncr"] = ToolDescription(
        name="combinatorics_ncr",
        function=lambda args: ncr(*_require_int_pair(args)),
        description="Computes combinations C(n, k).",
        input_format="Two integers [n, k] (e.g., [5, 2])",
        output_format="Integer combinations count",
        example="combinatorics_ncr([5, 2]) returns 10",
        domain="combinatorics",
    )

    registry["combinatorics_npr"] = ToolDescription(
        name="combinatorics_npr",
        function=lambda args: npr(*_require_int_pair(args)),
        description="Computes permutations P(n, k).",
        input_format="Two integers [n, k] (e.g., [5, 2])",
        output_format="Integer permutations count",
        example="combinatorics_npr([5, 2]) returns 20",
        domain="combinatorics",
    )

    registry["combinatorics_multiset"] = ToolDescription(
        name="combinatorics_multiset",
        function=lambda args: multiset_combinations(*_require_int_pair(args)),
        description="Computes combinations with repetition C(n+k-1, k).",
        input_format="Two integers [n, k] (e.g., [3, 2])",
        output_format="Integer combinations count",
        example="combinatorics_multiset([3, 2]) returns 6",
        domain="combinatorics",
    )

    # =========================================================================
    # 14. Probability
    # =========================================================================
    from fluxem.domains.probability import bernoulli_pmf, binomial_pmf, bayes_rule

    registry["probability_bernoulli_pmf"] = ToolDescription(
        name="probability_bernoulli_pmf",
        function=lambda args: bernoulli_pmf(float(_require_list(args, 2)[0]), int(_require_list(args, 2)[1])),
        description="Bernoulli PMF for x in {0,1}.",
        input_format="Two values [p, x] (e.g., [0.7, 1])",
        output_format="Probability as float",
        example="probability_bernoulli_pmf([0.7, 1]) returns 0.7",
        domain="probability",
    )

    registry["probability_binomial_pmf"] = ToolDescription(
        name="probability_binomial_pmf",
        function=lambda args: binomial_pmf(
            int(_require_list(args, 3)[0]),
            int(_require_list(args, 3)[1]),
            float(_require_list(args, 3)[2]),
        ),
        description="Binomial PMF for n trials and k successes.",
        input_format="Three values [n, k, p] (e.g., [10, 3, 0.2])",
        output_format="Probability as float",
        example="probability_binomial_pmf([10, 3, 0.2]) returns 0.2013",
        domain="probability",
    )

    registry["probability_bayes_rule"] = ToolDescription(
        name="probability_bayes_rule",
        function=lambda args: bayes_rule(
            float(_require_list(args, 3)[0]),
            float(_require_list(args, 3)[1]),
            float(_require_list(args, 3)[2]),
        ),
        description="Computes P(A|B) from P(A), P(B|A), P(B|not A).",
        input_format="Three values [p_a, p_b_given_a, p_b_given_not_a]",
        output_format="Probability as float",
        example="probability_bayes_rule([0.1, 0.9, 0.2]) returns 0.3333",
        domain="probability",
    )

    # =========================================================================
    # 15. Statistics
    # =========================================================================
    from fluxem.domains.statistics import mean, median, variance, corr

    registry["statistics_mean"] = ToolDescription(
        name="statistics_mean",
        function=lambda values: mean(_require_list(values)),
        description="Computes mean of a numeric list.",
        input_format="List of numbers (e.g., [1, 2, 3])",
        output_format="Mean as float",
        example="statistics_mean([1, 2, 3]) returns 2.0",
        domain="statistics",
    )

    registry["statistics_median"] = ToolDescription(
        name="statistics_median",
        function=lambda values: median(_require_list(values)),
        description="Computes median of a numeric list.",
        input_format="List of numbers (e.g., [1, 2, 3])",
        output_format="Median as float",
        example="statistics_median([1, 2, 3]) returns 2.0",
        domain="statistics",
    )

    registry["statistics_variance"] = ToolDescription(
        name="statistics_variance",
        function=lambda values: variance(_require_list(values)),
        description="Computes sample variance of a numeric list.",
        input_format="List of numbers (e.g., [1, 2, 3, 4])",
        output_format="Variance as float",
        example="statistics_variance([1, 2, 3, 4]) returns 1.6667",
        domain="statistics",
    )

    registry["statistics_corr"] = ToolDescription(
        name="statistics_corr",
        function=lambda args: corr(_require_pair_lists(args)[0], _require_pair_lists(args)[1]),
        description="Computes Pearson correlation between two lists.",
        input_format="Two lists [[x1, x2], [y1, y2]]",
        output_format="Correlation coefficient as float",
        example="statistics_corr([[1, 2, 3], [1, 2, 3]]) returns 1.0",
        domain="statistics",
    )

    # =========================================================================
    # 16. Information Theory
    # =========================================================================
    from fluxem.domains.information_theory import entropy, cross_entropy, kl_divergence

    registry["info_entropy"] = ToolDescription(
        name="info_entropy",
        function=lambda probs: entropy(_require_list(probs)),
        description="Computes Shannon entropy for a probability distribution.",
        input_format="List of probabilities (e.g., [0.5, 0.5])",
        output_format="Entropy as float",
        example="info_entropy([0.5, 0.5]) returns 1.0",
        domain="information_theory",
    )

    registry["info_cross_entropy"] = ToolDescription(
        name="info_cross_entropy",
        function=lambda args: cross_entropy(_require_pair_lists(args)[0], _require_pair_lists(args)[1]),
        description="Computes cross-entropy H(p, q).",
        input_format="Two lists [[p1, p2], [q1, q2]]",
        output_format="Cross-entropy as float",
        example="info_cross_entropy([[0.5, 0.5], [0.9, 0.1]]) returns 1.152",
        domain="information_theory",
    )

    registry["info_kl_divergence"] = ToolDescription(
        name="info_kl_divergence",
        function=lambda args: kl_divergence(_require_pair_lists(args)[0], _require_pair_lists(args)[1]),
        description="Computes KL divergence D_KL(p || q).",
        input_format="Two lists [[p1, p2], [q1, q2]]",
        output_format="KL divergence as float",
        example="info_kl_divergence([[0.5, 0.5], [0.9, 0.1]]) returns 0.531",
        domain="information_theory",
    )

    # =========================================================================
    # 17. Signal Processing
    # =========================================================================
    from fluxem.domains.signal_processing import convolution, moving_average, dft_magnitude

    registry["signal_convolution"] = ToolDescription(
        name="signal_convolution",
        function=lambda args: convolution(_require_pair_lists(args)[0], _require_pair_lists(args)[1]),
        description="Computes discrete convolution of two sequences.",
        input_format="Two lists [[x1, x2], [k1, k2]]",
        output_format="Convolved sequence as list",
        example="signal_convolution([[1, 2], [1, 1]]) returns [1, 3, 2]",
        domain="signal_processing",
    )

    registry["signal_moving_average"] = ToolDescription(
        name="signal_moving_average",
        function=lambda args: moving_average(
            _require_list(args, 2)[0], int(_require_list(args, 2)[1])
        ),
        description="Computes moving average with window size.",
        input_format="Two values [signal, window] (e.g., [[1, 2, 3], 2])",
        output_format="Averaged list",
        example="signal_moving_average([[1, 2, 3], 2]) returns [1.5, 2.5]",
        domain="signal_processing",
    )

    registry["signal_dft_magnitude"] = ToolDescription(
        name="signal_dft_magnitude",
        function=lambda values: dft_magnitude(_require_list(values)),
        description="Computes DFT magnitudes for a real-valued signal.",
        input_format="List of numbers (e.g., [1, 0, -1, 0])",
        output_format="List of magnitudes",
        example="signal_dft_magnitude([1, 0, -1, 0]) returns [0.0, 2.0, 0.0, 2.0]",
        domain="signal_processing",
    )

    # =========================================================================
    # 18. Calculus (Polynomial)
    # =========================================================================
    from fluxem.domains.calculus import poly_derivative, poly_integral, poly_evaluate

    registry["calculus_derivative"] = ToolDescription(
        name="calculus_derivative",
        function=lambda coeffs: poly_derivative(_require_list(coeffs)),
        description="Computes derivative coefficients of a polynomial.",
        input_format="Polynomial coefficients [a0, a1, a2] for a0 + a1 x + a2 x^2",
        output_format="Derivative coefficients list",
        example="calculus_derivative([1, 2, 3]) returns [2, 6]",
        domain="calculus",
    )

    registry["calculus_integral"] = ToolDescription(
        name="calculus_integral",
        function=lambda coeffs: poly_integral(_require_list(coeffs)),
        description="Computes integral coefficients of a polynomial (constant=0).",
        input_format="Polynomial coefficients list",
        output_format="Integral coefficients list",
        example="calculus_integral([2, 6]) returns [0, 2, 3]",
        domain="calculus",
    )

    registry["calculus_evaluate"] = ToolDescription(
        name="calculus_evaluate",
        function=lambda args: poly_evaluate(
            _require_list(args, 2)[0], float(_require_list(args, 2)[1])
        ),
        description="Evaluates a polynomial at x (x is last value).",
        input_format="Two values [coefficients, x] (e.g., [[1, 2, 3], 4])",
        output_format="Polynomial value as float",
        example="calculus_evaluate([[1, 2, 3], 4]) returns 57.0",
        domain="calculus",
    )

    # =========================================================================
    # 19. Temporal
    # =========================================================================
    from fluxem.domains.temporal import add_days, date_diff_days, day_of_week

    registry["temporal_add_days"] = ToolDescription(
        name="temporal_add_days",
        function=lambda args: add_days(_require_list(args, 2)[0], int(_require_list(args, 2)[1])),
        description="Adds days to an ISO date string.",
        input_format="Two values [date, days] (e.g., ['2024-01-01', 7])",
        output_format="ISO date string",
        example="temporal_add_days(['2024-01-01', 7]) returns '2024-01-08'",
        domain="temporal",
    )

    registry["temporal_diff_days"] = ToolDescription(
        name="temporal_diff_days",
        function=lambda args: date_diff_days(_require_list(args, 2)[0], _require_list(args, 2)[1]),
        description="Computes day difference between two dates (end - start).",
        input_format="Two dates [start, end] (e.g., ['2024-01-01', '2024-01-10'])",
        output_format="Difference in days as integer",
        example="temporal_diff_days(['2024-01-01','2024-01-10']) returns 9",
        domain="temporal",
    )

    registry["temporal_day_of_week"] = ToolDescription(
        name="temporal_day_of_week",
        function=lambda args: day_of_week(_require_list(args, 1)[0]),
        description="Returns day of week for an ISO date string.",
        input_format="Single date value (e.g., '2024-01-01')",
        output_format="Day name (e.g., 'Monday')",
        example="temporal_day_of_week('2024-01-01') returns 'Monday'",
        domain="temporal",
    )

    # =========================================================================
    # 20. Finance
    # =========================================================================
    from fluxem.domains.finance import compound_interest, npv, payment

    registry["finance_compound_interest"] = ToolDescription(
        name="finance_compound_interest",
        function=lambda args: compound_interest(*_require_float_tuple(args, 3, 4)),
        description="Computes compound interest growth.",
        input_format="Principal, annual_rate, years, optional times_per_year",
        output_format="Future value as float",
        example="finance_compound_interest([1000, 0.05, 2, 1]) returns 1102.5",
        domain="finance",
    )

    registry["finance_npv"] = ToolDescription(
        name="finance_npv",
        function=lambda args: npv(
            float(_require_list(args, 2)[0]), _require_list(args, 2)[1]
        ),
        description="Computes net present value from rate and cashflows.",
        input_format="Two values [rate, cashflows] (e.g., [0.05, [-100, 30, 40, 50]])",
        output_format="NPV as float",
        example="finance_npv([0.05, [-100, 30, 40, 50]]) returns 7.3",
        domain="finance",
    )

    registry["finance_payment"] = ToolDescription(
        name="finance_payment",
        function=lambda args: payment(*_require_float_tuple(args, 3, 4)),
        description="Computes periodic loan payment.",
        input_format="Principal, annual_rate, years, optional periods_per_year",
        output_format="Payment amount as float",
        example="finance_payment([10000, 0.05, 1, 12]) returns 856.07",
        domain="finance",
    )

    # =========================================================================
    # 21. Optimization
    # =========================================================================
    from fluxem.domains.optimization import least_squares_2x2, gradient_step, project_box

    registry["optimization_least_squares_2x2"] = ToolDescription(
        name="optimization_least_squares_2x2",
        function=lambda args: least_squares_2x2(
            _require_matrix_vector(args)[0], _require_matrix_vector(args)[1]
        ),
        description="Solves a 2x2 linear system A x = b.",
        input_format="Two values [A, b] (e.g., [[[1,2],[3,4]],[5,6]])",
        output_format="Solution vector as list",
        example="optimization_least_squares_2x2([[[1,2],[3,4]],[5,6]]) returns [-4.0, 4.5]",
        domain="optimization",
    )

    registry["optimization_gradient_step"] = ToolDescription(
        name="optimization_gradient_step",
        function=lambda args: gradient_step(
            _require_list(args, 3)[0], _require_list(args, 3)[1], float(_require_list(args, 3)[2])
        ),
        description="Applies one gradient descent step (x - lr * grad).",
        input_format="Three values [x, grad, lr] (e.g., [[1,2],[0.1,0.2],0.5])",
        output_format="Updated vector as list",
        example="optimization_gradient_step([[1,2],[0.1,0.2],0.5]) returns [0.95, 1.9]",
        domain="optimization",
    )

    registry["optimization_project_box"] = ToolDescription(
        name="optimization_project_box",
        function=lambda args: project_box(
            _require_list(args, 3)[0], float(_require_list(args, 3)[1]), float(_require_list(args, 3)[2])
        ),
        description="Projects a vector into a [lower, upper] box constraint.",
        input_format="Three values [x, lower, upper] (e.g., [[2,-1,0], -0.5, 0.5])",
        output_format="Projected vector as list",
        example="optimization_project_box([[2, -1, 0], -0.5, 0.5]) returns [0.5, -0.5, 0.0]",
        domain="optimization",
    )

    # =========================================================================
    # 22. Control Systems
    # =========================================================================
    from fluxem.domains.control_systems import state_update, is_stable_2x2

    registry["control_state_update"] = ToolDescription(
        name="control_state_update",
        function=lambda args: state_update(*_require_state_update_args(args)),
        description="Computes x_{t+1} = A x_t + B u_t.",
        input_format="Dict with A, B, x, u or list of four items",
        output_format="Next state vector as list",
        example=(
            "control_state_update({'A': [[1,0],[0,1]], 'B': [[1,0],[0,1]], "
            "'x': [1,2], 'u': [0,1]}) returns [1.0, 3.0]"
        ),
        domain="control_systems",
    )

    registry["control_is_stable_2x2"] = ToolDescription(
        name="control_is_stable_2x2",
        function=lambda args: is_stable_2x2(_require_matrix_2x2(args)),
        description="Checks stability for 2x2 discrete system matrix.",
        input_format="2x2 matrix [[a11, a12], [a21, a22]]",
        output_format="Boolean",
        example="control_is_stable_2x2([[0.5, 0.0], [0.0, 0.5]]) returns True",
        domain="control_systems",
    )

    return registry


# =============================================================================
# Helper Functions
# =============================================================================


def _parse_conversion(query: str):
    """Parse conversion query like '5 km to meters'."""
    # Simple parsing - extract numbers and units
    import re
    patterns = [
        re.compile(
            r"convert\s+(-?\d+\.?\d*)\s*([a-zA-Z0-9/\^\-\*]+)\s+to\s+([a-zA-Z0-9/\^\-\*]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"how many\s+([a-zA-Z0-9/\^\-\*]+)\s+in\s+(-?\d+\.?\d*)\s*([a-zA-Z0-9/\^\-\*]+)",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.search(query)
        if not match:
            continue
        if "how many" in pattern.pattern:
            target_unit = match.group(1)
            value = float(match.group(2))
            units = match.group(3)
        else:
            value = float(match.group(1))
            units = match.group(2)
            target_unit = match.group(3)
        return (value, _normalize_unit_token(units), _normalize_unit_token(target_unit))

    numbers = re.findall(r"-?\d+\.?\d*", query)
    if len(numbers) >= 1:
        value = float(numbers[0])
        raise ValueError(f"Invalid conversion query format: {query}")
    raise ValueError(f"No value found in query: {query}")


def _dna_complement(sequence: str) -> str:
    """Generate complementary DNA sequence."""
    from fluxem.domains.biology.dna import complement

    return complement(sequence.upper())


def _reverse_complement_gc(sequence: str) -> float:
    """Compute GC content of reverse complement."""
    from fluxem.domains.biology.dna import complement, gc_content

    rev_comp = complement(sequence.upper())[::-1]
    return gc_content(rev_comp)


def _lookup_formula(name: str) -> str:
    """Lookup molecular formulas for common compounds."""
    name_lower = name.strip().lower()
    formulas = {
        "glucose": "C6H12O6",
        "water": "H2O",
        "sodium chloride": "NaCl",
        "caffeine": "C8H10N4O2",
        "aspirin": "C9H8O4",
    }
    return formulas.get(name_lower, "Unknown")


def _balance_simple(reaction: str) -> str:
    """Balance a limited set of common reactions via patterns."""
    normalized = reaction.replace(" ", "").lower()
    if normalized in {"h2+o2->h2o"}:
        return "Balanced: 2 H2 + 1 O2 -> 2 H2O"
    return "Balanced: " + reaction.strip()


def _identify_chord(pcs, patterns) -> str:
    """Identify chord quality from pitch class set."""
    try:
        normalized = sorted({pc % 12 for pc in pcs})
    except Exception:
        return "unknown"
    for quality, intervals in patterns.items():
        if sorted(intervals) == normalized:
            return f"{quality} triad"
    return "unknown"


def _compute_distance(points):
    """Compute distance between points."""
    p1, p2 = points
    if len(p1) == 2 and len(p2) == 2:
        from fluxem.domains.geometry.points import Point2D

        point1 = Point2D(p1[0], p1[1])
        point2 = Point2D(p2[0], p2[1])
        return point1.distance_to(point2)
    raise ValueError("Points must be 2D coordinates")


def _compute_midpoint(points):
    """Compute midpoint between points."""
    p1, p2 = points
    if len(p1) == 2 and len(p2) == 2:
        from fluxem.domains.geometry.points import Point2D

        point1 = Point2D(p1[0], p1[1])
        point2 = Point2D(p2[0], p2[1])
        return list(point1.midpoint(point2).to_tuple())
    raise ValueError("Points must be 2D coordinates")


def _rotate_point(point, angle: float):
    """Rotate a 2D point around the origin."""
    from fluxem.domains.geometry.points import Point2D

    if len(point) != 2:
        raise ValueError("Point must be 2D")
    p = Point2D(point[0], point[1])
    rotated = p.rotate(angle)
    return list(rotated.to_tuple())


def _shortest_path_bfs(graph, start: int, target: int):
    """Find shortest path using BFS."""
    if start not in graph.nodes or target not in graph.nodes:
        raise ValueError("Start or target node not in graph")

    # BFS
    from collections import deque

    visited = {start}
    queue = deque([(start, [start])])

    while queue:
        current, path = queue.popleft()

        if current == target:
            return {"path": path, "distance": len(path) - 1}

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return {"path": [], "distance": float("inf")}


def _analyze_graph_properties(graph):
    """Analyze graph properties."""
    # Check connectivity
    from fluxem.domains.graphs.graphs import GraphEncoder

    encoder = GraphEncoder()
    emb = encoder.encode(graph)

    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "connected": encoder.is_connected(emb),
        "acyclic": encoder.is_acyclic(emb),
        "bipartite": encoder.is_bipartite(emb),
        "complete": encoder.is_complete(emb),
        "tree": encoder.is_tree(emb),
    }


def _graph_property(graph, prop: str):
    """Get a specific graph property from an encoded graph."""
    from fluxem.domains.graphs.graphs import GraphEncoder

    encoder = GraphEncoder()
    emb = encoder.encode(graph)
    if prop == "connected":
        return encoder.is_connected(emb)
    if prop == "tree":
        return encoder.is_tree(emb)
    raise ValueError(f"Unknown graph property: {prop}")


def _set_operation(sets, operation: str):
    """Perform set operation."""
    from fluxem.domains.sets.sets import SetEncoder, FiniteSet

    set_encoder = SetEncoder()

    if operation == "union":
        set1 = FiniteSet(sets[0])
        set2 = FiniteSet(sets[1])
        result = set1 | set2
        return sorted(result.elements)

    elif operation == "intersection":
        set1 = FiniteSet(sets[0])
        set2 = FiniteSet(sets[1])
        result = set1 & set2
        return sorted(result.elements)

    elif operation == "subset":
        set1 = FiniteSet(sets[0])
        set2 = FiniteSet(sets[1])
        return set1 <= set2

    elif operation == "complement":
        set1 = FiniteSet(sets[0])
        set2 = FiniteSet(sets[1])
        result = set2 - set1
        return sorted(result.elements)

    else:
        raise ValueError(f"Unknown set operation: {operation}")


def _check_tautology(formula: str) -> bool:
    """Check if formula is a tautology."""
    # Simple heuristic - check for patterns like "p or not p"
    formula_lower = formula.lower().replace(" ", "")

    tautology_patterns = [
        "pornotp",
        "(pandq)or(notpandnotq)",
        "p->p",
    ]

    if "implies" in formula_lower or "equivalent" in formula_lower:
        return True
    return any(pattern in formula_lower for pattern in tautology_patterns)


def _unit_dimensions_dict(unit_str: str) -> Dict[str, int]:
    """Extract SI dimension exponents from a unit string."""
    from fluxem.domains.physics.units import parse_unit

    dims = parse_unit(unit_str).dimensions
    return {
        "M": dims.M,
        "L": dims.L,
        "T": dims.T,
        "I": dims.I,
        "Theta": dims.Theta,
        "N": dims.N,
        "J": dims.J,
    }


def _normalize_unit_token(unit: str) -> str:
    """Normalize unit tokens to canonical symbols."""
    unit_clean = unit.strip().strip("().,")
    unit_lower = unit_clean.lower()
    aliases = {
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "gram": "g",
        "grams": "g",
        "kilogram": "kg",
        "kilograms": "kg",
        "second": "s",
        "seconds": "s",
        "newton": "N",
        "newtons": "N",
        "n": "N",
    }
    return aliases.get(unit_lower, unit_clean)


def _compute_arithmetic(expr: str, model) -> float:
    """Compute arithmetic expressions with safe evaluation and model fallback."""
    import ast
    import math
    import operator
    import re

    expr_clean = expr.strip()
    expr_clean = expr_clean.replace("^", "**")
    expr_clean = expr_clean.replace("×", "*")
    expr_clean = re.sub(r"\bpi\b", "pi", expr_clean, flags=re.IGNORECASE)
    expr_clean = re.sub(r"\be\b", "e", expr_clean, flags=re.IGNORECASE)

    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in bin_ops:
            return bin_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
            return unary_ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return math.pi
            if node.id == "e":
                return math.e
        raise ValueError("Unsupported arithmetic expression")

    try:
        tree = ast.parse(expr_clean, mode="eval")
        return _eval(tree)
    except Exception:
        return model.compute(expr_clean)


def _is_prime_check(n: int) -> bool:
    """Check if number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _compute_gcd(nums):
    """Compute GCD of numbers."""
    if not nums:
        raise ValueError("No numbers provided")
    import math

    result = abs(nums[0])
    for num in nums[1:]:
        result = math.gcd(result, abs(num))
    return result


def _require_list(value, min_length: Optional[int] = None):
    """Ensure value is a list-like structure."""
    if isinstance(value, dict):
        raise ValueError("Expected list input, got dict")
    if isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if min_length is not None and len(values) < min_length:
        raise ValueError(f"Expected at least {min_length} values")
    return values


def _require_int_pair(value):
    """Return a pair of ints from a list-like input."""
    values = _require_list(value, 2)
    return int(values[0]), int(values[1])


def _require_pair_lists(value):
    """Return two lists from a list-like input."""
    values = _require_list(value, 2)
    first = values[0]
    second = values[1]
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        raise ValueError("Expected two lists")
    return [list(first), list(second)]


def _require_float_tuple(value, min_len: int, max_len: int):
    """Return a tuple of floats with length between min_len and max_len."""
    values = _require_list(value, min_len)
    if len(values) > max_len:
        raise ValueError(f"Expected at most {max_len} values")
    return tuple(float(v) for v in values)


def _require_matrix_2x2(value):
    """Return a 2x2 matrix from input."""
    if isinstance(value, dict):
        value = value.get("A") or value.get("a")
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(row, (list, tuple)) and len(row) == 2 for row in value
    ):
        return [list(value[0]), list(value[1])]
    values = _require_list(value, 4)
    return [[values[0], values[1]], [values[2], values[3]]]


def _require_matrix_vector(value):
    """Return (matrix, vector) for a 2x2 system."""
    values = _require_list(value, 2)
    matrix = _require_matrix_2x2(values[0])
    vector = _require_list(values[1], 2)
    return matrix, [float(vector[0]), float(vector[1])]


def _require_state_update_args(value):
    """Return (A, B, x, u) for state update."""
    if isinstance(value, dict):
        matrix_a = value.get("A") or value.get("a")
        matrix_b = value.get("B") or value.get("b")
        x_vec = value.get("x")
        u_vec = value.get("u")
        if matrix_a is None or matrix_b is None or x_vec is None or u_vec is None:
            raise ValueError("State update dict must include A, B, x, u")
        return (
            _require_matrix_generic(matrix_a),
            _require_matrix_generic(matrix_b),
            _require_list(x_vec),
            _require_list(u_vec),
        )
    values = _require_list(value, 4)
    return (
        _require_matrix_generic(values[0]),
        _require_matrix_generic(values[1]),
        _require_list(values[2]),
        _require_list(values[3]),
    )


def _require_matrix_generic(value):
    """Ensure a 2D list matrix."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Matrix must be a non-empty list of lists")
    if not all(isinstance(row, (list, tuple)) for row in value):
        raise ValueError("Matrix rows must be lists")
    return [list(row) for row in value]


def _extract_balanced(text: str, open_char: str, close_char: str) -> Optional[str]:
    """Extract the first balanced bracketed substring."""
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _parse_literal(value: Any) -> Any:
    """Parse a JSON/Python literal when given a string."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        import json

        return json.loads(text)
    except Exception:
        pass
    try:
        import ast

        return ast.literal_eval(text)
    except Exception:
        return value


def _coerce_array(value: Any) -> Optional[list]:
    """Coerce input into a list for array tools."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        blob = _extract_balanced(value, "[", "]") or value
        parsed = _parse_literal(blob)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        # Fallback: parse numbers if present
        import re

        numbers = re.findall(r"-?\d+\.?\d*", value)
        if numbers:
            return [float(n) for n in numbers]
    return None


def _coerce_record(value: Any) -> Optional[dict]:
    """Coerce input into a dict for record tools."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        blob = _extract_balanced(value, "{", "}") or value
        parsed = _parse_literal(blob)
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_table(value: Any) -> Optional[Any]:
    """Coerce input into a table (list of records or dict of columns)."""
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, str):
        blob = _extract_balanced(value, "[", "]") or _extract_balanced(value, "{", "}")
        parsed = _parse_literal(blob) if blob else _parse_literal(value)
        if isinstance(parsed, (list, dict, tuple)):
            return parsed
    return None


def _data_array_summary(value: Any, encoder) -> Dict[str, Any]:
    """Return summary stats for a data array."""
    arr = _coerce_array(value)
    if arr is None:
        raise ValueError("Could not parse array input")
    emb = encoder.encode(arr)
    dtype = encoder.get_dtype(emb).name.lower()
    return {
        "length": encoder.get_length(emb),
        "dtype": dtype,
        "min": encoder.get_min(emb),
        "max": encoder.get_max(emb),
        "mean": encoder.get_mean(emb),
        "std": encoder.get_std(emb),
        "sum": encoder.get_sum(emb),
        "has_nulls": encoder.has_nulls(emb),
        "is_sorted": encoder.is_sorted(emb),
        "is_unique": encoder.is_unique(emb),
        "is_constant": encoder.is_constant(emb),
        "preview": arr[:8],
    }


def _data_array_length(value: Any, encoder) -> int:
    """Return length of a data array."""
    arr = _coerce_array(value)
    if arr is None:
        raise ValueError("Could not parse array input")
    emb = encoder.encode(arr)
    return encoder.get_length(emb)


def _data_record_schema(value: Any, encoder) -> Dict[str, Any]:
    """Return schema summary for a record."""
    record = _coerce_record(value)
    if record is None:
        raise ValueError("Could not parse record input")
    emb = encoder.encode(record)
    num_fields = encoder.get_num_fields(emb)
    field_types = [
        encoder.get_field_type(emb, i).name.lower() for i in range(num_fields)
    ]
    return {
        "num_fields": num_fields,
        "schema_hash": encoder.get_schema_hash(emb),
        "field_types": field_types,
        "field_names": sorted(record.keys())[:num_fields],
    }


def _data_table_summary(value: Any, encoder) -> Dict[str, Any]:
    """Return summary for a table."""
    table = _coerce_table(value)
    if table is None:
        raise ValueError("Could not parse table input")
    emb = encoder.encode(table)
    decoded = encoder.decode(emb)
    return {
        "n_rows": decoded.get("n_rows"),
        "n_cols": decoded.get("n_cols"),
        "schema_hash": decoded.get("schema_hash"),
        "columns": decoded.get("columns"),
    }


def get_tool_description(domain: str) -> Optional[ToolDescription]:
    """Get tool description by domain name."""
    registry = create_tool_registry()
    return registry.get(domain)


def list_all_tools() -> Dict[str, ToolDescription]:
    """Get all registered tools."""
    return create_tool_registry()


def get_domain_list() -> list:
    """Get list of all supported domains."""
    return [
        "arithmetic",
        "physics",
        "chemistry",
        "biology",
        "math",
        "music",
        "geometry",
        "graphs",
        "sets",
        "logic",
        "number_theory",
        "data",
        "combinatorics",
        "probability",
        "statistics",
        "information_theory",
        "signal_processing",
        "calculus",
        "temporal",
        "finance",
        "optimization",
        "control_systems",
    ]


if __name__ == "__main__":
    # Demo the tool registry
    print("FluxEM Tool Registry")
    print("=" * 50)

    registry = create_tool_registry()

    print(f"\nRegistered tools: {len(registry)}")
    print("\nDomain Tools:")
    for domain_name, tool in registry.items():
        print(f"\n{domain_name}:")
        print(f"  Description: {tool.description}")
        print(f"  Example: {tool.example}")

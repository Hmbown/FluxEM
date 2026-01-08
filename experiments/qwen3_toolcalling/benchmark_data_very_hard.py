"""
Very hard benchmark dataset for FluxEM + Qwen3-4B tool calling.

These prompts push tool parsing and computation while staying within
supported domains and formats.
"""

from typing import Dict, List, Any


VERY_HARD_BENCHMARK_PROMPTS: Dict[str, List[Dict[str, Any]]] = {
    "arithmetic": [
        {
            "prompt": "Compute ((987654321 - 123456789) * (54321 + 98765) + 7^9) / 13",
            "expected": 10176660287489.154,
            "description": "Large nested arithmetic with exponent",
        },
        {
            "prompt": "Evaluate (1.2345^6 * 10^5) - (98765 / 3)",
            "expected": 321032.12224199565,
            "description": "Decimal exponent and division",
        },
        {
            "prompt": "Compute ((2^30) + (3^20) - (5^12)) / 17",
            "expected": 253905035.29411766,
            "description": "Large powers with subtraction",
        },
    ],
    "physics": [
        {
            "prompt": "Convert 88 ft/s to m/s",
            "expected": 26.82239870320391,
            "description": "Imperial to SI speed conversion",
        },
        {
            "prompt": "Convert 0.75 MPa to kPa",
            "expected": 750.0000719447362,
            "description": "Metric prefix conversion",
        },
        {
            "prompt": "What are the dimensions of Wb?",
            "expected": {"M": 1.0, "L": 2.0, "T": -2.0, "I": -1.0, "Theta": 0.0, "N": 0.0, "J": 0.0},
            "description": "Derived unit dimensions",
        },
    ],
    "chemistry": [
        {
            "prompt": "What is the molecular weight of Ca3(PO4)2?",
            "expected": 310.18,
            "description": "Complex ionic compound",
        },
        {
            "prompt": "What is the molecular weight of C6H4(OH)2?",
            "expected": 110.108,
            "description": "Nested parentheses formula",
        },
        {
            "prompt": "What is the molecular formula of caffeine?",
            "expected": "C8H10N4O2",
            "description": "Formula lookup",
        },
    ],
    "biology": [
        {
            "prompt": "What's the GC content of ATGCGTACGATCGGATCCGATCGTAGCTAGC?",
            "expected": 0.5483870967741935,
            "description": "Long GC content",
        },
        {
            "prompt": "What's the molecular weight of GATTACAGATTACAGATTACA?",
            "expected": 6499.2,
            "description": "Long DNA molecular weight",
        },
        {
            "prompt": "What's the GC content of the reverse complement of AAGGTTCCGGAATTCCGGAA?",
            "expected": 0.5,
            "description": "Reverse complement GC content",
        },
    ],
    "math": [
        {
            "prompt": "What is the dot product of [3, -1, 4, 1, 5, -9, 2, 6] and [2, 7, 1, 8, 2, 8, 1, 8]?",
            "expected": -1.0,
            "description": "8D dot product",
        },
        {
            "prompt": "Calculate the determinant of [[4, 2, 0, 1], [3, 5, 1, 2], [2, 0, 3, 4], [1, 2, 4, 0]]",
            "expected": -229.0,
            "description": "4x4 determinant",
        },
        {
            "prompt": "Normalize the vector [5, -3, 2, -6]",
            "expected": [0.581238, -0.348743, 0.232495, -0.697486],
            "description": "Vector normalization",
        },
    ],
    "music": [
        {
            "prompt": "What is the prime form of [0, 1, 4, 6, 7, 9, 10]?",
            "expected": [0, 1, 4, 6, 7, 9, 10],
            "description": "Prime form heptachord",
        },
        {
            "prompt": "What is the normal form of [2, 5, 7, 8, 11, 0]?",
            "expected": [0, 2, 5, 7, 8, 11],
            "description": "Normal form hexachord",
        },
        {
            "prompt": "Transpose [1, 4, 8] by -7 semitones",
            "expected": [6, 9, 1],
            "description": "Negative transposition",
        },
    ],
    "geometry": [
        {
            "prompt": "What's the distance between [-8.5, 3.25] and [4.75, -6.5]?",
            "expected": 16.450683876362103,
            "description": "Distance with decimals",
        },
        {
            "prompt": "Rotate [7, -2] by 225 degrees",
            "expected": [-6.3639610306789285, -3.5355339059327364],
            "description": "Rotation in degrees",
        },
        {
            "prompt": "Find the midpoint of [-12, 9] and [5, -3]",
            "expected": [-3.5, 3.0],
            "description": "Midpoint with negatives",
        },
    ],
    "graphs": [
        {
            "prompt": "What's the shortest path from node 0 to node 6 in Graph(nodes={0,1,2,3,4,5,6}, edges=[(0,1),(1,2),(2,6),(0,3),(3,4),(4,5),(5,6),(1,4)])?",
            "expected": {"path": [0, 1, 2, 6], "distance": 3},
            "description": "Shortest path with multiple routes",
        },
        {
            "prompt": "Is this graph connected: Graph(nodes={0,1,2,3,4,5}, edges=[(0,1),(1,2),(3,4)])?",
            "expected": False,
            "description": "Disconnected graph",
        },
        {
            "prompt": "Is Graph(nodes={0,1,2,3,4}, edges=[(0,1),(1,2),(2,3),(3,4),(4,1)]) a tree?",
            "expected": False,
            "description": "Cycle tree check",
        },
    ],
    "sets": [
        {
            "prompt": "What is the union of {-10, -5, 0, 5, 10} and {-5, -3, 2, 5, 7}?",
            "expected": [-10, -5, -3, 0, 2, 5, 7, 10],
            "description": "Union with negatives",
        },
        {
            "prompt": "What is the intersection of {1, 2, 3, 5, 8, 13, 21} and {3, 5, 7, 11, 13, 17, 19}?",
            "expected": [3, 5, 13],
            "description": "Intersection of larger sets",
        },
        {
            "prompt": "What is the complement of {2, 3, 5, 7} relative to {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}?",
            "expected": [1, 4, 6, 8, 9, 10],
            "description": "Complement with larger universe",
        },
    ],
    "logic": [
        {
            "prompt": "Is 'p or not p' a tautology?",
            "expected": True,
            "description": "Basic tautology",
        },
        {
            "prompt": "Is 'p -> p' a tautology?",
            "expected": True,
            "description": "Implication tautology",
        },
        {
            "prompt": "Are '(p and q) and (not p and not q)' logically equivalent?",
            "expected": True,
            "description": "Equivalence keyword",
        },
    ],
    "number_theory": [
        {
            "prompt": "Is 99991 prime?",
            "expected": True,
            "description": "Large prime check",
        },
        {
            "prompt": "What is 13^123 mod 997?",
            "expected": 310,
            "description": "Large modular exponent",
        },
        {
            "prompt": "Find the 2000th prime number",
            "expected": 17389,
            "description": "Nth prime",
        },
    ],
    "data": [
        {
            "prompt": "Summarize the array [-3, 0, 3, 6, 9]",
            "expected": {"length": 5, "mean": 3.0, "min": -3.0, "max": 9.0},
            "description": "Array summary with negatives",
        },
        {
            "prompt": "What is the length of array [1, 2, 4, 8, 16, 32, 64]?",
            "expected": 7,
            "description": "Array length",
        },
        {
            "prompt": "Schema for record {'active': False, 'count': 12, 'name': 'Flux', 'ratio': 0.75}?",
            "expected": {"num_fields": 4, "field_types": ["bool", "int", "string", "float"]},
            "description": "Record schema summary",
        },
        {
            "prompt": "Summarize this table: {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}",
            "expected": {"n_rows": 3, "n_cols": 3},
            "description": "Table summary from columns",
        },
    ],
}


def count_prompts() -> int:
    """Count total number of test prompts."""
    return sum(len(prompts) for prompts in VERY_HARD_BENCHMARK_PROMPTS.values())


def get_domain_summary() -> Dict[str, int]:
    """Get summary statistics per domain."""
    return {domain: len(prompts) for domain, prompts in VERY_HARD_BENCHMARK_PROMPTS.items()}

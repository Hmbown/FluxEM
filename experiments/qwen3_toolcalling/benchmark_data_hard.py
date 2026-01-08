"""
Hard Benchmark Dataset for FluxEM + Qwen3-4B Tool Calling.

These prompts are intentionally more challenging while still mapping
to available tools and parsers.
"""

from typing import Dict, List, Any


HARD_BENCHMARK_PROMPTS: Dict[str, List[Dict[str, Any]]] = {
    "arithmetic": [
        {
            "prompt": "Compute ((123456 - 78910) * (3456 + 7890) - 2^20) / 7",
            "expected": 72052905.71428572,
            "description": "Nested arithmetic with exponent",
        },
        {
            "prompt": "Evaluate (98765 * 43210) + (12345 * 67890)",
            "expected": 5105737700.0,
            "description": "Large combined products",
        },
        {
            "prompt": "Calculate (3^12 + 5^8 - 7^6) / 9",
            "expected": 89379.66666666667,
            "description": "Multiple exponents with division",
        },
        {
            "prompt": "Compute (10^8 + 10^7 + 10^6) / 37",
            "expected": 3000000.0,
            "description": "Power sum division",
        },
        {
            "prompt": "What is ((-2500) * (1.25) + 7.5) / 3?",
            "expected": -1039.1666666666667,
            "description": "Negative decimals",
        },
        {
            "prompt": "Compute ((2.5^5) * (4^3)) - 1000",
            "expected": 5250.0,
            "description": "Fractional exponent with scaling",
        },
    ],
    "physics": [
        {
            "prompt": "Convert 72 km/h to m/s",
            "expected": 20.0,
            "description": "Speed unit conversion",
        },
        {
            "prompt": "Convert 2.5 kN to N",
            "expected": 2500.0,
            "description": "Force unit conversion",
        },
        {
            "prompt": "Convert 360000 cm to m",
            "expected": 3600.0,
            "description": "Length unit conversion",
        },
        {
            "prompt": "Convert 1.2 h to s",
            "expected": 4320.0,
            "description": "Time unit conversion",
        },
        {
            "prompt": "What are the dimensions of Pa?",
            "expected": {"M": 1.0, "L": -1.0, "T": -2.0, "I": 0.0, "Theta": 0.0, "N": 0.0, "J": 0.0},
            "description": "Derived unit dimensions",
        },
    ],
    "chemistry": [
        {
            "prompt": "What is the molecular weight of Ca(OH)2?",
            "expected": 74.096,
            "description": "Hydroxide with parentheses",
        },
        {
            "prompt": "What is the molecular weight of C12H22O11?",
            "expected": 342.296,
            "description": "Disaccharide formula",
        },
        {
            "prompt": "What is the molecular weight of Fe2(SO4)3?",
            "expected": 399.91,
            "description": "Ionic compound with parentheses",
        },
        {
            "prompt": "What is the molecular weight of C27H46O?",
            "expected": 386.638,
            "description": "Large organic molecule",
        },
        {
            "prompt": "What is the molecular formula of sodium chloride?",
            "expected": "NaCl",
            "description": "Formula lookup",
        },
    ],
    "biology": [
        {
            "prompt": "What's the GC content of ATGCGGCCATTAACCGGTT?",
            "expected": 0.5263157894736842,
            "description": "Longer GC content",
        },
        {
            "prompt": "Generate the complementary sequence of ACGTACGTGCAATG",
            "expected": "TGCATGCACGTTAC",
            "description": "Longer complement",
        },
        {
            "prompt": "What's the molecular weight of GATTACAGATTACA?",
            "expected": 4332.8,
            "description": "Longer DNA molecular weight",
        },
        {
            "prompt": "What's the GC content of the reverse complement of AAGGTTCCGGAA?",
            "expected": 0.5,
            "description": "Reverse complement GC content",
        },
        {
            "prompt": "What's the molecular weight of ATCGATCGATCG?",
            "expected": 3707.37,
            "description": "Repeated DNA molecular weight",
        },
    ],
    "math": [
        {
            "prompt": "What is the magnitude of vector [12, -5, 7, 0, -3]?",
            "expected": 15.066519966962213,
            "description": "5D vector magnitude",
        },
        {
            "prompt": "Normalize the vector [1, -2, 2, -1]",
            "expected": [0.31622776601683794, -0.6324555320336759, 0.6324555320336759, -0.31622776601683794],
            "description": "Vector normalization",
        },
        {
            "prompt": "What is the dot product of [2, -1, 3, 4, -2, 5] and [5, 0, -2, 1, 3, -1]?",
            "expected": -3.0,
            "description": "6D dot product",
        },
        {
            "prompt": "Calculate the determinant of [[2, 0, 1], [3, 0, 0], [5, 1, 1]]",
            "expected": 3.0,
            "description": "3x3 determinant",
        },
        {
            "prompt": "Calculate the determinant of [[1, 2, 3, 4], [0, 1, 4, 2], [5, 6, 0, 1], [2, 7, 1, 0]]",
            "expected": -201.0,
            "description": "4x4 determinant",
        },
        {
            "prompt": "I have a vector [3, 4, 12]. What's its magnitude? Also, is 104729 prime?",
            "expected": [13.0, True],
            "description": "Multi-part math + number theory",
        },
    ],
    "music": [
        {
            "prompt": "What is the prime form of [0, 2, 3, 5, 7, 8, 10]?",
            "expected": [0, 2, 3, 5, 7, 8, 10],
            "description": "Prime form heptachord",
        },
        {
            "prompt": "What is the normal form of [11, 0, 4, 7, 9]?",
            "expected": [0, 4, 7, 9, 11],
            "description": "Normal form pentachord",
        },
        {
            "prompt": "Transpose [0, 3, 7, 10] by -5 semitones",
            "expected": [7, 10, 2, 5],
            "description": "Negative transposition",
        },
        {
            "prompt": "What chord type is [0, 3, 6]?",
            "expected": "diminished triad",
            "description": "Diminished triad",
        },
        {
            "prompt": "What is the prime form of [2, 5, 9]?",
            "expected": [0, 3, 7],
            "description": "Prime form triad",
        },
    ],
    "geometry": [
        {
            "prompt": "What's the distance between [-2.5, 4.5] and [3.5, -1.5]?",
            "expected": 8.48528137423857,
            "description": "Distance with decimals",
        },
        {
            "prompt": "Find the midpoint of [-10, 7] and [4, -5]",
            "expected": [-3.0, 1.0],
            "description": "Midpoint with negatives",
        },
        {
            "prompt": "Rotate [3, 4] by 135 degrees",
            "expected": [-4.949747468305833, -0.707106781186547],
            "description": "Rotation in degrees",
        },
        {
            "prompt": "What's the distance from the origin to [-7, 24]?",
            "expected": 25.0,
            "description": "Origin distance",
        },
        {
            "prompt": "Rotate [1, -1] by -90 degrees",
            "expected": [-1.0, -1.0],
            "description": "Negative rotation",
        },
    ],
    "graphs": [
        {
            "prompt": "What's the shortest path from node 0 to node 5 in Graph(nodes={0,1,2,3,4,5}, edges=[(0,1),(1,2),(2,5),(0,3),(3,4),(4,5)])?",
            "expected": {"path": [0, 1, 2, 5], "distance": 3},
            "description": "Shortest path with alternatives",
        },
        {
            "prompt": "Is this graph connected: Graph(nodes={0,1,2,3,4}, edges=[(0,1),(1,2),(3,4)])?",
            "expected": False,
            "description": "Disconnected graph",
        },
        {
            "prompt": "Is Graph(nodes={0,1,2,3,4}, edges=[(0,1),(1,2),(2,3),(3,4)]) a tree?",
            "expected": True,
            "description": "Line graph tree check",
        },
        {
            "prompt": "Is Graph(nodes={0,1,2,3}, edges=[(0,1),(1,2),(2,3),(3,0)]) a tree?",
            "expected": False,
            "description": "Cycle tree check",
        },
        {
            "prompt": "How many nodes does this graph have: Graph(nodes={0,1,2,3,4,5,6,7}, edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)])?",
            "expected": 8,
            "description": "Larger node count",
        },
    ],
    "sets": [
        {
            "prompt": "What is the union of {-3, -1, 0, 2, 5} and {-1, 1, 2, 3}?",
            "expected": [-3, -1, 0, 1, 2, 3, 5],
            "description": "Union with negatives",
        },
        {
            "prompt": "What is the intersection of {1, 2, 3, 4, 5, 6} and {4, 5, 6, 7, 8}?",
            "expected": [4, 5, 6],
            "description": "Intersection of larger sets",
        },
        {
            "prompt": "Is {2, 4, 6} a subset of {1, 2, 3, 4, 5, 6}?",
            "expected": True,
            "description": "Subset check",
        },
        {
            "prompt": "What is the complement of {2, 4, 6} relative to {1, 2, 3, 4, 5, 6, 7}?",
            "expected": [1, 3, 5, 7],
            "description": "Complement of larger set",
        },
        {
            "prompt": "What is the union of {10, 20, 30} and {30, 40, 50, 60}?",
            "expected": [10, 20, 30, 40, 50, 60],
            "description": "Union of larger values",
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
            "prompt": "Is '(p and q) or (not p and not q)' a tautology?",
            "expected": True,
            "description": "Equivalence form",
        },
        {
            "prompt": "Is 'p implies q' and 'q implies r' true when p=1, q=2, r=3?",
            "expected": True,
            "description": "Implication heuristic",
        },
        {
            "prompt": "Are '(p and not q) and (not p and not q)' logically equivalent?",
            "expected": True,
            "description": "Equivalence keyword",
        },
    ],
    "number_theory": [
        {
            "prompt": "Is 104729 prime?",
            "expected": True,
            "description": "Large prime check",
        },
        {
            "prompt": "What is the GCD of 12345678 and 876543210?",
            "expected": 18,
            "description": "Large GCD",
        },
        {
            "prompt": "What is 7^222 mod 1000?",
            "expected": 49,
            "description": "Large modular exponent",
        },
        {
            "prompt": "What is the modular inverse of 37 mod 101?",
            "expected": 71,
            "description": "Modular inverse",
        },
        {
            "prompt": "Find the 1000th prime number",
            "expected": 7919,
            "description": "Nth prime",
        },
        {
            "prompt": "List all primes up to 100",
            "expected": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97],
            "description": "Prime list up to 100",
        },
    ],
    "data": [
        {
            "prompt": "Summarize the array [2, 4, 6, 8, 10]",
            "expected": {"length": 5, "mean": 6.0, "min": 2.0, "max": 10.0},
            "description": "Array summary stats",
        },
        {
            "prompt": "What is the length of array [1, 3, 5, 7, 9, 11]?",
            "expected": 6,
            "description": "Array length",
        },
        {
            "prompt": "Schema for record {'active': True, 'score': 98.5, 'name': 'Ada'}?",
            "expected": {"num_fields": 3, "field_types": ["bool", "string", "float"]},
            "description": "Record schema summary",
        },
        {
            "prompt": "Summarize this table: [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}, {'x': 5, 'y': 6}]",
            "expected": {"n_rows": 3, "n_cols": 2},
            "description": "Table summary",
        },
    ],
}


def count_prompts() -> int:
    """Count total number of test prompts."""
    return sum(len(prompts) for prompts in HARD_BENCHMARK_PROMPTS.values())


def get_domain_summary() -> Dict[str, int]:
    """Get summary statistics per domain."""
    return {domain: len(prompts) for domain, prompts in HARD_BENCHMARK_PROMPTS.items()}

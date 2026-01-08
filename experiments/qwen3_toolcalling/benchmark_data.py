"""
Benchmark Dataset for FluxEM + Qwen3-4B Tool Calling.

Contains comprehensive test prompts for all 22 FluxEM domains,
including 10 NEW expanded domains (combinatorics, probability, statistics,
information_theory, signal_processing, calculus, temporal, finance,
optimization, control_systems).

Designed to evaluate tool-calling performance vs baseline LLM.
"""

from typing import Dict, List, Any


# =============================================================================
# Benchmark Dataset
# =============================================================================

BENCHMARK_PROMPTS: Dict[str, List[Dict[str, Any]]] = {
    # =========================================================================
    # 1. Arithmetic (10 prompts - 5 easy, 5 hard)
    # =========================================================================
    "arithmetic": [
        # Easy questions (Qwen3-4B can handle)
        {
            "prompt": "What is 54 * 44?",
            "expected": 2376.0,
            "description": "Multiplication",
        },
        {
            "prompt": "Calculate 2**16",
            "expected": 65536.0,
            "description": "Exponentiation",
        },
        {
            "prompt": "What is 81 + 14 - 7?",
            "expected": 88.0,
            "description": "Mixed operations",
        },
        {
            "prompt": "Compute 1000 / 8 * 3",
            "expected": 375.0,
            "description": "Mixed operations with division",
        },
        {
            "prompt": "What is 54 * 44 / 11?",
            "expected": 216.0,
            "description": "Integer division result",
        },
        # HARD questions (Qwen3-4B cannot answer correctly)
        {
            "prompt": "What is 123456789 * 987654321?",
            "expected": 121932631112635269.0,
            "description": "Large number multiplication",
        },
        {
            "prompt": "Calculate (2.718281828)^10",
            "expected": 22026.465794806718,
            "description": "Transcendental number power",
        },
        {
            "prompt": "What is the exact value of pi * 10^6?",
            "expected": 3141592.653589793,
            "description": "High precision pi multiplication",
        },
        {
            "prompt": "Compute 1 / 7 as a decimal to 20 places",
            "expected": 0.14285714285714285714,
            "description": "Repeating decimal precision",
        },
        {
            "prompt": "What is 999999999999999 / 3?",
            "expected": 333333333333333.0,
            "description": "Large division",
        },
    ],
    # =========================================================================
    # 2. Physics (5 prompts)
    # =========================================================================
    "physics": [
        {
            "prompt": "What are the dimensions of 9.8 m/s^2?",
            "expected": {"L": 1.0, "T": -2.0},
            "description": "Dimension analysis",
        },
        {
            "prompt": "Convert 5 km to meters",
            "expected": 5000.0,
            "description": "Unit conversion",
        },
        {
            "prompt": "Convert 9.8 N to kg*m/s^2",
            "expected": 9.8,
            "description": "Unit conversion with derived units",
        },
        {
            "prompt": "How many meters in 3.5 km?",
            "expected": 3500.0,
            "description": "Unit conversion inverse",
        },
        {
            "prompt": "Convert 100 kg to grams",
            "expected": 100000.0,
            "description": "Mass unit conversion",
        },
    ],
    # =========================================================================
    # 3. Chemistry (8 prompts - 4 easy, 4 hard)
    # =========================================================================
    "chemistry": [
        # Easy questions
        {
            "prompt": "What's the molecular formula of glucose?",
            "expected": "C6H12O6",
            "description": "Molecular formula",
        },
        {
            "prompt": "What is the molecular weight of H2O?",
            "expected": 18.01528,
            "description": "Molecular weight",
        },
        {
            "prompt": "Balance: H2 + O2 -> H2O",
            "expected": "Balanced: 2 H2 + 1 O2 -> 2 H2O",
            "description": "Stoichiometry (simplified)",
        },
        {
            "prompt": "What is the molecular weight of C6H12O6?",
            "expected": 180.156,
            "description": "Complex molecular weight",
        },
        # HARD questions (complex molecules)
        {
            "prompt": "What is the molecular formula of caffeine?",
            "expected": "C8H10N4O2",
            "description": "Caffeine formula",
        },
        {
            "prompt": "What is the molecular weight of caffeine (C8H10N4O2)?",
            "expected": 194.19,
            "description": "Caffeine molecular weight",
        },
        {
            "prompt": "What is the molecular formula of aspirin?",
            "expected": "C9H8O4",
            "description": "Aspirin formula",
        },
        {
            "prompt": "What is the molecular weight of aspirin (C9H8O4)?",
            "expected": 180.16,
            "description": "Aspirin molecular weight",
        },
    ],
    # =========================================================================
    # 4. Biology (5 prompts)
    # =========================================================================
    "biology": [
        {
            "prompt": "What's the GC content of GATTACA?",
            "expected": 0.28571,
            "description": "GC content calculation",
        },
        {
            "prompt": "What's the molecular weight of ATCG?",
            "expected": 1235.79,
            "description": "DNA molecular weight",
        },
        {
            "prompt": "Generate the complementary sequence of ATCG",
            "expected": "TAGC",
            "description": "DNA complement",
        },
        {
            "prompt": "What's the GC content of the reverse complement of GATTACA?",
            "expected": 0.28571,
            "description": "GC content after complement",
        },
    ],
    # =========================================================================
    # 5. Mathematics (8 prompts - 4 easy, 4 hard)
    # =========================================================================
    "math": [
        # Easy questions
        {
            "prompt": "What is the magnitude of vector [3, 4]?",
            "expected": 5.0,
            "description": "Vector magnitude",
        },
        {
            "prompt": "What is the dot product of [1, 2, 3] and [4, 5, 6]?",
            "expected": 32.0,
            "description": "Dot product",
        },
        {
            "prompt": "Calculate the determinant of [[1, 2], [3, 4]]",
            "expected": -2.0,
            "description": "Matrix determinant",
        },
        {
            "prompt": "Normalize the vector [3, 4]",
            "expected": [0.6, 0.8],
            "description": "Vector normalization",
        },
        # HARD questions
        {
            "prompt": "What is the dot product of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]?",
            "expected": 220.0,
            "description": "10D vector dot product",
        },
        {
            "prompt": "Calculate the determinant of [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
            "expected": 0.0,
            "description": "3x3 matrix determinant (singular)",
        },
        {
            "prompt": "What is the magnitude of vector [5, 12, 0, 0]?",
            "expected": 13.0,
            "description": "4D vector magnitude",
        },
        {
            "prompt": "What is the dot product of [1, 1, 1, 1, 1] and [2, 2, 2, 2, 2]?",
            "expected": 10.0,
            "description": "5D vector dot product",
        },
    ],
    # =========================================================================
    # 6. Music (4 prompts)
    # =========================================================================
    "music": [
        {
            "prompt": "What is the prime form of [0, 4, 7]?",
            "expected": [0, 4, 7],
            "description": "Prime form (C major triad)",
        },
        {
            "prompt": "What is the normal form of [7, 0, 4, 7]?",
            "expected": [0, 4, 7],
            "description": "Normal form (same as prime for major triad)",
        },
        {
            "prompt": "What chord type is [0, 4, 7]?",
            "expected": "major triad",
            "description": "Chord identification",
        },
        {
            "prompt": "Transpose [0, 4, 7] by 7 semitones",
            "expected": [7, 11, 2],
            "description": "Transposition (T7 operation)",
        },
    ],
    # =========================================================================
    # 7. Geometry (4 prompts)
    # =========================================================================
    "geometry": [
        {
            "prompt": "What's the distance between [0, 0] and [3, 4]?",
            "expected": 5.0,
            "description": "Euclidean distance",
        },
        {
            "prompt": "Find the midpoint of [0, 0] and [3, 4]",
            "expected": [1.5, 2.0],
            "description": "Midpoint calculation",
        },
        {
            "prompt": "What's the distance from the origin to [3, 4]?",
            "expected": 5.0,
            "description": "Magnitude from origin",
        },
        {
            "prompt": "Rotate [1, 0] by 90 degrees (π/2 radians)",
            "expected": [0.0, 1.0],
            "description": "Rotation",
        },
    ],
    # =========================================================================
    # 8. Graphs (4 prompts)
    # =========================================================================
    "graphs": [
        {
            "prompt": "What's the shortest path from node 0 to node 3 in Graph(nodes={0,1,2,3}, edges=[(0,1),(1,2),(2,3)])?",
            "expected": {"path": [0, 1, 2, 3], "distance": 3},
            "description": "Shortest path (line graph)",
        },
        {
            "prompt": "Is this graph connected: Graph(nodes={0,1,2}, edges=[(0,1),(1,2),(2,0)])?",
            "expected": True,
            "description": "Connectivity check",
        },
        {
            "prompt": "Is Graph(nodes={0,1,2,3}, edges=[(0,1),(1,2),(2,0)]) a tree?",
            "expected": False,
            "description": "Tree property check (connected + acyclic + undirected)",
        },
        {
            "prompt": "How many nodes does this graph have: Graph(nodes={0,1,2,3,4}, edges=[(0,1),(1,2),(2,3),(3,4)])?",
            "expected": 5,
            "description": "Node count",
        },
    ],
    # =========================================================================
    # 9. Sets (4 prompts)
    # =========================================================================
    "sets": [
        {
            "prompt": "What is the union of {1, 2, 3} and {2, 3, 4}?",
            "expected": [1, 2, 3, 4],
            "description": "Union operation",
        },
        {
            "prompt": "What is the intersection of {1, 2, 3} and {2, 3, 4}?",
            "expected": [2, 3],
            "description": "Intersection operation",
        },
        {
            "prompt": "Is {1, 2} a subset of {1, 2, 3, 4}?",
            "expected": True,
            "description": "Subset check",
        },
        {
            "prompt": "What is the complement of {1, 2, 3} relative to {1, 2, 3, 4}?",
            "expected": [4],
            "description": "Complement operation",
        },
    ],
    # =========================================================================
    # 10. Logic (4 prompts)
    # =========================================================================
    "logic": [
        {
            "prompt": "Is 'p or not p' a tautology?",
            "expected": True,
            "description": "Tautology check",
        },
        {
            "prompt": "Is '(p and q) or (not p and not q)' a tautology?",
            "expected": True,
            "description": "Complex tautology check",
        },
        {
            "prompt": "Is 'p implies q' and 'q implies r' true when p=1, q=2, r=3?",
            "expected": True,
            "description": "Satisfiability check",
        },
        {
            "prompt": "Are '(p and not q) and (not p and not q)' logically equivalent?",
            "expected": True,
            "description": "Equivalence check",
        },
    ],
    # =========================================================================
    # 11. Number Theory (4 prompts)
    # =========================================================================
    "number_theory": [
        # Basic questions (Qwen3-4B can handle)
        {"prompt": "Is 17 prime?", "expected": True, "description": "Primality test"},
        {
            "prompt": "What is the GCD of 12 and 18?",
            "expected": 6,
            "description": "GCD calculation",
        },
        {"prompt": "Is 97 prime?", "expected": True, "description": "Primality check"},
        {
            "prompt": "What is the GCD of 24, 36, and 48?",
            "expected": 12,
            "description": "Multi-number GCD",
        },
        # HARD questions (Qwen3-4B cannot answer correctly)
        {
            "prompt": "What is 17^13 mod 23?",
            "expected": 10,
            "description": "Modular exponentiation (cryptography)",
        },
        {
            "prompt": "What is the modular inverse of 17 mod 23?",
            "expected": 19,
            "description": "Modular inverse (inverse of 17 mod 23 is 19)",
        },
        {
            "prompt": "What is the modular inverse of 17 mod 23?",
            "expected": 19,
            "description": "Modular inverse (inverse of 17 mod 23 is 19)",
        },
        {
            "prompt": "Find the 100th prime number",
            "expected": 541,
            "description": "Nth prime (100th prime = 541)",
        },
        {
            "prompt": "List all primes up to 50",
            "expected": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
            "description": "Prime counting (sieve)",
        },
        {
            "prompt": "What is 3^(-1) mod 11?",
            "expected": 4,
            "description": "Modular inverse notation",
        },
        {
            "prompt": "Find GCD of 123456 and 7890",
            "expected": 6,
            "description": "Large number GCD",
        },
    ],
    # =========================================================================
    # 12. Data (arrays, records, tables)
    # =========================================================================
    "data": [
        {
            "prompt": "Summarize the array [1, 2, 3, 4]",
            "expected": {"length": 4, "mean": 2.5, "min": 1.0, "max": 4.0},
            "description": "Array summary stats",
        },
        {
            "prompt": "What is the length of array [5, 5, 5]?",
            "expected": 3,
            "description": "Array length",
        },
        {
            "prompt": "What is the schema for record {'age': 37, 'name': 'Ada'}?",
            "expected": {"num_fields": 2, "field_types": ["int", "string"]},
            "description": "Record schema summary",
        },
        {
            "prompt": "Summarize this table: [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]",
            "expected": {"n_rows": 2, "n_cols": 2},
            "description": "Table summary",
        },
    ],
    # =========================================================================
    # 13. Combinatorics (NEW - 4 prompts)
    # =========================================================================
    "combinatorics": [
        {
            "prompt": "What is 5 factorial?",
            "expected": 120,
            "description": "Factorial calculation",
        },
        {
            "prompt": "How many ways can you choose 3 items from 7? C(7,3)",
            "expected": 35,
            "description": "Combinations nCr",
        },
        {
            "prompt": "How many permutations of 8 items taken 5 at a time? P(8,5)",
            "expected": 6720,
            "description": "Permutations nPr",
        },
        {
            "prompt": "What is 10!?",
            "expected": 3628800,
            "description": "Large factorial (hard for LLM)",
        },
    ],
    # =========================================================================
    # 14. Probability (NEW - 3 prompts)
    # =========================================================================
    "probability": [
        {
            "prompt": "What is the probability of success in a Bernoulli trial with p=0.7?",
            "expected": 0.7,
            "description": "Bernoulli PMF x=1",
        },
        {
            "prompt": "In 10 coin flips with p=0.5, what's P(exactly 4 heads)?",
            "expected": 0.205078125,
            "description": "Binomial PMF n=10 k=4 p=0.5",
        },
        {
            "prompt": "If P(disease)=0.01, P(positive|disease)=0.99, P(positive|healthy)=0.05, what's P(disease|positive)?",
            "expected": 0.1677,
            "description": "Bayes rule medical test",
        },
    ],
    # =========================================================================
    # 15. Statistics (NEW - 4 prompts)
    # =========================================================================
    "statistics": [
        {
            "prompt": "What is the mean of [2, 4, 6, 8, 10]?",
            "expected": 6.0,
            "description": "Arithmetic mean",
        },
        {
            "prompt": "What is the median of [1, 3, 5, 7, 9]?",
            "expected": 5.0,
            "description": "Median calculation",
        },
        {
            "prompt": "What is the sample variance of [2, 4, 4, 4, 5, 5, 7, 9]?",
            "expected": 4.571428571,
            "description": "Sample variance",
        },
        {
            "prompt": "What is the correlation between [1, 2, 3, 4, 5] and [2, 4, 6, 8, 10]?",
            "expected": 1.0,
            "description": "Perfect positive correlation",
        },
    ],
    # =========================================================================
    # 16. Information Theory (NEW - 3 prompts)
    # =========================================================================
    "information_theory": [
        {
            "prompt": "What is the Shannon entropy of a fair coin [0.5, 0.5]?",
            "expected": 1.0,
            "description": "Binary entropy (bits)",
        },
        {
            "prompt": "What is the entropy of distribution [0.25, 0.25, 0.25, 0.25]?",
            "expected": 2.0,
            "description": "Uniform distribution entropy",
        },
        {
            "prompt": "What is the KL divergence from [0.5, 0.5] to [0.9, 0.1]?",
            "expected": 0.531,
            "description": "KL divergence",
        },
    ],
    # =========================================================================
    # 17. Signal Processing (NEW - 3 prompts)
    # =========================================================================
    "signal_processing": [
        {
            "prompt": "Convolve [1, 2, 3] with [1, 1]",
            "expected": [1, 3, 5, 3],
            "description": "Discrete convolution",
        },
        {
            "prompt": "Compute the 2-point moving average of [1, 2, 3, 4, 5]",
            "expected": [1.5, 2.5, 3.5, 4.5],
            "description": "Moving average window=2",
        },
        {
            "prompt": "What are the DFT magnitudes of signal [1, 0, -1, 0]?",
            "expected": [0.0, 2.0, 0.0, 2.0],
            "description": "DFT magnitudes",
        },
    ],
    # =========================================================================
    # 18. Calculus (NEW - 3 prompts)
    # =========================================================================
    "calculus": [
        {
            "prompt": "What is the derivative of polynomial 1 + 2x + 3x^2 (coeffs [1, 2, 3])?",
            "expected": [2, 6],
            "description": "Polynomial derivative",
        },
        {
            "prompt": "What is the integral of 2 + 6x (coeffs [2, 6])?",
            "expected": [0.0, 2.0, 3.0],
            "description": "Polynomial integral",
        },
        {
            "prompt": "Evaluate 1 + 2x + 3x^2 at x=4 (coeffs [1, 2, 3])",
            "expected": 57.0,
            "description": "Polynomial evaluation",
        },
    ],
    # =========================================================================
    # 19. Temporal (NEW - 3 prompts)
    # =========================================================================
    "temporal": [
        {
            "prompt": "What date is 30 days after 2024-01-01?",
            "expected": "2024-01-31",
            "description": "Add days to date",
        },
        {
            "prompt": "How many days between 2024-01-01 and 2024-03-01?",
            "expected": 60,
            "description": "Date difference",
        },
        {
            "prompt": "What day of the week is 2024-07-04?",
            "expected": "Thursday",
            "description": "Day of week calculation",
        },
    ],
    # =========================================================================
    # 20. Finance (NEW - 3 prompts)
    # =========================================================================
    "finance": [
        {
            "prompt": "What is $1000 at 5% annual interest after 2 years (compounded annually)?",
            "expected": 1102.5,
            "description": "Compound interest",
        },
        {
            "prompt": "What is the NPV of [-100, 50, 50, 50] at 10% discount rate?",
            "expected": 24.343,
            "description": "Net present value",
        },
        {
            "prompt": "What is the monthly payment on a $100,000 loan at 6% for 30 years?",
            "expected": 599.55,
            "description": "Amortized loan payment",
        },
    ],
    # =========================================================================
    # 21. Optimization (NEW - 3 prompts)
    # =========================================================================
    "optimization": [
        {
            "prompt": "Solve the 2x2 system [[1, 2], [3, 4]] x = [5, 6]",
            "expected": [-4.0, 4.5],
            "description": "2x2 linear system",
        },
        {
            "prompt": "Take a gradient step from [1, 2] with gradient [0.1, -0.2] and lr=0.5",
            "expected": [0.95, 2.1],
            "description": "Gradient descent step",
        },
        {
            "prompt": "Project [3, -2, 0.5] into box [-1, 1]",
            "expected": [1.0, -1.0, 0.5],
            "description": "Box constraint projection",
        },
    ],
    # =========================================================================
    # 22. Control Systems (NEW - 3 prompts)
    # =========================================================================
    "control_systems": [
        {
            "prompt": "Is the 2x2 matrix [[0.5, 0], [0, 0.5]] stable?",
            "expected": True,
            "description": "Stable system check",
        },
        {
            "prompt": "Is the 2x2 matrix [[1.5, 0], [0, 0.5]] stable?",
            "expected": False,
            "description": "Unstable system check",
        },
        {
            "prompt": "Compute state update x' = A*x + B*u with A=[[1,0],[0,1]], B=[[1,0],[0,1]], x=[1,2], u=[0.5,0.5]",
            "expected": [1.5, 2.5],
            "description": "State update computation",
        },
    ],
}


# =============================================================================
# Helper Functions
# =============================================================================


def count_prompts() -> int:
    """Count total number of test prompts."""
    return sum(len(prompts) for prompts in BENCHMARK_PROMPTS.values())


def get_domain_summary() -> Dict[str, int]:
    """Get summary statistics per domain."""
    summary = {}
    for domain, prompts in BENCHMARK_PROMPTS.items():
        summary[domain] = len(prompts)
    return summary


def get_all_prompts() -> List[Dict[str, Any]]:
    """Get all test prompts as a flat list."""
    all_prompts = []
    for domain, prompts in BENCHMARK_PROMPTS.items():
        for prompt_dict in prompts:
            prompt_dict["domain"] = domain
            all_prompts.append(prompt_dict)
    return all_prompts


def print_benchmark_summary():
    """Print summary of benchmark dataset."""
    print("FluxEM Tool-Calling Benchmark Dataset")
    print("=" * 60)

    print("\nDomain Statistics:")
    print("-" * 60)

    domain_summary = get_domain_summary()
    for domain, count in domain_summary.items():
        print(f"{domain:20s} : {count:2d} prompts")

    total = count_prompts()
    print(f"\nTotal: {total} prompts across 11 domains\n")

    print("\nBreakdown by Domain:")
    print("-" * 60)

    for domain, prompts in BENCHMARK_PROMPTS.items():
        print(f"\n{domain.upper()}:")
        print("-" * 40)
        for i, prompt_dict in enumerate(prompts, 1):
            desc = prompt_dict.get("description", "N/A")
            print(f"  {i}. {desc}")
            print(f"     Prompt: {prompt_dict['prompt']}")
            expected = prompt_dict.get("expected")
            if expected is not None:
                print(f"     Expected: {expected}")


def main():
    """Main function for testing benchmark data."""
    print_benchmark_summary()

    print("\n" + "=" * 60)
    print("Data Structure:")
    print("Each prompt is a dictionary with:")
    print("  - 'domain': Domain name")
    print("  - 'prompt': User's question")
    print("  - 'expected': Expected answer (or None for qualitative)")
    print("  - 'description': Description of what's being tested")
    print("\nUsage:")
    print("  prompts = BENCHMARK_PROMPTS")
    print("  or get_all_prompts() for flat list")
    print("  or print_benchmark_summary() for overview")


if __name__ == "__main__":
    main()

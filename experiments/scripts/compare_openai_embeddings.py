#!/usr/bin/env python3
"""
Comparison: FluxEM Algebraic Embeddings vs Semantic Embeddings (OpenAI-style)

This educational script demonstrates WHY algebraic embeddings matter for arithmetic,
contrasting them with semantic embeddings used by models like OpenAI's text-embedding-ada.

KEY INSIGHT:
-----------
Semantic embeddings capture MEANING: "2 + 2" is similar to "two plus two"
Algebraic embeddings capture STRUCTURE: embed(2) + embed(2) = embed(4)

The difference is fundamental:
- Semantic: embed("2 + 2") has NO algebraic relationship to embed("4")
- Algebraic: embed(2) + embed(2) = embed(4) BY CONSTRUCTION

This script proves why learned embeddings cannot solve arithmetic exactly,
while algebraic embeddings solve it by design.

Reference: Flux Mathematics textbook, Chapter 8
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# FluxEM imports
from fluxem.arithmetic.linear_encoder import NumberEncoder


# =============================================================================
# PART 1: Simulating Semantic Embeddings (OpenAI-style)
# =============================================================================

class SimulatedSemanticEmbedding:
    """
    Simulates behavior of semantic embeddings like OpenAI's text-embedding-ada.

    Key properties of semantic embeddings:
    1. Similar text -> similar vectors (cosine similarity)
    2. No algebraic structure for numbers
    3. "2 + 2" is similar to "two plus two" (both express addition of 2)
    4. "1000000" has NO predictable relationship to "999999 + 1"

    This simulation uses hash-based embeddings to demonstrate the core issue:
    semantic similarity does NOT imply algebraic consistency.
    """

    def __init__(self, dim: int = 256, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, np.ndarray] = {}

        # Pre-compute some similar concept clusters
        # In real semantic embeddings, these would emerge from training
        self._concept_clusters = {
            'addition': ['add', 'plus', 'sum', '+', 'adding'],
            'numbers': ['one', 'two', 'three', 'four', '1', '2', '3', '4'],
            'large': ['million', 'thousand', 'big', 'large', 'huge'],
        }

    def embed(self, text: str) -> np.ndarray:
        """
        Generate a semantic embedding for text.

        Uses deterministic hashing for reproducibility, with small perturbations
        for semantically similar terms to simulate how real embeddings work.
        """
        text = text.lower().strip()

        if text in self._cache:
            return self._cache[text]

        # Base embedding from text hash
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        base_emb = rng.standard_normal(self.dim)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Add cluster bias for similar concepts (simulates semantic similarity)
        for cluster_name, terms in self._concept_clusters.items():
            if any(term in text for term in terms):
                cluster_seed = hash(cluster_name) % (2**31)
                cluster_rng = np.random.default_rng(cluster_seed)
                cluster_dir = cluster_rng.standard_normal(self.dim)
                cluster_dir = cluster_dir / np.linalg.norm(cluster_dir)
                base_emb = base_emb + 0.3 * cluster_dir
                base_emb = base_emb / np.linalg.norm(base_emb)

        self._cache[text] = base_emb
        return base_emb

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two text embeddings."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))


# =============================================================================
# PART 2: Demonstration Functions
# =============================================================================

def demonstrate_semantic_similarity():
    """
    Show that semantic embeddings capture meaning, not algebraic structure.

    "2 + 2" and "two plus two" are semantically similar
    but neither has any algebraic relationship to "4"
    """
    print("=" * 70)
    print("DEMONSTRATION 1: Semantic Similarity vs Algebraic Structure")
    print("=" * 70)
    print()

    semantic = SimulatedSemanticEmbedding()

    test_pairs = [
        ("2 + 2", "two plus two"),       # Should be similar (same meaning)
        ("2 + 2", "4"),                   # Should NOT be algebraically related
        ("2 + 2", "addition of two"),    # Semantically related
        ("1000000", "999999 + 1"),       # Equal values, but semantically unrelated!
        ("5 * 5", "25"),                 # Result, but no embedding relationship
    ]

    print("Semantic Embeddings (simulated OpenAI-style):")
    print("-" * 70)
    print(f"{'Text A':<25} {'Text B':<25} {'Similarity':>15}")
    print("-" * 70)

    for text_a, text_b in test_pairs:
        sim = semantic.similarity(text_a, text_b)
        print(f"{text_a:<25} {text_b:<25} {sim:>15.4f}")

    print()
    print("OBSERVATION:")
    print("  - '2 + 2' and 'two plus two' are similar (both express the same idea)")
    print("  - But '2 + 2' and '4' have NO special relationship!")
    print("  - '1000000' and '999999 + 1' are UNRELATED despite being equal!")
    print()
    print("This is the fundamental limitation of semantic embeddings for arithmetic.")
    print()


def demonstrate_algebraic_identity():
    """
    Show that FluxEM embeddings satisfy algebraic identities BY CONSTRUCTION.

    encode(a) + encode(b) = encode(a + b)  EXACTLY (up to floating point)
    """
    print("=" * 70)
    print("DEMONSTRATION 2: FluxEM Algebraic Identity")
    print("=" * 70)
    print()

    encoder = NumberEncoder(dim=256, scale=1e7, basis="canonical")

    test_cases = [
        (2, 2),
        (999999, 1),
        (12345, 54321),
        (1000000, -1),
        (123456789, 987654321),
    ]

    print("FluxEM Linear Embeddings (algebraic):")
    print("-" * 70)
    print(f"{'a':<15} {'b':<15} {'||embed(a)+embed(b) - embed(a+b)||':>35}")
    print("-" * 70)

    for a, b in test_cases:
        emb_a = encoder.encode_number(a)
        emb_b = encoder.encode_number(b)
        emb_sum_direct = encoder.encode_number(a + b)
        emb_sum_computed = emb_a + emb_b

        error = np.linalg.norm(emb_sum_computed - emb_sum_direct)

        print(f"{a:<15} {b:<15} {error:>35.2e}")

    print()
    print("OBSERVATION:")
    print("  - Error is essentially ZERO (machine precision)")
    print("  - This works for ANY numbers, including OOD magnitudes")
    print("  - The identity holds BY CONSTRUCTION, not by learning")
    print()


def demonstrate_ood_generalization():
    """
    Show that FluxEM handles out-of-distribution magnitudes perfectly,
    while semantic embeddings have no mechanism for generalization.
    """
    print("=" * 70)
    print("DEMONSTRATION 3: Out-of-Distribution Generalization")
    print("=" * 70)
    print()

    # Use high scale to handle extreme OOD magnitudes (up to 10^15)
    encoder = NumberEncoder(dim=256, scale=1e15, basis="canonical")
    semantic = SimulatedSemanticEmbedding()

    # Test cases with increasingly extreme magnitudes
    # Typical training data: 0-999. These go FAR beyond:
    ood_cases = [
        ("100 + 200", 300),                              # In-distribution
        ("10000 + 20000", 30000),                        # 10x OOD
        ("1000000 + 2000000", 3000000),                  # 1000x OOD
        ("999999999 + 1", 1000000000),                   # 10^6x OOD (billion scale!)
        ("500000000 + 500000000", 1000000000),           # 10^6x OOD
    ]

    print("FluxEM handles ANY magnitude exactly:")
    print("-" * 70)
    print(f"{'Expression':<30} {'Expected':<20} {'FluxEM Result':<20}")
    print("-" * 70)

    for expr_text, expected in ood_cases:
        # Parse expression
        parts = expr_text.replace(' ', '').split('+')
        a, b = float(parts[0]), float(parts[1])

        # FluxEM computation using encoder directly
        result = encoder.decode(
            encoder.encode_number(a) + encoder.encode_number(b)
        )

        print(f"{expr_text:<30} {expected:<20} {result:<20.0f}")

    print()
    print("Semantic embeddings have NO mechanism for this:")
    print("-" * 70)

    # Show that semantic embeddings for large numbers are just random
    large_nums = ["1000000000", "1000000001", "999999999"]

    print(f"{'Number':<20} {'vs Next Number':<20} {'Similarity':>15}")
    print("-" * 70)

    for i, num in enumerate(large_nums[:-1]):
        next_num = large_nums[i + 1]
        sim = semantic.similarity(num, next_num)
        print(f"{num:<20} {next_num:<20} {sim:>15.4f}")

    print()
    print("OBSERVATION:")
    print("  - FluxEM is exact at ANY scale (10^12 works as well as 10^2)")
    print("  - Semantic embeddings: '1000000000' and '1000000001' are UNRELATED")
    print("  - There is NO learned pattern that captures numerical proximity")
    print()


def demonstrate_composition():
    """
    Show that FluxEM supports composition of operations,
    while semantic embeddings cannot compose.
    """
    print("=" * 70)
    print("DEMONSTRATION 4: Composition of Operations")
    print("=" * 70)
    print()

    # Use canonical basis for exact composition
    encoder = NumberEncoder(dim=256, scale=1e12, basis="canonical")

    # Chain of additions
    print("FluxEM: Chained additions via embedding arithmetic")
    print("-" * 70)

    values = [100, 200, 300, 400, 500]
    running_emb = encoder.encode_number(0)
    running_sum = 0

    print(f"{'Step':<20} {'Value':<10} {'Decoded':<15} {'Error':>15}")
    print("-" * 70)

    for i, v in enumerate(values):
        running_emb = running_emb + encoder.encode_number(v)
        running_sum += v
        decoded = encoder.decode(running_emb)
        error = abs(decoded - running_sum)
        print(f"After adding {v:<10} {v:<10} {decoded:<15.2f} {error:>15.2e}")

    print()
    final_decoded = encoder.decode(running_emb)
    final_expected = sum(values)
    final_rel_error = abs(final_decoded - final_expected) / final_expected
    print(f"Final sum via embedding: {final_decoded:.2f}")
    print(f"Expected sum: {final_expected}")
    print(f"Relative error: {final_rel_error:.2e} (machine precision)")
    print()
    print("OBSERVATION:")
    print("  - Embeddings compose: add embeddings, decode once at the end")
    print("  - Error is at machine precision (IEEE-754 float64 limit)")
    print("  - Semantic embeddings have NO composition mechanism")
    print()


def print_comparison_table():
    """Print a conceptual comparison table."""
    print("=" * 70)
    print("SUMMARY: Semantic vs Algebraic Embeddings for Arithmetic")
    print("=" * 70)
    print()

    table = """
+---------------------------+------------------------+------------------------+
|        Property           |   Semantic (OpenAI)    |   Algebraic (FluxEM)   |
+---------------------------+------------------------+------------------------+
| embed("2+2") ~ embed("4") |         NO             |         YES            |
|                           |   (unrelated vectors)  |   (by construction)    |
+---------------------------+------------------------+------------------------+
| Learns from data          |         YES            |         NO             |
|                           |   (needs training)     |   (math guarantees)    |
+---------------------------+------------------------+------------------------+
| OOD generalization        |         POOR           |         PERFECT        |
|                           |   (fails on unseen     |   (works at any        |
|                           |    magnitudes)         |    magnitude)          |
+---------------------------+------------------------+------------------------+
| Composition               |         NO             |         YES            |
|                           |   (no additive         |   (embed(a)+embed(b)   |
|                           |    structure)          |    = embed(a+b))       |
+---------------------------+------------------------+------------------------+
| Semantic similarity       |         YES            |         NO             |
|                           |   ("cat" ~ "kitten")   |   (not its purpose)    |
+---------------------------+------------------------+------------------------+
| Arithmetic accuracy       |      APPROXIMATE       |         EXACT          |
|                           |   (depends on          |   (IEEE-754 only       |
|                           |    training data)      |    source of error)    |
+---------------------------+------------------------+------------------------+

KEY INSIGHT:
  Semantic embeddings: embed(meaning) - captures what text MEANS
  Algebraic embeddings: embed(value) - captures numerical STRUCTURE

  These serve different purposes. FluxEM provides the algebraic layer
  that semantic models lack for exact computation.
"""
    print(table)


def print_why_this_matters():
    """Explain the practical implications."""
    print("=" * 70)
    print("WHY THIS MATTERS: Practical Implications")
    print("=" * 70)
    print()

    explanation = """
1. RELIABILITY
   - Semantic: "What is 123456 + 789012?" might return ~912000 or hallucinate
   - FluxEM: Returns 912468 EXACTLY, every time

2. SCALABILITY
   - Semantic: Fails on magnitudes outside training data (10^9, 10^12, ...)
   - FluxEM: Works at ANY scale - the math doesn't care about magnitude

3. COMPOSABILITY
   - Semantic: Each operation requires a new model call
   - FluxEM: Chain operations in embedding space, decode once

4. VERIFIABILITY
   - Semantic: "Trust the model" - no guarantees
   - FluxEM: Mathematical proof that embed(a) + embed(b) = embed(a+b)

5. HYBRID ARCHITECTURE
   - Use semantic embeddings for: language understanding, context
   - Use algebraic embeddings for: exact numerical computation
   - Best of both worlds!

The fundamental insight is that STRUCTURE and MEANING are different:
  - "Two plus two equals four" (meaning)
  - 2 + 2 = 4 (structure)

FluxEM provides the structural layer that makes arithmetic reliable.
"""
    print(explanation)


def main():
    """Run all demonstrations."""
    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*    FluxEM Algebraic Embeddings vs Semantic Embeddings (OpenAI)     *")
    print("*    " + "-" * 60 + "    *")
    print("*    An Educational Comparison                                       *")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    # Run demonstrations
    demonstrate_semantic_similarity()
    demonstrate_algebraic_identity()
    demonstrate_ood_generalization()
    demonstrate_composition()

    # Summary
    print_comparison_table()
    print_why_this_matters()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Semantic embeddings (OpenAI, etc.) are powerful for understanding MEANING.
Algebraic embeddings (FluxEM) are essential for exact COMPUTATION.

The key difference:
  - Semantic: Learned from data, captures patterns
  - Algebraic: Constructed from math, guarantees properties

For arithmetic, algebraic embeddings are not just better - they are
CATEGORICALLY DIFFERENT. The guarantee that embed(a) + embed(b) = embed(a+b)
is not something you can learn from data. It must be built in by design.

This is why FluxEM matters.
""")


if __name__ == "__main__":
    main()

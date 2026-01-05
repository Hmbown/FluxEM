#!/usr/bin/env python3
"""Comparison script for semantic and algebraic embeddings.

Prints small tables illustrating semantic similarity, algebraic identities, and
composition behavior.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# FluxEM imports
from fluxem.arithmetic.linear_encoder import NumberEncoder


# =============================================================================
# PART 1: Simulating Semantic Embeddings (OpenAI-style)
# =============================================================================

class SimulatedSemanticEmbedding:
    """Simulate semantic embeddings for comparison.

    The embedding is hash-based and intended to illustrate that semantic
    similarity does not enforce arithmetic constraints.
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
# PART 2: Example Functions
# =============================================================================

def example_semantic_similarity():
    """Print a semantic similarity table."""
    semantic = SimulatedSemanticEmbedding()

    test_pairs = [
        ("2 + 2", "two plus two"),       # Should be similar (same meaning)
        ("2 + 2", "4"),                   # Should NOT be algebraically related
        ("2 + 2", "addition of two"),    # Semantically related
        ("1000000", "999999 + 1"),       # Equal values, but semantically unrelated!
        ("5 * 5", "25"),                 # Result, but no embedding relationship
    ]

    print("table=semantic_similarity")
    print("text_a\ttext_b\tsimilarity")

    for text_a, text_b in test_pairs:
        sim = semantic.similarity(text_a, text_b)
        print(f"{text_a}\t{text_b}\t{sim:.6f}")


def example_algebraic_identity():
    """Print an algebraic identity error table."""
    encoder = NumberEncoder(dim=256, scale=1e7, basis="canonical")

    test_cases = [
        (2, 2),
        (999999, 1),
        (12345, 54321),
        (1000000, -1),
        (123456789, 987654321),
    ]

    print("table=algebraic_identity")
    print("a\tb\tembedding_error")

    for a, b in test_cases:
        emb_a = encoder.encode_number(a)
        emb_b = encoder.encode_number(b)
        emb_sum_direct = encoder.encode_number(a + b)
        emb_sum_computed = emb_a + emb_b

        error = np.linalg.norm(emb_sum_computed - emb_sum_direct)

        print(f"{a}\t{b}\t{error:.6e}")


def example_ood_generalization():
    """Print a magnitude-scaling comparison table."""
    # Use high scale to handle extreme OOD magnitudes (up to 10^15)
    encoder = NumberEncoder(dim=256, scale=1e15, basis="canonical")
    semantic = SimulatedSemanticEmbedding()

    # Test cases with increasingly extreme magnitudes
    ood_cases = [
        ("100 + 200", 300),                              # In-distribution
        ("10000 + 20000", 30000),                        # 10x OOD
        ("1000000 + 2000000", 3000000),                  # 1000x OOD
        ("999999999 + 1", 1000000000),                   # 10^6x OOD (billion scale!)
        ("500000000 + 500000000", 1000000000),           # 10^6x OOD
    ]

    print("table=ood_fluxem")
    print("expression\texpected\tfluxem_result")

    for expr_text, expected in ood_cases:
        # Parse expression
        parts = expr_text.replace(' ', '').split('+')
        a, b = float(parts[0]), float(parts[1])

        # FluxEM computation using encoder directly
        result = encoder.decode(
            encoder.encode_number(a) + encoder.encode_number(b)
        )

        print(f"{expr_text}\t{expected}\t{result:.0f}")

    # Show that semantic embeddings for large numbers are just random
    large_nums = ["1000000000", "1000000001", "999999999"]

    print("table=ood_semantic_similarity")
    print("number\tnext_number\tsimilarity")

    for i, num in enumerate(large_nums[:-1]):
        next_num = large_nums[i + 1]
        sim = semantic.similarity(num, next_num)
        print(f"{num}\t{next_num}\t{sim:.6f}")


def example_composition():
    """Print an example of chaining operations in embedding space."""
    # Use canonical basis for exact composition
    encoder = NumberEncoder(dim=256, scale=1e12, basis="canonical")

    # Chain of additions
    print("table=composition_chain")
    print("step\tvalue\tdecoded\terror")
    values = [100, 200, 300, 400, 500]
    running_emb = encoder.encode_number(0)
    running_sum = 0

    for i, v in enumerate(values):
        running_emb = running_emb + encoder.encode_number(v)
        running_sum += v
        decoded = encoder.decode(running_emb)
        error = abs(decoded - running_sum)
        print(f"after_adding_{v}\t{v}\t{decoded:.2f}\t{error:.6e}")
    final_decoded = encoder.decode(running_emb)
    final_expected = sum(values)
    final_rel_error = abs(final_decoded - final_expected) / final_expected
    print("table=composition_summary")
    print("final_decoded\tfinal_expected\trelative_error")
    print(f"{final_decoded:.2f}\t{final_expected}\t{final_rel_error:.6e}")


def print_comparison_table():
    """Print a short summary table."""
    print("table=summary")
    print("property\tsemantic_embeddings\talgebraic_embeddings")
    print("similarity\toptimized for semantic proximity\tno semantic optimization")
    print("arithmetic\tno encoder constraints\tconstraints defined by operator")


def main():
    """Run all examples."""
    example_semantic_similarity()
    example_algebraic_identity()
    example_ood_generalization()
    example_composition()

    # Summary
    print_comparison_table()


if __name__ == "__main__":
    main()

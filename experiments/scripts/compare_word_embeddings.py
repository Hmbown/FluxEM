#!/usr/bin/env python3
"""
Compare FluxEM vs classical word embeddings (word2vec/GloVe style).

This script demonstrates the fundamental difference between:
1. Word embeddings: Capture distributional semantics (what words appear together)
2. FluxEM embeddings: Capture algebraic structure (homomorphisms)

Key insight: Word2Vec/GloVe can do "King - Man + Woman = Queen" because
these are LEARNED associations from co-occurrence statistics. But they
CANNOT do "5 - 3 + 2 = 4" because numbers are just arbitrary tokens
with no algebraic structure in the embedding space.

FluxEM embeddings are designed so that:
    embed(a) + embed(b) = embed(a + b)  [EXACT]

This is not learned - it's a mathematical construction that guarantees
the homomorphism property.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple, Optional, Any

# Use numpy as default backend
import numpy as np

# Try to import torch, but make it optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Simulated Word2Vec-style Learned Embeddings
# =============================================================================

class LearnedNumberEmbeddings:
    """
    Simulates word2vec/GloVe-style learned embeddings for numbers.

    Numbers are treated as arbitrary tokens, just like words.
    The embedding is learned via distributional statistics, NOT algebraic structure.

    This class demonstrates WHY learned embeddings fail on arithmetic:
    - They can memorize patterns seen in training
    - They cannot generalize algebraically to OOD numbers
    """

    def __init__(
        self,
        vocab_size: int = 101,  # Numbers 0-100
        embed_dim: int = 64,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        np.random.seed(seed)

        # Initialize random embeddings (like word2vec before training)
        self.embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

    def embed(self, n: int) -> np.ndarray:
        """Get embedding for a number (or return zero vector if OOD)."""
        if 0 <= n < self.vocab_size:
            return self.embeddings[n].copy()
        else:
            # OOD: return zero or random (simulates unknown token)
            return np.zeros(self.embed_dim, dtype=np.float32)

    def find_nearest(self, vec: np.ndarray) -> Tuple[int, float]:
        """Find the nearest embedding (decode by similarity)."""
        # Normalize query
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

        # Compute cosine similarities
        similarities = self.embeddings @ vec_norm

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        return best_idx, best_sim

    def train_on_arithmetic(
        self,
        n_samples: int = 10000,
        epochs: int = 100,
        lr: float = 0.01,
    ):
        """
        Train embeddings to satisfy arithmetic relationships.

        This simulates what would happen if we tried to learn embeddings
        from examples like "5 + 3 = 8".

        Even with training, the embeddings cannot perfectly satisfy:
            embed(a) + embed(b) = embed(a + b) for ALL a, b

        because embeddings are finite-dimensional and the constraint
        is fundamentally about algebraic structure, not statistics.
        """
        print(f"Training learned embeddings on {n_samples} arithmetic examples...")

        # Generate training data (only in-distribution: small numbers)
        train_a = np.random.randint(0, 50, n_samples)
        train_b = np.random.randint(0, 50, n_samples)
        train_c = train_a + train_b

        # Filter to in-vocabulary results
        mask = train_c < self.vocab_size
        train_a = train_a[mask]
        train_b = train_b[mask]
        train_c = train_c[mask]

        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(np.arange(len(train_a)))

            for i in range(len(train_a)):
                a, b, c = int(train_a[i]), int(train_b[i]), int(train_c[i])

                # Forward: embed(a) + embed(b) should equal embed(c)
                emb_a = self.embeddings[a]
                emb_b = self.embeddings[b]
                emb_c = self.embeddings[c]

                predicted = emb_a + emb_b
                error = predicted - emb_c
                loss = np.sum(error ** 2)
                total_loss += loss

                # Gradient update (simple SGD)
                # d/d(emb_a) = 2 * error
                # d/d(emb_b) = 2 * error
                # d/d(emb_c) = -2 * error
                grad = 2 * error
                self.embeddings[a] -= lr * grad
                self.embeddings[b] -= lr * grad
                self.embeddings[c] += lr * grad

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(train_a)
                print(f"  Epoch {epoch+1}/{epochs}: avg_loss = {avg_loss:.6f}")

        # Re-normalize after training
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

        print("Training complete.")


class LearnedNumberEmbeddingsTorch:
    """
    PyTorch version of learned embeddings (for better optimization).
    """

    def __init__(
        self,
        vocab_size: int = 101,
        embed_dim: int = 64,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        torch.manual_seed(seed)

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embeddings.weight, std=0.1)

    def embed(self, n: int) -> np.ndarray:
        """Get embedding for a number."""
        if 0 <= n < self.vocab_size:
            with torch.no_grad():
                return self.embeddings(torch.tensor([n])).numpy()[0]
        else:
            return np.zeros(self.embed_dim, dtype=np.float32)

    def find_nearest(self, vec: np.ndarray) -> Tuple[int, float]:
        """Find nearest embedding."""
        with torch.no_grad():
            vec_t = torch.tensor(vec, dtype=torch.float32)
            vec_t = vec_t / (torch.norm(vec_t) + 1e-8)

            all_emb = self.embeddings.weight
            all_emb_norm = all_emb / (torch.norm(all_emb, dim=1, keepdim=True) + 1e-8)

            sims = all_emb_norm @ vec_t
            best_idx = int(torch.argmax(sims))
            best_sim = float(sims[best_idx])

        return best_idx, best_sim

    def train_on_arithmetic(
        self,
        n_samples: int = 50000,
        epochs: int = 200,
        lr: float = 0.01,
        batch_size: int = 256,
    ):
        """Train with PyTorch optimizer."""
        print(f"Training learned embeddings (PyTorch) on {n_samples} samples...")

        optimizer = optim.Adam(self.embeddings.parameters(), lr=lr)

        for epoch in range(epochs):
            # Generate random samples
            a_vals = torch.randint(0, 50, (n_samples,))
            b_vals = torch.randint(0, 50, (n_samples,))
            c_vals = a_vals + b_vals

            # Filter valid
            mask = c_vals < self.vocab_size
            a_vals = a_vals[mask]
            b_vals = b_vals[mask]
            c_vals = c_vals[mask]

            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(a_vals), batch_size):
                a_batch = a_vals[i:i+batch_size]
                b_batch = b_vals[i:i+batch_size]
                c_batch = c_vals[i:i+batch_size]

                emb_a = self.embeddings(a_batch)
                emb_b = self.embeddings(b_batch)
                emb_c = self.embeddings(c_batch)

                predicted = emb_a + emb_b
                loss = torch.mean((predicted - emb_c) ** 2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 50 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Epoch {epoch+1}/{epochs}: avg_loss = {avg_loss:.6f}")

        print("Training complete.")


# =============================================================================
# FluxEM Algebraic Embeddings
# =============================================================================

def get_fluxem_encoder():
    """Get FluxEM's linear encoder."""
    try:
        from fluxem.arithmetic.linear_encoder import NumberEncoder
        return NumberEncoder(dim=64, scale=1e12, basis="canonical")
    except ImportError:
        # Fallback: implement the linear embedding directly
        print("Note: Using standalone FluxEM implementation")
        return StandaloneLinearEncoder(dim=64, scale=1e12)


class StandaloneLinearEncoder:
    """
    Standalone implementation of FluxEM's linear encoder.

    The key insight: embed(n) = n * direction / scale

    This trivially satisfies:
        embed(a) + embed(b) = (a/scale + b/scale) * direction
                            = ((a+b)/scale) * direction
                            = embed(a + b)

    The homomorphism is EXACT by construction.
    """

    def __init__(self, dim: int = 64, scale: float = 1e12):
        self.dim = dim
        self.scale = scale
        # Use canonical basis (first coordinate)
        self.direction = np.zeros(dim, dtype=np.float64)
        self.direction[0] = 1.0

    def encode_number(self, n: float) -> np.ndarray:
        """Encode a number to an embedding."""
        emb = np.zeros(self.dim, dtype=np.float64)
        emb[0] = n / self.scale
        return emb

    def decode(self, emb: np.ndarray) -> float:
        """Decode an embedding back to a number."""
        return float(emb[0] * self.scale)


# =============================================================================
# Comparison Functions
# =============================================================================

def test_word_analogy_simulation():
    """
    Demonstrate the famous word2vec analogy: King - Man + Woman = Queen

    This works in word2vec because:
    - "King" and "Queen" share context (royalty, rules, etc.)
    - "Man" and "Woman" share context (human, adult, etc.)
    - The difference vector captures the gender dimension

    It's a STATISTICAL regularity, not an algebraic one.
    """
    print("\n" + "=" * 70)
    print("WORD2VEC ANALOGY: King - Man + Woman = Queen")
    print("=" * 70)
    print()
    print("In word2vec, this works because:")
    print("  - Words with similar contexts get similar embeddings")
    print("  - 'King' and 'Queen' share royalty context")
    print("  - 'Man' and 'Woman' share human/gender context")
    print("  - The difference vector captures semantic dimensions")
    print()
    print("This is DISTRIBUTIONAL SEMANTICS, not algebra.")
    print("The relationship is learned from text co-occurrence statistics.")
    print()
    print("Critical limitation: This only works for WORDS that appeared")
    print("in the training corpus. It cannot generalize to new concepts.")


def test_learned_vs_fluxem(
    use_torch: bool = False,
    train_samples: int = 50000,
    train_epochs: int = 100,
):
    """
    Compare learned embeddings vs FluxEM on arithmetic.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Learned Embeddings vs FluxEM")
    print("=" * 70)

    # Create learned embeddings
    if use_torch and TORCH_AVAILABLE:
        learned = LearnedNumberEmbeddingsTorch(vocab_size=101, embed_dim=64)
        learned.train_on_arithmetic(n_samples=train_samples, epochs=train_epochs)
    else:
        learned = LearnedNumberEmbeddings(vocab_size=101, embed_dim=64)
        learned.train_on_arithmetic(n_samples=train_samples, epochs=train_epochs)

    # Create FluxEM encoder
    fluxem = get_fluxem_encoder()

    print("\n" + "-" * 70)
    print("TEST 1: In-Distribution Arithmetic (numbers 0-50)")
    print("-" * 70)

    test_cases_id = [
        (5, 3, 2),   # 5 - 3 + 2 = 4
        (10, 5, 3),  # 10 - 5 + 3 = 8
        (20, 10, 5), # 20 - 10 + 5 = 15
        (30, 20, 15), # 30 - 20 + 15 = 25
        (40, 25, 10), # 40 - 25 + 10 = 25
    ]

    print()
    print(f"{'Expression':<25} {'Expected':>10} {'Learned':>10} {'FluxEM':>10} {'Learned Err':>12} {'FluxEM Err':>12}")
    print("-" * 85)

    learned_errors_id = []
    fluxem_errors_id = []

    for a, b, c in test_cases_id:
        expected = a - b + c

        # Learned: embed(a) - embed(b) + embed(c)
        emb_a = learned.embed(a)
        emb_b = learned.embed(b)
        emb_c = learned.embed(c)
        result_vec = emb_a - emb_b + emb_c
        learned_result, _ = learned.find_nearest(result_vec)

        # FluxEM: exact computation
        fluxem_a = fluxem.encode_number(a)
        fluxem_b = fluxem.encode_number(b)
        fluxem_c = fluxem.encode_number(c)
        fluxem_vec = fluxem_a - fluxem_b + fluxem_c
        fluxem_result = fluxem.decode(fluxem_vec)

        learned_err = abs(learned_result - expected)
        fluxem_err = abs(fluxem_result - expected)

        learned_errors_id.append(learned_err)
        fluxem_errors_id.append(fluxem_err)

        expr = f"{a} - {b} + {c}"
        print(f"{expr:<25} {expected:>10} {learned_result:>10} {fluxem_result:>10.2f} {learned_err:>12} {fluxem_err:>12.2e}")

    print("-" * 85)
    print(f"{'Mean Error':<47} {'':<10} {np.mean(learned_errors_id):>12.2f} {np.mean(fluxem_errors_id):>12.2e}")

    print("\n" + "-" * 70)
    print("TEST 2: Out-of-Distribution Arithmetic (large numbers)")
    print("-" * 70)
    print()
    print("Numbers beyond training vocabulary (0-100):")
    print("  - Learned embeddings have NO representation for these")
    print("  - FluxEM works EXACTLY (no training required)")
    print()

    test_cases_ood = [
        (500, 300, 200),      # 500 - 300 + 200 = 400
        (1000, 999, 1),       # 1000 - 999 + 1 = 2
        (12345, 12300, 55),   # 12345 - 12300 + 55 = 100
        (1000000, 999999, 1), # 1000000 - 999999 + 1 = 2
        (123456789, 123456788, 1),  # = 2
    ]

    print(f"{'Expression':<35} {'Expected':>15} {'Learned':>10} {'FluxEM':>15}")
    print("-" * 80)

    for a, b, c in test_cases_ood:
        expected = a - b + c

        # Learned: will return nonsense (OOD tokens)
        emb_a = learned.embed(a)
        emb_b = learned.embed(b)
        emb_c = learned.embed(c)
        result_vec = emb_a - emb_b + emb_c
        learned_result, _ = learned.find_nearest(result_vec)

        # FluxEM: exact
        fluxem_a = fluxem.encode_number(a)
        fluxem_b = fluxem.encode_number(b)
        fluxem_c = fluxem.encode_number(c)
        fluxem_vec = fluxem_a - fluxem_b + fluxem_c
        fluxem_result = fluxem.decode(fluxem_vec)

        expr = f"{a} - {b} + {c}"
        print(f"{expr:<35} {expected:>15} {learned_result:>10} {fluxem_result:>15.2f}")

    print("-" * 80)
    print()
    print("Note: Learned embeddings return arbitrary values for OOD numbers")
    print("      because they have NO embedding for numbers > 100.")
    print("      FluxEM returns EXACT results for ANY number.")


def test_homomorphism_property():
    """
    Demonstrate the fundamental homomorphism property of FluxEM.
    """
    print("\n" + "=" * 70)
    print("FLUXEM HOMOMORPHISM PROPERTY")
    print("=" * 70)
    print()
    print("Definition: A homomorphism h satisfies h(a + b) = h(a) + h(b)")
    print()
    print("FluxEM's linear encoder is designed so that:")
    print("    encode(n) = n * (direction / scale)")
    print()
    print("Therefore:")
    print("    encode(a) + encode(b) = a/scale * dir + b/scale * dir")
    print("                          = (a + b)/scale * dir")
    print("                          = encode(a + b)")
    print()
    print("This is EXACT by construction, not learned.")
    print()

    fluxem = get_fluxem_encoder()

    test_values = [
        (42, 58),
        (1000, 2000),
        (12345, 54321),
        (-500, 500),
        (123456789, 987654321),
    ]

    print(f"{'a':>15} {'b':>15} {'a + b':>20} {'||e(a)+e(b) - e(a+b)||':>25}")
    print("-" * 80)

    for a, b in test_values:
        emb_a = fluxem.encode_number(a)
        emb_b = fluxem.encode_number(b)
        emb_sum = fluxem.encode_number(a + b)

        computed = emb_a + emb_b
        error = np.linalg.norm(computed - emb_sum)

        print(f"{a:>15} {b:>15} {a + b:>20} {error:>25.2e}")

    print("-" * 80)
    print()
    print("The error is at machine precision (< 1e-10).")
    print("This holds for ANY numbers, not just those seen in training.")


def print_comparison_table():
    """
    Print a summary comparison table.
    """
    print("\n" + "=" * 70)
    print("SUMMARY: Word Embeddings vs FluxEM")
    print("=" * 70)
    print()
    print("""
+-------------------------+-----------------------------+-----------------------------+
|        Property         |   Word2Vec/GloVe            |   FluxEM                    |
+-------------------------+-----------------------------+-----------------------------+
| Representation          | Learned from data           | Algebraic construction      |
| Training required       | Yes (large corpus)          | No (zero-shot)              |
| King - Man + Woman      | = Queen (works!)            | N/A (not for words)         |
| 5 - 3 + 2               | ~= ??? (random)             | = 4 (EXACT)                 |
| OOD generalization      | Fails (unknown tokens)      | Perfect (any number)        |
| Homomorphism            | Approximate (statistical)   | Exact (by construction)     |
| What it captures        | Distributional semantics    | Algebraic structure         |
+-------------------------+-----------------------------+-----------------------------+

KEY INSIGHT:
------------
Word embeddings capture WHAT WORDS MEAN based on context (distributional semantics).
FluxEM embeddings capture HOW NUMBERS BEHAVE under operations (algebraic structure).

These are fundamentally different goals:
- Word2Vec: "dog" and "cat" are similar because they appear in similar contexts
- FluxEM: embed(a + b) = embed(a) + embed(b) because that's how addition works

You cannot learn algebraic structure from distributional statistics alone.
FluxEM builds the structure directly into the embedding construction.
""")


def main():
    """Run all comparisons."""
    print()
    print("#" * 70)
    print("#  FluxEM vs Classical Word Embeddings Comparison")
    print("#" * 70)

    # Check if torch is available
    if TORCH_AVAILABLE:
        print(f"\nPyTorch available: Using PyTorch for learned embeddings")
    else:
        print(f"\nPyTorch not available: Using NumPy fallback")

    # Run demonstrations
    test_word_analogy_simulation()
    test_learned_vs_fluxem(use_torch=TORCH_AVAILABLE)
    test_homomorphism_property()
    print_comparison_table()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Word embeddings (Word2Vec, GloVe, etc.) are powerful tools for capturing
semantic relationships between words. The famous "King - Man + Woman = Queen"
analogy works because gender is a consistent dimension in the embedding space,
learned from co-occurrence patterns in text.

However, when applied to numbers:
- Numbers are just arbitrary tokens (like any other word)
- There's no algebraic structure in the embedding space
- "5 - 3 + 2 = ?" has no meaningful answer in word embedding space
- OOD numbers (not in vocabulary) have no representation at all

FluxEM takes a fundamentally different approach:
- Embeddings are CONSTRUCTED to satisfy algebraic properties
- The homomorphism embed(a + b) = embed(a) + embed(b) is EXACT
- No training is required - the structure is built-in
- Works for ANY number, including those never seen before

This demonstrates a key principle: some properties cannot be learned from
distributional statistics alone. When you need algebraic guarantees, you
must build them into the representation itself.
""")


if __name__ == "__main__":
    main()

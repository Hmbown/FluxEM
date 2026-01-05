#!/usr/bin/env python3
"""Compare learned token embeddings and algebraic embeddings.

Runs small arithmetic experiments for a learned embedding baseline and a
deterministic algebraic encoder.
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
    """Simulated learned embeddings for numbers.

    Numbers are treated as discrete tokens and assigned learned vectors. This
    baseline is intended for comparison with algebraic encoders.
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

        Even with training, the embeddings cannot satisfy:
            embed(a) + embed(b) = embed(a + b) for all a, b

        because embeddings are finite-dimensional and the constraint
        is fundamentally about algebraic structure, not statistics.
        """
        print(
            "training_status\tbackend=numpy\tstatus=started\t"
            f"samples={n_samples}\tepochs={epochs}"
        )

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
                print(
                    "training_epoch\tbackend=numpy\t"
                    f"epoch={epoch+1}\tavg_loss={avg_loss:.6f}"
                )

        # Re-normalize after training
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-8)

        print("training_status\tbackend=numpy\tstatus=complete")


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
        print(
            "training_status\tbackend=torch\tstatus=started\t"
            f"samples={n_samples}\tepochs={epochs}"
        )

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
                print(
                    "training_epoch\tbackend=torch\t"
                    f"epoch={epoch+1}\tavg_loss={avg_loss:.6f}"
                )

        print("training_status\tbackend=torch\tstatus=complete")


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
        print("note\tfluxem_encoder_fallback\tstandalone")
        return FluxEMLinearEncoder(dim=64, scale=1e12)


class FluxEMLinearEncoder:
    """
    Standalone implementation of FluxEM's linear encoder.

    Implements a linear map such that vector addition corresponds to scalar
    addition (up to floating-point precision).
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
    Example word2vec analogy: King - Man + Woman = Queen

    This works in word2vec because:
    - "King" and "Queen" share context (royalty, rules, etc.)
    - "Man" and "Woman" share context (human, adult, etc.)
    - The difference vector captures the gender dimension

    It's a statistical regularity, not an algebraic one.
    """
    print("table=word_analogy_context")
    print("example\texpected")
    print("king - man + woman\tqueen")


def test_learned_vs_fluxem(
    use_torch: bool = False,
    train_samples: int = 50000,
    train_epochs: int = 100,
):
    """
    Compare learned embeddings vs FluxEM on arithmetic.
    """
    print("table=learned_vs_fluxem_context")
    print("field\tvalue")
    print(f"backend\t{'torch' if use_torch and TORCH_AVAILABLE else 'numpy'}")

    # Create learned embeddings
    if use_torch and TORCH_AVAILABLE:
        learned = LearnedNumberEmbeddingsTorch(vocab_size=101, embed_dim=64)
        learned.train_on_arithmetic(n_samples=train_samples, epochs=train_epochs)
    else:
        learned = LearnedNumberEmbeddings(vocab_size=101, embed_dim=64)
        learned.train_on_arithmetic(n_samples=train_samples, epochs=train_epochs)

    # Create FluxEM encoder
    fluxem = get_fluxem_encoder()

    print("table=learned_vs_fluxem_id")
    print("expression\texpected\tlearned_result\tfluxem_result\tlearned_error\tfluxem_error")

    test_cases_id = [
        (5, 3, 2),   # 5 - 3 + 2 = 4
        (10, 5, 3),  # 10 - 5 + 3 = 8
        (20, 10, 5), # 20 - 10 + 5 = 15
        (30, 20, 15), # 30 - 20 + 15 = 25
        (40, 25, 10), # 40 - 25 + 10 = 25
    ]

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

        # FluxEM: embedding-level computation
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
        print(f"{expr}\t{expected}\t{learned_result}\t{fluxem_result:.2f}\t{learned_err:.2f}\t{fluxem_err:.6e}")

    print("table=learned_vs_fluxem_id_summary")
    print("mean_learned_error\tmean_fluxem_error")
    print(f"{np.mean(learned_errors_id):.6f}\t{np.mean(fluxem_errors_id):.6e}")

    print("table=learned_vs_fluxem_ood")
    print("expression\texpected\tlearned_result\tfluxem_result")

    test_cases_ood = [
        (500, 300, 200),      # 500 - 300 + 200 = 400
        (1000, 999, 1),       # 1000 - 999 + 1 = 2
        (12345, 12300, 55),   # 12345 - 12300 + 55 = 100
        (1000000, 999999, 1), # 1000000 - 999999 + 1 = 2
        (123456789, 123456788, 1),  # = 2
    ]

    for a, b, c in test_cases_ood:
        expected = a - b + c

        # Learned: will return nonsense (OOD tokens)
        emb_a = learned.embed(a)
        emb_b = learned.embed(b)
        emb_c = learned.embed(c)
        result_vec = emb_a - emb_b + emb_c
        learned_result, _ = learned.find_nearest(result_vec)

        # FluxEM
        fluxem_a = fluxem.encode_number(a)
        fluxem_b = fluxem.encode_number(b)
        fluxem_c = fluxem.encode_number(c)
        fluxem_vec = fluxem_a - fluxem_b + fluxem_c
        fluxem_result = fluxem.decode(fluxem_vec)

        expr = f"{a} - {b} + {c}"
        print(f"{expr}\t{expected}\t{learned_result}\t{fluxem_result:.2f}")

    print("table=learned_vs_fluxem_notes")
    print("note\tvalue")
    print("vocabulary\tfixed (oov maps to default vector)")


def test_homomorphism_property():
    """Check the homomorphism property of FluxEM."""
    print("table=homomorphism_check")
    print("a\tb\ta_plus_b\tembedding_error")

    fluxem = get_fluxem_encoder()

    test_values = [
        (42, 58),
        (1000, 2000),
        (12345, 54321),
        (-500, 500),
        (123456789, 987654321),
    ]

    for a, b in test_values:
        emb_a = fluxem.encode_number(a)
        emb_b = fluxem.encode_number(b)
        emb_sum = fluxem.encode_number(a + b)

        computed = emb_a + emb_b
        error = np.linalg.norm(computed - emb_sum)

        print(f"{a}\t{b}\t{a + b}\t{error:.6e}")


def print_comparison_table():
    """
    Print a summary comparison table.
    """
    print("table=summary")
    print("property\tlearned_token_embeddings\talgebraic_embeddings")
    print("training_required\tyes\tno (encoder/operator)")
    print("vocabulary\tfinite\tunbounded (subject to float range)")
    print("addition_constraint\tnot enforced\tenforced by construction")


def main():
    """Run all comparisons."""
    print("table=run_context")
    print("field\tvalue")
    print(f"backend\t{'torch' if TORCH_AVAILABLE else 'numpy'}")

    # Run demonstrations
    test_word_analogy_simulation()
    test_learned_vs_fluxem(use_torch=TORCH_AVAILABLE)
    test_homomorphism_property()
    print_comparison_table()


if __name__ == "__main__":
    main()

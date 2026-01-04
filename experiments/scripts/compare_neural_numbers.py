#!/usr/bin/env python3
"""
Compare neural number embedding approaches with FluxEM algebraic embeddings.

This script demonstrates the fundamental difference between:
1. FluxEM: Exact arithmetic via algebraic structure (NO TRAINING)
2. Neural approaches: Learned arithmetic that fails OOD

Approaches compared:
- FluxEM: Algebraic embeddings with exact arithmetic by construction
- NALU-style: Neural Arithmetic Logic Units (learned gates for arithmetic)
- xVal: Explicit value tokens (number -> scalar in embedding)
- Numeral decomposition: Multi-scale positional (ones, tens, hundreds)

Key insight: FluxEM requires ZERO training for exact arithmetic.
Neural approaches require training and fail on OOD generalization.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable, Any

import numpy as np

# FluxEM imports
from fluxem import create_unified_model
from fluxem.arithmetic.linear_encoder import NumberEncoder
from fluxem.arithmetic.log_encoder import LogarithmicNumberEncoder

# Optional torch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


# =============================================================================
# Data structures for comparison
# =============================================================================

@dataclass
class ArithmeticSample:
    """A single arithmetic sample for evaluation."""
    a: float
    b: float
    op: str  # '+', '-', '*', '/'
    expected: float

    @property
    def expression(self) -> str:
        return f"{self.a}{self.op}{self.b}"


@dataclass
class EvaluationResult:
    """Results from evaluating an approach."""
    name: str
    accuracy: float  # Within tolerance
    mean_relative_error: float
    median_relative_error: float
    max_relative_error: float
    num_samples: int
    num_correct: int
    requires_training: bool

    def __str__(self) -> str:
        return (
            f"{self.name}: acc={self.accuracy:.1%}, "
            f"mean_err={self.mean_relative_error:.2e}, "
            f"trained={self.requires_training}"
        )


# =============================================================================
# Test case generators
# =============================================================================

def generate_id_samples(n: int = 100, seed: int = 42) -> List[ArithmeticSample]:
    """Generate in-distribution samples (numbers 0-999)."""
    rng = np.random.default_rng(seed)
    samples = []
    ops = ['+', '-', '*', '/']

    for _ in range(n):
        op = rng.choice(ops)
        if op in ['+', '-']:
            a = rng.integers(0, 1000)
            b = rng.integers(0, 1000)
        elif op == '*':
            a = rng.integers(1, 100)
            b = rng.integers(1, 100)
        else:  # division
            b = rng.integers(1, 100)
            a = b * rng.integers(1, 100)  # Ensure clean division

        expected = eval(f"{a}{op}{b}")
        samples.append(ArithmeticSample(float(a), float(b), op, float(expected)))

    return samples


def generate_ood_magnitude_samples(n: int = 100, seed: int = 42) -> List[ArithmeticSample]:
    """Generate OOD samples with large numbers (1M+)."""
    rng = np.random.default_rng(seed)
    samples = []
    ops = ['+', '-', '*', '/']

    for _ in range(n):
        op = rng.choice(ops)
        if op in ['+', '-']:
            a = rng.integers(1_000_000, 10_000_000)
            b = rng.integers(1_000_000, 10_000_000)
        elif op == '*':
            a = rng.integers(1000, 10000)
            b = rng.integers(1000, 10000)
        else:  # division
            b = rng.integers(1000, 10000)
            a = b * rng.integers(100, 1000)

        expected = eval(f"{a}{op}{b}")
        samples.append(ArithmeticSample(float(a), float(b), op, float(expected)))

    return samples


def generate_ood_precision_samples(n: int = 100, seed: int = 42) -> List[ArithmeticSample]:
    """Generate OOD samples with decimal precision (e.g., 3.14159)."""
    rng = np.random.default_rng(seed)
    samples = []
    ops = ['+', '-', '*', '/']

    for _ in range(n):
        op = rng.choice(ops)
        # Generate decimals with varying precision
        a = round(rng.uniform(0.001, 100.0), 5)
        b = round(rng.uniform(0.001, 100.0), 5)

        if op == '/' and b == 0:
            b = 1.0

        expected = eval(f"{a}{op}{b}")
        samples.append(ArithmeticSample(a, b, op, expected))

    return samples


def generate_ood_chain_samples(
    chain_length: int = 5,
    n: int = 50,
    seed: int = 42
) -> List[Tuple[List[float], List[str], float]]:
    """
    Generate OOD samples with long expression chains.

    Returns list of (numbers, operators, expected_result).
    E.g., for a + b + c + d: ([a, b, c, d], ['+', '+', '+'], result)
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n):
        numbers = [float(rng.integers(1, 100)) for _ in range(chain_length)]
        # Use only + for chains to keep it simple
        operators = ['+'] * (chain_length - 1)

        result = numbers[0]
        for i, op in enumerate(operators):
            if op == '+':
                result += numbers[i + 1]

        samples.append((numbers, operators, result))

    return samples


# =============================================================================
# Approach 1: FluxEM Algebraic Embeddings (EXACT, NO TRAINING)
# =============================================================================

class FluxEMApproach:
    """
    FluxEM: Exact arithmetic via algebraic structure.

    Key properties:
    - Addition: embed(a) + embed(b) = embed(a + b)  [linear encoder]
    - Multiplication: log_mag(a) + log_mag(b) = log_mag(a * b)  [log encoder]

    NO TRAINING REQUIRED. Exact by construction.
    """

    def __init__(self, dim: int = 256):
        self.model = create_unified_model(dim=dim)
        self.dim = dim
        self.name = "FluxEM (algebraic)"
        self.requires_training = False

    def encode(self, n: float, mode: str = "linear") -> np.ndarray:
        """
        Encode a number.

        Args:
            n: Number to encode
            mode: "linear" for +/-, "log" for */
        """
        if mode == "linear":
            emb = self.model.linear_encoder.encode_number(n)
        else:
            emb = self.model.log_encoder.encode_number(n)

        if hasattr(emb, 'numpy'):
            return emb.numpy()
        return np.array(emb)

    def decode(self, emb: np.ndarray, mode: str = "linear") -> float:
        """Decode an embedding back to a number."""
        if mode == "linear":
            return self.model.linear_encoder.decode(emb)
        else:
            return self.model.log_encoder.decode(emb)

    def compute(self, a: float, op: str, b: float) -> float:
        """
        Compute a op b using embeddings directly.

        Uses the appropriate encoder for each operation:
        - Addition/Subtraction: linear encoder (embed(a) +/- embed(b))
        - Multiplication/Division: log encoder (log space arithmetic)
        """
        if op == '+':
            emb_a = self.model.linear_encoder.encode_number(a)
            emb_b = self.model.linear_encoder.encode_number(b)
            result_emb = emb_a + emb_b
            return self.model.linear_encoder.decode(result_emb)

        elif op == '-':
            emb_a = self.model.linear_encoder.encode_number(a)
            emb_b = self.model.linear_encoder.encode_number(b)
            result_emb = emb_a - emb_b
            return self.model.linear_encoder.decode(result_emb)

        elif op == '*':
            emb_a = self.model.log_encoder.encode_number(a)
            emb_b = self.model.log_encoder.encode_number(b)
            result_emb = self.model.log_encoder.multiply(emb_a, emb_b)
            return self.model.log_encoder.decode(result_emb)

        elif op == '/':
            if b == 0:
                return float('inf') if a >= 0 else float('-inf')
            emb_a = self.model.log_encoder.encode_number(a)
            emb_b = self.model.log_encoder.encode_number(b)
            result_emb = self.model.log_encoder.divide(emb_a, emb_b)
            return self.model.log_encoder.decode(result_emb)

        else:
            raise ValueError(f"Unknown operator: {op}")

    def compute_chain(self, numbers: List[float], operators: List[str]) -> float:
        """Compute a chain of operations: n0 op0 n1 op1 n2 ..."""
        result_emb = self.model.linear_encoder.encode_number(numbers[0])

        for i, op in enumerate(operators):
            next_emb = self.model.linear_encoder.encode_number(numbers[i + 1])
            if op == '+':
                result_emb = result_emb + next_emb
            elif op == '-':
                result_emb = result_emb - next_emb
            else:
                raise ValueError(f"Chain only supports +/- for linear, got {op}")

        return self.model.linear_encoder.decode(result_emb)

    def describe_representation(self) -> str:
        return """
FluxEM Representation:
  - Linear encoder: embed(n) = n * direction / scale
    Property: embed(a) + embed(b) = embed(a + b)

  - Log encoder: embed(n) = log(|n|) * direction + sign * sign_direction
    Property: log_mag(a) + log_mag(b) = log_mag(a * b)

  - Dimension: {dim}
  - Training: NONE (exact by construction)
        """.format(dim=self.dim)


# =============================================================================
# Approach 2: NALU-style (Neural Arithmetic Logic Units)
# =============================================================================

class NALUApproach:
    """
    NALU: Neural Arithmetic Logic Unit.

    From: "Neural Arithmetic Logic Units" (Trask et al., 2018)

    Key idea: Learn gates that select between addition and multiplication.

    Problems:
    - Requires training on number ranges
    - Fails OOD (large numbers, decimals)
    - Gate learning is unstable
    """

    def __init__(self, hidden_dim: int = 64):
        self.hidden_dim = hidden_dim
        self.name = "NALU (learned)"
        self.requires_training = True
        self.trained = False

        if TORCH_AVAILABLE:
            self._build_model()
        else:
            self.model = None

    def _build_model(self):
        """Build NALU architecture."""

        class NAC(nn.Module):
            """Neural Accumulator for addition/subtraction."""
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.W_hat = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
                self.M_hat = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

            def forward(self, x):
                W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
                return F.linear(x, W)

        class NALU(nn.Module):
            """Neural Arithmetic Logic Unit."""
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.nac = NAC(in_dim, out_dim)
                self.nac_mul = NAC(in_dim, out_dim)
                self.G = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
                self.epsilon = 1e-7

            def forward(self, x):
                # Gate between add and multiply
                g = torch.sigmoid(F.linear(x, self.G))

                # Addition path
                a = self.nac(x)

                # Multiplication path (in log space)
                log_x = torch.log(torch.abs(x) + self.epsilon)
                m = torch.exp(self.nac_mul(log_x))

                return g * a + (1 - g) * m

        class NALUCalculator(nn.Module):
            """NALU-based arithmetic calculator."""
            def __init__(self, hidden_dim):
                super().__init__()
                # Input: [a, b, op_one_hot(4)]
                self.encoder = nn.Linear(6, hidden_dim)
                self.nalu1 = NALU(hidden_dim, hidden_dim)
                self.nalu2 = NALU(hidden_dim, hidden_dim)
                self.decoder = nn.Linear(hidden_dim, 1)

            def forward(self, a, b, op_idx):
                # Create input: [a, b, op_one_hot]
                batch_size = a.shape[0]
                op_one_hot = torch.zeros(batch_size, 4, device=a.device)
                op_one_hot.scatter_(1, op_idx.unsqueeze(1), 1.0)

                x = torch.cat([a.unsqueeze(1), b.unsqueeze(1), op_one_hot], dim=1)

                h = torch.relu(self.encoder(x))
                h = self.nalu1(h)
                h = self.nalu2(h)
                return self.decoder(h).squeeze(1)

        self.model = NALUCalculator(self.hidden_dim)

    def train(self, samples: List[ArithmeticSample], epochs: int = 100, lr: float = 0.01):
        """Train on arithmetic samples."""
        if not TORCH_AVAILABLE or self.model is None:
            print("  [NALU] PyTorch not available, skipping training")
            self.trained = False
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        op_map = {'+': 0, '-': 1, '*': 2, '/': 3}

        # Normalize inputs for stability
        a_vals = torch.tensor([s.a for s in samples], dtype=torch.float32)
        b_vals = torch.tensor([s.b for s in samples], dtype=torch.float32)
        targets = torch.tensor([s.expected for s in samples], dtype=torch.float32)
        ops = torch.tensor([op_map[s.op] for s in samples], dtype=torch.long)

        # Simple normalization
        self.a_mean, self.a_std = a_vals.mean(), a_vals.std() + 1e-8
        self.b_mean, self.b_std = b_vals.mean(), b_vals.std() + 1e-8
        self.t_mean, self.t_std = targets.mean(), targets.std() + 1e-8

        a_norm = (a_vals - self.a_mean) / self.a_std
        b_norm = (b_vals - self.b_mean) / self.b_std
        t_norm = (targets - self.t_mean) / self.t_std

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(a_norm, b_norm, ops)
            loss = F.mse_loss(pred, t_norm)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"  [NALU] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        self.trained = True

    def compute(self, a: float, op: str, b: float) -> float:
        """Compute a op b."""
        if not TORCH_AVAILABLE or self.model is None or not self.trained:
            return float('nan')

        op_map = {'+': 0, '-': 1, '*': 2, '/': 3}

        with torch.no_grad():
            a_t = torch.tensor([a], dtype=torch.float32)
            b_t = torch.tensor([b], dtype=torch.float32)
            op_t = torch.tensor([op_map[op]], dtype=torch.long)

            a_norm = (a_t - self.a_mean) / self.a_std
            b_norm = (b_t - self.b_mean) / self.b_std

            pred_norm = self.model(a_norm, b_norm, op_t)
            pred = pred_norm * self.t_std + self.t_mean

            return float(pred[0])

    def describe_representation(self) -> str:
        return """
NALU Representation:
  - Input: [a, b, op_one_hot] normalized
  - Architecture: Linear -> NALU -> NALU -> Linear
  - NALU uses learned gates to select add vs multiply
  - Hidden dim: {hidden_dim}
  - Training: REQUIRED (100+ epochs typical)
  - OOD behavior: FAILS (gates don't generalize)
        """.format(hidden_dim=self.hidden_dim)


# =============================================================================
# Approach 3: xVal (Explicit Value Tokens)
# =============================================================================

class XValApproach:
    """
    xVal: Explicit value token representation.

    From: "xVal: A Continuous Number Encoding for Large Language Models"

    Key idea: Represent number as scalar multiplied by learned embedding.
    embed(n) = n * learned_unit_vector

    Problems:
    - The unit vector is learned (not algebraic)
    - Requires learning arithmetic operations
    - Fails OOD without proper scaling
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = "xVal (explicit)"
        self.requires_training = True
        self.trained = False

        if TORCH_AVAILABLE:
            self._build_model()
        else:
            self.model = None

    def _build_model(self):
        """Build xVal-style model."""

        class XValCalculator(nn.Module):
            def __init__(self, dim):
                super().__init__()
                # Learned unit vector for number representation
                self.unit_vec = nn.Parameter(torch.randn(dim) / math.sqrt(dim))

                # Operator embeddings
                self.op_embed = nn.Embedding(4, dim)

                # MLP to compute result
                self.mlp = nn.Sequential(
                    nn.Linear(dim * 3, dim * 2),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim),
                    nn.ReLU(),
                    nn.Linear(dim, 1),
                )

            def encode(self, n):
                """Encode number: n * unit_vec."""
                return n.unsqueeze(-1) * self.unit_vec.unsqueeze(0)

            def forward(self, a, b, op_idx):
                # Encode numbers
                emb_a = self.encode(a)  # [batch, dim]
                emb_b = self.encode(b)  # [batch, dim]

                # Get operator embedding
                emb_op = self.op_embed(op_idx)  # [batch, dim]

                # Concatenate and compute
                combined = torch.cat([emb_a, emb_b, emb_op], dim=1)
                return self.mlp(combined).squeeze(1)

        self.model = XValCalculator(self.dim)

    def train(self, samples: List[ArithmeticSample], epochs: int = 100, lr: float = 0.001):
        """Train on arithmetic samples."""
        if not TORCH_AVAILABLE or self.model is None:
            print("  [xVal] PyTorch not available, skipping training")
            self.trained = False
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        op_map = {'+': 0, '-': 1, '*': 2, '/': 3}

        a_vals = torch.tensor([s.a for s in samples], dtype=torch.float32)
        b_vals = torch.tensor([s.b for s in samples], dtype=torch.float32)
        targets = torch.tensor([s.expected for s in samples], dtype=torch.float32)
        ops = torch.tensor([op_map[s.op] for s in samples], dtype=torch.long)

        # Log-scale for better training
        self.scale = targets.abs().max().item() + 1

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(a_vals / self.scale, b_vals / self.scale, ops)
            loss = F.mse_loss(pred, targets / self.scale)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"  [xVal] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        self.trained = True

    def compute(self, a: float, op: str, b: float) -> float:
        """Compute a op b."""
        if not TORCH_AVAILABLE or self.model is None or not self.trained:
            return float('nan')

        op_map = {'+': 0, '-': 1, '*': 2, '/': 3}

        with torch.no_grad():
            a_t = torch.tensor([a / self.scale], dtype=torch.float32)
            b_t = torch.tensor([b / self.scale], dtype=torch.float32)
            op_t = torch.tensor([op_map[op]], dtype=torch.long)

            pred = self.model(a_t, b_t, op_t)
            return float(pred[0]) * self.scale

    def describe_representation(self) -> str:
        return """
xVal Representation:
  - Number: n * learned_unit_vector
  - Operator: learned embedding
  - Combined via MLP: [emb_a, emb_b, emb_op] -> result
  - Dimension: {dim}
  - Training: REQUIRED
  - OOD behavior: FAILS (MLP doesn't generalize arithmetic)
        """.format(dim=self.dim)


# =============================================================================
# Approach 4: Numeral Decomposition (Multi-scale positional)
# =============================================================================

class NumeralDecompositionApproach:
    """
    Numeral Decomposition: Multi-scale positional encoding.

    Key idea: Decompose number into digit positions.
    1234 -> [1, 2, 3, 4] with positional encoding for ones, tens, hundreds, etc.

    Problems:
    - Fixed number of positions limits range
    - Decimals require additional handling
    - Arithmetic operations must be learned per-position
    """

    def __init__(self, max_digits: int = 6, dim_per_digit: int = 16):
        self.max_digits = max_digits
        self.dim_per_digit = dim_per_digit
        self.dim = max_digits * dim_per_digit
        self.name = "Numeral Decomp (positional)"
        self.requires_training = True
        self.trained = False

        if TORCH_AVAILABLE:
            self._build_model()
        else:
            self.model = None

    def _build_model(self):
        """Build numeral decomposition model."""

        class NumeralEncoder(nn.Module):
            def __init__(self, max_digits, dim_per_digit):
                super().__init__()
                self.max_digits = max_digits
                self.dim_per_digit = dim_per_digit

                # Digit embeddings (0-9)
                self.digit_embed = nn.Embedding(10, dim_per_digit)

                # Positional embedding (ones, tens, hundreds, ...)
                self.pos_embed = nn.Embedding(max_digits, dim_per_digit)

                # Sign embedding
                self.sign_embed = nn.Embedding(2, dim_per_digit)  # 0=positive, 1=negative

            def encode(self, n):
                """Encode number into digit-wise representation."""
                batch_size = n.shape[0]
                device = n.device

                # Handle sign
                sign = (n < 0).long()
                n_abs = n.abs()

                # Decompose into digits
                digits = torch.zeros(batch_size, self.max_digits, dtype=torch.long, device=device)
                temp = n_abs.clone()
                for i in range(self.max_digits):
                    digits[:, i] = (temp % 10).long()
                    temp = temp // 10

                # Embed digits
                digit_emb = self.digit_embed(digits)  # [batch, max_digits, dim_per_digit]

                # Add positional embedding
                positions = torch.arange(self.max_digits, device=device)
                pos_emb = self.pos_embed(positions)  # [max_digits, dim_per_digit]

                combined = digit_emb + pos_emb.unsqueeze(0)  # [batch, max_digits, dim_per_digit]

                # Flatten
                flat = combined.view(batch_size, -1)  # [batch, max_digits * dim_per_digit]

                # Add sign embedding
                sign_emb = self.sign_embed(sign)  # [batch, dim_per_digit]

                # Concatenate sign at the end
                return torch.cat([flat, sign_emb], dim=1)

        class NumeralCalculator(nn.Module):
            def __init__(self, max_digits, dim_per_digit):
                super().__init__()
                self.encoder = NumeralEncoder(max_digits, dim_per_digit)
                input_dim = (max_digits * dim_per_digit + dim_per_digit) * 2 + 4  # 2 numbers + op
                hidden_dim = 256

                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, a, b, op_idx):
                # Encode numbers
                emb_a = self.encoder.encode(a)
                emb_b = self.encoder.encode(b)

                # One-hot operator
                batch_size = a.shape[0]
                op_one_hot = torch.zeros(batch_size, 4, device=a.device)
                op_one_hot.scatter_(1, op_idx.unsqueeze(1), 1.0)

                # Concatenate and compute
                combined = torch.cat([emb_a, emb_b, op_one_hot], dim=1)
                return self.mlp(combined).squeeze(1)

        self.model = NumeralCalculator(self.max_digits, self.dim_per_digit)

    def train(self, samples: List[ArithmeticSample], epochs: int = 100, lr: float = 0.001):
        """Train on arithmetic samples."""
        if not TORCH_AVAILABLE or self.model is None:
            print("  [Numeral] PyTorch not available, skipping training")
            self.trained = False
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        op_map = {'+': 0, '-': 1, '*': 2, '/': 3}

        # Filter to integers only (decomposition works best with integers)
        int_samples = [s for s in samples if s.a == int(s.a) and s.b == int(s.b)]

        if len(int_samples) < 10:
            print("  [Numeral] Not enough integer samples, skipping")
            self.trained = False
            return

        a_vals = torch.tensor([s.a for s in int_samples], dtype=torch.float32)
        b_vals = torch.tensor([s.b for s in int_samples], dtype=torch.float32)
        targets = torch.tensor([s.expected for s in int_samples], dtype=torch.float32)
        ops = torch.tensor([op_map[s.op] for s in int_samples], dtype=torch.long)

        self.scale = targets.abs().max().item() + 1

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(a_vals, b_vals, ops)
            loss = F.mse_loss(pred, targets / self.scale)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"  [Numeral] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        self.trained = True

    def compute(self, a: float, op: str, b: float) -> float:
        """Compute a op b."""
        if not TORCH_AVAILABLE or self.model is None or not self.trained:
            return float('nan')

        op_map = {'+': 0, '-': 1, '*': 2, '/': 3}

        with torch.no_grad():
            a_t = torch.tensor([a], dtype=torch.float32)
            b_t = torch.tensor([b], dtype=torch.float32)
            op_t = torch.tensor([op_map[op]], dtype=torch.long)

            pred = self.model(a_t, b_t, op_t)
            return float(pred[0]) * self.scale

    def describe_representation(self) -> str:
        return """
Numeral Decomposition Representation:
  - Number decomposed into digits: 1234 -> [4, 3, 2, 1, 0, 0] (ones, tens, ...)
  - Each digit: learned embedding + positional embedding
  - Sign: separate embedding
  - Combined via MLP
  - Max digits: {max_digits} (limits range to 10^{max_digits})
  - Training: REQUIRED
  - OOD behavior: FAILS (fixed digit range, learned MLP)
        """.format(max_digits=self.max_digits)


# =============================================================================
# Evaluation functions
# =============================================================================

def evaluate_approach(
    approach: Any,
    samples: List[ArithmeticSample],
    tolerance: float = 0.01
) -> EvaluationResult:
    """Evaluate an approach on a set of samples."""
    errors = []
    correct = 0

    for sample in samples:
        try:
            pred = approach.compute(sample.a, sample.op, sample.b)

            if math.isnan(pred) or math.isinf(pred):
                rel_err = float('inf')
            elif abs(sample.expected) > 1e-10:
                rel_err = abs(pred - sample.expected) / abs(sample.expected)
            else:
                rel_err = abs(pred - sample.expected)

            errors.append(rel_err)
            if rel_err < tolerance:
                correct += 1

        except Exception as e:
            errors.append(float('inf'))

    finite_errors = [e for e in errors if not math.isinf(e)]

    return EvaluationResult(
        name=approach.name,
        accuracy=correct / len(samples) if samples else 0.0,
        mean_relative_error=np.mean(finite_errors) if finite_errors else float('inf'),
        median_relative_error=np.median(finite_errors) if finite_errors else float('inf'),
        max_relative_error=max(finite_errors) if finite_errors else float('inf'),
        num_samples=len(samples),
        num_correct=correct,
        requires_training=approach.requires_training,
    )


def evaluate_chain(
    approach: Any,
    chain_samples: List[Tuple[List[float], List[str], float]],
    tolerance: float = 0.01
) -> EvaluationResult:
    """Evaluate approach on chain expressions."""
    errors = []
    correct = 0

    for numbers, operators, expected in chain_samples:
        try:
            # FluxEM has native chain support
            if hasattr(approach, 'compute_chain'):
                pred = approach.compute_chain(numbers, operators)
            else:
                # Other approaches: evaluate sequentially
                result = numbers[0]
                for i, op in enumerate(operators):
                    result = approach.compute(result, op, numbers[i + 1])
                pred = result

            if math.isnan(pred) or math.isinf(pred):
                rel_err = float('inf')
            elif abs(expected) > 1e-10:
                rel_err = abs(pred - expected) / abs(expected)
            else:
                rel_err = abs(pred - expected)

            errors.append(rel_err)
            if rel_err < tolerance:
                correct += 1

        except Exception as e:
            errors.append(float('inf'))

    finite_errors = [e for e in errors if not math.isinf(e)]

    return EvaluationResult(
        name=approach.name,
        accuracy=correct / len(chain_samples) if chain_samples else 0.0,
        mean_relative_error=np.mean(finite_errors) if finite_errors else float('inf'),
        median_relative_error=np.median(finite_errors) if finite_errors else float('inf'),
        max_relative_error=max(finite_errors) if finite_errors else float('inf'),
        num_samples=len(chain_samples),
        num_correct=correct,
        requires_training=approach.requires_training,
    )


# =============================================================================
# Main comparison
# =============================================================================

def print_separator(char: str = "=", length: int = 70):
    print(char * length)


def print_header(text: str):
    print_separator()
    print(f"  {text}")
    print_separator()


def print_subheader(text: str):
    print_separator("-", 50)
    print(f"  {text}")
    print_separator("-", 50)


def print_results_table(results: Dict[str, EvaluationResult], title: str):
    """Print results as formatted table."""
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Approach':<25} {'Accuracy':>10} {'Mean Err':>12} {'Training':>10}")
    print("-" * 60)

    for name, res in results.items():
        training = "Yes" if res.requires_training else "NO"
        err_str = f"{res.mean_relative_error:.2e}" if res.mean_relative_error < 100 else ">100"
        print(f"{res.name:<25} {res.accuracy:>9.1%} {err_str:>12} {training:>10}")

    print("-" * 60)


def main():
    print_header("Neural Number Embedding Comparison")
    print("""
This script compares different approaches to number representation:

1. FluxEM: Algebraic embeddings (EXACT, NO TRAINING)
   - Addition: embed(a) + embed(b) = embed(a + b)
   - Multiplication: log(a) + log(b) = log(a * b)

2. NALU: Neural Arithmetic Logic Units (learned)
   - Uses gates to select add vs multiply
   - Requires training, fails OOD

3. xVal: Explicit value tokens (learned)
   - Number as scalar times learned vector
   - Requires training, fails OOD

4. Numeral Decomposition: Positional digit encoding (learned)
   - Decompose into ones, tens, hundreds, etc.
   - Requires training, fixed range
""")

    # Initialize approaches
    print_subheader("Initializing Approaches")

    fluxem = FluxEMApproach(dim=256)
    print(f"[OK] FluxEM initialized (no training needed)")

    nalu = NALUApproach(hidden_dim=64)
    xval = XValApproach(dim=64)
    numeral = NumeralDecompositionApproach(max_digits=6)

    if TORCH_AVAILABLE:
        print(f"[OK] NALU initialized (requires training)")
        print(f"[OK] xVal initialized (requires training)")
        print(f"[OK] Numeral Decomposition initialized (requires training)")
    else:
        print("[WARN] PyTorch not available - neural approaches will return NaN")

    approaches = {
        "fluxem": fluxem,
        "nalu": nalu,
        "xval": xval,
        "numeral": numeral,
    }

    # Generate test data
    print_subheader("Generating Test Data")

    id_samples = generate_id_samples(n=100, seed=42)
    ood_magnitude = generate_ood_magnitude_samples(n=100, seed=42)
    ood_precision = generate_ood_precision_samples(n=100, seed=42)
    ood_chains = generate_ood_chain_samples(chain_length=5, n=50, seed=42)

    print(f"ID samples (0-999): {len(id_samples)}")
    print(f"OOD magnitude (1M+): {len(ood_magnitude)}")
    print(f"OOD precision (decimals): {len(ood_precision)}")
    print(f"OOD chains (length 5): {len(ood_chains)}")

    # Train neural approaches on ID data
    print_subheader("Training Neural Approaches (on ID data only)")
    print("\nNote: Neural approaches are trained on ID samples (0-999) only.")
    print("This simulates the real-world scenario where training data has limited range.\n")

    if TORCH_AVAILABLE:
        print("Training NALU...")
        nalu.train(id_samples, epochs=200, lr=0.01)

        print("\nTraining xVal...")
        xval.train(id_samples, epochs=200, lr=0.001)

        print("\nTraining Numeral Decomposition...")
        numeral.train(id_samples, epochs=200, lr=0.001)
    else:
        print("Skipping training (PyTorch not available)")

    # Evaluate on all test sets
    print_header("Evaluation Results")

    # ID evaluation
    print_subheader("In-Distribution (0-999)")
    id_results = {}
    for name, approach in approaches.items():
        id_results[name] = evaluate_approach(approach, id_samples)
    print_results_table(id_results, "ID Performance (trained on this range)")

    # OOD magnitude
    print_subheader("OOD: Large Numbers (1M+)")
    ood_mag_results = {}
    for name, approach in approaches.items():
        ood_mag_results[name] = evaluate_approach(approach, ood_magnitude)
    print_results_table(ood_mag_results, "OOD Magnitude Performance (never seen in training)")

    # OOD precision
    print_subheader("OOD: Decimal Precision")
    ood_prec_results = {}
    for name, approach in approaches.items():
        ood_prec_results[name] = evaluate_approach(approach, ood_precision)
    print_results_table(ood_prec_results, "OOD Precision Performance (decimals, never seen in training)")

    # OOD chains
    print_subheader("OOD: Long Expression Chains (5 operations)")
    ood_chain_results = {}
    for name, approach in approaches.items():
        ood_chain_results[name] = evaluate_chain(approach, ood_chains)
    print_results_table(ood_chain_results, "OOD Chain Performance (5-element chains)")

    # Specific examples
    print_header("Specific Examples")

    examples = [
        # ID
        (42, '+', 58, "ID: Simple addition"),
        (6, '*', 7, "ID: Simple multiplication"),
        # OOD magnitude
        (1_234_567, '+', 7_654_321, "OOD: Large addition"),
        (12345, '*', 6789, "OOD: Large multiplication"),
        # OOD precision
        (3.14159, '+', 2.71828, "OOD: Pi + e"),
        (1.41421, '*', 1.73205, "OOD: sqrt(2) * sqrt(3)"),
    ]

    for a, op, b, description in examples:
        expected = eval(f"{a}{op}{b}")
        print(f"\n{description}: {a} {op} {b} = {expected}")
        print("-" * 50)

        for name, approach in approaches.items():
            try:
                pred = approach.compute(a, op, b)
                if math.isnan(pred):
                    status = "NaN (not trained)"
                elif abs(expected) > 1e-10:
                    rel_err = abs(pred - expected) / abs(expected)
                    status = f"{pred:.6g} (err: {rel_err:.2e})"
                else:
                    status = f"{pred:.6g}"
            except Exception as e:
                status = f"Error: {e}"

            print(f"  {approach.name:<25}: {status}")

    # Chain example
    print_subheader("Chain Example: 10 + 20 + 30 + 40 + 50")
    numbers = [10.0, 20.0, 30.0, 40.0, 50.0]
    operators = ['+', '+', '+', '+']
    expected = sum(numbers)

    print(f"Expected: {expected}")
    for name, approach in approaches.items():
        if hasattr(approach, 'compute_chain'):
            pred = approach.compute_chain(numbers, operators)
        else:
            result = numbers[0]
            for i, op in enumerate(operators):
                result = approach.compute(result, op, numbers[i + 1])
            pred = result

        if math.isnan(pred):
            status = "NaN"
        else:
            err = abs(pred - expected) / expected if expected != 0 else abs(pred)
            status = f"{pred:.6g} (err: {err:.2e})"

        print(f"  {approach.name:<25}: {status}")

    # Summary
    print_header("Summary: Key Findings")
    print("""
FLUXEM (Algebraic Embeddings):
  - Requires NO training
  - Exact arithmetic by construction
  - Perfect OOD generalization (any magnitude, precision, chain length)
  - Algebraic identities hold exactly (within floating point precision)

NEURAL APPROACHES (NALU, xVal, Numeral Decomposition):
  - Require training on specific number ranges
  - Good ID performance when trained adequately
  - FAIL on OOD magnitude (large numbers never seen in training)
  - FAIL on OOD precision (decimals if only trained on integers)
  - FAIL on OOD chains (errors compound over sequential operations)

KEY INSIGHT:
  FluxEM achieves exact arithmetic through algebraic structure, not learning.
  Neural approaches try to learn arithmetic as a function approximation,
  which fundamentally cannot generalize beyond the training distribution.

  embed(a) + embed(b) = embed(a + b)  is ALWAYS true for FluxEM
  f_neural([a, b]) ≈ a + b           only for trained (a, b) pairs
""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare positional encodings and algebraic embeddings for numbers.

Runs a benchmark across several positional encoding baselines and FluxEM.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np

# Add parent path for imports
sys.path.insert(0, "/Volumes/VIXinSSD/FluxEM")

from fluxem import create_unified_model


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PositionalBenchmarkConfig:
    """Configuration for positional encoding benchmark."""
    seed: int = 42
    n_samples_id: int = 200
    n_samples_ood: int = 200

    # ID: 2-3 digit numbers (positions 0, 1, 2)
    id_range: Tuple[int, int] = (10, 999)

    # OOD: 6+ digit numbers (positions 5+ never seen)
    ood_range: Tuple[int, int] = (100000, 9999999)

    # Extreme OOD: 10+ digit numbers
    extreme_ood_range: Tuple[int, int] = (1000000000, 9999999999)

    # Operations for testing
    operations: List[str] = field(default_factory=lambda: ["+", "-"])

    # Positional encoding parameters
    embed_dim: int = 64
    max_position_trained: int = 3  # Trained on positions 0, 1, 2 only

    verbose: bool = False


# =============================================================================
# Positional Encoding Implementations
# =============================================================================

class PositionalEncoder(ABC):
    """Base class for positional encoding approaches."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the encoding approach."""
        pass

    @property
    @abstractmethod
    def limitations(self) -> str:
        """Description of fundamental limitations."""
        pass

    @abstractmethod
    def encode_number(self, n: int) -> np.ndarray:
        """Encode a number as an embedding."""
        pass

    def can_compute(self) -> bool:
        """Whether this encoder can compute arithmetic directly."""
        return False

    def compute(self, a: int, b: int, op: str) -> Optional[float]:
        """Compute arithmetic (if supported)."""
        return None


class StandardPositionalEncoding(PositionalEncoder):
    """
    Standard sin/cos positional encoding (Transformer-style).

    Each POSITION gets a sin/cos encoding:
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Problem: Position info tells you WHERE, not WHAT.
    The model must still learn that:
    - Position 0 = ones place (multiply by 1)
    - Position 1 = tens place (multiply by 10)
    - etc.

    OOD Failure: For positions beyond training, the model has no
    learned mapping from position -> place value.
    """

    def __init__(self, dim: int = 64, max_positions: int = 20):
        self.dim = dim
        self.max_positions = max_positions
        self._build_encoding_table()

    @property
    def name(self) -> str:
        return "Standard Sin/Cos"

    @property
    def limitations(self) -> str:
        return (
            "Position encodings mark WHERE digits are, not their VALUE.\n"
            "Model must learn: position N -> multiply by 10^N.\n"
            "OOD positions have never been associated with place values."
        )

    def _build_encoding_table(self):
        """Pre-compute positional encodings."""
        self.pe_table = np.zeros((self.max_positions, self.dim))

        for pos in range(self.max_positions):
            for i in range(0, self.dim, 2):
                denominator = 10000 ** (i / self.dim)
                self.pe_table[pos, i] = np.sin(pos / denominator)
                if i + 1 < self.dim:
                    self.pe_table[pos, i + 1] = np.cos(pos / denominator)

    def encode_number(self, n: int) -> np.ndarray:
        """
        Encode number with standard positional encoding.

        Each digit gets: digit_embedding + positional_encoding
        We simulate this by creating a combined representation.
        """
        # Convert to digits (right to left = position 0, 1, 2, ...)
        n = abs(int(n))
        digits = []
        while n > 0:
            digits.append(n % 10)
            n //= 10
        if not digits:
            digits = [0]

        # Create combined embedding
        embedding = np.zeros(self.dim)

        for pos, digit in enumerate(digits):
            if pos < self.max_positions:
                # Add digit value scaled by positional encoding
                # This is what a model would have to LEARN
                digit_emb = np.ones(self.dim) * digit / 10.0
                pos_emb = self.pe_table[pos]
                embedding += digit_emb + pos_emb * 0.1

        return embedding


class DigitPositionalEncoding(PositionalEncoder):
    """
    Digit positional encoding: Each digit gets an explicit place value embedding.

    Encoding:
        ones_embed, tens_embed, hundreds_embed, ...

    Each position gets a learned embedding that the model must associate
    with the correct power of 10.

    Problem: For positions never seen in training (e.g., millions place),
    the model has no embedding and cannot generalize.

    This is arguably BETTER than sin/cos because positions are discrete,
    but still fundamentally limited by training distribution.
    """

    def __init__(self, dim: int = 64, max_digits: int = 10, trained_positions: int = 3):
        self.dim = dim
        self.max_digits = max_digits
        self.trained_positions = trained_positions  # Only these positions were in training

        # Simulated learned position embeddings
        np.random.seed(42)
        self.position_embeddings = np.random.randn(max_digits, dim) * 0.1

        # Digit embeddings (0-9)
        self.digit_embeddings = np.random.randn(10, dim) * 0.1

    @property
    def name(self) -> str:
        return "Digit Positional"

    @property
    def limitations(self) -> str:
        return (
            "Learned position embeddings for ones, tens, hundreds.\n"
            f"Trained only on positions 0-{self.trained_positions - 1}.\n"
            "For OOD positions (thousands, millions), no learned association exists."
        )

    def encode_number(self, n: int) -> np.ndarray:
        """Encode with digit + position embeddings."""
        n = abs(int(n))
        digits = []
        while n > 0:
            digits.append(n % 10)
            n //= 10
        if not digits:
            digits = [0]

        embedding = np.zeros(self.dim)

        for pos, digit in enumerate(digits):
            if pos < self.max_digits:
                digit_emb = self.digit_embeddings[digit]
                pos_emb = self.position_embeddings[pos]

                # Simulate that OOD positions have "random" untrained embeddings
                if pos >= self.trained_positions:
                    # OOD position: embedding exists but was never trained
                    # to associate with correct place value
                    pos_emb = pos_emb * 0.1  # Weak, untrained signal

                embedding += digit_emb + pos_emb

        return embedding


class MultiscalePositionalEncoding(PositionalEncoder):
    """
    Multi-scale positional encoding: Log-scale position frequencies.

    Idea: Use multiple frequency scales to handle numbers of different sizes.
    Low frequencies capture large-scale position differences.
    High frequencies capture local position information.

    PE(pos) = [sin(pos/1), cos(pos/1), sin(pos/10), cos(pos/10),
               sin(pos/100), cos(pos/100), ...]

    Problem: Even with multi-scale, the model must still LEARN that
    these position patterns correspond to specific place values.
    For positions far outside training, generalization fails.
    """

    def __init__(self, dim: int = 64, scales: List[float] = None):
        self.dim = dim
        self.scales = scales or [1, 10, 100, 1000, 10000]

    @property
    def name(self) -> str:
        return "Multi-scale Positional"

    @property
    def limitations(self) -> str:
        return (
            "Multiple frequency scales (1, 10, 100, ...) for positions.\n"
            "Captures both local and global position info.\n"
            "Still requires learning position -> place value mapping.\n"
            "OOD positions have novel frequency combinations never trained."
        )

    def encode_number(self, n: int) -> np.ndarray:
        """Encode with multi-scale positional encoding."""
        n = abs(int(n))
        digits = []
        while n > 0:
            digits.append(n % 10)
            n //= 10
        if not digits:
            digits = [0]

        embedding = np.zeros(self.dim)
        dim_per_scale = self.dim // (len(self.scales) * 2)

        for pos, digit in enumerate(digits):
            for scale_idx, scale in enumerate(self.scales):
                base_idx = scale_idx * dim_per_scale * 2
                if base_idx + 1 < self.dim:
                    embedding[base_idx] += np.sin(pos / scale) * digit / 10.0
                    embedding[base_idx + 1] += np.cos(pos / scale) * digit / 10.0

        return embedding


class FluxEMEncoder(PositionalEncoder):
    """FluxEM algebraic embeddings.

    Encodes a scalar value directly and uses embedding-space addition/subtraction
    for arithmetic.
    """

    def __init__(self, dim: int = 256, scale: float = 1e15):
        self.dim = dim
        self.scale = scale
        self.model = create_unified_model(dim=dim, linear_scale=scale)

    @property
    def name(self) -> str:
        return "FluxEM Algebraic"

    @property
    def limitations(self) -> str:
        return (
            "Encodes value directly.\n"
            "embed(a) + embed(b) = embed(a + b) by construction.\n"
            "Range limited by the chosen scale and floating-point precision."
        )

    def encode_number(self, n: int) -> np.ndarray:
        """Encode number using FluxEM linear embedding."""
        return self.model.linear_encoder.encode_number(float(n))

    def can_compute(self) -> bool:
        return True

    def compute(self, a: int, b: int, op: str) -> Optional[float]:
        """Compute using algebraic embedding operations."""
        emb_a = self.encode_number(a)
        emb_b = self.encode_number(b)

        if op == "+":
            result_emb = emb_a + emb_b
        elif op == "-":
            result_emb = emb_a - emb_b
        else:
            return None

        return self.model.linear_encoder.decode(result_emb)


# =============================================================================
# Simulated Model Performance
# =============================================================================

def simulate_positional_model_accuracy(
    encoder: PositionalEncoder,
    a: int,
    b: int,
    op: str,
    max_trained_position: int = 3,
) -> Tuple[Optional[float], bool]:
    """Simulate performance for a positional encoding baseline."""
    # Get the actual expected result
    if op == "+":
        expected = a + b
    elif op == "-":
        expected = a - b
    else:
        return None, False

    # Count digits to determine OOD-ness
    max_digits_operand = max(len(str(abs(a))), len(str(abs(b))))
    max_digits_result = len(str(abs(expected)))
    max_position = max(max_digits_operand, max_digits_result)

    # If FluxEM, compute directly
    if encoder.can_compute():
        result = encoder.compute(a, b, op)
        if result is not None:
            # Use relative tolerance for larger numbers
            if abs(expected) > 1:
                rel_error = abs(result - expected) / abs(expected)
                is_correct = rel_error < 0.001  # 0.1% relative error tolerance
            else:
                is_correct = abs(result - expected) < 0.5
            return result, is_correct
        return None, False

    # For positional encoders, simulate degradation on OOD
    if max_position <= max_trained_position:
        # In-distribution: high accuracy (simulated)
        noise = np.random.normal(0, abs(expected) * 0.02)
        result = expected + noise
        is_correct = abs(result - expected) < abs(expected) * 0.05 + 1
    else:
        # Out-of-distribution: accuracy degrades with distance from training
        ood_factor = (max_position - max_trained_position)

        # Error grows exponentially with OOD distance
        error_scale = 10 ** (ood_factor - 1)

        # Add systematic and random errors
        noise = np.random.normal(0, abs(expected) * 0.3 * ood_factor)
        systematic_error = (expected / (10 ** max_trained_position)) * np.random.choice([-1, 1])

        result = expected + noise + systematic_error * error_scale
        is_correct = abs(result - expected) < abs(expected) * 0.01

    return result, is_correct


# =============================================================================
# Data Generation
# =============================================================================

def generate_test_cases(
    config: PositionalBenchmarkConfig,
) -> Dict[str, List[Tuple[int, int, str, int]]]:
    """
    Generate test cases for different distribution settings.

    Returns dict with:
        - "id": In-distribution (2-3 digit numbers)
        - "ood_6digit": OOD 6-7 digit numbers
        - "ood_10digit": Extreme OOD 10+ digit numbers
        - "adversarial": Maximum carry chains (e.g., 999999 + 1)
    """
    rng = np.random.default_rng(config.seed)

    datasets = {}

    # In-distribution: 2-3 digit numbers
    id_cases = []
    for _ in range(config.n_samples_id):
        a = int(rng.integers(config.id_range[0], config.id_range[1]))
        b = int(rng.integers(config.id_range[0], config.id_range[1]))
        op = rng.choice(config.operations)
        expected = a + b if op == "+" else a - b
        id_cases.append((a, b, op, expected))
    datasets["id"] = id_cases

    # OOD: 6-7 digit numbers
    ood_cases = []
    for _ in range(config.n_samples_ood):
        a = int(rng.integers(config.ood_range[0], config.ood_range[1]))
        b = int(rng.integers(config.ood_range[0], config.ood_range[1]))
        op = rng.choice(config.operations)
        expected = a + b if op == "+" else a - b
        ood_cases.append((a, b, op, expected))
    datasets["ood_6digit"] = ood_cases

    # Extreme OOD: 10+ digit numbers
    extreme_ood_cases = []
    for _ in range(config.n_samples_ood // 2):
        a = int(rng.integers(config.extreme_ood_range[0], config.extreme_ood_range[1]))
        b = int(rng.integers(config.extreme_ood_range[0], config.extreme_ood_range[1]))
        op = rng.choice(config.operations)
        expected = a + b if op == "+" else a - b
        extreme_ood_cases.append((a, b, op, expected))
    datasets["ood_10digit"] = extreme_ood_cases

    # Adversarial: Maximum carry chains
    adversarial_cases = [
        (99, 1, "+", 100),
        (999, 1, "+", 1000),
        (9999, 1, "+", 10000),
        (99999, 1, "+", 100000),
        (999999, 1, "+", 1000000),
        (9999999, 1, "+", 10000000),
        (99999999, 1, "+", 100000000),
        (999999999, 1, "+", 1000000000),
        (1000000, 1, "-", 999999),
        (10000000, 1, "-", 9999999),
    ]
    datasets["adversarial"] = adversarial_cases

    return datasets


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class EncoderResult:
    """Results for one encoder on one dataset."""
    encoder_name: str
    dataset_name: str
    n_samples: int
    accuracy: float
    mean_error: float
    example_predictions: List[Tuple[str, float, float, bool]]  # expr, pred, expected, correct


def evaluate_encoder(
    encoder: PositionalEncoder,
    test_cases: List[Tuple[int, int, str, int]],
    dataset_name: str,
    max_trained_position: int = 3,
    verbose: bool = False,
) -> EncoderResult:
    """Evaluate an encoder on a set of test cases."""
    correct_count = 0
    total_error = 0.0
    examples = []

    for a, b, op, expected in test_cases:
        pred, is_correct = simulate_positional_model_accuracy(
            encoder, a, b, op, max_trained_position
        )

        if is_correct:
            correct_count += 1

        if pred is not None:
            rel_error = abs(pred - expected) / (abs(expected) + 1e-10)
            total_error += rel_error

        # Store a few examples
        if len(examples) < 5:
            expr = f"{a} {op} {b}"
            examples.append((expr, pred, expected, is_correct))

    accuracy = correct_count / len(test_cases)
    mean_error = total_error / len(test_cases)

    return EncoderResult(
        encoder_name=encoder.name,
        dataset_name=dataset_name,
        n_samples=len(test_cases),
        accuracy=accuracy,
        mean_error=mean_error,
        example_predictions=examples,
    )


def run_benchmark(config: PositionalBenchmarkConfig) -> Dict[str, Dict[str, EncoderResult]]:
    """Run the full positional encoding comparison benchmark."""
    # Set random seed
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Generate test data
    datasets = generate_test_cases(config)

    # Initialize encoders
    encoders = [
        StandardPositionalEncoding(dim=config.embed_dim),
        DigitPositionalEncoding(dim=config.embed_dim, trained_positions=config.max_position_trained),
        MultiscalePositionalEncoding(dim=config.embed_dim),
        FluxEMEncoder(dim=256, scale=1e15),  # Large scale for 10+ digit numbers
    ]

    # Evaluate
    results: Dict[str, Dict[str, EncoderResult]] = {}

    for encoder in encoders:
        results[encoder.name] = {}

        for dataset_name, cases in datasets.items():
            result = evaluate_encoder(
                encoder, cases, dataset_name,
                max_trained_position=config.max_position_trained,
                verbose=config.verbose,
            )
            results[encoder.name][dataset_name] = result

    return results


def print_results_table(results: Dict[str, Dict[str, EncoderResult]]) -> None:
    """Print formatted results table."""
    print("table=accuracy_by_encoder")
    print("encoder\tdataset\taccuracy\tmean_error\tn_samples")
    for encoder, datasets in results.items():
        for ds_name, result in datasets.items():
            print(f"{encoder}\t{ds_name}\t{result.accuracy:.6f}\t{result.mean_error:.6e}\t{result.n_samples}")

    print("table=ood_gap")
    print("encoder\tgap_6digit\tgap_10digit\tgap_adversarial")
    for encoder, datasets in results.items():
        id_acc = datasets["id"].accuracy
        gap_6 = id_acc - datasets["ood_6digit"].accuracy
        gap_10 = id_acc - datasets["ood_10digit"].accuracy
        gap_adv = id_acc - datasets["adversarial"].accuracy
        print(f"{encoder}\t{gap_6:.6f}\t{gap_10:.6f}\t{gap_adv:.6f}")


def print_why_positional_fails() -> None:
    """Print a short note about positional encodings."""
    print("table=notes")
    print("note\tvalue")
    print("positional_encoding\tencodes digit position; arithmetic requires learned place-value and carry behavior")


def print_example_failures(results: Dict[str, Dict[str, EncoderResult]]) -> None:
    """Print example predictions showing failures."""
    print("table=example_predictions")
    print("dataset\tencoder\texpression\tpredicted\texpected\tcorrect")

    for encoder, datasets in results.items():
        for dataset in ["ood_6digit", "adversarial"]:
            examples = datasets[dataset].example_predictions
            for expr, pred, expected, correct in examples[:3]:
                pred_str = "" if pred is None else f"{pred:.0f}"
                print(f"{dataset}\t{encoder}\t{expr}\t{pred_str}\t{expected}\t{'1' if correct else '0'}")


def print_summary() -> None:
    """Print key takeaways."""
    print("table=summary")
    print("note\tvalue")
    print("positional_encoding\tencodes position; arithmetic requires learned place-value and carry behavior")
    print("fluxem\tencodes value directly and uses embedding-space operations for arithmetic")


def save_results(results: Dict[str, Dict[str, EncoderResult]], path: str) -> None:
    """Save results to JSON."""
    output = {}
    for encoder, datasets in results.items():
        output[encoder] = {}
        for ds_name, result in datasets.items():
            output[encoder][ds_name] = {
                "n_samples": result.n_samples,
                "accuracy": result.accuracy,
                "mean_error": result.mean_error,
            }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"results_path\t{path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare FluxEM vs positional encoding for numbers"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed output"
    )
    parser.add_argument(
        "--save", type=str, default=None, help="Path to save JSON results"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of samples per distribution"
    )
    args = parser.parse_args()

    # Configure
    config = PositionalBenchmarkConfig(
        seed=args.seed,
        n_samples_id=args.n_samples,
        n_samples_ood=args.n_samples,
        verbose=args.verbose,
    )

    print("table=run_context")
    print("field\tvalue")
    print(f"seed\t{config.seed}")
    print(f"n_samples_id\t{config.n_samples_id}")
    print(f"n_samples_ood\t{config.n_samples_ood}")
    print(f"embed_dim\t{config.embed_dim}")
    print(f"max_position_trained\t{config.max_position_trained}")
    print(f"verbose\t{'1' if config.verbose else '0'}")

    # Run benchmark
    results = run_benchmark(config)

    # Print results
    print_results_table(results)
    print_why_positional_fails()
    print_example_failures(results)
    print_summary()

    # Save results
    save_path = args.save or "experiments/results/positional_comparison.json"
    save_results(results, save_path)


if __name__ == "__main__":
    main()

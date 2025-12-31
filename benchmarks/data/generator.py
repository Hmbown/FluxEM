"""Dataset generation for FluxEM benchmark."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .grammar import (
    GrammarConfig,
    OOD_A_CONFIG,
    OOD_B_CONFIG,
    OOD_C_CONFIG,
    TRAIN_CONFIG,
)


@dataclass
class Sample:
    """A single benchmark sample."""

    expression: str
    ground_truth: float
    num_ops: int
    distribution: str
    seed: int


def generate_number(rng: random.Random, config: GrammarConfig) -> int:
    """Generate a random integer within config bounds."""
    return rng.randint(config.min_int, config.max_int)


def generate_expression(
    rng: random.Random,
    config: GrammarConfig,
) -> Tuple[str, int]:
    """
    Generate a random arithmetic expression.

    Returns (expression_string, num_operations).
    """
    num_ops = rng.randint(config.min_ops, config.max_ops)

    # Start with a number
    result = str(generate_number(rng, config))
    op_count = 0

    for i in range(num_ops):
        # Choose operator
        if config.include_power and rng.random() < 0.2:
            op = "**"
            # Small exponents to avoid overflow
            operand = str(rng.randint(config.power_min, config.power_max))
        else:
            op = rng.choice(config.operators)
            operand = str(generate_number(rng, config))

            # Avoid division by zero
            if op == "/" and operand == "0":
                operand = str(rng.randint(1, max(1, config.max_int)))

        # Optionally wrap previous expression in parentheses
        if config.allow_parentheses and rng.random() < 0.3 and op_count > 0:
            result = f"({result})"

        result = f"{result} {op} {operand}"
        op_count += 1

    return result, op_count


def evaluate_expression(expr: str) -> Optional[float]:
    """
    Evaluate expression using Python eval with safety checks.

    Returns None if evaluation fails.
    """
    try:
        # Safe eval with no builtins
        result = eval(expr, {"__builtins__": {}}, {})

        # Validate result
        if not isinstance(result, (int, float)):
            return None
        if abs(result) > 1e15:
            return None
        if result != result:  # NaN check
            return None

        return float(result)
    except (ZeroDivisionError, OverflowError, ValueError, SyntaxError):
        return None


def generate_dataset(
    config: GrammarConfig,
    n_samples: int,
    distribution_name: str,
    seed: int = 42,
) -> List[Sample]:
    """Generate a dataset of arithmetic expressions."""
    rng = random.Random(seed)
    samples = []
    attempts = 0
    max_attempts = n_samples * 20

    while len(samples) < n_samples and attempts < max_attempts:
        sample_seed = seed + attempts
        rng_sample = random.Random(sample_seed)

        expr, num_ops = generate_expression(rng_sample, config)
        result = evaluate_expression(expr)

        if result is not None:
            samples.append(
                Sample(
                    expression=expr,
                    ground_truth=result,
                    num_ops=num_ops,
                    distribution=distribution_name,
                    seed=sample_seed,
                )
            )

        attempts += 1

    return samples


def save_dataset(samples: List[Sample], path: Path) -> None:
    """Save samples as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample)) + "\n")


def load_dataset(path: Path) -> List[Sample]:
    """Load samples from JSONL."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            samples.append(Sample(**data))
    return samples


def generate_all_datasets(
    output_dir: Path,
    train_samples: int = 10_000,
    test_samples: int = 1_000,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generate all train/test datasets."""
    datasets = {}

    print("Generating datasets...")

    # Training set
    print(f"  train: {train_samples} samples")
    train = generate_dataset(TRAIN_CONFIG, train_samples, "train", seed)
    train_path = output_dir / "train.jsonl"
    save_dataset(train, train_path)
    datasets["train"] = train_path

    # In-distribution test
    print(f"  test_id: {test_samples} samples")
    test_id = generate_dataset(TRAIN_CONFIG, test_samples, "test_id", seed + 100_000)
    test_id_path = output_dir / "test_id.jsonl"
    save_dataset(test_id, test_id_path)
    datasets["test_id"] = test_id_path

    # OOD-A: Large integers
    print(f"  ood_a: {test_samples} samples")
    ood_a = generate_dataset(OOD_A_CONFIG, test_samples, "ood_a", seed + 200_000)
    ood_a_path = output_dir / "test_ood_a.jsonl"
    save_dataset(ood_a, ood_a_path)
    datasets["ood_a"] = ood_a_path

    # OOD-B: Longer expressions
    print(f"  ood_b: {test_samples} samples")
    ood_b = generate_dataset(OOD_B_CONFIG, test_samples, "ood_b", seed + 300_000)
    ood_b_path = output_dir / "test_ood_b.jsonl"
    save_dataset(ood_b, ood_b_path)
    datasets["ood_b"] = ood_b_path

    # OOD-C: Mixed with exponentiation
    print(f"  ood_c: {test_samples} samples")
    ood_c = generate_dataset(OOD_C_CONFIG, test_samples, "ood_c", seed + 400_000)
    ood_c_path = output_dir / "test_ood_c.jsonl"
    save_dataset(ood_c, ood_c_path)
    datasets["ood_c"] = ood_c_path

    print(f"All datasets saved to {output_dir}")
    return datasets

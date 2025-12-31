"""Expression grammar configurations for benchmark distributions."""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class GrammarConfig:
    """Configuration for expression generation."""

    # Number ranges
    min_int: int
    max_int: int

    # Operation counts
    min_ops: int
    max_ops: int

    # Allowed operators
    operators: List[Literal["+", "-", "*", "/"]]

    # Whether to include parentheses
    allow_parentheses: bool = True

    # Include exponentiation (OOD-C only)
    include_power: bool = False

    # Power exponent range (only used if include_power=True)
    power_min: int = 2
    power_max: int = 5


# Training distribution
TRAIN_CONFIG = GrammarConfig(
    min_int=0,
    max_int=999,
    min_ops=1,
    max_ops=3,
    operators=["+", "-", "*", "/"],
    allow_parentheses=True,
    include_power=False,
)

# OOD-A: Large integers
OOD_A_CONFIG = GrammarConfig(
    min_int=10_000,
    max_int=1_000_000,
    min_ops=1,
    max_ops=3,
    operators=["+", "-", "*", "/"],
    allow_parentheses=True,
    include_power=False,
)

# OOD-B: Longer expressions
OOD_B_CONFIG = GrammarConfig(
    min_int=0,
    max_int=999,
    min_ops=4,
    max_ops=8,
    operators=["+", "-", "*", "/"],
    allow_parentheses=True,
    include_power=False,
)

# OOD-C: Mixed with exponentiation
OOD_C_CONFIG = GrammarConfig(
    min_int=2,
    max_int=20,
    min_ops=2,
    max_ops=4,
    operators=["+", "-", "*", "/"],
    allow_parentheses=True,
    include_power=True,
    power_min=2,
    power_max=4,
)

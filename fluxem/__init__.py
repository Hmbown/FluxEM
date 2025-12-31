"""
FluxEM: Algebraic Embeddings for Deterministic Arithmetic.

This package provides algebraic number embeddings that encode arithmetic
properties directly into the representation, enabling exact arithmetic
operations in embedding space.

Key Properties:
- Addition/Subtraction: encode(a) +/- encode(b) = encode(a +/- b)
- Multiplication/Division: log_mag(a) +/- log_mag(b) = log_mag(a */ b)
- Powers: log_mag(a^n) = n * log_mag(a)

Example Usage
-------------
>>> from fluxem import UnifiedArithmeticModel, create_unified_model
>>> model = create_unified_model()
>>> model.compute("42+58=")
100.0
>>> model.compute("6*7=")
42.0

For extended operations:
>>> from fluxem import ExtendedOps, create_extended_ops
>>> ops = create_extended_ops()
>>> ops.sqrt(16)
4.0
>>> ops.power(2, 10)
1024.0

Reference: Flux Mathematics textbook, Chapter 8
"""

from .linear import (
    NumberEncoder,
    parse_arithmetic_expression,
    verify_linear_property,
)
from .logarithmic import (
    LogarithmicNumberEncoder,
    verify_multiplication_theorem,
    verify_division_theorem,
)
from .unified import (
    UnifiedArithmeticModel,
    create_unified_model,
    evaluate_all_operations_ood,
)
from .extended import (
    ExtendedOps,
    create_extended_ops,
)

__version__ = "0.1.0"

__all__ = [
    # Linear encoder (addition, subtraction)
    "NumberEncoder",
    "parse_arithmetic_expression",
    "verify_linear_property",
    # Logarithmic encoder (multiplication, division)
    "LogarithmicNumberEncoder",
    "verify_multiplication_theorem",
    "verify_division_theorem",
    # Unified model (all four operations)
    "UnifiedArithmeticModel",
    "create_unified_model",
    "evaluate_all_operations_ood",
    # Extended operations (powers, roots, exp, ln)
    "ExtendedOps",
    "create_extended_ops",
]

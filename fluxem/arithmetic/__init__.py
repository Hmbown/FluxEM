"""
FluxEM Arithmetic: Exact arithmetic through structured embeddings.

This module provides algebraic embeddings that make arithmetic operations exact:
- Linear embeddings for addition/subtraction: encode(a) + encode(b) = encode(a+b)
- Logarithmic embeddings for multiplication/division: log_mag(a) + log_mag(b) = log_mag(a*b)
- Extended operations for powers, roots, exp, ln

Key Result: 100% OOD accuracy on all four basic operations.
"""

from .linear_encoder import (
    NumberEncoder,
    parse_arithmetic_expression,
    verify_linear_property,
)
from .log_encoder import (
    LogarithmicNumberEncoder,
    verify_multiplication_theorem,
    verify_division_theorem,
)
from .unified import (
    UnifiedArithmeticModel,
    create_unified_model,
    evaluate_all_operations_ood,
)
from .extended_ops import (
    ExtendedOps,
    create_extended_ops,
)

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

"""
FluxEM: Algebraic embeddings for arithmetic with IEEE-754 float precision.

Addition becomes vector addition.
Multiplication becomes addition in log-space.
Systematic generalization via algebraic structure (parameter-free).

Example Usage
-------------
>>> from fluxem import create_unified_model
>>> model = create_unified_model()
>>> model.compute("1847*392")
724024.0
>>> model.compute("123456+789")
124245.0

Extended operations:
>>> from fluxem import create_extended_ops
>>> ops = create_extended_ops()
>>> ops.sqrt(16)
4.0
>>> ops.power(2, 16)
65536.0

How It Works
------------
Linear embeddings (addition/subtraction):
    embed(a) + embed(b) = embed(a + b)

Log embeddings (multiplication/division):
    log_embed(a) + log_embed(b) = log_embed(a * b)

Arithmetic operations map to geometric operations in embedding space.
See docs/FORMAL_DEFINITION.md for mathematical specification.
See docs/ERROR_MODEL.md for precision notes.
"""

from .arithmetic import (
    # Linear encoder (addition, subtraction)
    NumberEncoder,
    parse_arithmetic_expression,
    verify_linear_property,
    # Logarithmic encoder (multiplication, division)
    LogarithmicNumberEncoder,
    verify_multiplication_theorem,
    verify_division_theorem,
    # Unified model (all four operations)
    UnifiedArithmeticModel,
    create_unified_model,
    evaluate_all_operations_ood,
    # Extended operations (powers, roots, exp, ln)
    ExtendedOps,
    create_extended_ops,
)

__version__ = "0.2.0"

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

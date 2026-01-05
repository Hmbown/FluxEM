"""
Number Theory Module for FluxEM-Domains

Provides embeddings for number-theoretic objects and operations.
Operations use integer arithmetic and standard algorithms.

Supports:
- Integers with arbitrary-precision arithmetic
- Prime numbers and factorization
- Divisibility and GCD/LCM
- Modular arithmetic
- Rational numbers (fractions in canonical form)
"""

from .integers import Integer, IntegerEncoder
from .primes import (
    is_prime,
    prime_factorization,
    nth_prime,
    primes_up_to,
    PrimeEncoder,
)
from .divisibility import (
    gcd,
    lcm,
    divides,
    divisors,
    is_coprime,
    euler_totient,
)
from .modular import (
    ModularInt,
    ModularEncoder,
    mod_add,
    mod_mul,
    mod_pow,
    mod_inverse,
)
from .rationals import Rational, RationalEncoder

__all__ = [
    # Integers
    "Integer",
    "IntegerEncoder",
    # Primes
    "is_prime",
    "prime_factorization",
    "nth_prime",
    "primes_up_to",
    "PrimeEncoder",
    # Divisibility
    "gcd",
    "lcm",
    "divides",
    "divisors",
    "is_coprime",
    "euler_totient",
    # Modular arithmetic
    "ModularInt",
    "ModularEncoder",
    "mod_add",
    "mod_mul",
    "mod_pow",
    "mod_inverse",
    # Rationals
    "Rational",
    "RationalEncoder",
]

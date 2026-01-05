"""
Divisibility functions for FluxEM-Domains.

Provides deterministic operations for divisibility, GCD, LCM, and related concepts.
"""

from typing import List
import math


def gcd(a: int, b: int) -> int:
    """
    Greatest common divisor via the Euclidean algorithm.

    Returns the largest positive integer that divides both a and b.
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a if a else 1


def lcm(a: int, b: int) -> int:
    """
    Least common multiple via lcm(a,b) = |a*b| / gcd(a,b).

    Returns the smallest positive integer divisible by both a and b.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)


def divides(a: int, b: int) -> bool:
    """
    Check if a divides b evenly.

    Returns True if b % a == 0 and a != 0.
    """
    if a == 0:
        return False
    return b % a == 0


def divisors(n: int) -> List[int]:
    """
    Return all positive divisors of n.

    Example: divisors(12) -> [1, 2, 3, 4, 6, 12]
    """
    if n <= 0:
        return []

    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1

    return sorted(divs)


def is_coprime(a: int, b: int) -> bool:
    """
    Check if gcd(a, b) == 1.

    Two integers are coprime if their only common divisor is 1.
    """
    return gcd(a, b) == 1


def euler_totient(n: int) -> int:
    """
    Euler's totient function φ(n) from factorization.

    Returns the count of integers 1 <= k <= n that are coprime to n.

    For prime p: φ(p) = p - 1
    For n = ∏ p_i^e_i: φ(n) = n * ∏ (1 - 1/p_i)
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1

    from .primes import prime_factorization

    factors = prime_factorization(n)
    result = n

    for p in factors:
        result = result * (p - 1) // p

    return result


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Extended Euclidean algorithm.

    Returns (g, x, y) such that g = gcd(a, b) and g = a*x + b*y.
    """
    if a == 0:
        return (abs(b), 0, 1 if b > 0 else -1)

    if b == 0:
        return (abs(a), 1 if a > 0 else -1, 0)

    x0, x1, y0, y1 = 1, 0, 0, 1
    orig_a, orig_b = a, b
    a, b = abs(a), abs(b)

    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1

    # Adjust signs
    if orig_a < 0:
        x0 = -x0
    if orig_b < 0:
        y0 = -y0

    return (a, x0, y0)

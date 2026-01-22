"""Number theory domain - primes, GCD, modular arithmetic.

This module provides deterministic number theory computations.
"""

import math
from typing import Any, Dict, List, Optional, Union

from ..registry import ToolSpec, ToolRegistry


def is_prime(n: int) -> bool:
    """Check if a number is prime.

    Args:
        n: Integer to check

    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor.

    Args:
        a: First integer
        b: Second integer

    Returns:
        GCD of a and b
    """
    return math.gcd(abs(int(a)), abs(int(b)))


def gcd_list(numbers: List[int]) -> int:
    """Compute GCD of multiple numbers.

    Args:
        numbers: List of integers

    Returns:
        GCD of all numbers
    """
    if not numbers:
        raise ValueError("No numbers provided")
    result = abs(int(numbers[0]))
    for num in numbers[1:]:
        result = math.gcd(result, abs(int(num)))
    return result


def lcm(a: int, b: int) -> int:
    """Compute least common multiple.

    Args:
        a: First integer
        b: Second integer

    Returns:
        LCM of a and b
    """
    a, b = abs(int(a)), abs(int(b))
    return (a * b) // math.gcd(a, b) if a and b else 0


def mod_pow(base: int, exponent: int, modulus: int) -> int:
    """Compute modular exponentiation: base^exponent mod modulus.

    Uses fast exponentiation for efficiency.

    Args:
        base: Base number
        exponent: Exponent
        modulus: Modulus

    Returns:
        (base ** exponent) % modulus
    """
    return pow(int(base), int(exponent), int(modulus))


def mod_inverse(a: int, m: int) -> int:
    """Compute modular multiplicative inverse.

    Finds x such that (a * x) % m == 1.

    Args:
        a: Number to find inverse of
        m: Modulus

    Returns:
        Modular inverse of a mod m

    Raises:
        ValueError: If inverse doesn't exist (a and m not coprime)
    """
    a, m = int(a), int(m)
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Modular inverse doesn't exist: gcd({a}, {m}) = {g}")
    return x % m


def _extended_gcd(a: int, b: int) -> tuple:
    """Extended Euclidean algorithm."""
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def primes_up_to(n: int) -> List[int]:
    """Find all primes up to n using Sieve of Eratosthenes.

    Args:
        n: Upper limit (inclusive)

    Returns:
        List of primes <= n
    """
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def nth_prime(n: int) -> int:
    """Find the nth prime number (1-indexed).

    Args:
        n: Which prime to find (1 = first prime = 2)

    Returns:
        The nth prime number
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return 2
    count = 1
    candidate = 3
    while count < n:
        if is_prime(candidate):
            count += 1
        if count < n:
            candidate += 2
    return candidate


def totient(n: int) -> int:
    """Compute Euler's totient function phi(n).

    Returns count of integers 1 to n that are coprime with n.

    Args:
        n: Positive integer

    Returns:
        phi(n)
    """
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def divisors(n: int) -> List[int]:
    """Find all positive divisors of n.

    Args:
        n: Positive integer

    Returns:
        Sorted list of divisors
    """
    n = abs(int(n))
    if n == 0:
        raise ValueError("0 has infinite divisors")
    result = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)
    return sorted(result)


def _parse_args(args: Any, keys: List[str]) -> tuple:
    """Parse arguments that might be dict or list."""
    if isinstance(args, dict):
        return tuple(args.get(k) for k in keys)
    if isinstance(args, (list, tuple)):
        return tuple(args)
    return (args,)


def register_tools(registry: ToolRegistry) -> None:
    """Register number theory tools in the registry."""

    registry.register(ToolSpec(
        name="number_theory_is_prime",
        function=lambda n: is_prime(int(n) if not isinstance(n, dict) else int(n.get("n", n.get("number", list(n.values())[0])))),
        description="Checks if a number is prime.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number to check"}
            },
            "required": ["n"]
        },
        returns="Boolean - True if prime, False otherwise",
        examples=[
            {"input": {"n": 17}, "output": True},
            {"input": {"n": 18}, "output": False},
        ],
        domain="number_theory",
        tags=["prime", "primality", "integer"],
    ))

    def _gcd_handler(args):
        if isinstance(args, dict):
            if "a" in args and "b" in args:
                return gcd(args["a"], args["b"])
            if "numbers" in args:
                return gcd_list(args["numbers"])
            if "values" in args:
                return gcd_list(args["values"])
            vals = list(args.values())
            if len(vals) >= 2:
                return gcd_list(vals)
        if isinstance(args, (list, tuple)):
            return gcd_list(args)
        raise ValueError(f"Invalid GCD arguments: {args}")

    registry.register(ToolSpec(
        name="number_theory_gcd",
        function=_gcd_handler,
        description="Computes greatest common divisor of two or more numbers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"]
        },
        returns="GCD as integer",
        examples=[
            {"input": {"a": 48, "b": 18}, "output": 6},
            {"input": {"numbers": [12, 18, 24]}, "output": 6},
        ],
        domain="number_theory",
        tags=["gcd", "divisor", "common"],
    ))

    def _mod_pow_handler(args):
        if isinstance(args, dict):
            base = args.get("base", args.get("a"))
            exp = args.get("exponent", args.get("exp", args.get("b")))
            mod = args.get("modulus", args.get("mod", args.get("m")))
            return mod_pow(base, exp, mod)
        return mod_pow(*args)

    registry.register(ToolSpec(
        name="number_theory_mod_pow",
        function=_mod_pow_handler,
        description="Computes modular exponentiation: base^exponent mod modulus.",
        parameters={
            "type": "object",
            "properties": {
                "base": {"type": "integer", "description": "Base number"},
                "exponent": {"type": "integer", "description": "Exponent"},
                "modulus": {"type": "integer", "description": "Modulus"},
            },
            "required": ["base", "exponent", "modulus"]
        },
        returns="Result as integer",
        examples=[
            {"input": {"base": 2, "exponent": 10, "modulus": 1000}, "output": 24},
            {"input": {"base": 13, "exponent": 123, "modulus": 997}, "output": 649},
        ],
        domain="number_theory",
        tags=["modular", "exponentiation", "power", "cryptography"],
    ))

    def _mod_inverse_handler(args):
        if isinstance(args, dict):
            a = args.get("a", args.get("n"))
            m = args.get("m", args.get("modulus", args.get("mod")))
            return mod_inverse(a, m)
        return mod_inverse(*args)

    registry.register(ToolSpec(
        name="number_theory_mod_inverse",
        function=_mod_inverse_handler,
        description="Computes modular multiplicative inverse: finds x such that (a * x) % m == 1.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "Number to find inverse of"},
                "m": {"type": "integer", "description": "Modulus"},
            },
            "required": ["a", "m"]
        },
        returns="Modular inverse as integer",
        examples=[
            {"input": {"a": 3, "m": 7}, "output": 5},
        ],
        domain="number_theory",
        tags=["modular", "inverse", "cryptography"],
    ))

    registry.register(ToolSpec(
        name="number_theory_primes_up_to",
        function=lambda n: primes_up_to(int(n) if not isinstance(n, dict) else int(n.get("n", list(n.values())[0]))),
        description="Finds all prime numbers up to n using Sieve of Eratosthenes.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Upper limit (inclusive)"}
            },
            "required": ["n"]
        },
        returns="List of primes <= n",
        examples=[
            {"input": {"n": 20}, "output": [2, 3, 5, 7, 11, 13, 17, 19]},
        ],
        domain="number_theory",
        tags=["prime", "sieve", "list"],
    ))

    registry.register(ToolSpec(
        name="number_theory_nth_prime",
        function=lambda n: nth_prime(int(n) if not isinstance(n, dict) else int(n.get("n", list(n.values())[0]))),
        description="Finds the nth prime number (1-indexed: nth_prime(1) = 2).",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Which prime to find (1 = first = 2)"}
            },
            "required": ["n"]
        },
        returns="The nth prime number",
        examples=[
            {"input": {"n": 1}, "output": 2},
            {"input": {"n": 10}, "output": 29},
        ],
        domain="number_theory",
        tags=["prime", "nth"],
    ))

    def _lcm_handler(args):
        if isinstance(args, dict):
            return lcm(args.get("a"), args.get("b"))
        return lcm(*args)

    registry.register(ToolSpec(
        name="number_theory_lcm",
        function=_lcm_handler,
        description="Computes least common multiple of two numbers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"]
        },
        returns="LCM as integer",
        examples=[
            {"input": {"a": 12, "b": 18}, "output": 36},
        ],
        domain="number_theory",
        tags=["lcm", "multiple", "common"],
    ))

    registry.register(ToolSpec(
        name="number_theory_totient",
        function=lambda n: totient(int(n) if not isinstance(n, dict) else int(n.get("n", list(n.values())[0]))),
        description="Computes Euler's totient function phi(n) - count of integers 1 to n coprime with n.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Positive integer"}
            },
            "required": ["n"]
        },
        returns="phi(n) as integer",
        examples=[
            {"input": {"n": 10}, "output": 4},
            {"input": {"n": 12}, "output": 4},
        ],
        domain="number_theory",
        tags=["euler", "totient", "phi", "coprime"],
    ))

    registry.register(ToolSpec(
        name="number_theory_divisors",
        function=lambda n: divisors(int(n) if not isinstance(n, dict) else int(n.get("n", list(n.values())[0]))),
        description="Finds all positive divisors of n.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Positive integer"}
            },
            "required": ["n"]
        },
        returns="Sorted list of divisors",
        examples=[
            {"input": {"n": 12}, "output": [1, 2, 3, 4, 6, 12]},
        ],
        domain="number_theory",
        tags=["divisor", "factor"],
    ))

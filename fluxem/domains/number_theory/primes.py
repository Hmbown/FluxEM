"""
Prime number utilities and encoder for FluxEM-Domains.

Provides deterministic primality testing, factorization, and prime-related operations.
"""

from typing import Any, Dict, List, Tuple
import math
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    EPSILON,
)


def is_prime(n: int) -> bool:
    """
    Check if n is prime using trial division up to sqrt(n).
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False

    # Check divisors of form 6k ± 1 up to sqrt(n)
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def prime_factorization(n: int) -> Dict[int, int]:
    """
    Return prime factorization as {prime: exponent}.

    Example: 12 -> {2: 2, 3: 1}
    """
    if n <= 1:
        return {}

    factors: Dict[int, int] = {}

    # Extract factor 2
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    # Extract odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2

    # If n is still > 1, it's a prime factor
    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    return factors


def primes_up_to(n: int) -> List[int]:
    """
    Return all primes <= n using Sieve of Eratosthenes.
    """
    if n < 2:
        return []

    # Sieve of Eratosthenes
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False

    i = 2
    while i * i <= n:
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
        i += 1

    return [i for i, is_prime_val in enumerate(sieve) if is_prime_val]


def nth_prime(n: int) -> int:
    """
    Return the nth prime (1-indexed).

    Example: nth_prime(1) = 2, nth_prime(4) = 7
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # Approximate upper bound using prime number theorem
    if n < 6:
        bound = 15
    else:
        # p_n < n * (log n + log log n) for n >= 6
        bound = int(n * (math.log(n) + math.log(math.log(n)))) + 1

    # Sieve up to bound and get nth prime
    primes = primes_up_to(bound)

    if len(primes) < n:
        # Bound was too low, extend
        bound *= 2
        primes = primes_up_to(bound)

    if len(primes) < n:
        raise ValueError(
            f"Could not find {n}th prime (found {len(primes)} up to {bound})"
        )

    return primes[n - 1]


def prime_index(p: int) -> int:
    """
    Return the index of prime p (1-indexed).

    Example: prime_index(2) = 1, prime_index(7) = 4

    Returns -1 if p is not prime.
    """
    if not is_prime(p):
        return -1

    # Count primes less than p
    # Use prime number theorem to estimate count
    if p < 2:
        return -1

    # For small primes, just count
    count = 0
    for i in range(2, p + 1):
        if is_prime(i):
            count += 1
            if i == p:
                return count

    return -1


class PrimeEncoder:
    """
    Encoder for prime numbers and their properties.

    Embedding layout (domain-specific dims 8-71):
    - dim 8: prime value (or 0 for composite)
    - dim 9: prime index (which prime it is, 0 for composite)
    - dims 10-19: prime factors (for composite numbers)
    - dims 72-79: properties (is_prime, is_prime_power, etc.)
    """

    domain_tag = DOMAIN_TAGS["num_prime"]
    domain_name = "num_prime"

    VALUE_POS = 8
    INDEX_POS = 9
    FACTORS_START = 10
    FACTORS_COUNT = 10

    IS_PRIME_FLAG_POS = 72
    IS_PRIME_POWER_POS = 73
    IS_COMPOSITE_POS = 74
    NUM_FACTORS_POS = 75

    def __init__(self):
        """Initialize the prime encoder."""
        pass

    def encode(self, n: int) -> Any:
        """
        Encode a number with its prime properties.

        Encodes the value and its prime factorization.
        """
        backend = get_backend()
        embedding = backend.zeros((EMBEDDING_DIM,))

        # Domain tag
        embedding = backend.at_add(embedding, slice(0, 8), self.domain_tag)

        # Store value
        embedding = backend.at_add(embedding, self.VALUE_POS, float(n))

        # Check if prime and get index
        prime = is_prime(n)
        if prime:
            idx = prime_index(n)
            embedding = backend.at_add(embedding, self.INDEX_POS, float(idx))
            embedding = backend.at_add(embedding, self.IS_PRIME_FLAG_POS, 1.0)
            embedding = backend.at_add(embedding, self.NUM_FACTORS_POS, 0.0)
        else:
            embedding = backend.at_add(embedding, self.INDEX_POS, 0.0)
            embedding = backend.at_add(embedding, self.IS_COMPOSITE_POS, 1.0)

            # Store prime factorization
            factors = prime_factorization(n)
            embedding = backend.at_add(embedding, self.NUM_FACTORS_POS, float(len(factors)))

            # Store first 10 prime factors (as values)
            for i, (prime_factor, exponent) in enumerate(
                sorted(factors.items())[: self.FACTORS_COUNT]
            ):
                pos = self.FACTORS_START + (i * 2)
                embedding = backend.at_add(embedding, pos, float(prime_factor))
                embedding = backend.at_add(embedding, pos + 1, float(exponent))

            # Check if prime power
            if len(factors) == 1:
                embedding = backend.at_add(embedding, self.IS_PRIME_POWER_POS, 1.0)

        return embedding

    def decode(self, embedding: Any) -> int:
        """Decode embedding to integer value."""
        value = int(round(float(embedding[self.VALUE_POS])))
        return value

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid for this domain."""
        backend = get_backend()
        tag = emb[0:8]
        return backend.allclose(tag, self.domain_tag, atol=0.1).item()

    # === QUERIES ===

    def is_prime(self, embedding: Any) -> bool:
        """Check if the encoded number is prime."""
        return float(embedding[self.IS_PRIME_FLAG_POS]) > 0.5

    def is_composite(self, embedding: Any) -> bool:
        """Check if the encoded number is composite (> 1, not prime)."""
        return float(embedding[self.IS_COMPOSITE_POS]) > 0.5

    def is_prime_power(self, embedding: Any) -> bool:
        """Check if the encoded number is a prime power."""
        return float(embedding[self.IS_PRIME_POWER_POS]) > 0.5

    def get_prime_index(self, embedding: Any) -> int:
        """
        Get the index of a prime number.

        Returns 0 if not prime.
        """
        if not self.is_prime(embedding):
            return 0
        return int(round(float(embedding[self.INDEX_POS])))

    def get_prime_factors(self, embedding: Any) -> Dict[int, int]:
        """
        Get prime factorization from embedding.

        Returns empty dict if prime or 1.
        """
        if self.is_prime(embedding):
            return {}

        n = self.decode(embedding)
        if n <= 1:
            return {}

        # Reconstruct from factors in embedding
        num_factors = int(round(float(embedding[self.NUM_FACTORS_POS])))
        factors: Dict[int, int] = {}

        for i in range(num_factors):
            pos = self.FACTORS_START + (i * 2)
            if pos + 1 < EMBEDDING_DIM:
                prime_val = int(round(float(embedding[pos])))
                exponent = int(round(float(embedding[pos + 1])))
                if prime_val > 1:
                    factors[prime_val] = exponent

        # Fallback: compute directly
        if not factors:
            factors = prime_factorization(n)

        return factors

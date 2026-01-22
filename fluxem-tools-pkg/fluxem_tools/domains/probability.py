"""Probability domain - distributions, Bayes rule.

This module provides deterministic probability computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def bernoulli_pmf(p: float, k: int) -> float:
    """Bernoulli distribution PMF.

    Args:
        p: Success probability (0-1)
        k: Outcome (0 or 1)

    Returns:
        P(X = k)
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if k not in (0, 1):
        raise ValueError("k must be 0 or 1")
    return p if k == 1 else 1 - p


def binomial_pmf(n: int, k: int, p: float) -> float:
    """Binomial distribution PMF.

    Args:
        n: Number of trials
        k: Number of successes
        p: Success probability (0-1)

    Returns:
        P(X = k)
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    if k < 0 or k > n:
        return 0.0

    # Calculate C(n, k)
    c = 1
    for i in range(min(k, n - k)):
        c = c * (n - i) // (i + 1)

    return c * (p ** k) * ((1 - p) ** (n - k))


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson distribution PMF.

    Args:
        k: Number of events
        lam: Rate parameter (λ)

    Returns:
        P(X = k)
    """
    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if k < 0:
        return 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def bayes_rule(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """Apply Bayes' theorem: P(A|B) = P(B|A)P(A) / P(B).

    Args:
        p_a: Prior probability P(A)
        p_b_given_a: Likelihood P(B|A)
        p_b_given_not_a: P(B|¬A)

    Returns:
        Posterior probability P(A|B)
    """
    p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
    if p_b == 0:
        return 0.0
    return (p_b_given_a * p_a) / p_b


def geometric_pmf(k: int, p: float) -> float:
    """Geometric distribution PMF (number of trials until first success).

    Args:
        k: Number of trials (k >= 1)
        p: Success probability (0-1)

    Returns:
        P(X = k)
    """
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    if k < 1:
        return 0.0
    return p * ((1 - p) ** (k - 1))


def expected_value(outcomes: List[float], probabilities: List[float]) -> float:
    """Calculate expected value E[X] = Σ x·P(x).

    Args:
        outcomes: Possible outcomes
        probabilities: Corresponding probabilities

    Returns:
        Expected value
    """
    if len(outcomes) != len(probabilities):
        raise ValueError("Outcomes and probabilities must have same length")
    if abs(sum(probabilities) - 1.0) > 1e-9:
        raise ValueError("Probabilities must sum to 1")
    return sum(x * p for x, p in zip(outcomes, probabilities))


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_bernoulli(args) -> Tuple[float, int]:
    if isinstance(args, dict):
        p = float(args.get("p", args.get("probability")))
        k = int(args.get("k", args.get("x", args.get("outcome"))))
        return p, k
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return float(args[0]), int(args[1])
    raise ValueError(f"Cannot parse Bernoulli args: {args}")


def _parse_binomial(args) -> Tuple[int, int, float]:
    if isinstance(args, dict):
        n = int(args.get("n", args.get("trials")))
        k = int(args.get("k", args.get("successes")))
        p = float(args.get("p", args.get("probability")))
        return n, k, p
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return int(args[0]), int(args[1]), float(args[2])
    raise ValueError(f"Cannot parse binomial args: {args}")


def _parse_poisson(args) -> Tuple[int, float]:
    if isinstance(args, dict):
        k = int(args.get("k", args.get("events")))
        lam = float(args.get("lambda", args.get("lam", args.get("rate"))))
        return k, lam
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        return int(args[0]), float(args[1])
    raise ValueError(f"Cannot parse Poisson args: {args}")


def _parse_bayes(args) -> Tuple[float, float, float]:
    if isinstance(args, dict):
        p_a = float(args.get("p_a", args.get("prior")))
        p_b_a = float(args.get("p_b_given_a", args.get("likelihood")))
        p_b_not_a = float(args.get("p_b_given_not_a", args.get("p_b_not_a")))
        return p_a, p_b_a, p_b_not_a
    if isinstance(args, (list, tuple)) and len(args) >= 3:
        return float(args[0]), float(args[1]), float(args[2])
    raise ValueError(f"Cannot parse Bayes args: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register probability tools in the registry."""

    registry.register(ToolSpec(
        name="probability_bernoulli_pmf",
        function=lambda args: bernoulli_pmf(*_parse_bernoulli(args)),
        description="Computes Bernoulli distribution PMF: P(X = k) for k in {0, 1}.",
        parameters={
            "type": "object",
            "properties": {
                "p": {"type": "number", "description": "Success probability (0-1)"},
                "k": {"type": "integer", "description": "Outcome (0 or 1)"},
            },
            "required": ["p", "k"]
        },
        returns="Probability as float",
        examples=[
            {"input": {"p": 0.7, "k": 1}, "output": 0.7},
            {"input": {"p": 0.7, "k": 0}, "output": 0.3},
        ],
        domain="probability",
        tags=["bernoulli", "pmf", "distribution"],
    ))

    registry.register(ToolSpec(
        name="probability_binomial_pmf",
        function=lambda args: binomial_pmf(*_parse_binomial(args)),
        description="Computes Binomial distribution PMF: P(X = k) for n trials with success probability p.",
        parameters={
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "Number of trials"},
                "k": {"type": "integer", "description": "Number of successes"},
                "p": {"type": "number", "description": "Success probability"},
            },
            "required": ["n", "k", "p"]
        },
        returns="Probability as float",
        examples=[
            {"input": {"n": 10, "k": 3, "p": 0.5}, "output": 0.1172},
        ],
        domain="probability",
        tags=["binomial", "pmf", "distribution"],
    ))

    registry.register(ToolSpec(
        name="probability_bayes_rule",
        function=lambda args: bayes_rule(*_parse_bayes(args)),
        description="Computes P(A|B) using Bayes' theorem from P(A), P(B|A), P(B|¬A).",
        parameters={
            "type": "object",
            "properties": {
                "p_a": {"type": "number", "description": "Prior P(A)"},
                "p_b_given_a": {"type": "number", "description": "Likelihood P(B|A)"},
                "p_b_given_not_a": {"type": "number", "description": "P(B|¬A)"},
            },
            "required": ["p_a", "p_b_given_a", "p_b_given_not_a"]
        },
        returns="Posterior P(A|B) as float",
        examples=[
            {"input": {"p_a": 0.01, "p_b_given_a": 0.9, "p_b_given_not_a": 0.1}, "output": 0.0833},
        ],
        domain="probability",
        tags=["bayes", "posterior", "conditional"],
    ))

    registry.register(ToolSpec(
        name="probability_poisson_pmf",
        function=lambda args: poisson_pmf(*_parse_poisson(args)),
        description="Computes Poisson distribution PMF: P(X = k) for rate λ.",
        parameters={
            "type": "object",
            "properties": {
                "k": {"type": "integer", "description": "Number of events"},
                "lambda": {"type": "number", "description": "Rate parameter (λ)"},
            },
            "required": ["k", "lambda"]
        },
        returns="Probability as float",
        examples=[
            {"input": {"k": 3, "lambda": 2.5}, "output": 0.2138},
        ],
        domain="probability",
        tags=["poisson", "pmf", "distribution"],
    ))

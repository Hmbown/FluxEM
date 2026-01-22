"""Information theory domain - entropy, cross-entropy, KL divergence.

This module provides deterministic information-theoretic computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..registry import ToolSpec, ToolRegistry


# =============================================================================
# Core Functions
# =============================================================================

def entropy(probs: List[float], base: float = 2) -> float:
    """Compute Shannon entropy H(X) = -Σ p(x) log p(x).

    Args:
        probs: Probability distribution (must sum to 1)
        base: Logarithm base (2 for bits, e for nats)

    Returns:
        Entropy value
    """
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("Probabilities must sum to 1")

    result = 0.0
    for p in probs:
        if p > 0:
            result -= p * math.log(p, base)
    return result


def cross_entropy(p: List[float], q: List[float], base: float = 2) -> float:
    """Compute cross-entropy H(p, q) = -Σ p(x) log q(x).

    Args:
        p: True distribution
        q: Predicted distribution
        base: Logarithm base

    Returns:
        Cross-entropy value
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            result -= pi * math.log(qi, base)
    return result


def kl_divergence(p: List[float], q: List[float], base: float = 2) -> float:
    """Compute KL divergence D_KL(P || Q) = Σ p(x) log(p(x)/q(x)).

    Args:
        p: True distribution
        q: Approximate distribution
        base: Logarithm base

    Returns:
        KL divergence (always >= 0)
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")

    result = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            result += pi * math.log(pi / qi, base)
    return result


def mutual_information(joint: List[List[float]], base: float = 2) -> float:
    """Compute mutual information I(X; Y) from joint distribution.

    Args:
        joint: Joint probability table P(X, Y)
        base: Logarithm base

    Returns:
        Mutual information I(X; Y)
    """
    rows = len(joint)
    cols = len(joint[0]) if joint else 0

    # Compute marginals
    p_x = [sum(joint[i]) for i in range(rows)]
    p_y = [sum(joint[i][j] for i in range(rows)) for j in range(cols)]

    result = 0.0
    for i in range(rows):
        for j in range(cols):
            if joint[i][j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                result += joint[i][j] * math.log(joint[i][j] / (p_x[i] * p_y[j]), base)

    return result


def conditional_entropy(joint: List[List[float]], base: float = 2) -> float:
    """Compute conditional entropy H(Y|X) from joint distribution.

    Args:
        joint: Joint probability table P(X, Y)
        base: Logarithm base

    Returns:
        Conditional entropy H(Y|X)
    """
    rows = len(joint)
    cols = len(joint[0]) if joint else 0

    p_x = [sum(joint[i]) for i in range(rows)]

    result = 0.0
    for i in range(rows):
        if p_x[i] > 0:
            for j in range(cols):
                p_y_given_x = joint[i][j] / p_x[i]
                if p_y_given_x > 0:
                    result -= joint[i][j] * math.log(p_y_given_x, base)

    return result


# =============================================================================
# Argument Parsing
# =============================================================================

def _parse_probs(args) -> List[float]:
    if isinstance(args, dict):
        probs = args.get("probs", args.get("distribution", args.get("p")))
        if probs is not None:
            return [float(x) for x in probs]
        return [float(x) for x in list(args.values())[0]]
    if isinstance(args, (list, tuple)):
        return [float(x) for x in args]
    raise ValueError(f"Cannot parse probabilities: {args}")


def _parse_two_dists(args) -> Tuple[List[float], List[float]]:
    if isinstance(args, dict):
        p = args.get("p", args.get("true", args.get("first")))
        q = args.get("q", args.get("predicted", args.get("second")))
        if p is not None and q is not None:
            return [float(x) for x in p], [float(x) for x in q]
    if isinstance(args, (list, tuple)) and len(args) >= 2:
        if isinstance(args[0], (list, tuple)):
            return [float(x) for x in args[0]], [float(x) for x in args[1]]
    raise ValueError(f"Cannot parse two distributions: {args}")


# =============================================================================
# Tool Registration
# =============================================================================

def register_tools(registry: ToolRegistry) -> None:
    """Register information theory tools in the registry."""

    registry.register(ToolSpec(
        name="info_entropy",
        function=lambda args: entropy(_parse_probs(args)),
        description="Computes Shannon entropy H(X) = -Σ p(x) log₂ p(x) in bits.",
        parameters={
            "type": "object",
            "properties": {
                "probs": {"type": "array", "items": {"type": "number"}, "description": "Probability distribution (must sum to 1)"}
            },
            "required": ["probs"]
        },
        returns="Entropy in bits",
        examples=[
            {"input": {"probs": [0.5, 0.5]}, "output": 1.0},
            {"input": {"probs": [0.25, 0.25, 0.25, 0.25]}, "output": 2.0},
        ],
        domain="information_theory",
        tags=["entropy", "shannon", "uncertainty"],
    ))

    registry.register(ToolSpec(
        name="info_cross_entropy",
        function=lambda args: cross_entropy(*_parse_two_dists(args)),
        description="Computes cross-entropy H(p, q) = -Σ p(x) log q(x).",
        parameters={
            "type": "object",
            "properties": {
                "p": {"type": "array", "items": {"type": "number"}, "description": "True distribution"},
                "q": {"type": "array", "items": {"type": "number"}, "description": "Predicted distribution"},
            },
            "required": ["p", "q"]
        },
        returns="Cross-entropy in bits",
        examples=[
            {"input": {"p": [0.5, 0.5], "q": [0.9, 0.1]}, "output": 1.152},
        ],
        domain="information_theory",
        tags=["cross-entropy", "loss"],
    ))

    registry.register(ToolSpec(
        name="info_kl_divergence",
        function=lambda args: kl_divergence(*_parse_two_dists(args)),
        description="Computes KL divergence D_KL(P || Q) = Σ p(x) log(p(x)/q(x)).",
        parameters={
            "type": "object",
            "properties": {
                "p": {"type": "array", "items": {"type": "number"}, "description": "True distribution"},
                "q": {"type": "array", "items": {"type": "number"}, "description": "Approximate distribution"},
            },
            "required": ["p", "q"]
        },
        returns="KL divergence in bits (always >= 0)",
        examples=[
            {"input": {"p": [0.5, 0.5], "q": [0.9, 0.1]}, "output": 0.531},
        ],
        domain="information_theory",
        tags=["kl-divergence", "divergence", "distance"],
    ))

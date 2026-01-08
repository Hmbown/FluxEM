"""Domain-specific regex patterns for span detection.

Each pattern is a tuple of (compiled_regex, domain_name, parser_function).
The parser_function extracts structured data from the match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple
from enum import Enum, auto


class DomainType(Enum):
    """Domain types for span detection."""
    ARITHMETIC = auto()
    COMBINATORICS = auto()
    PROBABILITY = auto()
    STATISTICS = auto()
    TEMPORAL = auto()
    FINANCE = auto()
    PHYSICS = auto()
    CHEMISTRY = auto()
    CALCULUS = auto()
    CONTROL = auto()
    SIGNAL = auto()
    INFO_THEORY = auto()
    OPTIMIZATION = auto()


@dataclass
class DomainPattern:
    """A pattern for detecting domain spans."""

    name: str
    domain: DomainType
    pattern: Pattern[str]
    priority: int  # Higher = more specific, wins in overlap resolution
    parser: Callable[[re.Match], Any]

    def match(self, text: str) -> List[Tuple[int, int, Any]]:
        """Find all matches in text.

        Returns:
            List of (start, end, parsed_value) tuples.
        """
        results = []
        for match in self.pattern.finditer(text):
            try:
                parsed = self.parser(match)
                results.append((match.start(), match.end(), parsed))
            except (ValueError, TypeError):
                # Skip invalid matches
                continue
        return results


# =============================================================================
# Parser Functions
# =============================================================================

def _parse_integer(match: re.Match) -> int:
    """Parse an integer, handling commas and underscores."""
    text = match.group(0).replace(",", "").replace("_", "")
    return int(text)


def _parse_float(match: re.Match) -> float:
    """Parse a float, handling commas."""
    text = match.group(0).replace(",", "")
    return float(text)


def _parse_scientific(match: re.Match) -> float:
    """Parse scientific notation."""
    return float(match.group(0).replace(" ", ""))


def _parse_factorial(match: re.Match) -> dict:
    """Parse factorial: n! or n!!"""
    text = match.group(0)
    if text.endswith("!!"):
        n = int(text[:-2])
        return {"type": "double_factorial", "n": n}
    else:
        n = int(text[:-1])
        return {"type": "factorial", "n": n}


def _parse_choose(match: re.Match) -> dict:
    """Parse combination: C(n,k) or n choose k."""
    groups = match.groups()
    if len(groups) >= 2:
        n = int(groups[0])
        k = int(groups[1])
        return {"type": "combination", "n": n, "k": k}
    return {}


def _parse_permutation(match: re.Match) -> dict:
    """Parse permutation: P(n,k)."""
    groups = match.groups()
    if len(groups) >= 2:
        n = int(groups[0])
        k = int(groups[1])
        return {"type": "permutation", "n": n, "k": k}
    return {}


def _parse_probability(match: re.Match) -> dict:
    """Parse probability expressions: P(X=k), Bernoulli(p), etc."""
    text = match.group(0)

    # Bernoulli(p)
    if "bernoulli" in text.lower():
        p_match = re.search(r"[\d.]+", text)
        if p_match:
            return {"kind": "bernoulli", "p": float(p_match.group())}

    # Binomial(n, p)
    if "binomial" in text.lower():
        nums = re.findall(r"[\d.]+", text)
        if len(nums) >= 2:
            return {"kind": "binomial", "n": int(nums[0]), "p": float(nums[1])}

    # P(X=k)
    prob_match = re.search(r"P\s*\(\s*X\s*=\s*(\d+)\s*\)", text)
    if prob_match:
        return {"kind": "pmf_query", "k": int(prob_match.group(1))}

    return {"kind": "unknown", "text": text}


def _parse_statistics(match: re.Match) -> dict:
    """Parse statistics expressions: mean, variance, etc."""
    text = match.group(0).lower()

    if "mean" in text or "average" in text:
        return {"operation": "mean"}
    elif "variance" in text or "var" in text:
        return {"operation": "variance"}
    elif "std" in text or "standard deviation" in text:
        return {"operation": "std"}
    elif "median" in text:
        return {"operation": "median"}
    elif "mode" in text:
        return {"operation": "mode"}
    elif "correlation" in text or "corr" in text:
        return {"operation": "correlation"}

    return {"operation": "unknown"}


def _parse_date(match: re.Match) -> dict:
    """Parse date expressions."""
    text = match.group(0)

    # ISO format: YYYY-MM-DD
    iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso_match:
        return {
            "year": int(iso_match.group(1)),
            "month": int(iso_match.group(2)),
            "day": int(iso_match.group(3)),
        }

    # US format: MM/DD/YYYY
    us_match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if us_match:
        return {
            "year": int(us_match.group(3)),
            "month": int(us_match.group(1)),
            "day": int(us_match.group(2)),
        }

    return {"raw": text}


def _parse_currency(match: re.Match) -> dict:
    """Parse currency amounts."""
    text = match.group(0)

    # Extract amount
    amount_match = re.search(r"[\d,]+\.?\d*", text)
    if amount_match:
        amount = float(amount_match.group().replace(",", ""))

        # Detect currency
        if "$" in text:
            currency = "USD"
        elif "€" in text:
            currency = "EUR"
        elif "£" in text:
            currency = "GBP"
        elif "¥" in text:
            currency = "JPY"
        else:
            currency = "USD"  # default

        return {"amount": amount, "currency": currency}

    return {}


def _parse_unit_quantity(match: re.Match) -> dict:
    """Parse physical quantity with unit."""
    text = match.group(0)

    # Extract number and unit
    num_match = re.match(r"([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*(\S+)", text)
    if num_match:
        value = float(num_match.group(1))
        unit = num_match.group(2)
        return {"value": value, "unit": unit}

    return {}


def _parse_chemical_formula(match: re.Match) -> dict:
    """Parse chemical formula."""
    text = match.group(0)

    # Parse elements and counts
    elements = {}
    for elem_match in re.finditer(r"([A-Z][a-z]?)(\d*)", text):
        element = elem_match.group(1)
        count = int(elem_match.group(2)) if elem_match.group(2) else 1
        elements[element] = elements.get(element, 0) + count

    return {"formula": text, "elements": elements}


def _parse_polynomial(match: re.Match) -> dict:
    """Parse polynomial expression."""
    text = match.group(0)
    return {"expression": text}


def _parse_derivative(match: re.Match) -> dict:
    """Parse derivative notation."""
    text = match.group(0)

    # d/dx or dy/dx
    var_match = re.search(r"d\s*([a-z]?)\s*/\s*d\s*([a-z])", text)
    if var_match:
        return {
            "operation": "derivative",
            "dependent": var_match.group(1) or "y",
            "independent": var_match.group(2),
        }

    return {"operation": "derivative", "expression": text}


def _parse_integral(match: re.Match) -> dict:
    """Parse integral notation."""
    text = match.group(0)
    return {"operation": "integral", "expression": text}


def _parse_matrix(match: re.Match) -> dict:
    """Parse matrix notation."""
    text = match.group(0)

    # Try to extract 2x2 matrix [[a,b],[c,d]]
    nums = re.findall(r"[-+]?[\d.]+", text)
    if len(nums) == 4:
        return {
            "type": "matrix_2x2",
            "values": [[float(nums[0]), float(nums[1])],
                       [float(nums[2]), float(nums[3])]],
        }

    return {"type": "matrix", "raw": text}


def _parse_entropy(match: re.Match) -> dict:
    """Parse entropy expressions."""
    text = match.group(0).lower()

    if "shannon" in text:
        return {"type": "shannon_entropy"}
    elif "kl" in text or "divergence" in text:
        return {"type": "kl_divergence"}
    elif "mutual" in text:
        return {"type": "mutual_information"}

    return {"type": "entropy"}


def _parse_signal(match: re.Match) -> dict:
    """Parse signal processing expressions."""
    text = match.group(0).lower()

    if "convolve" in text or "convolution" in text:
        return {"operation": "convolution"}
    elif "fft" in text or "fourier" in text:
        return {"operation": "fft"}
    elif "filter" in text:
        return {"operation": "filter"}
    elif "moving average" in text:
        return {"operation": "moving_average"}

    return {"operation": "signal"}


def _parse_optimization(match: re.Match) -> dict:
    """Parse optimization expressions."""
    text = match.group(0).lower()

    if "gradient" in text:
        return {"operation": "gradient_descent"}
    elif "minimize" in text:
        return {"operation": "minimize"}
    elif "maximize" in text:
        return {"operation": "maximize"}
    elif "learning rate" in text or "lr" in text:
        lr_match = re.search(r"[\d.]+", text)
        if lr_match:
            return {"operation": "learning_rate", "value": float(lr_match.group())}

    return {"operation": "optimization"}


# =============================================================================
# Pattern Definitions
# =============================================================================

# Arithmetic patterns
ARITHMETIC_PATTERNS = [
    DomainPattern(
        name="scientific_notation",
        domain=DomainType.ARITHMETIC,
        pattern=re.compile(r"[-+]?\d+\.?\d*\s*[eE]\s*[-+]?\d+"),
        priority=90,
        parser=_parse_scientific,
    ),
    DomainPattern(
        name="decimal_number",
        domain=DomainType.ARITHMETIC,
        pattern=re.compile(r"[-+]?\d+\.\d+"),
        priority=80,
        parser=_parse_float,
    ),
    DomainPattern(
        name="integer",
        domain=DomainType.ARITHMETIC,
        pattern=re.compile(r"[-+]?\d{1,3}(?:[,_]\d{3})*|\d+"),
        priority=70,
        parser=_parse_integer,
    ),
]

# Combinatorics patterns
COMBINATORICS_PATTERNS = [
    DomainPattern(
        name="factorial",
        domain=DomainType.COMBINATORICS,
        pattern=re.compile(r"\d+!{1,2}"),
        priority=95,
        parser=_parse_factorial,
    ),
    DomainPattern(
        name="combination_C",
        domain=DomainType.COMBINATORICS,
        pattern=re.compile(r"C\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE),
        priority=95,
        parser=_parse_choose,
    ),
    DomainPattern(
        name="combination_choose",
        domain=DomainType.COMBINATORICS,
        pattern=re.compile(r"(\d+)\s+choose\s+(\d+)", re.IGNORECASE),
        priority=95,
        parser=_parse_choose,
    ),
    DomainPattern(
        name="permutation_P",
        domain=DomainType.COMBINATORICS,
        pattern=re.compile(r"P\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE),
        priority=95,
        parser=_parse_permutation,
    ),
    DomainPattern(
        name="binomial_coefficient",
        domain=DomainType.COMBINATORICS,
        pattern=re.compile(r"\(\s*(\d+)\s*\\?\\?choose\s*(\d+)\s*\)"),
        priority=95,
        parser=_parse_choose,
    ),
]

# Probability patterns
PROBABILITY_PATTERNS = [
    DomainPattern(
        name="bernoulli",
        domain=DomainType.PROBABILITY,
        pattern=re.compile(r"[Bb]ernoulli\s*\(\s*[\d.]+\s*\)"),
        priority=95,
        parser=_parse_probability,
    ),
    DomainPattern(
        name="binomial",
        domain=DomainType.PROBABILITY,
        pattern=re.compile(r"[Bb]inomial\s*\(\s*\d+\s*,\s*[\d.]+\s*\)"),
        priority=95,
        parser=_parse_probability,
    ),
    DomainPattern(
        name="probability_pmf",
        domain=DomainType.PROBABILITY,
        pattern=re.compile(r"P\s*\(\s*X\s*=\s*\d+\s*\)"),
        priority=90,
        parser=_parse_probability,
    ),
]

# Statistics patterns
STATISTICS_PATTERNS = [
    DomainPattern(
        name="mean",
        domain=DomainType.STATISTICS,
        pattern=re.compile(r"\b(?:mean|average)\s+(?:of\s+)?\[[\d\s,.-]+\]", re.IGNORECASE),
        priority=95,
        parser=_parse_statistics,
    ),
    DomainPattern(
        name="variance",
        domain=DomainType.STATISTICS,
        pattern=re.compile(r"\b(?:variance|var)\s+(?:of\s+)?\[[\d\s,.-]+\]", re.IGNORECASE),
        priority=95,
        parser=_parse_statistics,
    ),
    DomainPattern(
        name="std_dev",
        domain=DomainType.STATISTICS,
        pattern=re.compile(r"\b(?:std|standard\s+deviation)\s+(?:of\s+)?\[[\d\s,.-]+\]", re.IGNORECASE),
        priority=95,
        parser=_parse_statistics,
    ),
    DomainPattern(
        name="median",
        domain=DomainType.STATISTICS,
        pattern=re.compile(r"\bmedian\s+(?:of\s+)?\[[\d\s,.-]+\]", re.IGNORECASE),
        priority=95,
        parser=_parse_statistics,
    ),
    DomainPattern(
        name="correlation",
        domain=DomainType.STATISTICS,
        pattern=re.compile(r"\b(?:correlation|corr)\b", re.IGNORECASE),
        priority=90,
        parser=_parse_statistics,
    ),
]

# Temporal patterns
TEMPORAL_PATTERNS = [
    DomainPattern(
        name="iso_date",
        domain=DomainType.TEMPORAL,
        pattern=re.compile(r"\d{4}-\d{2}-\d{2}"),
        priority=95,
        parser=_parse_date,
    ),
    DomainPattern(
        name="us_date",
        domain=DomainType.TEMPORAL,
        pattern=re.compile(r"\d{1,2}/\d{1,2}/\d{4}"),
        priority=90,
        parser=_parse_date,
    ),
]

# Finance patterns
FINANCE_PATTERNS = [
    DomainPattern(
        name="currency_usd",
        domain=DomainType.FINANCE,
        pattern=re.compile(r"\$[\d,]+\.?\d*"),
        priority=95,
        parser=_parse_currency,
    ),
    DomainPattern(
        name="currency_eur",
        domain=DomainType.FINANCE,
        pattern=re.compile(r"€[\d,]+\.?\d*"),
        priority=95,
        parser=_parse_currency,
    ),
    DomainPattern(
        name="npv",
        domain=DomainType.FINANCE,
        pattern=re.compile(r"\bNPV\b", re.IGNORECASE),
        priority=90,
        parser=lambda m: {"operation": "npv"},
    ),
    DomainPattern(
        name="irr",
        domain=DomainType.FINANCE,
        pattern=re.compile(r"\bIRR\b", re.IGNORECASE),
        priority=90,
        parser=lambda m: {"operation": "irr"},
    ),
]

# Physics patterns
PHYSICS_PATTERNS = [
    DomainPattern(
        name="quantity_with_unit",
        domain=DomainType.PHYSICS,
        pattern=re.compile(
            r"[-+]?[\d.]+(?:[eE][-+]?\d+)?\s*"
            r"(?:m/s\^?2|m/s|kg|g|mol|K|°C|°F|N|J|W|Pa|Hz|V|A|Ω|F|H|T|lm|lx|Bq|Gy|Sv|m|km|cm|mm|μm|nm|s|ms|μs|ns)"
        ),
        priority=85,
        parser=_parse_unit_quantity,
    ),
]

# Chemistry patterns
CHEMISTRY_PATTERNS = [
    DomainPattern(
        name="molecular_formula",
        domain=DomainType.CHEMISTRY,
        pattern=re.compile(r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)+\b"),
        priority=85,
        parser=_parse_chemical_formula,
    ),
    DomainPattern(
        name="simple_element",
        domain=DomainType.CHEMISTRY,
        pattern=re.compile(r"\b(?:H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Fe|Cu|Zn|Ag|Au|Hg|Pb|U)\d*\b"),
        priority=80,
        parser=_parse_chemical_formula,
    ),
]

# Calculus patterns
CALCULUS_PATTERNS = [
    DomainPattern(
        name="derivative",
        domain=DomainType.CALCULUS,
        pattern=re.compile(r"d\s*[a-z]?\s*/\s*d\s*[a-z]"),
        priority=95,
        parser=_parse_derivative,
    ),
    DomainPattern(
        name="integral",
        domain=DomainType.CALCULUS,
        pattern=re.compile(r"∫|\\int"),
        priority=95,
        parser=_parse_integral,
    ),
    DomainPattern(
        name="polynomial",
        domain=DomainType.CALCULUS,
        pattern=re.compile(r"[-+]?\d*x\^?\d*(?:\s*[-+]\s*\d*x?\^?\d*)*"),
        priority=80,
        parser=_parse_polynomial,
    ),
]

# Control systems patterns
CONTROL_PATTERNS = [
    DomainPattern(
        name="matrix_2x2",
        domain=DomainType.CONTROL,
        pattern=re.compile(r"\[\s*\[\s*[-+]?[\d.]+\s*,\s*[-+]?[\d.]+\s*\]\s*,\s*\[\s*[-+]?[\d.]+\s*,\s*[-+]?[\d.]+\s*\]\s*\]"),
        priority=95,
        parser=_parse_matrix,
    ),
    DomainPattern(
        name="state_space",
        domain=DomainType.CONTROL,
        pattern=re.compile(r"\b(?:state\s+space|transfer\s+function|eigenvalue|spectral\s+radius)\b", re.IGNORECASE),
        priority=90,
        parser=lambda m: {"type": "control_concept"},
    ),
]

# Information theory patterns
INFO_THEORY_PATTERNS = [
    DomainPattern(
        name="entropy",
        domain=DomainType.INFO_THEORY,
        pattern=re.compile(r"\b(?:shannon\s+)?entropy\b", re.IGNORECASE),
        priority=95,
        parser=_parse_entropy,
    ),
    DomainPattern(
        name="kl_divergence",
        domain=DomainType.INFO_THEORY,
        pattern=re.compile(r"\b(?:KL\s+divergence|Kullback[-\s]?Leibler)\b", re.IGNORECASE),
        priority=95,
        parser=_parse_entropy,
    ),
    DomainPattern(
        name="mutual_information",
        domain=DomainType.INFO_THEORY,
        pattern=re.compile(r"\bmutual\s+information\b", re.IGNORECASE),
        priority=95,
        parser=_parse_entropy,
    ),
]

# Signal processing patterns
SIGNAL_PATTERNS = [
    DomainPattern(
        name="convolution",
        domain=DomainType.SIGNAL,
        pattern=re.compile(r"\b(?:convolve|convolution)\b", re.IGNORECASE),
        priority=95,
        parser=_parse_signal,
    ),
    DomainPattern(
        name="fft",
        domain=DomainType.SIGNAL,
        pattern=re.compile(r"\b(?:FFT|DFT|Fourier\s+transform)\b", re.IGNORECASE),
        priority=95,
        parser=_parse_signal,
    ),
    DomainPattern(
        name="moving_average",
        domain=DomainType.SIGNAL,
        pattern=re.compile(r"\bmoving\s+average\b", re.IGNORECASE),
        priority=95,
        parser=_parse_signal,
    ),
]

# Optimization patterns
OPTIMIZATION_PATTERNS = [
    DomainPattern(
        name="gradient_descent",
        domain=DomainType.OPTIMIZATION,
        pattern=re.compile(r"\bgradient\s+(?:descent|step)\b", re.IGNORECASE),
        priority=95,
        parser=_parse_optimization,
    ),
    DomainPattern(
        name="learning_rate",
        domain=DomainType.OPTIMIZATION,
        pattern=re.compile(r"\b(?:learning\s+rate|lr)\s*[=:]\s*[\d.]+", re.IGNORECASE),
        priority=95,
        parser=_parse_optimization,
    ),
    DomainPattern(
        name="minimize_maximize",
        domain=DomainType.OPTIMIZATION,
        pattern=re.compile(r"\b(?:minimize|maximize|argmin|argmax)\b", re.IGNORECASE),
        priority=90,
        parser=_parse_optimization,
    ),
]


# =============================================================================
# Combined Pattern Registry
# =============================================================================

DOMAIN_PATTERNS: Dict[DomainType, List[DomainPattern]] = {
    DomainType.ARITHMETIC: ARITHMETIC_PATTERNS,
    DomainType.COMBINATORICS: COMBINATORICS_PATTERNS,
    DomainType.PROBABILITY: PROBABILITY_PATTERNS,
    DomainType.STATISTICS: STATISTICS_PATTERNS,
    DomainType.TEMPORAL: TEMPORAL_PATTERNS,
    DomainType.FINANCE: FINANCE_PATTERNS,
    DomainType.PHYSICS: PHYSICS_PATTERNS,
    DomainType.CHEMISTRY: CHEMISTRY_PATTERNS,
    DomainType.CALCULUS: CALCULUS_PATTERNS,
    DomainType.CONTROL: CONTROL_PATTERNS,
    DomainType.INFO_THEORY: INFO_THEORY_PATTERNS,
    DomainType.SIGNAL: SIGNAL_PATTERNS,
    DomainType.OPTIMIZATION: OPTIMIZATION_PATTERNS,
}


def get_all_patterns() -> List[DomainPattern]:
    """Get all patterns as a flat list, sorted by priority (highest first)."""
    all_patterns = []
    for patterns in DOMAIN_PATTERNS.values():
        all_patterns.extend(patterns)
    return sorted(all_patterns, key=lambda p: -p.priority)

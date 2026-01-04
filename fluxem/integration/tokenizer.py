"""
Multi-Domain Tokenizer for FluxEM-LLM integration.

Detects and tokenizes domain-specific patterns in text, allowing
the LLM to recognize when to use algebraic embeddings.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class DomainType(Enum):
    """Domain types for tokenization."""

    TEXT = 0  # Regular text (pass to LLM)
    DIMENSION = 1  # Physical dimensions [M L T^-2]
    FORMULA = 2  # Chemical formula H2O, C6H12O6
    REACTION = 3  # Chemical reaction ->
    COMPLEX = 4  # Complex number 3+4j
    RATIONAL = 5  # Rational number 3/4
    VECTOR = 6  # Vector [1, 2, 3]
    MATRIX = 7  # Matrix [[1, 2], [3, 4]]
    POLYNOMIAL = 8  # Polynomial x^2 + 2x + 1
    LOGICAL = 9  # Logical expression p ∧ q
    UNIT = 10  # Physical unit with prefix (kg, km/s)
    QUANTITY = 11  # Quantity with unit (10 m/s)
    DATE = 12  # Date: 2024-01-15, Jan 15, 2024
    TIME = 13  # Time: 14:30:00, 2:30 PM
    DATETIME = 14  # Combined date and time
    DURATION = 15  # Duration: 3 days, 2h 30m
    ARITHMETIC = 16  # Arithmetic expression: 123 + 456
    # Biology domains
    DNA = 17  # DNA sequence: ATGCCGTAGC
    RNA = 18  # RNA sequence: AUGCCGUAGC
    PROTEIN = 19  # Protein sequence: MVLSPADKTN
    TAXONOMY = 20  # Taxonomic classification
    # Music domains
    PITCH = 21  # Musical pitch: A4, C#5
    CHORD = 22  # Musical chord: Cmaj7, F#m9
    SCALE = 23  # Musical scale: C major, A minor
    RHYTHM = 24  # Rhythm/time signature: 4/4, 3/4
    ATONAL = 25  # Atonal pitch class set: [0,4,7]


@dataclass
class DomainToken:
    """A token with domain information."""

    text: str  # Original text
    domain: DomainType  # Detected domain
    start: int  # Start position in original text
    end: int  # End position in original text
    metadata: Optional[Dict] = None  # Parsed metadata if available

    def __repr__(self):
        return f"DomainToken({self.domain.name}: '{self.text}')"


class MultiDomainTokenizer:
    """
    Tokenizer that detects domain-specific patterns in text.

    This enables FluxEM-LLM to:
    1. Recognize when exact algebraic computation is needed
    2. Route to appropriate domain encoders
    3. Mix neural and algebraic processing
    """

    # Patterns for domain detection (ordered by priority)
    PATTERNS = {
        # Physical dimensions: [M L T^-2], [L^2 T^-2], [THETA]
        "dimension": r"\[([MLTINΘJ](?:\^-?\d+)?(?:\s*[MLTINΘJ](?:\^-?\d+)?)*)\]",
        # ISO datetime: 2024-01-15T14:30:00, 2024-01-15 14:30
        "datetime": r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?",
        # ISO date: 2024-01-15, 2024/01/15
        "date": r"\d{4}[-/]\d{2}[-/]\d{2}",
        # Time: 14:30:00, 14:30, 2:30 PM
        "time": r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AP]M)?",
        # Duration: 3 days, 2 hours, 30 minutes, 2h 30m
        "duration": r"\d+\s*(?:days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[dhms])\b",
        # Arithmetic expression: 123 + 456, 10 * 20
        "arithmetic": r"\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*",
        # Quantity with unit: 10 m/s, 3.14 kg*m/s^2, 9.8 m/s²
        "quantity": r"(\d+\.?\d*(?:e[+-]?\d+)?)\s*((?:[a-zA-Z]+(?:\^-?\d+)?[*/]?)+)",
        # Chemical reaction: 2H2 + O2 -> 2H2O
        "reaction": r"((?:[A-Z][a-z]?\d*)+(?:\s*\+\s*(?:[A-Z][a-z]?\d*)+)*)\s*(?:->|→|⟶)\s*((?:[A-Z][a-z]?\d*)+(?:\s*\+\s*(?:[A-Z][a-z]?\d*)+)*)",
        # Chemical formula: H2O, C6H12O6, NaCl
        "formula": r"\b([A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*)\b",
        # DNA sequence: ATGCCGTAGC (longer sequences to avoid false positives)
        "dna": r"\b[ATGC]{10,}\b",
        # RNA sequence: AUGCCGUAGC
        "rna": r"\b[AUGC]{10,}\b",
        # Protein sequence: MVLSPADKTN (amino acids)
        "protein": r"\b[ACDEFGHIKLMNPQRSTVWY]{8,}\b",
        # Musical pitch: A4, C#5, Bb3
        "pitch": r"\b([A-Ga-g])([#b])?(\d)\b",
        # Musical chord: Cmaj7, F#m9, Bbdim7
        "chord": r"\b([A-Ga-g])([#b])?(?:maj|min|m|dim|aug|dom|aug)?[0-9]*(?:maj|min|dim|aug|7|9|11|13)?\b",
        # Musical scale: C major, A minor, D dorian
        "scale": r"\b([A-Ga-g])([#b])?\s+(?:major|minor|dorian|phrygian|lydian|mixolydian|locrian|pentatonic|blues)\b",
        # Complex number: 3+4j, -2-5i, 3.14+2.71j
        "complex": r"(-?\d+\.?\d*)\s*([+-])\s*(\d+\.?\d*)\s*[ij]",
        # Rational number: 3/4, -7/11, 22/7
        "rational": r"(-?\d+)\s*/\s*(\d+)",
        # Vector: [1, 2, 3], (1, 2, 3)
        "vector": r"[\[(](-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)+)[\])]",
        # Matrix: [[1, 2], [3, 4]]
        "matrix": r"\[\s*\[(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\](?:\s*,\s*\[(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)*)\])+\s*\]",
        # Polynomial: x^2 + 2x + 1, 3x³ - 2x² + x - 5
        "polynomial": r"(?:\d*\.?\d*\s*)?[xX](?:\^|\*\*)?(\d+)?(?:\s*[+-]\s*(?:\d*\.?\d*\s*)?[xX](?:\^|\*\*)?(\d+)?)*(?:\s*[+-]\s*\d+\.?\d*)?",
        # Logical: p ∧ q, A ∨ B, ¬P, P → Q, P ↔ Q
        "logical": r"[A-Za-z](?:\s*[∧∨¬→↔⊕⊤⊥]\s*[A-Za-z])+",
    }

    # Map pattern names to domain types
    PATTERN_TO_DOMAIN = {
        "dimension": DomainType.DIMENSION,
        "datetime": DomainType.DATETIME,
        "date": DomainType.DATE,
        "time": DomainType.TIME,
        "duration": DomainType.DURATION,
        "arithmetic": DomainType.ARITHMETIC,
        "quantity": DomainType.QUANTITY,
        "reaction": DomainType.REACTION,
        "formula": DomainType.FORMULA,
        "dna": DomainType.DNA,
        "rna": DomainType.RNA,
        "protein": DomainType.PROTEIN,
        "pitch": DomainType.PITCH,
        "chord": DomainType.CHORD,
        "scale": DomainType.SCALE,
        "complex": DomainType.COMPLEX,
        "rational": DomainType.RATIONAL,
        "vector": DomainType.VECTOR,
        "matrix": DomainType.MATRIX,
        "polynomial": DomainType.POLYNOMIAL,
        "logical": DomainType.LOGICAL,
    }

    def __init__(self):
        """Initialize the tokenizer with compiled patterns."""
        self._compiled = {
            name: re.compile(
                pattern, re.IGNORECASE if name not in ("formula", "reaction") else 0
            )
            for name, pattern in self.PATTERNS.items()
        }

    def detect_domains(self, text: str) -> List[DomainToken]:
        """
        Detect domain-specific patterns in text.

        Args:
            text: Input text to analyze

        Returns:
            List of DomainToken objects for detected patterns
        """
        tokens = []

        # Try each pattern type
        for pattern_name, regex in self._compiled.items():
            for match in regex.finditer(text):
                domain = self.PATTERN_TO_DOMAIN[pattern_name]
                metadata = self._parse_metadata(pattern_name, match)

                # Skip formulas that look like regular words
                if pattern_name == "formula":
                    if not self._is_valid_formula(match.group(0)):
                        continue
                # Skip DNA/protein sequences that are too short
                if pattern_name in ["dna", "rna", "protein"]:
                    # Pattern already enforces minimum length
                    pass

                token = DomainToken(
                    text=match.group(0),
                    domain=domain,
                    start=match.start(),
                    end=match.end(),
                    metadata=metadata,
                )
                tokens.append(token)

        # Remove overlapping tokens (keep longer/more specific ones)
        tokens = self._resolve_overlaps(tokens)

        # Sort by position
        tokens.sort(key=lambda t: t.start)

        return tokens

    def _is_valid_formula(self, text: str) -> bool:
        """Check if text looks like a valid chemical formula."""
        # Must have at least one capital letter followed by lowercase or number
        if len(text) < 2:
            return False

        # Common false positives to exclude
        common_words = {"I", "A", "In", "As", "At", "Be", "He", "Me", "We", "No", "So"}
        if text in common_words:
            return False

        # Should have a number or multiple elements
        has_number = any(c.isdigit() for c in text)
        has_multiple_caps = sum(1 for c in text if c.isupper()) > 1

        return has_number or has_multiple_caps

    def _parse_metadata(self, pattern_name: str, match: re.Match) -> Dict[str, Any]:
        """Parse metadata from regex match."""
        if pattern_name == "complex":
            return {
                "real": float(match.group(1)),
                "sign": match.group(2),
                "imag": float(match.group(3)),
            }
        elif pattern_name == "rational":
            return {
                "numerator": int(match.group(1)),
                "denominator": int(match.group(2)),
            }
        elif pattern_name == "quantity":
            return {
                "value": float(match.group(1)),
                "unit": match.group(2),
            }
        elif pattern_name == "reaction":
            return {
                "reactants": match.group(1),
                "products": match.group(2),
            }
        return {}

    def _resolve_overlaps(self, tokens: List[DomainToken]) -> List[DomainToken]:
        """Resolve overlapping tokens by keeping the most specific."""
        if not tokens:
            return tokens

        # Sort by (start, -length, domain priority)
        priority = {
            DomainType.REACTION: 0,  # Highest priority
            DomainType.QUANTITY: 1,
            DomainType.DIMENSION: 2,
            DomainType.MATRIX: 3,
            DomainType.CHORD: 4,
            DomainType.SCALE: 5,
            DomainType.COMPLEX: 6,
            DomainType.RATIONAL: 7,
            DomainType.VECTOR: 8,
            DomainType.PITCH: 9,
            DomainType.POLYNOMIAL: 10,
            DomainType.LOGICAL: 11,
            DomainType.DNA: 12,
            DomainType.RNA: 12,
            DomainType.PROTEIN: 12,
            DomainType.FORMULA: 13,  # Lower priority (many false positives)
        }

        tokens.sort(
            key=lambda t: (t.start, -(t.end - t.start), priority.get(t.domain, 99))
        )

        result = []
        last_end = -1

        for token in tokens:
            if token.start >= last_end:
                result.append(token)
                last_end = token.end

        return result

    def tokenize(self, text: str) -> List[DomainToken]:
        """
        Tokenize text into domain-specific and regular text tokens.

        Args:
            text: Input text

        Returns:
            List of DomainToken objects covering the entire text
        """
        domain_tokens = self.detect_domains(text)

        # Fill gaps with TEXT tokens
        result = []
        pos = 0

        for token in domain_tokens:
            # Add text before this token
            if pos < token.start:
                result.append(
                    DomainToken(
                        text=text[pos : token.start],
                        domain=DomainType.TEXT,
                        start=pos,
                        end=token.start,
                    )
                )

            result.append(token)
            pos = token.end

        # Add remaining text
        if pos < len(text):
            result.append(
                DomainToken(
                    text=text[pos:],
                    domain=DomainType.TEXT,
                    start=pos,
                    end=len(text),
                )
            )

        return result

    def get_domain_spans(self, text: str) -> List[Tuple[int, int, DomainType]]:
        """
        Get list of (start, end, domain) tuples for the text.

        Useful for downstream processing.
        """
        tokens = self.tokenize(text)
        return [(t.start, t.end, t.domain) for t in tokens]

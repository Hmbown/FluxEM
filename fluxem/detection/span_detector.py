"""Span detection for FluxEM.

Automatically identifies domain-relevant spans in input text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from .patterns import DomainType, DomainPattern, DOMAIN_PATTERNS, get_all_patterns


@dataclass
class DetectedSpan:
    """A detected domain span in text.

    Attributes:
        start: Start character index in original text.
        end: End character index in original text.
        text: The matched text.
        domain: The detected domain type.
        confidence: Detection confidence (0-1).
        parsed_value: Pre-parsed structured value for encoder.
        pattern_name: Name of the pattern that matched.
    """

    start: int
    end: int
    text: str
    domain: DomainType
    confidence: float
    parsed_value: Any
    pattern_name: str = ""

    def __repr__(self) -> str:
        return (
            f"DetectedSpan({self.start}:{self.end}, "
            f"'{self.text[:20]}...', {self.domain.name}, "
            f"conf={self.confidence:.2f})"
        )

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: DetectedSpan) -> bool:
        """Check if this span overlaps with another."""
        return not (self.end <= other.start or self.start >= other.end)

    def contains(self, other: DetectedSpan) -> bool:
        """Check if this span fully contains another."""
        return self.start <= other.start and self.end >= other.end


@dataclass
class SpanDetectorConfig:
    """Configuration for span detection.

    Attributes:
        domains: Set of domains to detect (None = all).
        min_confidence: Minimum confidence threshold for detection.
        resolve_overlaps: Strategy for handling overlapping spans.
        max_spans: Maximum number of spans to return (None = unlimited).
    """

    domains: Optional[Set[DomainType]] = None
    min_confidence: float = 0.5
    resolve_overlaps: str = "priority"  # "priority", "longest", "first", "none"
    max_spans: Optional[int] = None


class SpanDetector:
    """Detect domain-relevant spans in text.

    This is the main interface for span detection in FluxEM.
    It uses pattern-based detection with configurable overlap resolution.

    Example:
        >>> detector = SpanDetector()
        >>> spans = detector.detect("Calculate 5! + C(10,3)")
        >>> for span in spans:
        ...     print(f"{span.text}: {span.domain.name}")
        5!: COMBINATORICS
        C(10,3): COMBINATORICS
    """

    def __init__(
        self,
        config: Optional[SpanDetectorConfig] = None,
        custom_patterns: Optional[List[DomainPattern]] = None,
    ):
        """Initialize the span detector.

        Args:
            config: Detection configuration.
            custom_patterns: Additional patterns to use.
        """
        self.config = config or SpanDetectorConfig()

        # Build pattern list
        self._patterns: List[DomainPattern] = []

        # Add built-in patterns for enabled domains
        for domain, patterns in DOMAIN_PATTERNS.items():
            if self.config.domains is None or domain in self.config.domains:
                self._patterns.extend(patterns)

        # Add custom patterns
        if custom_patterns:
            self._patterns.extend(custom_patterns)

        # Sort by priority (highest first)
        self._patterns.sort(key=lambda p: -p.priority)

    def detect(self, text: str) -> List[DetectedSpan]:
        """Detect all domain spans in text.

        Args:
            text: Input text to analyze.

        Returns:
            List of detected spans, sorted by position.
        """
        if not text:
            return []

        # Find all matches from all patterns
        all_spans: List[DetectedSpan] = []

        for pattern in self._patterns:
            if (
                self.config.domains is not None
                and pattern.domain not in self.config.domains
            ):
                continue

            matches = pattern.match(text)
            for start, end, parsed in matches:
                # Calculate confidence based on pattern priority
                confidence = min(1.0, pattern.priority / 100.0)

                if confidence >= self.config.min_confidence:
                    span = DetectedSpan(
                        start=start,
                        end=end,
                        text=text[start:end],
                        domain=pattern.domain,
                        confidence=confidence,
                        parsed_value=parsed,
                        pattern_name=pattern.name,
                    )
                    all_spans.append(span)

        # Resolve overlaps
        if self.config.resolve_overlaps != "none" and all_spans:
            all_spans = self._resolve_overlaps(all_spans)

        # Sort by position
        all_spans.sort(key=lambda s: (s.start, -s.end))

        # Apply max_spans limit
        if self.config.max_spans is not None:
            all_spans = all_spans[: self.config.max_spans]

        return all_spans

    def detect_domain(
        self, text: str, domain: DomainType
    ) -> List[DetectedSpan]:
        """Detect spans for a specific domain.

        Args:
            text: Input text to analyze.
            domain: Domain type to detect.

        Returns:
            List of detected spans for the specified domain.
        """
        all_spans = self.detect(text)
        return [s for s in all_spans if s.domain == domain]

    def _resolve_overlaps(
        self, spans: List[DetectedSpan]
    ) -> List[DetectedSpan]:
        """Resolve overlapping spans according to configuration.

        Args:
            spans: List of possibly overlapping spans.

        Returns:
            List of non-overlapping spans.
        """
        if not spans:
            return spans

        strategy = self.config.resolve_overlaps

        if strategy == "priority":
            # Higher priority patterns win
            return self._resolve_by_priority(spans)
        elif strategy == "longest":
            # Longer spans win
            return self._resolve_by_length(spans)
        elif strategy == "first":
            # Earlier spans win
            return self._resolve_by_position(spans)
        else:
            return spans

    def _resolve_by_priority(
        self, spans: List[DetectedSpan]
    ) -> List[DetectedSpan]:
        """Keep spans with highest confidence when overlapping."""
        # Sort by confidence descending, then by length descending
        sorted_spans = sorted(
            spans, key=lambda s: (-s.confidence, -s.length, s.start)
        )

        result: List[DetectedSpan] = []
        for span in sorted_spans:
            if not any(span.overlaps(existing) for existing in result):
                result.append(span)

        return result

    def _resolve_by_length(
        self, spans: List[DetectedSpan]
    ) -> List[DetectedSpan]:
        """Keep longest spans when overlapping."""
        sorted_spans = sorted(
            spans, key=lambda s: (-s.length, -s.confidence, s.start)
        )

        result: List[DetectedSpan] = []
        for span in sorted_spans:
            if not any(span.overlaps(existing) for existing in result):
                result.append(span)

        return result

    def _resolve_by_position(
        self, spans: List[DetectedSpan]
    ) -> List[DetectedSpan]:
        """Keep earlier spans when overlapping."""
        sorted_spans = sorted(spans, key=lambda s: (s.start, -s.length))

        result: List[DetectedSpan] = []
        for span in sorted_spans:
            if not any(span.overlaps(existing) for existing in result):
                result.append(span)

        return result

    def get_non_span_regions(
        self, text: str, spans: List[DetectedSpan]
    ) -> List[Tuple[int, int]]:
        """Get regions of text that are not covered by spans.

        Args:
            text: Original text.
            spans: Detected spans.

        Returns:
            List of (start, end) tuples for non-span regions.
        """
        if not spans:
            return [(0, len(text))]

        # Sort spans by position
        sorted_spans = sorted(spans, key=lambda s: s.start)

        regions = []
        pos = 0

        for span in sorted_spans:
            if span.start > pos:
                regions.append((pos, span.start))
            pos = max(pos, span.end)

        if pos < len(text):
            regions.append((pos, len(text)))

        return regions

    def annotate(self, text: str) -> str:
        """Annotate text with detected span markers.

        Args:
            text: Input text.

        Returns:
            Text with span markers like [DOMAIN:text].
        """
        spans = self.detect(text)

        if not spans:
            return text

        # Sort by position descending to insert markers from end
        sorted_spans = sorted(spans, key=lambda s: -s.start)

        result = text
        for span in sorted_spans:
            marker = f"[{span.domain.name}:{span.text}]"
            result = result[: span.start] + marker + result[span.end :]

        return result


# =============================================================================
# Convenience Functions
# =============================================================================

_default_detector: Optional[SpanDetector] = None


def get_default_detector() -> SpanDetector:
    """Get the default span detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = SpanDetector()
    return _default_detector


def detect_spans(text: str) -> List[DetectedSpan]:
    """Detect spans using the default detector.

    Args:
        text: Input text.

    Returns:
        List of detected spans.
    """
    return get_default_detector().detect(text)


def detect_numbers(text: str) -> List[DetectedSpan]:
    """Detect arithmetic spans (numbers) in text."""
    detector = SpanDetector(
        SpanDetectorConfig(domains={DomainType.ARITHMETIC})
    )
    return detector.detect(text)


def detect_combinatorics(text: str) -> List[DetectedSpan]:
    """Detect combinatorics spans (factorials, combinations, etc.)."""
    detector = SpanDetector(
        SpanDetectorConfig(domains={DomainType.COMBINATORICS})
    )
    return detector.detect(text)


def detect_dates(text: str) -> List[DetectedSpan]:
    """Detect temporal spans (dates) in text."""
    detector = SpanDetector(
        SpanDetectorConfig(domains={DomainType.TEMPORAL})
    )
    return detector.detect(text)


def detect_formulas(text: str) -> List[DetectedSpan]:
    """Detect chemistry spans (molecular formulas) in text."""
    detector = SpanDetector(
        SpanDetectorConfig(domains={DomainType.CHEMISTRY})
    )
    return detector.detect(text)


def detect_quantities(text: str) -> List[DetectedSpan]:
    """Detect physics spans (quantities with units) in text."""
    detector = SpanDetector(
        SpanDetectorConfig(domains={DomainType.PHYSICS})
    )
    return detector.detect(text)

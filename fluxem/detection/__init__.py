"""Span detection module for FluxEM.

Automatically identifies domain-relevant spans in input text
and prepares them for encoding.
"""

from .span_detector import SpanDetector, DetectedSpan
from .patterns import DOMAIN_PATTERNS

__all__ = [
    "SpanDetector",
    "DetectedSpan",
    "DOMAIN_PATTERNS",
]

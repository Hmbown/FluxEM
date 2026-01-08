#!/usr/bin/env python3
"""
Demo: FluxEM end-to-end tool chaining for a factorial -> binomial entropy query.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path for local imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from fluxem.detection.span_detector import SpanDetector
from fluxem.pipeline.inference import FluxEMPipeline

try:
    from fluxem.composition.operators import CompositionOperator
except Exception:
    CompositionOperator = None

from experiments.qwen3_toolcalling.tool_registry import create_tool_registry


def main() -> None:
    prompt = (
        "What's the entropy of a binomial distribution with n=5! trials and p=0.3?"
    )

    detector = SpanDetector()
    tool_registry = create_tool_registry()

    composer = None
    if CompositionOperator is not None:
        try:
            composer = CompositionOperator()
        except Exception:
            composer = None

    pipeline = FluxEMPipeline(
        detector=detector,
        tool_registry=tool_registry,
        composer=composer,
    )

    spans = detector.detect(prompt)
    print("Detected spans:")
    for span in spans:
        print(f"- {span.domain.name}: '{span.text}' -> {span.parsed_value}")

    result = pipeline(prompt)
    print("\nResult:")
    print(result)


if __name__ == "__main__":
    main()


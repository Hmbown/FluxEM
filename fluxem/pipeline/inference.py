"""
End-to-end inference pipeline for FluxEM + LLM integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import ast
import json
import math
import re

from ..detection.span_detector import SpanDetector, DetectedSpan
from ..detection.patterns import DomainType
from ..domains.combinatorics import CombinatoricsEncoder, CombinatorialTerm
from ..domains.probability import ProbabilityEncoder, ProbabilityDistribution
from ..domains.information_theory import entropy as info_entropy
from ..domains.math.arithmetic import ArithmeticEncoder
from ..composition.operators import CompositionOperator


@dataclass
class PipelineConfig:
    """Configuration for FluxEMPipeline."""

    max_tool_loops: int = 3
    composition_operation: str = "chain"


@dataclass
class ToolCall:
    """A single tool call."""

    name: str
    args: Any
    domain: Optional[str] = None
    span: Optional[DetectedSpan] = None
    result: Any = None
    error: Optional[str] = None


class SpanEncoderRegistry:
    """Map detected spans to FluxEM embeddings."""

    def __init__(self) -> None:
        self.arithmetic = ArithmeticEncoder()
        self.combinatorics = CombinatoricsEncoder()
        self.probability = ProbabilityEncoder()

    def encode_span(self, span: DetectedSpan) -> Optional[Any]:
        if span.domain == DomainType.ARITHMETIC:
            return self.arithmetic.encode(span.parsed_value)
        if span.domain == DomainType.COMBINATORICS:
            return self._encode_combinatorics(span.parsed_value)
        if span.domain == DomainType.PROBABILITY:
            return self._encode_probability(span.parsed_value)
        return None

    def encode_tool_result(self, result: Any, domain: Optional[str]) -> Optional[Any]:
        if isinstance(result, (int, float)):
            return self.arithmetic.encode(result)
        if isinstance(result, list) and result and all(
            isinstance(v, (int, float)) for v in result
        ):
            try:
                return self.probability.encode(result)
            except Exception:
                return None
        return None

    def _encode_combinatorics(self, parsed: Any) -> Optional[Any]:
        if not isinstance(parsed, dict):
            return None
        kind = parsed.get("type")
        n_val = parsed.get("n")
        k_val = parsed.get("k")
        if kind == "factorial" and n_val is not None:
            term = CombinatorialTerm(kind="factorial", n=int(n_val))
            return self.combinatorics.encode(term)
        if kind == "combination" and n_val is not None and k_val is not None:
            term = CombinatorialTerm(kind="ncr", n=int(n_val), k=int(k_val))
            return self.combinatorics.encode(term)
        if kind == "permutation" and n_val is not None and k_val is not None:
            term = CombinatorialTerm(kind="npr", n=int(n_val), k=int(k_val))
            return self.combinatorics.encode(term)
        return None

    def _encode_probability(self, parsed: Any) -> Optional[Any]:
        if not isinstance(parsed, dict):
            return None
        kind = parsed.get("kind")
        if kind == "bernoulli":
            p_val = parsed.get("p")
            if p_val is None:
                return None
            dist = ProbabilityDistribution(kind="bernoulli", p=float(p_val))
            return self.probability.encode(dist)
        if kind == "binomial":
            p_val = parsed.get("p")
            n_val = parsed.get("n")
            if p_val is None or n_val is None:
                return None
            dist = ProbabilityDistribution(
                kind="binomial", n=int(n_val), p=float(p_val)
            )
            return self.probability.encode(dist)
        return None


class ToolExecutor:
    """Execute tool calls from a registry."""

    def __init__(self, tool_registry: Optional[Dict[str, Any]] = None) -> None:
        self.tool_registry = tool_registry or {}

    def call(self, name: str, args: Any) -> Any:
        tool = self.tool_registry.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found.")
        func = getattr(tool, "function", tool)
        return func(args)


class FluxEMPipeline:
    """Complete inference pipeline for FluxEM + LLM integration."""

    def __init__(
        self,
        model: Optional[Any] = None,
        detector: Optional[SpanDetector] = None,
        span_encoder: Optional[SpanEncoderRegistry] = None,
        composer: Optional[CompositionOperator] = None,
        tool_registry: Optional[Dict[str, Any]] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.model = model
        self.detector = detector or SpanDetector()
        self.span_encoder = span_encoder or SpanEncoderRegistry()
        self.composer = composer
        self.tool_executor = ToolExecutor(tool_registry)
        self.config = config or PipelineConfig()

    def __call__(self, prompt: str) -> str:
        spans = self.detector.detect(prompt)
        span_embeddings = self._encode_spans(spans)

        output = self._generate(prompt, spans, span_embeddings)
        tool_calls = self._extract_tool_calls(output)
        if not tool_calls:
            tool_calls = self._plan_tool_chain(prompt, spans)

        loop_count = 0
        while tool_calls and loop_count < self.config.max_tool_loops:
            results = self._execute_tools(tool_calls, prompt, spans)
            composed = self._compose_results(results)

            if self.model is not None and composed is not None:
                output = self.model.continue_generation(prompt, composed)
                tool_calls = self._extract_tool_calls(output)
            else:
                return self._format_tool_results(prompt, results)

            loop_count += 1

        return output

    def _generate(
        self,
        prompt: str,
        spans: List[DetectedSpan],
        span_embeddings: List[Any],
    ) -> str:
        if self.model is None:
            return prompt
        return self.model.generate(
            prompt,
            spans=spans,
            span_embeddings=span_embeddings,
        )

    def _encode_spans(self, spans: List[DetectedSpan]) -> List[Any]:
        return [self.span_encoder.encode_span(span) for span in spans]

    def _plan_tool_chain(
        self,
        prompt: str,
        spans: List[DetectedSpan],
    ) -> List[ToolCall]:
        lower_prompt = prompt.lower()
        calls: List[ToolCall] = []

        factorial_spans = [
            span
            for span in spans
            if span.domain == DomainType.COMBINATORICS
            and isinstance(span.parsed_value, dict)
            and span.parsed_value.get("type") == "factorial"
        ]
        binomial_spans = [
            span
            for span in spans
            if span.domain == DomainType.PROBABILITY
            and isinstance(span.parsed_value, dict)
            and span.parsed_value.get("kind") == "binomial"
        ]

        if not binomial_spans:
            match = re.search(
                r"binomial.*?n\s*=\s*(\d+)\s*!?\s*.*?p\s*=\s*([0-9.]+)",
                prompt,
                re.IGNORECASE,
            )
            if match:
                n_val = int(match.group(1))
                p_val = float(match.group(2))
                binomial_spans = [
                    DetectedSpan(
                        start=match.start(),
                        end=match.end(),
                        text=match.group(0),
                        domain=DomainType.PROBABILITY,
                        confidence=0.5,
                        parsed_value={"kind": "binomial", "n": n_val, "p": p_val},
                        pattern_name="binomial_prompt",
                    )
                ]

        for span in factorial_spans:
            n_val = span.parsed_value.get("n")
            if n_val is not None:
                calls.append(
                    ToolCall(
                        name="combinatorics_factorial",
                        args=[int(n_val)],
                        domain="combinatorics",
                        span=span,
                    )
                )

        if "entropy" in lower_prompt and binomial_spans:
            for span in binomial_spans:
                calls.append(
                    ToolCall(
                        name="info_entropy",
                        args=span.parsed_value,
                        domain="information_theory",
                        span=span,
                    )
                )
        return calls

    def _execute_tools(
        self,
        calls: List[ToolCall],
        prompt: str,
        spans: List[DetectedSpan],
    ) -> List[ToolCall]:
        results: List[ToolCall] = []
        factorial_value = None
        factorial_input = None

        for call in calls:
            if call.name == "combinatorics_factorial":
                factorial_input = call.args[0] if isinstance(call.args, list) else None
                try:
                    call.result = self.tool_executor.call(call.name, call.args)
                    factorial_value = call.result
                except Exception as exc:
                    call.error = str(exc)
                results.append(call)

        for call in calls:
            if call.name == "info_entropy":
                parsed = call.args if isinstance(call.args, dict) else {}
                if parsed.get("kind") == "binomial":
                    n_val = parsed.get("n")
                    p_val = parsed.get("p")
                    if n_val is None or p_val is None:
                        call.error = "Missing n or p for binomial entropy."
                        results.append(call)
                        continue
                    if (
                        factorial_value is not None
                        and factorial_input is not None
                        and n_val == factorial_input
                    ):
                        n_val = factorial_value
                    probs = self._binomial_probs(int(n_val), float(p_val))
                    call.args = probs
                    try:
                        call.result = self._call_entropy_tool(probs)
                    except Exception as exc:
                        call.error = str(exc)
                    results.append(call)
                else:
                    call.error = "Unsupported info_entropy args."
                    results.append(call)

        return results

    def _call_entropy_tool(self, probs: List[float]) -> float:
        try:
            return float(self.tool_executor.call("info_entropy", probs))
        except Exception:
            return float(info_entropy(probs))

    def _binomial_probs(self, n_val: int, p_val: float) -> List[float]:
        probs = []
        for k in range(n_val + 1):
            coeff = math.comb(n_val, k)
            probs.append(coeff * (p_val ** k) * ((1.0 - p_val) ** (n_val - k)))
        return probs

    def _compose_results(self, results: List[ToolCall]) -> Optional[Any]:
        if self.composer is None:
            return None
        embeddings = []
        for result in results:
            emb = self.span_encoder.encode_tool_result(result.result, result.domain)
            if emb is not None:
                embeddings.append(self._to_torch(emb))
        if not embeddings:
            return None
        stacked = self._stack_embeddings(embeddings)
        return self.composer.compose_many(
            stacked, operation=self.config.composition_operation
        )

    def _stack_embeddings(self, embeddings: List[Any]) -> Any:
        import torch

        return torch.stack(embeddings, dim=0).unsqueeze(0)

    def _to_torch(self, emb: Any) -> Any:
        import torch
        import numpy as np

        if isinstance(emb, torch.Tensor):
            return emb.float()
        array = np.array(emb, dtype=np.float32).reshape(-1)
        return torch.tensor(array, dtype=torch.float32)

    def _extract_tool_calls(self, output: str) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        pattern = re.compile(
            r"\{\s*\"tool\"\s*:\s*\"(?P<tool>[^\"]+)\"\s*,\s*\"args\"\s*:\s*(?P<args>[^}]+)\}"
        )
        for match in pattern.finditer(output):
            tool_name = match.group("tool")
            raw_args = match.group("args").strip()
            parsed_args = raw_args
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                try:
                    parsed_args = ast.literal_eval(raw_args)
                except Exception:
                    parsed_args = raw_args
            tool_calls.append(
                ToolCall(name=tool_name, args=parsed_args)
            )
        return tool_calls

    def _format_tool_results(self, prompt: str, results: List[ToolCall]) -> str:
        lines = ["FluxEM tool execution results:"]
        for result in results:
            if result.error:
                lines.append(f"- {result.name}: error={result.error}")
            else:
                lines.append(f"- {result.name}: {result.result}")
        return "\n".join(lines)

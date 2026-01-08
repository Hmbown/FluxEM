"""
Hybrid model wrapper for injecting FluxEM embeddings into LLM hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    _TORCH_IMPORT_ERROR = exc

    class _MissingTorch:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "torch is required for fluxem.injection.hybrid_model."
                ) from _TORCH_IMPORT_ERROR

        def __getattr__(self, name: str):
            raise ImportError(
                "torch is required for fluxem.injection.hybrid_model."
            ) from _TORCH_IMPORT_ERROR

    torch = _MissingTorch()
    nn = _MissingTorch()

from ..detection.span_detector import DetectedSpan, SpanDetector
from .projector import FluxEMProjector
from .injector import EmbeddingInjector, SpanBatch


SpanEncoder = Callable[[DetectedSpan], Optional[Any]]


@dataclass
class HybridModelConfig:
    """Configuration for HybridModel."""

    max_spans: int = 16
    span_strategy: str = "first"  # first or all
    device: Optional[str] = None


class HybridModel(nn.Module):
    """
    Wraps an LLM to inject FluxEM embeddings during forward/generation.

    The wrapper handles:
    - Span detection and encoding
    - Projection to LLM hidden space
    - Injection into token embeddings or hidden states
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        projector: FluxEMProjector,
        injector: EmbeddingInjector,
        detector: Optional[SpanDetector] = None,
        span_encoder: Optional[SpanEncoder] = None,
        config: Optional[HybridModelConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.projector = projector
        self.injector = injector
        self.detector = detector
        self.span_encoder = span_encoder
        self.config = config or HybridModelConfig()

        device = self.config.device
        if device is None and hasattr(base_model, "device"):
            device = str(getattr(base_model, "device"))
        self.device = torch.device(device) if device else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        spans: Optional[List[DetectedSpan]] = None,
        span_embeddings: Optional[Sequence[Any]] = None,
        span_positions: Optional[Sequence[int]] = None,
        offsets: Optional[Sequence[Tuple[int, int]]] = None,
        **kwargs,
    ) -> Any:
        """Forward pass with optional FluxEM injection."""
        input_ids = input_ids.to(self.device) if self.device else input_ids
        if attention_mask is not None and self.device:
            attention_mask = attention_mask.to(self.device)

        span_batch = self._build_span_batch(
            spans=spans,
            span_embeddings=span_embeddings,
            span_positions=span_positions,
            offsets=offsets,
        )

        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        if span_batch is not None:
            inputs_embeds = self.injector.inject(inputs_embeds, span_batch)

        return self.base_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def generate(
        self,
        prompt: str,
        spans: Optional[List[DetectedSpan]] = None,
        span_embeddings: Optional[Sequence[Any]] = None,
        span_positions: Optional[Sequence[int]] = None,
        **gen_kwargs,
    ) -> str:
        """Generate text with FluxEM injection."""
        encoding = self._encode_prompt(prompt)
        input_ids = encoding["input_ids"]
        attention_mask = encoding.get("attention_mask")
        offsets = encoding.get("offset_mapping")
        if offsets is not None:
            offsets = self._normalize_offsets(offsets)

        if spans is None and self.detector is not None:
            spans = self.detector.detect(prompt)

        span_batch = self._build_span_batch(
            spans=spans,
            span_embeddings=span_embeddings,
            span_positions=span_positions,
            offsets=offsets,
        )

        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        if span_batch is not None:
            inputs_embeds = self.injector.inject(inputs_embeds, span_batch)

        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        if hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return str(outputs)

    def continue_generation(
        self,
        prompt: str,
        composed_embeddings: Any,
        **gen_kwargs,
    ) -> str:
        """Continue generation using composed tool embeddings."""
        encoding = self._encode_prompt(prompt)
        input_ids = encoding["input_ids"]
        seq_len = input_ids.shape[1]

        span_positions = [seq_len - 1]
        return self.generate(
            prompt,
            span_positions=span_positions,
            span_embeddings=[composed_embeddings],
            **gen_kwargs,
        )

    def _encode_prompt(self, prompt: str) -> dict:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for HybridModel.generate.")
        try:
            encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except TypeError:
            encoding = self.tokenizer(prompt, return_tensors="pt")

        if self.device:
            encoding["input_ids"] = encoding["input_ids"].to(self.device)
            if "attention_mask" in encoding:
                encoding["attention_mask"] = encoding["attention_mask"].to(self.device)
        return encoding

    def _normalize_offsets(self, offsets: Any) -> List[Tuple[int, int]]:
        if isinstance(offsets, torch.Tensor):
            offsets = offsets[0].tolist()
        elif isinstance(offsets, list) and offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        return [(int(start), int(end)) for start, end in offsets]

    def _build_span_batch(
        self,
        spans: Optional[List[DetectedSpan]],
        span_embeddings: Optional[Sequence[Any]],
        span_positions: Optional[Sequence[int]],
        offsets: Optional[Sequence[Tuple[int, int]]],
    ) -> Optional[SpanBatch]:
        if span_positions is None:
            if spans is None:
                return None
            if span_embeddings is None and self.span_encoder is not None:
                span_embeddings = [self.span_encoder(span) for span in spans]
            if span_embeddings is None:
                return None
            span_positions, span_embeddings = self._align_spans(
                spans, span_embeddings, offsets
            )

        if not span_positions or not span_embeddings:
            return None

        max_spans = self.config.max_spans
        span_positions = list(span_positions)[:max_spans]
        span_embeddings = list(span_embeddings)[:max_spans]

        input_dim = self.projector.config.input_dim
        position_tensor = torch.full(
            (max_spans,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        embedding_tensor = torch.zeros(
            (max_spans, input_dim),
            dtype=torch.float32,
            device=self.device,
        )

        for idx, (pos, emb) in enumerate(zip(span_positions, span_embeddings)):
            if emb is None:
                continue
            position_tensor[idx] = int(pos)
            embedding_tensor[idx] = self._to_torch(emb, input_dim)

        projected = self.projector(embedding_tensor.unsqueeze(0))
        mask = position_tensor >= 0
        return SpanBatch(
            positions=position_tensor.unsqueeze(0),
            embeddings=projected,
            mask=mask.unsqueeze(0),
        )

    def _align_spans(
        self,
        spans: Sequence[DetectedSpan],
        span_embeddings: Sequence[Any],
        offsets: Optional[Sequence[Tuple[int, int]]],
    ) -> Tuple[List[int], List[Any]]:
        if offsets is None:
            return [], []

        positions: List[int] = []
        embeddings: List[Any] = []

        for span, emb in zip(spans, span_embeddings):
            if emb is None:
                continue
            token_indices = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if start < span.end and end > span.start
            ]
            if not token_indices:
                continue

            if self.config.span_strategy == "all":
                for idx in token_indices:
                    positions.append(idx)
                    embeddings.append(emb)
            else:
                positions.append(token_indices[0])
                embeddings.append(emb)

        return positions, embeddings

    def _to_torch(self, emb: Any, input_dim: int) -> torch.Tensor:
        if isinstance(emb, torch.Tensor):
            return emb.to(self.device) if self.device else emb
        array = np.array(emb, dtype=np.float32).reshape(-1)
        if array.shape[-1] != input_dim:
            raise ValueError(
                f"Expected embedding dim {input_dim}, got {array.shape[-1]}"
            )
        return torch.tensor(array, dtype=torch.float32, device=self.device)

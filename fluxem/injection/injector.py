"""
Embedding injector for merging FluxEM projections into LLM hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    _TORCH_IMPORT_ERROR = exc

    class _MissingTorch:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "torch is required for fluxem.injection.injector."
                ) from _TORCH_IMPORT_ERROR

        def __getattr__(self, name: str):
            raise ImportError(
                "torch is required for fluxem.injection.injector."
            ) from _TORCH_IMPORT_ERROR

    torch = _MissingTorch()
    nn = _MissingTorch()


@dataclass
class InjectionConfig:
    """Configuration for EmbeddingInjector."""

    mode: str = "replace"  # replace, add, cross_attention
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.0
    cross_attention_scale: float = 1.0


@dataclass
class SpanBatch:
    """Batch of span positions and projected embeddings."""

    positions: torch.Tensor  # (batch, max_spans)
    embeddings: torch.Tensor  # (batch, max_spans, hidden_dim)
    mask: Optional[torch.Tensor] = None  # (batch, max_spans) True for valid

    def valid_mask(self) -> torch.Tensor:
        if self.mask is not None:
            return self.mask
        return self.positions >= 0


class EmbeddingInjector(nn.Module):
    """
    Injects projected FluxEM embeddings into hidden states.

    Strategies:
    - replace: replace token embeddings at span positions
    - add: add projected embeddings to token embeddings at span positions
    - cross_attention: attend from hidden states to FluxEM embeddings
    """

    def __init__(self, hidden_dim: int, config: Optional[InjectionConfig] = None):
        super().__init__()
        self.config = config or InjectionConfig()
        self.hidden_dim = hidden_dim

        self.cross_attn = None
        if self.config.mode == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=self.config.cross_attention_heads,
                dropout=self.config.cross_attention_dropout,
                batch_first=True,
            )

    def inject(
        self,
        hidden_states: torch.Tensor,
        span_batch: Optional[SpanBatch],
    ) -> torch.Tensor:
        """Inject projected embeddings into hidden states."""
        if span_batch is None:
            return hidden_states

        embeddings = span_batch.embeddings
        positions = span_batch.positions
        if embeddings.numel() == 0:
            return hidden_states

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)

        batch_size, seq_len, _ = hidden_states.shape
        mode = self.config.mode

        if mode in {"replace", "add"}:
            output = hidden_states.clone()
            for b in range(batch_size):
                for s in range(positions.shape[1]):
                    pos = int(positions[b, s].item())
                    if 0 <= pos < seq_len:
                        if mode == "replace":
                            output[b, pos] = embeddings[b, s]
                        else:
                            output[b, pos] = output[b, pos] + embeddings[b, s]
            return output

        if mode == "cross_attention":
            if self.cross_attn is None:
                raise RuntimeError("cross_attention mode requires attention module")
            valid_mask = span_batch.valid_mask()
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)
            key_padding_mask = ~valid_mask
            attn_output, _ = self.cross_attn(
                query=hidden_states,
                key=embeddings,
                value=embeddings,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            return hidden_states + attn_output * self.config.cross_attention_scale

        raise ValueError(f"Unknown injection mode: {mode}")


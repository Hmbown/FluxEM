"""
Composition operators for combining FluxEM embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..core.base import EMBEDDING_DIM

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    _TORCH_IMPORT_ERROR = exc

    class _MissingTorch:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "torch is required for fluxem.composition.operators."
                ) from _TORCH_IMPORT_ERROR

        def __getattr__(self, name: str):
            raise ImportError(
                "torch is required for fluxem.composition.operators."
            ) from _TORCH_IMPORT_ERROR

    torch = _MissingTorch()
    nn = _MissingTorch()


@dataclass
class CompositionConfig:
    """Configuration for CompositionOperator."""

    embedding_dim: int = EMBEDDING_DIM
    strategy: str = "concat_project"  # concat_project, gated_fusion, attention
    num_heads: int = 4
    dropout: float = 0.0


class CompositionOperator(nn.Module):
    """
    Combine embeddings from multiple tool outputs.

    Supported operations:
    - chain: sequential composition (a feeds into b)
    - parallel: independent results combined
    - conditional: choose based on embedding content
    """

    def __init__(self, config: Optional[CompositionConfig] = None):
        super().__init__()
        self.config = config or CompositionConfig()

        strategy = self.config.strategy
        dim = self.config.embedding_dim

        self.concat_project = None
        self.gate = None
        self.attn = None

        if strategy == "concat_project":
            self.concat_project = nn.Linear(dim * 2, dim)
        elif strategy == "gated_fusion":
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid(),
            )
        elif strategy == "attention":
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown composition strategy: {strategy}")

        self.conditional_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def compose(self, emb_a: torch.Tensor, emb_b: torch.Tensor, operation: str) -> torch.Tensor:
        """Compose two embeddings with the selected operation."""
        single = emb_a.dim() == 1
        if single:
            emb_a = emb_a.unsqueeze(0)
            emb_b = emb_b.unsqueeze(0)

        if operation not in {"chain", "parallel", "conditional"}:
            raise ValueError(f"Unknown composition operation: {operation}")

        if operation == "conditional":
            gate = self.conditional_gate(torch.cat([emb_a, emb_b], dim=-1))
            output = gate * emb_a + (1.0 - gate) * emb_b
            return output[0] if single else output

        output = self._combine(emb_a, emb_b)
        return output[0] if single else output

    def compose_many(
        self,
        embeddings: torch.Tensor,
        operation: str = "chain",
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compose a sequence of embeddings.

        Args:
            embeddings: Tensor of shape (batch, seq, dim) or (seq, dim).
            operation: Composition operation for pairwise reduction.
            mask: Optional mask for attention strategy (batch, seq).
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            single = True
        else:
            single = False

        if self.config.strategy == "attention":
            if self.attn is None:
                raise RuntimeError("Attention module not initialized.")
            key_padding_mask = None
            if mask is not None:
                key_padding_mask = ~mask
            attn_out, _ = self.attn(
                embeddings,
                embeddings,
                embeddings,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            output = attn_out.mean(dim=1)
            return output[0] if single else output

        output = embeddings[:, 0]
        for idx in range(1, embeddings.shape[1]):
            output = self.compose(output, embeddings[:, idx], operation=operation)
        return output[0] if single else output

    def _combine(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        strategy = self.config.strategy
        if strategy == "concat_project":
            return self.concat_project(torch.cat([emb_a, emb_b], dim=-1))
        if strategy == "gated_fusion":
            gate = self.gate(torch.cat([emb_a, emb_b], dim=-1))
            return gate * emb_a + (1.0 - gate) * emb_b
        if strategy == "attention":
            seq = torch.stack([emb_a, emb_b], dim=1)
            attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
            return attn_out.mean(dim=1)
        raise ValueError(f"Unknown composition strategy: {strategy}")


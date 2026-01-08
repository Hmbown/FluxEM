"""
Projection layers for mapping FluxEM embeddings into LLM hidden space.
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
                    "torch is required for fluxem.injection.projector."
                ) from _TORCH_IMPORT_ERROR

        def __getattr__(self, name: str):
            raise ImportError(
                "torch is required for fluxem.injection.projector."
            ) from _TORCH_IMPORT_ERROR

    torch = _MissingTorch()
    nn = _MissingTorch()


@dataclass
class ProjectorConfig:
    """Configuration for FluxEMProjector."""

    input_dim: int = EMBEDDING_DIM
    output_dim: int = 2048
    hidden_dim: int = 1024
    num_layers: int = 2
    dropout: float = 0.0
    use_layer_norm: bool = True


class FluxEMProjector(nn.Module):
    """
    Simple MLP projection from FluxEM space to LLM hidden space.

    Input:  FluxEM embeddings (default 256-dim)
    Output: LLM hidden states (default 2048-dim)
    """

    def __init__(self, config: Optional[ProjectorConfig] = None):
        super().__init__()
        self.config = config or ProjectorConfig()

        layers = []
        if self.config.num_layers <= 1:
            layers.append(nn.Linear(self.config.input_dim, self.config.output_dim))
        else:
            current_dim = self.config.input_dim
            for _ in range(self.config.num_layers - 1):
                layers.append(nn.Linear(current_dim, self.config.hidden_dim))
                layers.append(nn.GELU())
                if self.config.dropout > 0:
                    layers.append(nn.Dropout(self.config.dropout))
                current_dim = self.config.hidden_dim
            layers.append(nn.Linear(current_dim, self.config.output_dim))

        self.network = nn.Sequential(*layers)
        self.norm = (
            nn.LayerNorm(self.config.output_dim)
            if self.config.use_layer_norm
            else None
        )

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project FluxEM embeddings into LLM hidden space."""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        x = self.network(x)
        if self.norm is not None:
            x = self.norm(x)

        return x[0] if single else x


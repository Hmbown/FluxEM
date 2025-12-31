"""Centralized configuration for FluxEM benchmark."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class BenchmarkConfig:
    """All benchmark hyperparameters."""

    # Random seed
    seed: int = 42

    # Dataset sizes
    train_samples: int = 10_000
    test_samples: int = 1_000

    # FluxEM configuration
    fluxem_dim: int = 256
    fluxem_linear_scale: float = 1e10
    fluxem_log_scale: float = 35.0

    # Tokenizer
    max_seq_length: int = 64

    # Transformer baseline
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_ff: int = 128
    transformer_dropout: float = 0.1

    # GRU baseline
    gru_embed_dim: int = 32
    gru_hidden_dim: int = 64
    gru_num_layers: int = 1
    gru_bidirectional: bool = True
    gru_dropout: float = 0.1

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 50
    use_log_scale: bool = True

    # Evaluation
    tolerance: float = 0.01  # 1% relative error

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))

    @property
    def data_dir(self) -> Path:
        return self.output_dir / "datasets"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"


# Default configuration
DEFAULT_CONFIG = BenchmarkConfig()

# Quick test configuration
QUICK_CONFIG = BenchmarkConfig(
    train_samples=1000,
    test_samples=200,
    epochs=10,
)

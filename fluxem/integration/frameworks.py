"""
Framework adapters for FluxEM-LLM integration.

This module provides adapters to integrate MLX-based domain embeddings
with PyTorch and JAX-based LLMs.

Key design principles:
1. Keep domain encoders as MLX-native (they're compute-light)
2. Convert embeddings to target framework only when needed
3. Provide framework-specific projection heads and training utilities
"""

import warnings
from typing import Dict, Any, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
import importlib

from ..backend import get_backend

try:
    import mlx.nn as nn
except Exception as exc:
    _MLX_IMPORT_ERROR = exc

    class _MissingNN:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "MLX is required for fluxem.integration.frameworks."
                ) from _MLX_IMPORT_ERROR

        def __getattr__(self, name: str):
            raise ImportError(
                "MLX is required for fluxem.integration.frameworks."
            ) from _MLX_IMPORT_ERROR

    nn = _MissingNN()

T = TypeVar("T")

# =============================================================================
# Framework Detection and Import
# =============================================================================


class Framework:
    """Supported deep learning frameworks."""

    MLX = "mlx"
    PYTORCH = "torch"
    JAX = "jax"
    NUMPY = "numpy"


def detect_framework(tensor_like) -> str:
    """
    Detect the framework of a tensor-like object.

    Args:
        tensor_like: Any object that might be a tensor

    Returns:
        Framework name (mlx, torch, jax, or numpy)
    """
    if tensor_like is None:
        return Framework.NUMPY

    # Check MLX
    module_name = getattr(tensor_like.__class__, "__module__", "")
    if "mlx" in module_name:
        return Framework.MLX

    # Check PyTorch
    try:
        import torch

        if isinstance(tensor_like, torch.Tensor):
            return Framework.PYTORCH
    except ImportError:
        pass

    # Check JAX
    try:
        import jax.numpy as jnp

        if hasattr(tensor_like, "__jax_array__") or (
            hasattr(tensor_like, "__class__") and "jax" in str(tensor_like.__class__)
        ):
            return Framework.JAX
    except ImportError:
        pass

    # Check NumPy
    try:
        import numpy as np

        if isinstance(tensor_like, np.ndarray):
            return Framework.NUMPY
    except ImportError:
        pass

    # Default to numpy for Python scalars/lists
    return Framework.NUMPY


# =============================================================================
# Tensor Conversion Utilities
# =============================================================================


def to_framework(emb: Any, target_framework: str, device: Optional[str] = None):
    """
    Convert an MLX embedding to target framework.

    Args:
        emb: MLX array (128-dim embedding)
        target_framework: One of 'mlx', 'torch', 'jax', 'numpy'
        device: Optional device placement for target framework

    Returns:
        Embedding in target framework
    """
    if target_framework == Framework.MLX:
        return emb

    # First convert to numpy (common intermediate)
    np_emb = mx_to_numpy(emb)

    if target_framework == Framework.NUMPY:
        return np_emb

    elif target_framework == Framework.PYTORCH:
        import torch

        torch_emb = torch.from_numpy(np_emb).float()
        if device:
            torch_emb = torch_emb.to(device)
        return torch_emb

    elif target_framework == Framework.JAX:
        import jax.numpy as jnp

        jax_emb = jnp.array(np_emb)
        return jax_emb

    else:
        raise ValueError(f"Unsupported target framework: {target_framework}")


def from_framework(emb, source_framework: str) -> Any:
    """
    Convert any framework's embedding to MLX.

    Args:
        emb: Embedding in any framework
        source_framework: Framework of input embedding

    Returns:
        MLX array
    """
    if source_framework == Framework.MLX:
        return emb

    # Convert to numpy first
    if source_framework == Framework.NUMPY:
        np_emb = emb
    elif source_framework == Framework.PYTORCH:
        import torch

        np_emb = emb.detach().cpu().numpy()
    elif source_framework == Framework.JAX:
        import numpy as np

        np_emb = np.array(emb)
    else:
        raise ValueError(f"Unsupported source framework: {source_framework}")

    # Convert numpy to MLX
    return numpy_to_mx(np_emb)


def mx_to_numpy(emb: Any):
    """Convert MLX array to NumPy array."""
    import numpy as np

    return np.array(emb)


def numpy_to_mx(np_array):
    """Convert NumPy array to MLX array."""
    backend = get_backend()
    return backend.array(np_array)


# =============================================================================
# Framework-Specific Projection Heads
# =============================================================================


@dataclass
class ProjectorConfig:
    """Configuration for framework-specific projectors."""

    llm_hidden_dim: int = 4096
    domain_embed_dim: int = 128
    projection_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"


class MLXProjector(nn.Module):
    """MLX implementation of projection head."""

    def __init__(self, config: ProjectorConfig):
        super().__init__()

        layers = []
        current_dim = config.domain_embed_dim

        for i in range(config.num_layers):
            next_dim = (
                config.projection_dim
                if i < config.num_layers - 1
                else config.llm_hidden_dim
            )
            layers.append(nn.Linear(current_dim, next_dim))

            if i < config.num_layers - 1:
                if config.activation == "gelu":
                    layers.append(nn.GELU())
                elif config.activation == "relu":
                    layers.append(nn.ReLU())
                elif config.activation == "tanh":
                    layers.append(nn.Tanh())

                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))

            current_dim = next_dim

        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)


class PyTorchProjector:
    """PyTorch implementation of projection head."""

    def __init__(self, config: ProjectorConfig, device: str = "cuda"):
        import torch
        import torch.nn as tnn

        self.device = device
        self.config = config

        layers = []
        current_dim = config.domain_embed_dim

        for i in range(config.num_layers):
            next_dim = (
                config.projection_dim
                if i < config.num_layers - 1
                else config.llm_hidden_dim
            )
            layers.append(tnn.Linear(current_dim, next_dim))

            if i < config.num_layers - 1:
                if config.activation == "gelu":
                    layers.append(tnn.GELU())
                elif config.activation == "relu":
                    layers.append(tnn.ReLU())
                elif config.activation == "tanh":
                    layers.append(tnn.Tanh())

                if config.dropout > 0:
                    layers.append(tnn.Dropout(config.dropout))

            current_dim = next_dim

        self.layers = tnn.Sequential(*layers).to(device)

    def __call__(self, x):
        import torch

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(self.device)
        return self.layers(x)

    def to(self, device):
        self.device = device
        self.layers = self.layers.to(device)
        return self


class JAXProjector:
    """JAX/Flax implementation of projection head."""

    def __init__(self, config: ProjectorConfig):
        try:
            import flax.linen as nn
            import jax
            import jax.numpy as jnp

            self.nn = nn
            self.jax = jax
            self.jnp = jnp
        except ImportError:
            raise ImportError("JAX and Flax required for JAXProjector")

        self.config = config
        self._init_model()

    def _init_model(self):
        """Initialize Flax module."""

        class Projector(self.nn.Module):
            config: ProjectorConfig

            @self.nn.compact
            def __call__(self, x, training=False):
                dropout_rate = self.config.dropout if training else 0.0
                current_dim = self.config.domain_embed_dim

                for i in range(self.config.num_layers):
                    x = self.nn.Dense(
                        self.config.projection_dim
                        if i < self.config.num_layers - 1
                        else self.config.llm_hidden_dim
                    )(x)

                    if i < self.config.num_layers - 1:
                        if self.config.activation == "gelu":
                            x = self.nn.gelu(x)
                        elif self.config.activation == "relu":
                            x = self.nn.relu(x)
                        elif self.config.activation == "tanh":
                            x = jnp.tanh(x)

                        if dropout_rate > 0:
                            x = self.nn.Dropout(rate=dropout_rate)(
                                x, deterministic=not training
                            )

                return x

        self.model = Projector(self.config)

    def init(self, rng_key):
        """Initialize model parameters."""
        import jax

        dummy = self.jnp.ones((1, self.config.domain_embed_dim))
        return self.model.init(rng_key, dummy, training=False)

    def apply(self, params, x, training=False):
        """Apply model to input."""
        return self.model.apply(params, x, training=training)


# =============================================================================
# Unified Training Pipeline (Framework-Agnostic)
# =============================================================================


class FluxEMTrainingPipeline:
    """
    Unified training pipeline that works with multiple frameworks.

    This pipeline:
    1. Uses MLX domain encoders (lightweight, Apple Silicon optimized)
    2. Converts to target framework for projection
    3. Integrates with LLM training loop
    """

    def __init__(
        self,
        target_framework: str = Framework.MLX,
        llm_hidden_dim: int = 4096,
        device: Optional[str] = None,
        projector_config: Optional[ProjectorConfig] = None,
    ):
        """
        Initialize pipeline.

        Args:
            target_framework: Framework for projection/LLM integration
            llm_hidden_dim: Hidden dimension of target LLM
            device: Device for target framework (e.g., 'cuda', 'mps', 'cpu')
            projector_config: Optional custom projector configuration
        """
        self.target_framework = target_framework
        self.llm_hidden_dim = llm_hidden_dim
        self.device = device or self._default_device(target_framework)

        # Import tokenizer and registry (MLX-based)
        from .tokenizer import MultiDomainTokenizer, DomainType
        from .pipeline import DomainEncoderRegistry

        self.tokenizer = MultiDomainTokenizer()
        self.registry = DomainEncoderRegistry()

        # Projector configuration
        if projector_config is None:
            projector_config = ProjectorConfig(llm_hidden_dim=llm_hidden_dim)
        self.projector_config = projector_config

        # Initialize framework-specific projector
        self.projector = self._create_projector()

    def _default_device(self, framework: str) -> str:
        """Get default device for framework."""
        if framework == Framework.MLX:
            return "default"  # MLX handles device automatically
        elif framework == Framework.PYTORCH:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        elif framework == Framework.JAX:
            try:
                import jax

                devices = jax.devices()
                return str(devices[0]) if devices else "cpu"
            except ImportError:
                return "cpu"
        else:
            return "cpu"

    def _create_projector(self):
        """Create framework-specific projector."""
        if self.target_framework == Framework.MLX:
            return MLXProjector(self.projector_config)
        elif self.target_framework == Framework.PYTORCH:
            return PyTorchProjector(self.projector_config, device=self.device)
        elif self.target_framework == Framework.JAX:
            return JAXProjector(self.projector_config)
        else:
            raise ValueError(f"Unsupported framework: {self.target_framework}")

    def encode_text(self, text: str):
        """
        Encode text with domain embeddings.

        Returns framework-specific tensors ready for LLM integration.
        """
        backend = get_backend()
        # Tokenize (MLX-based)
        tokens = self.tokenizer.tokenize(text)

        # Process each token
        embeddings = []
        domain_mask = []

        for token in tokens:
            if token.domain.value == 0:  # DomainType.TEXT
                # Text token - return zeros (to be replaced by LLM embeddings)
                if self.target_framework == Framework.MLX:
                    emb = backend.zeros(self.llm_hidden_dim)
                elif self.target_framework == Framework.PYTORCH:
                    import torch

                    emb = torch.zeros(self.llm_hidden_dim, device=self.device)
                elif self.target_framework == Framework.JAX:
                    import jax.numpy as jnp

                    emb = jnp.zeros(self.llm_hidden_dim)
                else:
                    import numpy as np

                    emb = np.zeros(self.llm_hidden_dim)

                domain_mask.append(0.0)

            else:
                # Domain token - encode with MLX, convert, project
                domain_emb = self.registry.encode_token(token)
                if domain_emb is not None:
                    # Convert to target framework
                    target_emb = to_framework(
                        domain_emb, self.target_framework, self.device
                    )
                    # Project to LLM dimension
                    emb = self.projector(target_emb)
                    domain_mask.append(1.0)
                else:
                    # Fallback to zero embedding
                    if self.target_framework == Framework.MLX:
                        emb = backend.zeros(self.llm_hidden_dim)
                    elif self.target_framework == Framework.PYTORCH:
                        import torch

                        emb = torch.zeros(self.llm_hidden_dim, device=self.device)
                    elif self.target_framework == Framework.JAX:
                        import jax.numpy as jnp

                        emb = jnp.zeros(self.llm_hidden_dim)
                    else:
                        import numpy as np

                        emb = np.zeros(self.llm_hidden_dim)
                    domain_mask.append(0.0)

            embeddings.append(emb)

        # Stack embeddings
        if self.target_framework == Framework.MLX:
            stacked = backend.stack(embeddings, axis=0)
            mask = backend.array(domain_mask)
        elif self.target_framework == Framework.PYTORCH:
            import torch

            stacked = torch.stack(embeddings, dim=0)
            mask = torch.tensor(domain_mask, device=self.device)
        elif self.target_framework == Framework.JAX:
            import jax.numpy as jnp

            stacked = jnp.stack(embeddings, axis=0)
            mask = jnp.array(domain_mask)
        else:
            import numpy as np

            stacked = np.stack(embeddings, axis=0)
            mask = np.array(domain_mask)

        return {
            "embeddings": stacked,
            "domain_mask": mask,
            "tokens": tokens,
        }


# =============================================================================
# HuggingFace Integration Utilities
# =============================================================================


def create_hf_processor_class(base_tokenizer_class):
    """
    Create a HuggingFace processor class that adds domain tokenization.

    Usage:
        FluxEMProcessor = create_hf_processor_class(PreTrainedTokenizerBase)
    """
    from transformers import PreTrainedTokenizerBase

    class FluxEMProcessor(base_tokenizer_class):
        """HuggingFace processor with FluxEM domain tokenization."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.domain_tokenizer = MultiDomainTokenizer()

        def encode_plus_with_domains(self, text, **kwargs):
            """
            Encode text with domain annotations.

            Returns:
                Dictionary with 'input_ids', 'attention_mask', and 'domain_mask'
            """
            # First, get regular tokenization
            encoding = super().encode_plus(text, **kwargs)

            # Detect domains in original text
            tokens = self.domain_tokenizer.tokenize(text)

            # Map character positions to token positions
            # This is simplified - in practice need character-token alignment
            domain_mask = [0] * len(encoding["input_ids"])

            # TODO: Implement proper alignment between domain tokens and BPE tokens

            encoding["domain_mask"] = domain_mask
            return encoding

        def batch_encode_plus_with_domains(self, texts, **kwargs):
            """Batch version of encode_plus_with_domains."""
            return [self.encode_plus_with_domains(text, **kwargs) for text in texts]

    return FluxEMProcessor


# =============================================================================
# Module-level Exports
# =============================================================================

__all__ = [
    "Framework",
    "detect_framework",
    "to_framework",
    "from_framework",
    "mx_to_numpy",
    "numpy_to_mx",
    "ProjectorConfig",
    "MLXProjector",
    "PyTorchProjector",
    "JAXProjector",
    "FluxEMTrainingPipeline",
    "create_hf_processor_class",
]

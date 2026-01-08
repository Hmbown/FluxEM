"""Integration with FluxEM-LLM."""

import os

from .tokenizer import MultiDomainTokenizer, DomainToken, DomainType
from .pipeline import (
    TrainingPipeline,
    DomainEncoderRegistry,
    EncodedSequence,
    create_training_pipeline,
)
from .sample_format import (
    Sample,
    Span,
    ValidationResult,
    validate_sample,
    validate_jsonl_file,
    VALID_SPAN_TYPES,
)

PROJECTOR_AVAILABLE = False
if os.environ.get("FLUXEM_ENABLE_MLX") == "1":
    try:
        from .projector import (
            MultiDomainProjector,
            ProjectorConfig,
            DomainProjectionHead,
            HybridEmbedder,
        )

        PROJECTOR_AVAILABLE = True
    except Exception:
        PROJECTOR_AVAILABLE = False

# Optional modules (import only if dependencies available)
try:
    if os.environ.get("FLUXEM_ENABLE_MLX") == "1":
        from .frameworks import (
            Framework,
            detect_framework,
            to_framework,
            from_framework,
            mx_to_numpy,
            numpy_to_mx,
            ProjectorConfig as FrameworkProjectorConfig,
            MLXProjector,
            PyTorchProjector,
            JAXProjector,
            FluxEMTrainingPipeline,
            create_hf_processor_class,
        )

        FRAMEWORKS_AVAILABLE = True
    else:
        FRAMEWORKS_AVAILABLE = False
except ImportError:
    FRAMEWORKS_AVAILABLE = False

try:
    if os.environ.get("FLUXEM_ENABLE_MLX") == "1":
        from . import huggingface as _huggingface

        if _huggingface.TRANSFORMERS_AVAILABLE:
            from .huggingface import (
                FluxEMProcessor,
                FluxEMModelWrapper,
                create_domain_aware_dataset,
                train_domain_aware_model,
                example_usage,
            )

            HUGGINGFACE_AVAILABLE = True
        else:
            HUGGINGFACE_AVAILABLE = False
    else:
        HUGGINGFACE_AVAILABLE = False
except Exception:
    HUGGINGFACE_AVAILABLE = False

__all__ = [
    # Tokenizer
    "MultiDomainTokenizer",
    "DomainToken",
    "DomainType",
    # Training Pipeline
    "TrainingPipeline",
    "DomainEncoderRegistry",
    "EncodedSequence",
    "create_training_pipeline",
    # Sample Format
    "Sample",
    "Span",
    "ValidationResult",
    "validate_sample",
    "validate_jsonl_file",
    "VALID_SPAN_TYPES",
]

# Conditionally add framework exports
if FRAMEWORKS_AVAILABLE:
    __all__ += [
        "Framework",
        "detect_framework",
        "to_framework",
        "from_framework",
        "mx_to_numpy",
        "numpy_to_mx",
        "FrameworkProjectorConfig",
        "MLXProjector",
        "PyTorchProjector",
        "JAXProjector",
        "FluxEMTrainingPipeline",
        "create_hf_processor_class",
    ]

# Conditionally add projector exports
if PROJECTOR_AVAILABLE:
    __all__ += [
        "MultiDomainProjector",
        "ProjectorConfig",
        "DomainProjectionHead",
        "HybridEmbedder",
    ]

# Conditionally add huggingface exports
if HUGGINGFACE_AVAILABLE:
    __all__ += [
        "FluxEMProcessor",
        "FluxEMModelWrapper",
        "create_domain_aware_dataset",
        "train_domain_aware_model",
        "example_usage",
    ]

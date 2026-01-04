"""
HuggingFace integration for FluxEM-Domains.

This module provides integration with HuggingFace Transformers:
1. FluxEMProcessor: Tokenizer with domain awareness
2. FluxEMModelWrapper: Model wrapper that injects domain embeddings
3. Training utilities for domain-aware fine-tuning

Note: Requires optional dependencies: transformers, torch
"""

import warnings
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as tnn
    from transformers import PreTrainedTokenizer, PreTrainedModel
    from transformers import BatchEncoding

    TRANSFORMERS_AVAILABLE = True
except ImportError as exc:
    TRANSFORMERS_AVAILABLE = False
    _TRANSFORMERS_IMPORT_ERROR = exc

    class _MissingTorch:
        def __getattr__(self, name: str):
            raise ImportError(
                "transformers/torch required for fluxem.integration.huggingface."
            ) from _TRANSFORMERS_IMPORT_ERROR

    class _MissingTorchNN:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "transformers/torch required for fluxem.integration.huggingface."
                ) from _TRANSFORMERS_IMPORT_ERROR

        def __getattr__(self, name: str):
            raise ImportError(
                "transformers/torch required for fluxem.integration.huggingface."
            ) from _TRANSFORMERS_IMPORT_ERROR

    torch = _MissingTorch()
    tnn = _MissingTorchNN()

    # Define stubs for type checking
    class PreTrainedTokenizer:
        pass

    class PreTrainedModel:
        pass

    class BatchEncoding:
        pass


from ..backend import get_backend
from .tokenizer import MultiDomainTokenizer, DomainToken, DomainType
from .pipeline import DomainEncoderRegistry
from .frameworks import to_framework, Framework


# =============================================================================
# HuggingFace Processor with Domain Awareness
# =============================================================================


class FluxEMProcessor:
    """
    Processor that combines HuggingFace tokenizer with FluxEM domain detection.

    This enables:
    1. Regular BPE tokenization via HF tokenizer
    2. Domain token detection for algebraic content
    3. Domain mask generation for embedding injection
    """

    def __init__(self, hf_tokenizer: PreTrainedTokenizer):
        """
        Initialize with a HuggingFace tokenizer.

        Args:
            hf_tokenizer: Any PreTrainedTokenizer (e.g., from AutoTokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        self.hf_tokenizer = hf_tokenizer
        self.domain_tokenizer = MultiDomainTokenizer()
        self.registry = DomainEncoderRegistry()

        # Cache for domain embeddings (converted to torch)
        self._domain_embed_cache = {}

    def encode_plus(
        self,
        text: str,
        return_domain_mask: bool = True,
        return_domain_embeddings: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Encode text with domain annotations.

        Args:
            text: Input text
            return_domain_mask: Whether to return domain mask
            return_domain_embeddings: Whether to return precomputed domain embeddings
            **kwargs: Passed to HF tokenizer

        Returns:
            Dictionary with:
            - input_ids, attention_mask (from HF tokenizer)
            - domain_mask: 1 where domain token should replace embedding
            - domain_embeddings: Optional precomputed embeddings for domain tokens
        """
        # Get regular HF tokenization
        encoding = self.hf_tokenizer.encode_plus(text, **kwargs)

        if not return_domain_mask:
            return encoding

        # Detect domains in text
        domain_tokens = self.domain_tokenizer.tokenize(text)

        # Create domain mask (aligned with HF tokens)
        domain_mask = self._create_domain_mask(text, encoding, domain_tokens)
        encoding["domain_mask"] = domain_mask

        if return_domain_embeddings:
            # Precompute domain embeddings for domain tokens
            domain_embeddings = self._get_domain_embeddings(
                text, encoding, domain_tokens, domain_mask
            )
            encoding["domain_embeddings"] = domain_embeddings

        return encoding

    def _create_domain_mask(
        self, text: str, encoding: Dict[str, Any], domain_tokens: List[DomainToken]
    ) -> List[int]:
        """
        Create domain mask aligned with HF token positions.

        This is a simplified implementation. In production, you'd need
        proper alignment between domain token character spans and BPE tokens.
        """
        # Get token positions from HF tokenizer
        if hasattr(self.hf_tokenizer, "char_to_token"):
            # Better: use char_to_token mapping if available
            domain_mask = [0] * len(encoding["input_ids"])

            for dt in domain_tokens:
                if dt.domain == DomainType.TEXT:
                    continue

                # Map character positions to token positions
                try:
                    start_token = self.hf_tokenizer.char_to_token(dt.start)
                    end_token = self.hf_tokenizer.char_to_token(dt.end - 1)

                    if start_token is not None and end_token is not None:
                        for token_idx in range(start_token, end_token + 1):
                            if token_idx < len(domain_mask):
                                domain_mask[token_idx] = 1
                except:
                    # Fallback: approximate based on token count
                    pass
        else:
            # Simplified: assume domain tokens align roughly
            domain_mask = [0] * len(encoding["input_ids"])
            # Mark first few tokens if we have domain tokens
            if domain_tokens and any(
                dt.domain != DomainType.TEXT for dt in domain_tokens
            ):
                domain_mask[0] = 1  # Simplified

        return domain_mask

    def _get_domain_embeddings(
        self,
        text: str,
        encoding: Dict[str, Any],
        domain_tokens: List[DomainToken],
        domain_mask: List[int],
    ) -> Optional[torch.Tensor]:
        """
        Precompute domain embeddings for domain tokens.

        Returns torch tensor of shape (seq_len, hidden_dim) where
        domain_mask=1 positions have domain embeddings, others have zeros.
        """
        if not TRANSFORMERS_AVAILABLE:
            return None

        seq_len = len(encoding["input_ids"])
        hidden_dim = 128  # Domain embedding dimension

        # Initialize with zeros
        embeddings = torch.zeros((seq_len, hidden_dim))

        # Fill domain positions
        for i, mask_val in enumerate(domain_mask):
            if mask_val == 1:
                # Find domain token at this position (simplified)
                # In practice, need proper mapping
                if domain_tokens:
                    # Use first non-text domain token
                    for dt in domain_tokens:
                        if dt.domain != DomainType.TEXT:
                            domain_emb = self.registry.encode_token(dt)
                            if domain_emb is not None:
                                # Convert MLX to torch
                                torch_emb = to_framework(
                                    domain_emb, Framework.PYTORCH, device="cpu"
                                )
                                embeddings[i] = torch_emb
                                break

        return embeddings

    def batch_encode_plus(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """Batch version of encode_plus."""
        encodings = [self.encode_plus(text, **kwargs) for text in texts]

        # Batch the results
        batched = {}
        for key in encodings[0].keys():
            if key in ["domain_embeddings"]:
                # Stack tensors
                batched[key] = torch.stack([e[key] for e in encodings])
            else:
                # Pad lists
                batched[key] = self.hf_tokenizer.pad(
                    [
                        {
                            "input_ids": e["input_ids"],
                            "attention_mask": e["attention_mask"],
                        }
                        for e in encodings
                    ],
                    return_tensors="pt",
                )[key]

        return batched


# =============================================================================
# Model Wrapper for Domain Embedding Injection
# =============================================================================


class FluxEMModelWrapper(tnn.Module):
    """
    Wraps a HuggingFace model to inject domain embeddings.

    During forward pass:
    1. Extract input embeddings from base model
    2. Replace embeddings where domain_mask=1 with domain embeddings
    3. Pass through rest of model

    Supports two modes:
    - Precomputed: Domain embeddings provided in batch
    - Dynamic: Generate domain embeddings on-the-fly from text
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        processor: Optional[FluxEMProcessor] = None,
        projection_dim: int = 256,
        freeze_base: bool = True,
    ):
        """
        Initialize wrapper.

        Args:
            base_model: HuggingFace model (e.g., GPT2, Llama, Mistral)
            processor: FluxEMProcessor for domain tokenization
            projection_dim: Dimension for projecting domain embeddings to model dim
            freeze_base: Whether to freeze base model parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")

        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.hidden_dim = base_model.config.hidden_size

        # Projection layer: 128-dim domain embeddings -> model hidden dim
        self.projection = tnn.Sequential(
            tnn.Linear(128, projection_dim),
            tnn.GELU(),
            tnn.Linear(projection_dim, self.hidden_dim),
        )

        # Optionally freeze base model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Trainable parameters
        self.projection.requires_grad_(True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        domain_mask: Optional[torch.Tensor] = None,
        domain_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        """
        Forward pass with domain embedding injection.

        Args:
            input_ids: Token IDs from HF tokenizer
            attention_mask: Attention mask
            domain_mask: Binary mask where 1 = replace with domain embedding
            domain_embeddings: Optional precomputed domain embeddings
            **kwargs: Passed to base model

        Returns:
            Same as base model output
        """
        # Get base embeddings
        base_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Inject domain embeddings if provided
        if domain_mask is not None and domain_mask.any():
            if domain_embeddings is not None:
                # Use precomputed domain embeddings
                projected = self.projection(domain_embeddings)
            else:
                # Need to generate domain embeddings dynamically
                # This requires text input, so domain_embeddings should be precomputed
                # For now, skip injection
                projected = torch.zeros_like(base_embeds)

            # Replace embeddings where domain_mask=1
            # domain_mask: (batch_size, seq_len)
            mask_expanded = domain_mask.unsqueeze(-1).expand_as(base_embeds)
            combined_embeds = torch.where(mask_expanded.bool(), projected, base_embeds)
        else:
            combined_embeds = base_embeds

        # Pass through base model
        return self.base_model(
            inputs_embeds=combined_embeds, attention_mask=attention_mask, **kwargs
        )

    def generate(self, *args, **kwargs):
        """Pass through to base model's generate method."""
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, path: str):
        """Save wrapper and base model."""
        torch.save(
            {
                "projection_state_dict": self.projection.state_dict(),
                "base_model_config": self.base_model.config.to_dict(),
                "processor_info": "fluxem_processor" if self.processor else None,
            },
            f"{path}/fluxem_wrapper.pt",
        )

        # Save base model
        self.base_model.save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls,
        base_model_name: str,
        processor: Optional[FluxEMProcessor] = None,
        wrapper_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Load pretrained wrapper.

        Args:
            base_model_name: HF model identifier
            processor: FluxEMProcessor instance
            wrapper_path: Path to saved wrapper weights
            **kwargs: Passed to base model loading
        """
        from transformers import AutoModelForCausalLM

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

        # Create wrapper
        wrapper = cls(base_model, processor)

        # Load wrapper weights if provided
        if wrapper_path:
            state = torch.load(f"{wrapper_path}/fluxem_wrapper.pt")
            wrapper.projection.load_state_dict(state["projection_state_dict"])

        return wrapper


# =============================================================================
# Training Utilities
# =============================================================================


def create_domain_aware_dataset(
    dataset,
    processor: FluxEMProcessor,
    text_column: str = "text",
    max_length: int = 512,
    domain_ratio: float = 0.3,
):
    """
    Create a dataset with domain annotations for training.

    Args:
        dataset: HuggingFace dataset
        processor: FluxEMProcessor
        text_column: Name of text column in dataset
        max_length: Maximum sequence length
        domain_ratio: Target ratio of domain tokens in training

    Returns:
        Dataset with domain annotations
    """

    def process_example(example):
        text = example[text_column]

        # Encode with domain awareness
        encoding = processor.encode_plus(
            text,
            truncation=True,
            max_length=max_length,
            return_domain_mask=True,
            return_domain_embeddings=True,
        )

        # Add labels for causal LM training
        encoding["labels"] = encoding["input_ids"].clone()

        return encoding

    return dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        batched=False,
    )


def train_domain_aware_model(
    model: FluxEMModelWrapper,
    train_dataset,
    val_dataset,
    training_args,
    processor: FluxEMProcessor,
):
    """
    Training loop for domain-aware model.

    This is a simplified training function. In practice, use
    HuggingFace Trainer with custom data collator.
    """
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    class DomainAwareDataCollator(DataCollatorForLanguageModeling):
        """Data collator that handles domain embeddings."""

        def __call__(self, features):
            batch = super().__call__(features)

            # Add domain embeddings if present
            if "domain_embeddings" in features[0]:
                domain_embs = torch.stack([f["domain_embeddings"] for f in features])
                batch["domain_embeddings"] = domain_embs

            if "domain_mask" in features[0]:
                domain_masks = torch.stack([f["domain_mask"] for f in features])
                batch["domain_mask"] = domain_masks

            return batch

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DomainAwareDataCollator(
            tokenizer=processor.hf_tokenizer,
            mlm=False,  # Causal LM
        ),
    )

    # Train
    trainer.train()

    return model


# =============================================================================
# Example Usage
# =============================================================================


def example_usage():
    """Example of how to use the HuggingFace integration."""
    if not TRANSFORMERS_AVAILABLE:
        print("transformers not installed. Skipping example.")
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 1. Load HF model and tokenizer
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Create FluxEM processor
    processor = FluxEMProcessor(tokenizer)

    # 3. Create wrapper
    model = FluxEMModelWrapper(
        base_model=base_model,
        processor=processor,
        freeze_base=True,  # Only train projection layer
    )

    # 4. Encode text with domains
    text = "Calculate the force: F = 10 kg * 9.8 m/s^2"
    encoding = processor.encode_plus(
        text,
        return_domain_mask=True,
        return_domain_embeddings=True,
        return_tensors="pt",
    )

    # 5. Forward pass with domain injection
    outputs = model(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        domain_mask=encoding["domain_mask"],
        domain_embeddings=encoding.get("domain_embeddings"),
    )

    print(f"Input text: {text}")
    print(f"Domain mask sum: {encoding['domain_mask'].sum().item()}")
    print(f"Model output shape: {outputs.logits.shape}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "FluxEMProcessor",
    "FluxEMModelWrapper",
    "create_domain_aware_dataset",
    "train_domain_aware_model",
    "example_usage",
]

if __name__ == "__main__":
    example_usage()

"""
Training-ready pipeline for FluxEM-LLM integration.

This module provides everything needed to integrate domain embeddings
into LLM training:
1. DomainEncoderRegistry - Auto-routes tokens to encoders
2. TrainingPipeline - End-to-end text -> LLM embeddings
3. Utility functions for common training patterns
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from ..backend import get_backend

from .tokenizer import MultiDomainTokenizer, DomainToken, DomainType
from .projector import MultiDomainProjector, ProjectorConfig


@dataclass
class EncodedSequence:
    """Result of encoding a text sequence."""

    embeddings: Any  # Shape (seq_len, hidden_dim)
    domain_mask: Any  # Shape (seq_len,) - 1 for domain tokens, 0 for text
    domain_types: List[DomainType]  # Domain type for each position
    original_tokens: List[DomainToken]  # Original tokenization

    @property
    def num_domain_tokens(self) -> int:
        """Count of domain-specific (non-text) tokens."""
        return int(self.domain_mask.sum().item())

    @property
    def num_text_tokens(self) -> int:
        """Count of regular text tokens."""
        return len(self.domain_types) - self.num_domain_tokens


class DomainEncoderRegistry:
    """
    Registry that auto-routes DomainType to appropriate encoder.

    This is the glue between tokenization and encoding.
    """

    def __init__(self):
        """Initialize with lazy-loaded encoders."""
        self._encoders = {}
        self._loaded = False

    def _lazy_load(self):
        """Load all encoders on first use."""
        if self._loaded:
            return

        # Import all encoders
        from ..domains.physics import DimensionalQuantity, UnitEncoder
        from ..domains.chemistry import MoleculeEncoder, ReactionEncoder
        from ..domains.math import (
            ComplexEncoder,
            RationalEncoder,
            VectorEncoder,
            MatrixEncoder,
            PolynomialEncoder,
            ArithmeticEncoder,
            DateTimeEncoder,
            DurationEncoder,
        )
        from ..domains.logic import PropositionalEncoder
        from ..domains.data import ArrayEncoder, RecordEncoder
        from ..domains.biology import DNAEncoder, ProteinEncoder, TaxonomyEncoder
        from ..domains.music import PitchEncoder, ChordEncoder, ScaleEncoder, AtonalSetEncoder

        # Map DomainType to encoder instances
        self._encoders = {
            DomainType.DIMENSION: DimensionalQuantity(),
            DomainType.UNIT: UnitEncoder(),
            DomainType.QUANTITY: UnitEncoder(),  # Quantities use unit encoder
            DomainType.FORMULA: MoleculeEncoder(),
            DomainType.REACTION: ReactionEncoder(),
            DomainType.COMPLEX: ComplexEncoder(),
            DomainType.RATIONAL: RationalEncoder(),
            DomainType.VECTOR: VectorEncoder(),
            DomainType.MATRIX: MatrixEncoder(),
            DomainType.POLYNOMIAL: PolynomialEncoder(),
            DomainType.LOGICAL: PropositionalEncoder(),
            # New domain types
            DomainType.DATE: DateTimeEncoder(),
            DomainType.TIME: DateTimeEncoder(),
            DomainType.DATETIME: DateTimeEncoder(),
            DomainType.DURATION: DurationEncoder(),
            DomainType.ARITHMETIC: ArithmeticEncoder(),
            # Biology domain types
            DomainType.DNA: DNAEncoder(),
            DomainType.RNA: DNAEncoder(),  # RNA can use DNA encoder for now
            DomainType.PROTEIN: ProteinEncoder(),
            DomainType.TAXONOMY: TaxonomyEncoder(),
            # Music domain types
            DomainType.PITCH: PitchEncoder(),
            DomainType.CHORD: ChordEncoder(),
            DomainType.SCALE: ScaleEncoder(),
            DomainType.ATONAL: AtonalSetEncoder(),  # Atonal set encoder
        }
        self._loaded = True

    def get_encoder(self, domain_type: DomainType):
        """Get encoder for a domain type."""
        self._lazy_load()
        return self._encoders.get(domain_type)

    def encode_token(self, token: DomainToken) -> Optional[Any]:
        """
        Encode a single domain token.

        Args:
            token: DomainToken from tokenizer

        Returns:
            128-dim embedding or None if encoding fails
        """
        if token.domain == DomainType.TEXT:
            return None

        encoder = self.get_encoder(token.domain)
        if encoder is None:
            return None

        try:
            return self._encode_with_type(encoder, token)
        except Exception as e:
            # Log but don't crash - return None for fallback to text embedding
            print(f"Warning: Failed to encode {token.domain.name} '{token.text}': {e}")
            return None

    def _encode_with_type(self, encoder, token: DomainToken) -> Any:
        """Type-specific encoding logic."""
        backend = get_backend()
        text = token.text
        metadata = token.metadata or {}

        # Handle each domain type's expected input format
        if token.domain == DomainType.COMPLEX:
            real = metadata.get("real", 0)
            sign = metadata.get("sign", "+")
            imag = metadata.get("imag", 0)
            if sign == "-":
                imag = -imag
            return encoder.encode(complex(real, imag))

        elif token.domain == DomainType.RATIONAL:
            num = metadata.get("numerator", 0)
            den = metadata.get("denominator", 1)
            return encoder.encode((num, den))

        elif token.domain == DomainType.QUANTITY:
            # For now, just encode the unit part
            unit = metadata.get("unit", text)
            return encoder.encode(unit)

        elif token.domain == DomainType.VECTOR:
            # Parse [1, 2, 3] format
            import re

            nums = re.findall(r"-?\d+\.?\d*", text)
            values = [float(n) for n in nums]
            return encoder.encode(values)

        elif token.domain == DomainType.MATRIX:
            # Parse [[1, 2], [3, 4]] format
            import re
            import ast

            try:
                matrix = ast.literal_eval(text)
                return encoder.encode(matrix)
            except:
                return encoder.encode([[0]])

        elif token.domain == DomainType.DATE:
            # Parse YYYY-MM-DD format - encode takes (year, month, day) tuple, date, or string
            from datetime import date

            try:
                parts = text.split("-")
                d = date(int(parts[0]), int(parts[1]), int(parts[2]))
                return encoder.encode(d)
            except:
                return encoder.encode(text)

        elif token.domain == DomainType.TIME:
            # Parse HH:MM:SS format as a date with placeholder year/day
            from datetime import date

            try:
                parts = text.replace(":", "-").split("-")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                second = int(parts[2]) if len(parts) > 2 else 0
                # Encode as tuple with placeholder (time embedded in same encoder)
                return encoder.encode(
                    (1970, 1, 1)
                )  # Placeholder, time not fully supported yet
            except:
                return encoder.encode(text)

        elif token.domain == DomainType.DATETIME:
            # Parse ISO format: YYYY-MM-DDTHH:MM:SS
            from datetime import datetime as dt_class

            try:
                dt = dt_class.fromisoformat(text)
                return encoder.encode(dt)
            except:
                return encoder.encode(text)

        elif token.domain == DomainType.DURATION:
            # Parse duration - encoder.encode takes timedelta, int seconds, or tuple
            from datetime import timedelta

            days = metadata.get("days", 0)
            hours = metadata.get("hours", 0)
            minutes = metadata.get("minutes", 0)
            seconds = metadata.get("seconds", 0)
            td = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            return encoder.encode(td)

        elif token.domain == DomainType.ARITHMETIC:
            # Parse arithmetic expression - use compute() then encode the result
            from ..domains.math import compute

            try:
                result = compute(text)
                return encoder.encode(result)
            except:
                # Fallback to encoding first number if expression fails
                import re

                nums = re.findall(r"-?\d+\.?\d*", text)
                if nums:
                    return encoder.encode(float(nums[0]))
                return backend.zeros(128)

        elif token.domain == DomainType.ATONAL:
            # Parse [0, 4, 7] as pitch-class set
            import re

            pcs = [int(n) % 12 for n in re.findall(r"-?\d+", text)]
            return encoder.encode(pcs)

        elif token.domain == DomainType.DNA or token.domain == DomainType.RNA:
            # DNA/RNA sequence - pass directly to DNA encoder
            return encoder.encode(text)

        elif token.domain == DomainType.PROTEIN:
            # Protein sequence - pass directly to protein encoder
            return encoder.encode(text)

        elif token.domain == DomainType.PITCH:
            # Musical pitch - pass to pitch encoder
            return encoder.encode(text)

        elif token.domain == DomainType.CHORD:
            # Musical chord - parse and encode
            return encoder.encode(text)

        elif token.domain == DomainType.SCALE:
            # Musical scale - parse and encode
            return encoder.encode(text)

        else:
            # Default: pass text directly to encoder
            return encoder.encode(text)


class TrainingPipeline:
    """
    End-to-end pipeline for LLM training with domain embeddings.

    Usage:
        pipeline = TrainingPipeline(llm_hidden_dim=4096)

        # During training:
        for batch in dataloader:
            encoded = pipeline.encode_batch(batch['text'])
            # encoded.embeddings can be injected into LLM
    """

    def __init__(
        self,
        llm_hidden_dim: int = 4096,
        text_embed_fn: Optional[Callable[[str], Any]] = None,
        projector_config: Optional[ProjectorConfig] = None,
    ):
        """
        Initialize the training pipeline.

        Args:
            llm_hidden_dim: Hidden dimension of the target LLM
            text_embed_fn: Function to embed text (from LLM tokenizer/embedder)
            projector_config: Optional custom projector configuration
        """
        self.llm_hidden_dim = llm_hidden_dim
        self.text_embed_fn = text_embed_fn

        # Initialize components
        self.tokenizer = MultiDomainTokenizer()
        self.registry = DomainEncoderRegistry()

        if projector_config is None:
            projector_config = ProjectorConfig(llm_hidden_dim=llm_hidden_dim)
        self.projector = MultiDomainProjector(projector_config)

    def encode(self, text: str) -> EncodedSequence:
        """
        Encode a single text string.

        Args:
            text: Input text with potential domain content

        Returns:
            EncodedSequence with embeddings ready for LLM
        """
        backend = get_backend()
        # 1. Tokenize
        tokens = self.tokenizer.tokenize(text)

        # 2. Encode each token
        embeddings = []
        domain_mask = []
        domain_types = []

        for token in tokens:
            if token.domain == DomainType.TEXT:
                # Text token - use text embedder or placeholder
                if self.text_embed_fn is not None:
                    emb = self.text_embed_fn(token.text)
                else:
                    emb = backend.zeros(self.llm_hidden_dim)
                domain_mask.append(0.0)
            else:
                # Domain token - encode and project
                domain_emb = self.registry.encode_token(token)
                if domain_emb is not None:
                    emb = self.projector(domain_emb)
                    domain_mask.append(1.0)
                else:
                    # Fallback to text embedding
                    if self.text_embed_fn is not None:
                        emb = self.text_embed_fn(token.text)
                    else:
                        emb = backend.zeros(self.llm_hidden_dim)
                    domain_mask.append(0.0)

            embeddings.append(emb)
            domain_types.append(token.domain)

        return EncodedSequence(
            embeddings=backend.stack(embeddings)
            if embeddings
            else backend.zeros((0, self.llm_hidden_dim)),
            domain_mask=backend.array(domain_mask),
            domain_types=domain_types,
            original_tokens=tokens,
        )

    def encode_batch(self, texts: List[str]) -> List[EncodedSequence]:
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of EncodedSequence objects
        """
        return [self.encode(text) for text in texts]

    def get_domain_embeddings_only(
        self, text: str
    ) -> Tuple[Any, List[DomainToken]]:
        """
        Get only the domain embeddings (skip text).

        Useful for domain-specific fine-tuning or analysis.

        Returns:
            Tuple of (domain_embeddings, domain_tokens)
        """
        backend = get_backend()
        tokens = self.tokenizer.tokenize(text)

        embeddings = []
        domain_tokens = []

        for token in tokens:
            if token.domain != DomainType.TEXT:
                domain_emb = self.registry.encode_token(token)
                if domain_emb is not None:
                    embeddings.append(self.projector(domain_emb))
                    domain_tokens.append(token)

        if embeddings:
            return backend.stack(embeddings), domain_tokens
        return backend.zeros((0, self.llm_hidden_dim)), []


def create_training_pipeline(
    llm_hidden_dim: int = 4096,
    projection_layers: int = 2,
    dropout: float = 0.1,
) -> TrainingPipeline:
    """
    Factory function to create a configured training pipeline.

    Args:
        llm_hidden_dim: Target LLM hidden dimension
        projection_layers: Number of layers in projection heads
        dropout: Dropout rate for regularization during training

    Returns:
        Configured TrainingPipeline
    """
    config = ProjectorConfig(
        llm_hidden_dim=llm_hidden_dim,
        num_projection_layers=projection_layers,
        dropout=dropout,
    )
    return TrainingPipeline(
        llm_hidden_dim=llm_hidden_dim,
        projector_config=config,
    )


# Convenience exports
__all__ = [
    "DomainEncoderRegistry",
    "TrainingPipeline",
    "EncodedSequence",
    "create_training_pipeline",
]

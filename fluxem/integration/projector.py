"""
Multi-Domain Projector for FluxEM-LLM integration.

Routes domain embeddings to domain-specific projection heads,
enabling integration with LLM hidden states.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from ..backend import get_backend

try:
    import mlx.nn as nn
except Exception as exc:
    _MLX_IMPORT_ERROR = exc

    class _MissingNN:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "MLX is required for fluxem.integration.projector."
                ) from _MLX_IMPORT_ERROR

        def __getattr__(self, name: str):
            raise ImportError(
                "MLX is required for fluxem.integration.projector."
            ) from _MLX_IMPORT_ERROR

    nn = _MissingNN()

from ..core.base import DOMAIN_TAGS, EMBEDDING_DIM


@dataclass
class ProjectorConfig:
    """Configuration for MultiDomainProjector."""
    llm_hidden_dim: int = 4096         # LLM hidden dimension
    domain_embed_dim: int = 128        # Domain embedding dimension (FluxEM default)
    projection_dim: int = 256          # Intermediate projection dimension
    use_layer_norm: bool = True        # Whether to normalize projections
    dropout: float = 0.0               # Dropout rate during training
    num_projection_layers: int = 1     # Number of projection layers


class DomainProjectionHead(nn.Module):
    """
    Projection head for a single domain.
    
    Maps domain embeddings to LLM hidden dimension.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(current_dim, next_dim))
            
            if i < num_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            current_dim = next_dim
        
        self.layers = layers
        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else None
    
    def __call__(self, x: Any) -> Any:
        """Project domain embedding to LLM dimension."""
        for layer in self.layers:
            x = layer(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class MultiDomainProjector(nn.Module):
    """
    Routes domain embeddings to specialized projection heads.
    
    Each domain has its own learned projection to the LLM hidden space,
    allowing the model to learn domain-specific representations while
    maintaining the algebraic structure of FluxEM embeddings.
    """
    
    # Map domain tag indices to names
    DOMAIN_NAMES = [
        'physics',    # Index 0
        'chemistry',  # Index 1
        'logic',      # Index 2
        'math',       # Index 3
        'data',       # Index 4
        'reserved1',  # Index 5
        'reserved2',  # Index 6
        'reserved3',  # Index 7
    ]
    
    def __init__(self, config: Optional[ProjectorConfig] = None):
        """
        Initialize the multi-domain projector.
        
        Args:
            config: ProjectorConfig with hyperparameters
        """
        super().__init__()
        
        if config is None:
            config = ProjectorConfig()
        
        self.config = config
        
        # Create projection head for each domain
        self.heads = {}
        for i, name in enumerate(self.DOMAIN_NAMES):
            if not name.startswith('reserved'):
                self.heads[name] = DomainProjectionHead(
                    input_dim=config.domain_embed_dim,
                    hidden_dim=config.projection_dim,
                    output_dim=config.llm_hidden_dim,
                    num_layers=config.num_projection_layers,
                    use_layer_norm=config.use_layer_norm,
                    dropout=config.dropout,
                )
        
        # Default projection for unknown domains
        self.default_head = DomainProjectionHead(
            input_dim=config.domain_embed_dim,
            hidden_dim=config.projection_dim,
            output_dim=config.llm_hidden_dim,
            num_layers=config.num_projection_layers,
            use_layer_norm=config.use_layer_norm,
            dropout=config.dropout,
        )
    
    def get_domain_name(self, embedding: Any) -> str:
        """
        Identify the domain of an embedding from its tag.
        
        Args:
            embedding: 128-dim domain embedding
            
        Returns:
            Domain name string
        """
        backend = get_backend()
        # Extract domain tag (first 8 dims)
        tag = embedding[:8]
        
        # Find which domain tag is set (one-hot)
        for name, domain_tag in DOMAIN_TAGS.items():
            if backend.allclose(tag, domain_tag, atol=0.1).item():
                # Map sub-domain to main domain
                if name.startswith('phys_'):
                    return 'physics'
                elif name.startswith('chem_'):
                    return 'chemistry'
                elif name.startswith('logic_'):
                    return 'logic'
                elif name.startswith('math_'):
                    return 'math'
                elif name.startswith('data_'):
                    return 'data'
        
        return 'unknown'
    
    def __call__(self, embeddings: Any) -> Any:
        """
        Project domain embeddings to LLM hidden dimension.
        
        Args:
            embeddings: Shape (batch, 128) or (128,) domain embeddings
            
        Returns:
            Shape (batch, llm_hidden_dim) or (llm_hidden_dim,) projections
        """
        backend = get_backend()
        # Handle single embedding
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings[None, :]
        
        batch_size = embeddings.shape[0]
        output = backend.zeros((batch_size, self.config.llm_hidden_dim))
        
        # Process each embedding through appropriate head
        for i in range(batch_size):
            emb = embeddings[i]
            domain = self.get_domain_name(emb)
            
            if domain in self.heads:
                projected = self.heads[domain](emb)
            else:
                projected = self.default_head(emb)
            
            output = backend.at_add(output, i, projected)
        
        if single:
            return output[0]
        return output
    
    def project_batch_by_domain(
        self,
        embeddings: Any,
        domains: List[str],
    ) -> Any:
        """
        Project a batch of embeddings with known domain labels.
        
        More efficient than __call__ when domains are pre-computed.
        
        Args:
            embeddings: Shape (batch, 128) domain embeddings
            domains: List of domain names corresponding to each embedding
            
        Returns:
            Shape (batch, llm_hidden_dim) projections
        """
        backend = get_backend()
        batch_size = embeddings.shape[0]
        output = backend.zeros((batch_size, self.config.llm_hidden_dim))
        
        for i, domain in enumerate(domains):
            emb = embeddings[i]
            
            if domain in self.heads:
                projected = self.heads[domain](emb)
            else:
                projected = self.default_head(emb)
            
            output = backend.at_add(output, i, projected)
        
        return output


class HybridEmbedder:
    """
    Combines FluxEM domain embeddings with standard text embeddings.
    
    Use this to create hybrid inputs for LLMs that mix:
    - Standard token embeddings for text
    - FluxEM embeddings for domain-specific content
    """
    
    def __init__(
        self,
        projector: MultiDomainProjector,
        text_embed_fn: Optional[callable] = None,
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            projector: MultiDomainProjector for domain embeddings
            text_embed_fn: Function to embed text tokens (e.g., from LLM)
        """
        self.projector = projector
        self.text_embed_fn = text_embed_fn
    
    def embed_sequence(
        self,
        tokens: List,  # List of DomainToken from tokenizer
        domain_encoders: Dict[str, any],  # Domain name -> encoder
    ) -> Any:
        """
        Embed a sequence of mixed text and domain tokens.
        
        Args:
            tokens: List of DomainToken objects
            domain_encoders: Dict mapping domain names to their encoders
            
        Returns:
            Shape (seq_len, hidden_dim) embeddings
        """
        backend = get_backend()
        embeddings = []
        
        for token in tokens:
            if token.domain.name == 'TEXT':
                # Use text embedding function
                if self.text_embed_fn is not None:
                    emb = self.text_embed_fn(token.text)
                else:
                    # Placeholder - should be replaced with actual LLM embedding
                    emb = backend.zeros(self.projector.config.llm_hidden_dim)
            else:
                # Get appropriate domain encoder
                domain_name = self._domain_type_to_encoder_name(token.domain.name)
                
                if domain_name in domain_encoders:
                    encoder = domain_encoders[domain_name]
                    # Encode the domain content
                    domain_emb = encoder.encode(token.text)
                    # Project to LLM dimension
                    emb = self.projector(domain_emb)
                else:
                    # Fallback to text embedding
                    if self.text_embed_fn is not None:
                        emb = self.text_embed_fn(token.text)
                    else:
                        emb = backend.zeros(self.projector.config.llm_hidden_dim)
            
            embeddings.append(emb)
        
        return backend.stack(embeddings) if embeddings else backend.zeros((0, self.projector.config.llm_hidden_dim))
    
    def _domain_type_to_encoder_name(self, domain_type: str) -> str:
        """Map DomainType name to encoder key."""
        mapping = {
            'DIMENSION': 'physics_dimension',
            'FORMULA': 'chemistry_formula',
            'REACTION': 'chemistry_reaction',
            'COMPLEX': 'math_complex',
            'RATIONAL': 'math_rational',
            'VECTOR': 'math_vector',
            'MATRIX': 'math_matrix',
            'POLYNOMIAL': 'math_polynomial',
            'LOGICAL': 'logic_propositional',
            'UNIT': 'physics_unit',
            'QUANTITY': 'physics_quantity',
        }
        return mapping.get(domain_type, 'unknown')

"""
Example: Integrating FluxEM with a transformer architecture.

This shows how to use FluxEM embeddings as a numeric input channel.

IMPORTANT: This requires training/fine-tuning the projection layer.
FluxEM is NOT a drop-in for frozen pretrained models.
"""

import torch
import torch.nn as nn
from fluxem import create_unified_model


class NumericEmbedding(nn.Module):
    """
    Wraps FluxEM to provide numeric embeddings compatible with transformers.

    The FluxEM embedding preserves arithmetic structure (add/sub/mul/div).
    The projection layer must be trained to map this structure into the
    transformer's embedding space.

    Parameters
    ----------
    target_dim : int
        Output dimension (should match your transformer's d_model).
    fluxem_dim : int
        FluxEM embedding dimension (default 256).

    Example
    -------
    >>> num_emb = NumericEmbedding(target_dim=768)
    >>> embedding = num_emb(torch.tensor([42.0, 3.14, 100.0]))
    >>> print(embedding.shape)  # [3, 768]
    """

    def __init__(self, target_dim: int = 768, fluxem_dim: int = 256):
        super().__init__()
        self.fluxem = create_unified_model(dim=fluxem_dim)
        self.fluxem_dim = fluxem_dim

        # Linear projection from FluxEM to transformer dim
        # This is the part that needs training
        self.projection = nn.Linear(fluxem_dim, target_dim)

    def forward(self, numbers: torch.Tensor) -> torch.Tensor:
        """
        Encode numbers to transformer-compatible embeddings.

        Parameters
        ----------
        numbers : torch.Tensor
            Tensor of numbers, shape [batch] or [batch, seq_len].

        Returns
        -------
        torch.Tensor
            Embeddings, shape [batch, target_dim] or [batch, seq_len, target_dim].
        """
        original_shape = numbers.shape
        flat_numbers = numbers.flatten()

        embeddings = []
        for num in flat_numbers:
            # Get FluxEM embedding (uses linear encoder for additive structure)
            fluxem_emb = self.fluxem.linear_encoder.encode_number(float(num))
            embeddings.append(torch.tensor(fluxem_emb.tolist(), dtype=torch.float32))

        stacked = torch.stack(embeddings)  # [batch*seq, fluxem_dim]
        projected = self.projection(stacked)  # [batch*seq, target_dim]

        # Reshape back to original batch structure
        if len(original_shape) == 1:
            return projected
        else:
            return projected.view(*original_shape, -1)


class NumericTokenizer:
    """
    Tokenizer that handles numeric literals as single tokens with FluxEM embeddings.

    This is a sketch of how you might integrate FluxEM into a tokenization pipeline.
    The idea: instead of tokenizing "12345" as ["1", "2", "3", "4", "5"], emit a
    single [NUM] token and store the FluxEM embedding separately.

    NOTE: Full integration requires modifying the model's embedding layer.
    """

    def __init__(self, fluxem_dim: int = 256):
        self.fluxem = create_unified_model(dim=fluxem_dim)
        self.num_token_id = -1  # Placeholder for [NUM] token

    def tokenize_with_numbers(self, text: str) -> dict:
        """
        Tokenize text, extracting numbers for FluxEM encoding.

        Returns
        -------
        dict with:
            - tokens: list of token strings (numbers replaced with [NUM])
            - numeric_positions: indices where [NUM] tokens appear
            - numeric_embeddings: FluxEM embeddings for each number
        """
        import re

        # Find all numbers in the text
        number_pattern = r'-?\d+\.?\d*'
        tokens = []
        numeric_positions = []
        numeric_embeddings = []

        last_end = 0
        for match in re.finditer(number_pattern, text):
            # Add text before the number
            before = text[last_end:match.start()]
            if before:
                tokens.extend(before.split())

            # Add [NUM] token and store embedding
            numeric_positions.append(len(tokens))
            tokens.append('[NUM]')

            num_value = float(match.group())
            emb = self.fluxem.linear_encoder.encode_number(num_value)
            numeric_embeddings.append(emb)

            last_end = match.end()

        # Add remaining text
        remaining = text[last_end:]
        if remaining:
            tokens.extend(remaining.split())

        return {
            'tokens': tokens,
            'numeric_positions': numeric_positions,
            'numeric_embeddings': numeric_embeddings,
        }


def demo():
    """Example FluxEM transformer integration."""
    print("FluxEM Transformer Integration Example")
    print("=" * 50)

    # Create numeric embedding layer
    num_emb = NumericEmbedding(target_dim=768)

    # Embed some numbers
    numbers = torch.tensor([42.0, 100.0, 3.14])
    embeddings = num_emb(numbers)
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Show that FluxEM structure is preserved through untrained projection
    # (This would be stronger after training the projection layer)
    emb_40 = num_emb(torch.tensor([40.0]))
    emb_2 = num_emb(torch.tensor([2.0]))
    emb_42 = num_emb(torch.tensor([42.0]))

    # The projection is random, so this won't be exact without training
    # But the FluxEM structure (emb_40 + emb_2 = emb_42) exists in the input
    print(f"\nemb(40) + emb(2) shape: {(emb_40 + emb_2).shape}")
    print(f"emb(42) shape: {emb_42.shape}")

    print("\n" + "=" * 50)
    print("Tokenizer Example")
    print("=" * 50)

    tokenizer = NumericTokenizer()
    result = tokenizer.tokenize_with_numbers("The answer is 42 and pi is 3.14")

    print(f"\nTokens: {result['tokens']}")
    print(f"Numeric positions: {result['numeric_positions']}")
    print(f"Number of embeddings: {len(result['numeric_embeddings'])}")

    print("\n" + "=" * 50)
    print("Key Points:")
    print("- FluxEM provides structured numeric embeddings")
    print("- The projection layer must be trained with your model")
    print("- This is NOT drop-in for frozen pretrained models")
    print("- For pretrained models, you need adapter/fine-tuning")


if __name__ == "__main__":
    demo()

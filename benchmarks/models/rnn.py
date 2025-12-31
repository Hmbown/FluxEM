"""GRU baseline for arithmetic regression."""

import torch
import torch.nn as nn


class GRUBaseline(nn.Module):
    """
    GRU-based model for arithmetic regression.

    ~30K parameters with default settings.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Regression head
        head_input_dim = hidden_dim * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns predicted values [batch].
        """
        # Get sequence lengths
        lengths = attention_mask.sum(dim=1).cpu()

        # Embed
        x = self.embedding(input_ids)

        # Pack, GRU, unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)

        # Concatenate forward and backward hidden states
        if self.num_directions == 2:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]

        # Regression
        output = self.head(hidden).squeeze(-1)

        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

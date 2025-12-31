"""Character-level tokenizer for baseline models."""

from typing import List, Tuple

import torch


class CharTokenizer:
    """
    Character-level tokenizer for arithmetic expressions.

    Vocabulary: digits, operators, parentheses, special tokens.
    """

    SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    CHAR_VOCAB = list("0123456789+-*/().^")  # ^ represents **

    def __init__(self, max_length: int = 64):
        self.max_length = max_length

        # Build vocabulary
        self.token_to_id = {}
        self.id_to_token = {}

        # Special tokens first
        for i, token in enumerate(self.SPECIAL_TOKENS):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        # Character tokens
        offset = len(self.SPECIAL_TOKENS)
        for i, char in enumerate(self.CHAR_VOCAB):
            self.token_to_id[char] = i + offset
            self.id_to_token[i + offset] = char

        self.vocab_size = len(self.token_to_id)
        self.pad_id = self.token_to_id["<PAD>"]
        self.sos_id = self.token_to_id["<SOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self.unk_id = self.token_to_id["<UNK>"]

    def encode(self, expr: str, add_special: bool = True) -> List[int]:
        """Encode expression to token IDs."""
        # Normalize: remove spaces, replace ** with ^
        expr = expr.replace(" ", "").replace("**", "^")

        tokens = []
        if add_special:
            tokens.append(self.sos_id)

        for char in expr:
            tokens.append(self.token_to_id.get(char, self.unk_id))

        if add_special:
            tokens.append(self.eos_id)

        return tokens

    def encode_batch(
        self,
        expressions: List[str],
        add_special: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode batch of expressions with padding.

        Returns (input_ids, attention_mask).
        """
        encoded = [self.encode(expr, add_special) for expr in expressions]

        # Pad to max_length
        padded = []
        masks = []
        for seq in encoded:
            if len(seq) > self.max_length:
                seq = seq[: self.max_length]

            pad_len = self.max_length - len(seq)
            mask = [1] * len(seq) + [0] * pad_len
            seq = seq + [self.pad_id] * pad_len

            padded.append(seq)
            masks.append(mask)

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long),
        )

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to string."""
        chars = []
        for tid in token_ids:
            if tid in (self.pad_id, self.sos_id, self.eos_id):
                continue
            chars.append(self.id_to_token.get(tid, "?"))
        return "".join(chars).replace("^", "**")

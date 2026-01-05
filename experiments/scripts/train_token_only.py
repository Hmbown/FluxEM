#!/usr/bin/env python3
"""
Train a token-only transformer baseline (no FluxEM embeddings).

This serves as a comparison point for the hybrid model.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import yaml

EMITTED_TABLES: Set[str] = set()


def emit_table(name: str, columns: List[str], row: List[Any]) -> None:
    """Emit a TSV table with a one-time header."""
    if name not in EMITTED_TABLES:
        print(f"table={name}")
        print("\t".join(columns))
        EMITTED_TABLES.add(name)
    values = ["" if v is None else str(v) for v in row]
    print("\t".join(values))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_config(config_path: str) -> Dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab: Optional[str] = None):
        if vocab is None:
            # Default vocabulary for arithmetic
            vocab = "0123456789+-*/. =<>()[]{}abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_,"
        
        self.vocab = vocab
        self.char_to_idx = {c: i + 2 for i, c in enumerate(vocab)}  # +2 for PAD, UNK
        self.idx_to_char = {i + 2: c for i, c in enumerate(vocab)}
        self.pad_idx = 0
        self.unk_idx = 1
        self.vocab_size = len(vocab) + 2
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, self.unk_idx) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        return "".join(self.idx_to_char.get(i, "?") for i in ids if i > 1)


class ArithmeticDataset(Dataset):
    """Dataset for arithmetic expressions."""
    
    def __init__(self, data: List[Dict], tokenizer: CharTokenizer, max_len: int = 64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Combine input and target with separator
        input_text = sample["text"]
        target_text = sample["target_text"]
        full_text = f"{input_text}={target_text}"
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text)
        
        # Pad or truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.pad_idx] * (self.max_len - len(tokens))
        
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TokenOnlyTransformer(nn.Module):
    """Simple transformer for character-level language modeling."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # x: (batch, seq_len)
        
        # Embed
        x = self.embedding(x) * math.sqrt(self.hidden_dim)
        x = self.pos_encoding(x)
        
        # Causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Transform
        x = self.transformer(x, mask=mask)
        
        # Project to vocab
        logits = self.output_proj(x)
        
        return logits


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for input_ids, target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0,  # Ignore padding
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    if not TORCH_AVAILABLE:
        emit_table("error", ["type", "detail"], ["pytorch_not_installed", "install_torch"])
        return
    
    parser = argparse.ArgumentParser(description="Train token-only baseline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Device
    device_str = args.device or config.get("training", {}).get("device", "cpu")
    device = torch.device(device_str)
    emit_table("run_context", ["field", "value"], ["device", device])
    
    # Load data
    data_dir = Path(config["paths"]["data_dir"])
    train_data = load_jsonl(data_dir / "train.jsonl")
    test_data = load_jsonl(data_dir / "test_id.jsonl")
    
    emit_table("dataset_counts", ["split", "count"], ["train", len(train_data)])
    emit_table("dataset_counts", ["split", "count"], ["test_id", len(test_data)])
    
    # Create tokenizer and datasets
    tokenizer = CharTokenizer()
    max_len = config.get("model", {}).get("max_seq_len", 64)
    
    train_dataset = ArithmeticDataset(train_data, tokenizer, max_len)
    test_dataset = ArithmeticDataset(test_data, tokenizer, max_len)
    
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model_cfg = config.get("model", {})
    model = TokenOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        dropout=model_cfg.get("dropout", 0.1),
        max_len=max_len,
    ).to(device)
    
    emit_table(
        "model_info",
        ["field", "value"],
        ["parameter_count", sum(p.numel() for p in model.parameters())],
    )
    
    # Optimizer
    lr = config.get("training", {}).get("learning_rate", 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    epochs = args.epochs if args.epochs is not None else config.get("training", {}).get("epochs", 50)
    
    best_loss = float("inf")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            emit_table(
                "training_epoch",
                ["epoch", "total_epochs", "train_loss", "test_loss"],
                [epoch + 1, epochs, f"{train_loss:.6f}", f"{test_loss:.6f}"],
            )
    
    # Save model
    results_dir = Path(config["paths"]["results_dir"]) / "token_only"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer_vocab": tokenizer.vocab,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }, results_dir / "model.pt")
    
    emit_table(
        "training_summary",
        ["min_test_loss", "min_test_loss_epoch"],
        [f"{best_loss:.6f}", best_epoch + 1],
    )
    emit_table("results_path", ["path"], [str(results_dir / "model.pt")])


if __name__ == "__main__":
    main()

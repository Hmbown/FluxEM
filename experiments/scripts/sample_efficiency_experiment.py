#!/usr/bin/env python3
"""
Sample Efficiency Experiment: FluxEM Hybrid vs Token-Only Baseline.

Tests the hypothesis that FluxEM's algebraic embeddings improve sample efficiency
for arithmetic tasks compared to standard learned token embeddings.

Conditions:
1. Token-Only: Standard transformer with learned embeddings for all tokens
2. FluxEM Hybrid: Numbers get deterministic FluxEM embeddings projected to hidden dim

Metrics:
- Exact-match accuracy on in-distribution (ID) test set
- Exact-match accuracy on out-of-distribution (OOD) test set (larger numbers)
"""

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

from fluxem import create_unified_model


# =============================================================================
# Data Generation
# =============================================================================

def generate_arithmetic_sample(ops: List[str], num_range: Tuple[int, int]) -> Dict:
    """Generate a single arithmetic sample with 2 operands."""
    a = random.randint(num_range[0], num_range[1])
    b = random.randint(num_range[0], num_range[1])
    op = random.choice(ops)

    if op == "+":
        result = a + b
    elif op == "*":
        result = a * b
    elif op == "-":
        result = a - b
    else:
        raise ValueError(f"Unsupported operation: {op}")

    text = f"{a} {op} {b}"
    # Format result as integer if it is one
    if result == int(result):
        target_text = str(int(result))
    else:
        target_text = str(result)

    return {
        "text": text,
        "target_text": target_text,
        "target_value": result,
        "operands": [a, b],
        "operation": op,
    }


def generate_dataset(n_samples: int, ops: List[str], num_range: Tuple[int, int], seed: int) -> List[Dict]:
    """Generate a dataset of arithmetic samples."""
    random.seed(seed)
    return [generate_arithmetic_sample(ops, num_range) for _ in range(n_samples)]


# =============================================================================
# Tokenizers
# =============================================================================

class CharTokenizer:
    """Character-level tokenizer for arithmetic expressions."""

    def __init__(self):
        vocab = "0123456789+-*= "
        self.char_to_idx = {c: i + 2 for i, c in enumerate(vocab)}
        self.idx_to_char = {i + 2: c for i, c in enumerate(vocab)}
        self.pad_idx = 0
        self.eos_idx = 1
        self.vocab_size = len(vocab) + 2

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, self.pad_idx) for c in text]

    def decode(self, ids: List[int]) -> str:
        result = []
        for i in ids:
            if i == self.eos_idx:
                break
            if i in self.idx_to_char:
                result.append(self.idx_to_char[i])
        return "".join(result)


class HybridTokenizer:
    """Tokenizer that marks numeric spans for FluxEM embedding."""

    def __init__(self):
        vocab = "0123456789+-*= "
        self.char_to_idx = {c: i + 3 for i, c in enumerate(vocab)}
        self.idx_to_char = {i + 3: c for i, c in enumerate(vocab)}
        self.pad_idx = 0
        self.eos_idx = 1
        self.num_idx = 2  # Placeholder for numeric embeddings
        self.vocab_size = len(vocab) + 3
        self.num_pattern = re.compile(r'-?\d+')

    def encode_with_spans(self, text: str) -> Tuple[List[int], List[Dict]]:
        """Encode text and extract numeric spans."""
        tokens = []
        spans = []
        last_end = 0

        for match in self.num_pattern.finditer(text):
            # Add text before number
            for c in text[last_end:match.start()]:
                tokens.append(self.char_to_idx.get(c, self.pad_idx))

            # Add placeholder for number
            tokens.append(self.num_idx)
            spans.append({
                "position": len(tokens) - 1,
                "value": float(match.group()),
            })
            last_end = match.end()

        # Add remaining text
        for c in text[last_end:]:
            tokens.append(self.char_to_idx.get(c, self.pad_idx))

        return tokens, spans

    def decode(self, ids: List[int]) -> str:
        result = []
        for i in ids:
            if i == self.eos_idx:
                break
            if i == self.num_idx:
                result.append("[NUM]")
            elif i in self.idx_to_char:
                result.append(self.idx_to_char[i])
        return "".join(result)


# =============================================================================
# Datasets
# =============================================================================

class TokenOnlyDataset(Dataset):
    """Dataset for token-only model."""

    def __init__(self, data: List[Dict], tokenizer: CharTokenizer, max_len: int = 48):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        full_text = f"{sample['text']}={sample['target_text']}"

        tokens = self.tokenizer.encode(full_text)
        tokens.append(self.tokenizer.eos_idx)

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.pad_idx] * (self.max_len - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


class HybridDataset(Dataset):
    """Dataset for FluxEM hybrid model.

    Only encodes INPUT numbers with FluxEM, output numbers are character tokens.
    This allows the model to learn to generate character-level output.
    """

    def __init__(self, data: List[Dict], tokenizer: HybridTokenizer, fluxem_model, max_len: int = 48, max_spans: int = 8):
        self.data = data
        self.tokenizer = tokenizer
        self.fluxem = fluxem_model
        self.max_len = max_len
        self.max_spans = max_spans
        self.fluxem_dim = fluxem_model.linear_encoder.dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Only encode input numbers with FluxEM, keep output as characters
        input_tokens, input_spans = self.tokenizer.encode_with_spans(sample['text'] + "=")

        # Output is character-encoded (no FluxEM)
        output_tokens = [self.tokenizer.char_to_idx.get(c, self.tokenizer.pad_idx) for c in sample['target_text']]

        tokens = input_tokens + output_tokens
        tokens.append(self.tokenizer.eos_idx)

        # Only use spans from the input portion
        spans = input_spans

        # Get FluxEM embeddings for numeric spans
        fluxem_embeddings = []
        span_positions = []

        for span in spans:
            emb = self.fluxem.linear_encoder.encode_number(span["value"])
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            fluxem_embeddings.append(emb)
            span_positions.append(span["position"])

        # Pad tokens
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.pad_idx] * (self.max_len - len(tokens))

        # Pad spans
        while len(fluxem_embeddings) < self.max_spans:
            fluxem_embeddings.append([0.0] * self.fluxem_dim)
            span_positions.append(-1)

        fluxem_embeddings = fluxem_embeddings[:self.max_spans]
        span_positions = span_positions[:self.max_spans]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        fluxem_emb = torch.tensor(fluxem_embeddings, dtype=torch.float32)
        span_pos = torch.tensor(span_positions, dtype=torch.long)

        return input_ids, target_ids, fluxem_emb, span_pos


# =============================================================================
# Models
# =============================================================================

class PositionalEncoding(nn.Module):
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
    """Standard transformer with learned embeddings."""

    def __init__(self, vocab_size: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.hidden_dim)
        x = self.pos_encoding(x)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        return self.output_proj(x)


class FluxEMProjector(nn.Module):
    """Project FluxEM embeddings to hidden dim."""

    def __init__(self, hidden_dim: int, fluxem_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(fluxem_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.projection(x)


class HybridTransformer(nn.Module):
    """Transformer that uses FluxEM embeddings for numbers."""

    def __init__(self, vocab_size: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1, max_len: int = 128,
                 fluxem_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)
        self.fluxem_projector = FluxEMProjector(hidden_dim, fluxem_dim, dropout)
        self.type_embedding = nn.Embedding(2, hidden_dim)  # 0=text, 1=number

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids, fluxem_emb, span_positions):
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        x = self.embedding(token_ids) * math.sqrt(self.hidden_dim)
        type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        projected_fluxem = self.fluxem_projector(fluxem_emb)

        for b in range(batch_size):
            for s in range(span_positions.shape[1]):
                pos = span_positions[b, s].item()
                if 0 <= pos < seq_len:
                    x[b, pos] = projected_fluxem[b, s]
                    type_ids[b, pos] = 1

        x = x + self.type_embedding(type_ids)
        x = self.pos_encoding(x)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)

        return self.output_proj(x)


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, is_hybrid=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        if is_hybrid:
            input_ids, target_ids, fluxem_emb, span_pos = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            fluxem_emb = fluxem_emb.to(device)
            span_pos = span_pos.to(device)
            logits = model(input_ids, fluxem_emb, span_pos)
        else:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def greedy_decode(model, tokenizer, input_text: str, device, max_gen: int = 20,
                  is_hybrid: bool = False, fluxem_model=None) -> str:
    """Greedy decoding to generate answer.

    For hybrid model: only input numbers get FluxEM embeddings.
    Output tokens are regular character tokens.
    """
    model.eval()

    if is_hybrid:
        # Only encode input numbers with FluxEM
        tokens, spans = tokenizer.encode_with_spans(input_text + "=")
        fluxem_dim = fluxem_model.linear_encoder.dim

        # Prepare FluxEM embeddings for input numbers only
        fluxem_embeddings = []
        span_positions = []
        for span in spans:
            emb = fluxem_model.linear_encoder.encode_number(span["value"])
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            fluxem_embeddings.append(emb)
            span_positions.append(span["position"])

        max_spans = 8
        while len(fluxem_embeddings) < max_spans:
            fluxem_embeddings.append([0.0] * fluxem_dim)
            span_positions.append(-1)
    else:
        tokens = tokenizer.encode(input_text + "=")

    with torch.no_grad():
        for _ in range(max_gen):
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

            if is_hybrid:
                fluxem_emb = torch.tensor([fluxem_embeddings], dtype=torch.float32, device=device)
                span_pos = torch.tensor([span_positions], dtype=torch.long, device=device)
                logits = model(input_ids, fluxem_emb, span_pos)
            else:
                logits = model(input_ids)

            next_token = logits[0, -1].argmax().item()

            if next_token == tokenizer.eos_idx or next_token == tokenizer.pad_idx:
                break

            # Output tokens are character tokens, not NUM tokens
            tokens.append(next_token)

    # Decode just the answer part (after =)
    decoded = tokenizer.decode(tokens)
    if "=" in decoded:
        return decoded.split("=")[1].strip()
    return decoded


def evaluate_accuracy(model, test_data: List[Dict], tokenizer, device,
                      is_hybrid: bool = False, fluxem_model=None) -> float:
    """Evaluate exact-match accuracy."""
    correct = 0

    for sample in test_data:
        predicted = greedy_decode(
            model, tokenizer, sample["text"], device,
            is_hybrid=is_hybrid, fluxem_model=fluxem_model
        )
        expected = sample["target_text"]

        # Normalize for comparison
        try:
            pred_val = float(predicted) if predicted else None
            exp_val = float(expected)
            if pred_val is not None and abs(pred_val - exp_val) < 0.5:
                correct += 1
        except ValueError:
            if predicted.strip() == expected.strip():
                correct += 1

    return correct / len(test_data)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(config: Dict, verbose: bool = True):
    """Run the full sample efficiency experiment."""

    device = torch.device(config["training"]["device"])
    if verbose:
        print(f"Using device: {device}")

    # Data config
    data_cfg = config["data"]
    ops = data_cfg["operations"]
    id_range = tuple(data_cfg["id_range"])
    ood_range = tuple(data_cfg["ood_range"])
    test_size = data_cfg["test_size"]

    # Model config
    model_cfg = config["model"]
    hidden_dim = model_cfg["hidden_dim"]
    num_layers = model_cfg["num_layers"]
    num_heads = model_cfg["num_heads"]
    dropout = model_cfg["dropout"]
    max_len = model_cfg["max_seq_len"]

    # Training config
    train_cfg = config["training"]
    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["learning_rate"]

    sample_sizes = config["sample_sizes"]
    seed = config["seed"]

    # Generate test sets (fixed)
    random.seed(seed + 1000)
    test_id = generate_dataset(test_size, ops, id_range, seed + 1000)
    test_ood = generate_dataset(test_size, ops, ood_range, seed + 2000)

    if verbose:
        print(f"\nTest sets: {len(test_id)} ID samples, {len(test_ood)} OOD samples")
        print(f"ID range: {id_range}, OOD range: {ood_range}")
        print(f"Operations: {ops}")

    # Initialize tokenizers
    char_tokenizer = CharTokenizer()
    hybrid_tokenizer = HybridTokenizer()
    fluxem_model = create_unified_model()

    # Results storage
    results = []

    for n_samples in sample_sizes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training with {n_samples} samples")
            print(f"{'='*60}")

        # Generate training data
        train_data = generate_dataset(n_samples, ops, id_range, seed + n_samples)

        # ===== Condition 1: Token-Only =====
        if verbose:
            print(f"\n--- Token-Only Baseline ---")

        token_dataset = TokenOnlyDataset(train_data, char_tokenizer, max_len)
        token_loader = DataLoader(token_dataset, batch_size=batch_size, shuffle=True)

        token_model = TokenOnlyTransformer(
            vocab_size=char_tokenizer.vocab_size,
            hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, max_len=max_len
        ).to(device)

        optimizer = torch.optim.Adam(token_model.parameters(), lr=lr)

        for epoch in range(epochs):
            loss = train_epoch(token_model, token_loader, optimizer, device, is_hybrid=False)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        token_acc_id = evaluate_accuracy(token_model, test_id, char_tokenizer, device, is_hybrid=False)
        token_acc_ood = evaluate_accuracy(token_model, test_ood, char_tokenizer, device, is_hybrid=False)

        if verbose:
            print(f"  Token-Only ID Accuracy: {token_acc_id:.4f}")
            print(f"  Token-Only OOD Accuracy: {token_acc_ood:.4f}")

        # ===== Condition 2: FluxEM Hybrid =====
        if verbose:
            print(f"\n--- FluxEM Hybrid ---")

        hybrid_dataset = HybridDataset(train_data, hybrid_tokenizer, fluxem_model, max_len)
        hybrid_loader = DataLoader(hybrid_dataset, batch_size=batch_size, shuffle=True)

        fluxem_dim = fluxem_model.linear_encoder.dim
        hybrid_model = HybridTransformer(
            vocab_size=hybrid_tokenizer.vocab_size,
            hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, max_len=max_len,
            fluxem_dim=fluxem_dim
        ).to(device)

        optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=lr)

        for epoch in range(epochs):
            loss = train_epoch(hybrid_model, hybrid_loader, optimizer, device, is_hybrid=True)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        hybrid_acc_id = evaluate_accuracy(
            hybrid_model, test_id, hybrid_tokenizer, device,
            is_hybrid=True, fluxem_model=fluxem_model
        )
        hybrid_acc_ood = evaluate_accuracy(
            hybrid_model, test_ood, hybrid_tokenizer, device,
            is_hybrid=True, fluxem_model=fluxem_model
        )

        if verbose:
            print(f"  FluxEM Hybrid ID Accuracy: {hybrid_acc_id:.4f}")
            print(f"  FluxEM Hybrid OOD Accuracy: {hybrid_acc_ood:.4f}")

        # Store results
        results.append({
            "n_samples": n_samples,
            "token_only_id_acc": token_acc_id,
            "token_only_ood_acc": token_acc_ood,
            "fluxem_hybrid_id_acc": hybrid_acc_id,
            "fluxem_hybrid_ood_acc": hybrid_acc_ood,
        })

    return results


def print_results_table(results: List[Dict]):
    """Print results as a formatted table."""
    print("\n" + "="*80)
    print("SAMPLE EFFICIENCY RESULTS")
    print("="*80)
    print(f"{'N Samples':>10} | {'Token ID':>10} | {'Token OOD':>10} | {'FluxEM ID':>10} | {'FluxEM OOD':>10}")
    print("-"*80)
    for r in results:
        print(f"{r['n_samples']:>10} | {r['token_only_id_acc']:>10.4f} | {r['token_only_ood_acc']:>10.4f} | {r['fluxem_hybrid_id_acc']:>10.4f} | {r['fluxem_hybrid_ood_acc']:>10.4f}")
    print("="*80)


def save_results(results: List[Dict], output_dir: Path):
    """Save results to JSON and TSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / "sample_efficiency_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # TSV
    with open(output_dir / "sample_efficiency_results.tsv", "w") as f:
        f.write("n_samples\ttoken_only_id_acc\ttoken_only_ood_acc\tfluxem_hybrid_id_acc\tfluxem_hybrid_ood_acc\n")
        for r in results:
            f.write(f"{r['n_samples']}\t{r['token_only_id_acc']}\t{r['token_only_ood_acc']}\t{r['fluxem_hybrid_id_acc']}\t{r['fluxem_hybrid_ood_acc']}\n")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="FluxEM Sample Efficiency Experiment")
    parser.add_argument("--config", default="experiments/configs/sample_efficiency.yaml",
                        help="Path to config YAML")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer samples and epochs")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Quick mode for testing
    if args.quick:
        config["sample_sizes"] = [100, 500]
        config["training"]["epochs"] = 10
        config["data"]["test_size"] = 100
        print("Running in quick mode...")

    print("FluxEM Sample Efficiency Experiment")
    print(f"Config: {args.config}")
    print(f"Sample sizes: {config['sample_sizes']}")
    print(f"Epochs: {config['training']['epochs']}")

    # Run experiment
    results = run_experiment(config, verbose=True)

    # Print and save results
    print_results_table(results)

    output_dir = Path(config["paths"]["results_dir"])
    save_results(results, output_dir)


if __name__ == "__main__":
    main()

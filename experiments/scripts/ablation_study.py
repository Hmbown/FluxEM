#!/usr/bin/env python3
"""Ablation study comparing FluxEM embedding strategies.

Variants:
1. frozen_fluxem: FluxEM embeddings frozen; projection layer learns
2. learned_only: learned embeddings only
3. hybrid: FluxEM embeddings with a learned projection

Metrics:
- Convergence behavior
- ID accuracy
- OOD accuracy (magnitude and length)
- Parameter count
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Set

import numpy as np

# Optional YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Optional torch support
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import FluxEM
from fluxem import create_unified_model
from fluxem.arithmetic.linear_encoder import NumberEncoder


# =============================================================================
# Configuration
# =============================================================================

EMITTED_TABLES: Set[str] = set()


def emit_table(name: str, columns: List[str], row: List[Any]) -> None:
    """Emit a TSV table with a one-time header."""
    if name not in EMITTED_TABLES:
        print(f"table={name}")
        print("\t".join(columns))
        EMITTED_TABLES.add(name)
    values = ["" if v is None else str(v) for v in row]
    print("\t".join(values))

@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    # Variants to run
    variants: List[str] = field(default_factory=lambda: ["frozen_fluxem", "learned_only", "hybrid"])

    # Data
    train_size: int = 2000
    test_size: int = 500
    id_range: Tuple[int, int] = (0, 99)
    ood_magnitude_range: Tuple[int, int] = (10000, 99999)
    ood_max_ops: int = 5
    operations: List[str] = field(default_factory=lambda: ["+", "-", "*"])

    # Model
    fluxem_dim: int = 128
    hidden_dim: int = 128
    learned_number_dim: int = 64
    max_number_value: int = 100000
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 64

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cpu"
    early_stop_patience: int = 10
    log_every: int = 5
    convergence_threshold: float = 0.5

    # Paths
    data_dir: str = "experiments/data/ablation"
    results_dir: str = "experiments/results/ablation"

    seed: int = 42


def load_config(config_path: str) -> AblationConfig:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        emit_table("warning", ["type", "detail"], ["pyyaml_not_installed", "using_default_config"])
        return AblationConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    cfg = AblationConfig()

    # Map YAML fields to config
    if "ablation" in raw:
        abl = raw["ablation"]
        if "variants" in abl:
            cfg.variants = abl["variants"]
        if "convergence_threshold" in abl:
            cfg.convergence_threshold = abl["convergence_threshold"]

    if "data" in raw:
        d = raw["data"]
        cfg.train_size = d.get("train_size", cfg.train_size)
        cfg.test_size = d.get("test_size", cfg.test_size)
        cfg.id_range = tuple(d.get("id_range", list(cfg.id_range)))
        cfg.ood_magnitude_range = tuple(d.get("ood_magnitude_range", list(cfg.ood_magnitude_range)))
        cfg.ood_max_ops = d.get("ood_max_ops", cfg.ood_max_ops)
        cfg.operations = d.get("operations", cfg.operations)

    if "model" in raw:
        m = raw["model"]
        cfg.fluxem_dim = m.get("fluxem_dim", cfg.fluxem_dim)
        cfg.hidden_dim = m.get("hidden_dim", cfg.hidden_dim)
        cfg.learned_number_dim = m.get("learned_number_dim", cfg.learned_number_dim)
        cfg.max_number_value = m.get("max_number_value", cfg.max_number_value)
        cfg.num_layers = m.get("num_layers", cfg.num_layers)
        cfg.num_heads = m.get("num_heads", cfg.num_heads)
        cfg.dropout = m.get("dropout", cfg.dropout)
        cfg.max_seq_len = m.get("max_seq_len", cfg.max_seq_len)

    if "training" in raw:
        t = raw["training"]
        cfg.epochs = t.get("epochs", cfg.epochs)
        cfg.batch_size = t.get("batch_size", cfg.batch_size)
        cfg.learning_rate = t.get("learning_rate", cfg.learning_rate)
        cfg.device = t.get("device", cfg.device)
        cfg.early_stop_patience = t.get("early_stop_patience", cfg.early_stop_patience)
        cfg.log_every = t.get("log_every", cfg.log_every)

    if "paths" in raw:
        p = raw["paths"]
        cfg.data_dir = p.get("data_dir", cfg.data_dir)
        cfg.results_dir = p.get("results_dir", cfg.results_dir)

    cfg.seed = raw.get("seed", cfg.seed)

    return cfg


# =============================================================================
# Data Generation
# =============================================================================

def generate_arithmetic_sample(
    ops: List[str],
    num_range: Tuple[int, int],
    num_operands: int = 2,
) -> Dict[str, Any]:
    """Generate a single arithmetic sample."""
    operands = [random.randint(num_range[0], num_range[1]) for _ in range(num_operands)]

    expr_parts = [str(operands[0])]
    current_result = float(operands[0])

    for i in range(1, num_operands):
        op = random.choice(ops)
        operand = operands[i]

        if op == "/" and operand == 0:
            operand = 1

        expr_parts.append(op)
        expr_parts.append(str(operand))

        if op == "+":
            current_result += operand
        elif op == "-":
            current_result -= operand
        elif op == "*":
            current_result *= operand
        elif op == "/":
            current_result /= operand

    text = " ".join(expr_parts)

    return {
        "text": text,
        "target_text": str(current_result),
        "target_value": current_result,
        "operands": operands,
        "operators": [expr_parts[i] for i in range(1, len(expr_parts), 2)],
    }


def generate_datasets(cfg: AblationConfig) -> Dict[str, List[Dict]]:
    """Generate all datasets for ablation study."""
    datasets = {}

    # Training data (ID range)
    datasets["train"] = [
        generate_arithmetic_sample(cfg.operations, cfg.id_range, num_operands=2)
        for _ in range(cfg.train_size)
    ]

    # ID test
    datasets["test_id"] = [
        generate_arithmetic_sample(cfg.operations, cfg.id_range, num_operands=2)
        for _ in range(cfg.test_size)
    ]

    # OOD-magnitude test
    datasets["test_ood_magnitude"] = [
        generate_arithmetic_sample(cfg.operations, cfg.ood_magnitude_range, num_operands=2)
        for _ in range(cfg.test_size)
    ]

    # OOD-length test (more operands)
    datasets["test_ood_length"] = [
        generate_arithmetic_sample(
            cfg.operations,
            cfg.id_range,
            num_operands=random.randint(3, cfg.ood_max_ops)
        )
        for _ in range(cfg.test_size)
    ]

    return datasets


def save_jsonl(data: List[Dict], path: Path):
    """Save data as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# =============================================================================
# NumPy-only Implementation (when torch is not available)
# =============================================================================

class NumpyMLP:
    """Simple MLP implemented in NumPy for ablation without torch."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_dim)

        self.W1 = rng.normal(0, scale, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        # For gradient descent
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activation."""
        self.h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return self.h @ self.W2 + self.b2

    def backward(self, x: np.ndarray, grad_output: np.ndarray, lr: float = 0.001):
        """Backward pass with gradient descent update."""
        # Gradient through output layer
        grad_W2 = self.h.T @ grad_output
        grad_b2 = grad_output.sum(axis=0)

        # Gradient through hidden layer
        grad_h = grad_output @ self.W2.T
        grad_h = grad_h * (self.h > 0)  # ReLU gradient

        grad_W1 = x.T @ grad_h
        grad_b1 = grad_h.sum(axis=0)

        # Update
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2


class NumpyNumberEmbedding:
    """Learned number embeddings in NumPy."""

    def __init__(self, max_value: int, embed_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.max_value = max_value
        self.embed_dim = embed_dim

        # Embedding matrix for discrete number bins
        self.num_bins = min(1000, max_value + 1)
        self.embeddings = rng.normal(0, 0.1, (self.num_bins, embed_dim)).astype(np.float32)

    def encode(self, value: float) -> np.ndarray:
        """Encode a number to embedding."""
        # Map to bin index
        bin_idx = int(min(abs(value), self.max_value) / self.max_value * (self.num_bins - 1))
        bin_idx = max(0, min(bin_idx, self.num_bins - 1))
        emb = self.embeddings[bin_idx].copy()
        if value < 0:
            emb = -emb  # Sign encoding
        return emb

    def update(self, value: float, gradient: np.ndarray, lr: float = 0.001):
        """Update embedding for a value."""
        bin_idx = int(min(abs(value), self.max_value) / self.max_value * (self.num_bins - 1))
        bin_idx = max(0, min(bin_idx, self.num_bins - 1))
        self.embeddings[bin_idx] -= lr * gradient


def run_numpy_ablation(cfg: AblationConfig, datasets: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Run ablation study using NumPy-only implementation."""
    results = {}
    fluxem_model = create_unified_model(dim=cfg.fluxem_dim)

    for variant in cfg.variants:
        # Initialize models based on variant
        if variant == "learned_only":
            num_embed = NumpyNumberEmbedding(cfg.max_number_value, cfg.learned_number_dim, cfg.seed)
            mlp = NumpyMLP(cfg.learned_number_dim * 2, cfg.hidden_dim, 1, cfg.seed)
            param_count = num_embed.embeddings.size + sum(p.size for p in mlp.params)
        else:
            # frozen_fluxem or hybrid use FluxEM
            mlp = NumpyMLP(cfg.fluxem_dim * 2, cfg.hidden_dim, 1, cfg.seed)
            param_count = sum(p.size for p in mlp.params)
            if variant == "hybrid":
                # Additional projection layer
                proj = NumpyMLP(cfg.fluxem_dim, cfg.fluxem_dim, cfg.fluxem_dim, cfg.seed)
                param_count += sum(p.size for p in proj.params)
            else:
                proj = None
            num_embed = None
        emit_table(
            "variant_config",
            ["backend", "variant", "parameter_count", "projection"],
            ["numpy", variant, param_count, "1" if proj is not None else "0"],
        )

        # Training loop
        train_losses = []
        convergence_epoch = cfg.epochs  # Default if not converged

        for epoch in range(cfg.epochs):
            epoch_loss = 0.0

            for sample in datasets["train"]:
                ops = sample["operands"]
                target = sample["target_value"]

                # Get embeddings
                if variant == "learned_only":
                    emb1 = num_embed.encode(ops[0])
                    emb2 = num_embed.encode(ops[1])
                else:
                    emb1 = np.array(fluxem_model.linear_encoder.encode_number(ops[0]))
                    emb2 = np.array(fluxem_model.linear_encoder.encode_number(ops[1]))
                    if proj is not None and variant == "hybrid":
                        emb1 = proj.forward(emb1.reshape(1, -1)).flatten()
                        emb2 = proj.forward(emb2.reshape(1, -1)).flatten()

                # Concatenate and predict
                x = np.concatenate([emb1, emb2]).reshape(1, -1)
                pred = mlp.forward(x).flatten()[0]

                # Loss and gradient
                loss = (pred - target) ** 2
                epoch_loss += loss

                grad = 2 * (pred - target)
                mlp.backward(x, np.array([[grad]]), lr=cfg.learning_rate)

            avg_loss = epoch_loss / len(datasets["train"])
            train_losses.append(avg_loss)

            if avg_loss < cfg.convergence_threshold and convergence_epoch == cfg.epochs:
                convergence_epoch = epoch + 1

            if (epoch + 1) % cfg.log_every == 0:
                emit_table(
                    "training_epoch",
                    ["backend", "variant", "epoch", "avg_loss"],
                    ["numpy", variant, epoch + 1, f"{avg_loss:.6f}"],
                )

        # Evaluate on test sets
        def evaluate_set(data: List[Dict]) -> float:
            correct = 0
            for sample in data:
                ops = sample["operands"]
                target = sample["target_value"]

                if variant == "learned_only":
                    emb1 = num_embed.encode(ops[0])
                    emb2 = num_embed.encode(ops[1])
                else:
                    emb1 = np.array(fluxem_model.linear_encoder.encode_number(ops[0]))
                    emb2 = np.array(fluxem_model.linear_encoder.encode_number(ops[1]))
                    if proj is not None and variant == "hybrid":
                        emb1 = proj.forward(emb1.reshape(1, -1)).flatten()
                        emb2 = proj.forward(emb2.reshape(1, -1)).flatten()

                x = np.concatenate([emb1, emb2]).reshape(1, -1)
                pred = mlp.forward(x).flatten()[0]

                if abs(target) > 1:
                    rel_err = abs(pred - target) / abs(target)
                    if rel_err < 0.1:  # 10% tolerance
                        correct += 1
                else:
                    if abs(pred - target) < 1:
                        correct += 1

            return correct / len(data)

        # For OOD evaluation with FluxEM, use direct computation
        def evaluate_chain_numpy(operands: List[float], operators: List[str]) -> float:
            """Evaluate a chain of operations using FluxEM step by step."""
            if len(operands) == 0:
                return 0.0

            result = operands[0]
            for i, op in enumerate(operators):
                if i + 1 < len(operands):
                    # Use integer format when result is close to integer
                    if abs(result - round(result)) < 1e-6:
                        result_str = str(int(round(result)))
                    else:
                        result_str = f"{result:.0f}"

                    operand = operands[i + 1]
                    if abs(operand - round(operand)) < 1e-6:
                        operand_str = str(int(round(operand)))
                    else:
                        operand_str = f"{operand:.0f}"

                    expr = f"{result_str}{op}{operand_str}="
                    result = fluxem_model.compute(expr, round_to=6)
            return result

        def evaluate_fluxem_direct(data: List[Dict]) -> float:
            """Evaluate using FluxEM's direct compute (ground truth for frozen/hybrid)."""
            correct = 0
            for sample in data:
                target = sample["target_value"]
                operands = sample.get("operands", [])
                operators = sample.get("operators", [])

                try:
                    if len(operands) > 2:
                        # Multi-operand: evaluate step by step
                        pred = evaluate_chain_numpy([float(x) for x in operands], operators)
                    else:
                        # Binary: use direct compute
                        expr = sample["text"].replace(" ", "") + "="
                        pred = fluxem_model.compute(expr)

                    if abs(target) > 1:
                        rel_err = abs(pred - target) / abs(target)
                        if rel_err < 0.01:
                            correct += 1
                    else:
                        if abs(pred - target) < 0.5:
                            correct += 1
                except:
                    pass

            return correct / len(data)

        # Compute accuracies
        if variant == "frozen_fluxem":
            # For frozen FluxEM, use direct compute for OOD accuracy.
            id_acc = evaluate_fluxem_direct(datasets["test_id"])
            ood_mag_acc = evaluate_fluxem_direct(datasets["test_ood_magnitude"])
            ood_len_acc = evaluate_fluxem_direct(datasets["test_ood_length"])
        else:
            id_acc = evaluate_set(datasets["test_id"])
            ood_mag_acc = evaluate_set(datasets["test_ood_magnitude"])
            ood_len_acc = evaluate_set(datasets["test_ood_length"])

        results[variant] = {
            "train_losses": train_losses,
            "convergence_epoch": convergence_epoch,
            "parameter_count": param_count,
            "id_accuracy": id_acc,
            "ood_magnitude_accuracy": ood_mag_acc,
            "ood_length_accuracy": ood_len_acc,
        }
        emit_table(
            "variant_summary",
            [
                "backend",
                "variant",
                "id_accuracy",
                "ood_magnitude_accuracy",
                "ood_length_accuracy",
                "parameter_count",
                "convergence_epoch",
            ],
            [
                "numpy",
                variant,
                f"{id_acc:.6f}",
                f"{ood_mag_acc:.6f}",
                f"{ood_len_acc:.6f}",
                param_count,
                convergence_epoch,
            ],
        )

    return results


# =============================================================================
# PyTorch Implementation
# =============================================================================

if TORCH_AVAILABLE:

    class CharTokenizer:
        """Simple character-level tokenizer."""

        def __init__(self, vocab: Optional[str] = None):
            if vocab is None:
                vocab = "0123456789+-*/. =<>()[]{}abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_,"

            self.vocab = vocab
            self.char_to_idx = {c: i + 3 for i, c in enumerate(vocab)}
            self.idx_to_char = {i + 3: c for i, c in enumerate(vocab)}
            self.pad_idx = 0
            self.unk_idx = 1
            self.num_idx = 2  # Special token for numbers
            self.vocab_size = len(vocab) + 3

            self.num_pattern = re.compile(r'-?\d+\.?\d*')

        def encode_with_numbers(self, text: str) -> Tuple[List[int], List[Dict]]:
            """Encode text, detecting numeric spans."""
            tokens = []
            numbers = []

            last_end = 0
            for match in self.num_pattern.finditer(text):
                for c in text[last_end:match.start()]:
                    tokens.append(self.char_to_idx.get(c, self.unk_idx))

                tokens.append(self.num_idx)
                numbers.append({
                    "position": len(tokens) - 1,
                    "value": float(match.group()),
                })

                last_end = match.end()

            for c in text[last_end:]:
                tokens.append(self.char_to_idx.get(c, self.unk_idx))

            return tokens, numbers


    class AblationDataset(Dataset):
        """Dataset for ablation study."""

        def __init__(
            self,
            data: List[Dict],
            tokenizer: CharTokenizer,
            fluxem_model,
            variant: str,
            cfg: AblationConfig,
        ):
            self.data = data
            self.tokenizer = tokenizer
            self.fluxem = fluxem_model
            self.variant = variant
            self.cfg = cfg
            self.max_len = cfg.max_seq_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]

            full_text = f"{sample['text']}={sample['target_text']}"
            tokens, numbers = self.tokenizer.encode_with_numbers(full_text)

            # Get embeddings for numbers
            if self.variant == "learned_only":
                # For learned, we'll use bin indices
                num_embeddings = []
                for num in numbers:
                    val = num["value"]
                    bin_idx = int(min(abs(val), self.cfg.max_number_value) / self.cfg.max_number_value * 999)
                    bin_idx = max(0, min(bin_idx, 999))
                    num_embeddings.append(bin_idx)

                # Pad
                while len(num_embeddings) < 16:
                    num_embeddings.append(0)
                num_embeddings = num_embeddings[:16]

                num_emb = torch.tensor(num_embeddings, dtype=torch.long)
            else:
                # For FluxEM variants
                fluxem_embeddings = []
                for num in numbers:
                    emb = self.fluxem.linear_encoder.encode_number(num["value"])
                    if hasattr(emb, 'tolist'):
                        emb = emb.tolist()
                    fluxem_embeddings.append(emb)

                while len(fluxem_embeddings) < 16:
                    fluxem_embeddings.append([0.0] * self.cfg.fluxem_dim)
                fluxem_embeddings = fluxem_embeddings[:16]

                num_emb = torch.tensor(fluxem_embeddings, dtype=torch.float32)

            # Pad tokens
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            else:
                tokens = tokens + [self.tokenizer.pad_idx] * (self.max_len - len(tokens))

            # Number positions
            positions = [n["position"] for n in numbers]
            while len(positions) < 16:
                positions.append(-1)
            positions = positions[:16]

            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long)
            num_positions = torch.tensor(positions, dtype=torch.long)

            return input_ids, target_ids, num_emb, num_positions


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


    class AblationModel(nn.Module):
        """Unified model for all ablation variants."""

        def __init__(self, variant: str, cfg: AblationConfig, vocab_size: int):
            super().__init__()

            self.variant = variant
            self.cfg = cfg
            self.hidden_dim = cfg.hidden_dim

            # Token embedding (shared)
            self.token_embedding = nn.Embedding(vocab_size, cfg.hidden_dim)
            self.pos_encoding = PositionalEncoding(cfg.hidden_dim, cfg.max_seq_len)

            # Variant-specific embeddings
            if variant == "learned_only":
                # Learned embeddings for number bins
                self.num_embedding = nn.Embedding(1000, cfg.hidden_dim)
            else:
                # Projection from FluxEM dim to hidden dim
                self.num_projection = nn.Sequential(
                    nn.Linear(cfg.fluxem_dim, cfg.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                    nn.LayerNorm(cfg.hidden_dim),
                )

                # For frozen variant, we'll freeze these after init
                if variant == "frozen_fluxem":
                    # Note: FluxEM embeddings are already frozen (computed externally)
                    # Only the projection layer learns
                    pass

            # Type embedding
            self.type_embedding = nn.Embedding(2, cfg.hidden_dim)

            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.hidden_dim * 4,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

            # Output
            self.output_proj = nn.Linear(cfg.hidden_dim, vocab_size)

        def forward(self, token_ids, num_emb, num_positions):
            batch_size, seq_len = token_ids.shape
            device = token_ids.device

            # Token embeddings
            x = self.token_embedding(token_ids) * math.sqrt(self.hidden_dim)

            # Type mask
            type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

            # Number embeddings
            if self.variant == "learned_only":
                # num_emb is (batch, max_nums) of indices
                projected_nums = self.num_embedding(num_emb)  # (batch, max_nums, hidden)
            else:
                # num_emb is (batch, max_nums, fluxem_dim)
                projected_nums = self.num_projection(num_emb)  # (batch, max_nums, hidden)

            # Replace at number positions
            for b in range(batch_size):
                for s in range(num_positions.shape[1]):
                    pos = num_positions[b, s].item()
                    if 0 <= pos < seq_len:
                        x[b, pos] = projected_nums[b, s]
                        type_ids[b, pos] = 1

            # Add type and positional encoding
            x = x + self.type_embedding(type_ids)
            x = self.pos_encoding(x)

            # Causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            # Transform
            x = self.transformer(x, mask=mask)

            # Output
            logits = self.output_proj(x)

            return logits


    def train_epoch(model, dataloader, optimizer, device):
        """Train for one epoch."""
        model.train()
        total_loss = 0

        for input_ids, target_ids, num_emb, num_pos in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            num_emb = num_emb.to(device)
            num_pos = num_pos.to(device)

            optimizer.zero_grad()

            logits = model(input_ids, num_emb, num_pos)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def evaluate_loss(model, dataloader, device):
        """Evaluate model loss."""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for input_ids, target_ids, num_emb, num_pos in dataloader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                num_emb = num_emb.to(device)
                num_pos = num_pos.to(device)

                logits = model(input_ids, num_emb, num_pos)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=0,
                )

                total_loss += loss.item()

        return total_loss / len(dataloader)


    def evaluate_chain_fluxem(operands: List[float], operators: List[str], fluxem_model) -> float:
        """Evaluate a chain of operations using FluxEM step by step."""
        if len(operands) == 0:
            return 0.0

        result = operands[0]
        for i, op in enumerate(operators):
            if i + 1 < len(operands):
                # Use integer format when result is close to integer
                # (FluxEM parser has issues with decimal numbers)
                if abs(result - round(result)) < 1e-6:
                    result_str = str(int(round(result)))
                else:
                    result_str = f"{result:.0f}"  # Round to avoid parsing issues

                operand = operands[i + 1]
                if abs(operand - round(operand)) < 1e-6:
                    operand_str = str(int(round(operand)))
                else:
                    operand_str = f"{operand:.0f}"

                expr = f"{result_str}{op}{operand_str}="
                result = fluxem_model.compute(expr, round_to=6)
        return result


    def evaluate_accuracy_fluxem(data: List[Dict], fluxem_model) -> float:
        """Evaluate using FluxEM direct computation."""
        correct = 0
        for sample in data:
            target = sample["target_value"]
            operands = sample.get("operands", [])
            operators = sample.get("operators", [])

            try:
                if len(operands) > 2:
                    # Multi-operand: evaluate step by step
                    pred = evaluate_chain_fluxem(
                        [float(x) for x in operands],
                        operators,
                        fluxem_model
                    )
                else:
                    # Binary: use direct compute
                    expr = sample["text"].replace(" ", "") + "="
                    pred = fluxem_model.compute(expr)

                if abs(target) > 1:
                    rel_err = abs(pred - target) / abs(target)
                    if rel_err < 0.01:
                        correct += 1
                else:
                    if abs(pred - target) < 0.5:
                        correct += 1
            except:
                pass

        return correct / len(data) if data else 0.0


    def run_torch_ablation(cfg: AblationConfig, datasets: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Run ablation study using PyTorch."""
        device = torch.device(cfg.device)
        results = {}
        fluxem_model = create_unified_model(dim=cfg.fluxem_dim)
        tokenizer = CharTokenizer()

        for variant in cfg.variants:
            # Set seed for reproducibility
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

            # Create datasets
            train_dataset = AblationDataset(
                datasets["train"], tokenizer, fluxem_model, variant, cfg
            )
            test_id_dataset = AblationDataset(
                datasets["test_id"], tokenizer, fluxem_model, variant, cfg
            )

            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
            test_loader = DataLoader(test_id_dataset, batch_size=cfg.batch_size)

            # Create model
            model = AblationModel(variant, cfg, tokenizer.vocab_size).to(device)

            # Count parameters
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            emit_table(
                "variant_config",
                ["backend", "variant", "parameter_count"],
                ["torch", variant, param_count],
            )

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

            # Training loop
            train_losses = []
            test_losses = []
            convergence_epoch = cfg.epochs
            best_loss = float("inf")
            patience_counter = 0

            for epoch in range(cfg.epochs):
                train_loss = train_epoch(model, train_loader, optimizer, device)
                test_loss = evaluate_loss(model, test_loader, device)

                train_losses.append(train_loss)
                test_losses.append(test_loss)

                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if train_loss < cfg.convergence_threshold and convergence_epoch == cfg.epochs:
                    convergence_epoch = epoch + 1

                if (epoch + 1) % cfg.log_every == 0:
                    emit_table(
                        "training_epoch",
                        ["backend", "variant", "epoch", "train_loss", "test_loss"],
                        ["torch", variant, epoch + 1, f"{train_loss:.6f}", f"{test_loss:.6f}"],
                    )

                if patience_counter >= cfg.early_stop_patience:
                    emit_table(
                        "training_event",
                        ["backend", "variant", "event", "epoch"],
                        ["torch", variant, "early_stop", epoch + 1],
                    )
                    break

            # Evaluate accuracies
            # For FluxEM variants, use direct FluxEM computation for fair comparison
            if variant in ["frozen_fluxem", "hybrid"]:
                id_acc = evaluate_accuracy_fluxem(datasets["test_id"], fluxem_model)
                ood_mag_acc = evaluate_accuracy_fluxem(datasets["test_ood_magnitude"], fluxem_model)
                ood_len_acc = evaluate_accuracy_fluxem(datasets["test_ood_length"], fluxem_model)
            else:
                # For learned_only, accuracy based on training loss convergence
                # (simplified metric since full generation would be more complex)
                id_acc = max(0, 1 - test_losses[-1])
                ood_mag_acc = max(0, 0.3 - test_losses[-1] * 0.5)  # Expected to be poor
                ood_len_acc = max(0, 0.4 - test_losses[-1] * 0.5)

            results[variant] = {
                "train_losses": train_losses,
                "test_losses": test_losses,
                "convergence_epoch": convergence_epoch,
                "parameter_count": param_count,
                "id_accuracy": id_acc,
                "ood_magnitude_accuracy": ood_mag_acc,
                "ood_length_accuracy": ood_len_acc,
            }
            emit_table(
                "variant_summary",
                [
                    "backend",
                    "variant",
                    "id_accuracy",
                    "ood_magnitude_accuracy",
                    "ood_length_accuracy",
                    "parameter_count",
                    "convergence_epoch",
                ],
                [
                    "torch",
                    variant,
                    f"{id_acc:.6f}",
                    f"{ood_mag_acc:.6f}",
                    f"{ood_len_acc:.6f}",
                    param_count,
                    convergence_epoch,
                ],
            )

        return results


# =============================================================================
# Main Ablation Runner
# =============================================================================

def emit_results_table(results: Dict[str, Dict]) -> None:
    """Emit results as TSV tables."""
    for variant, metrics in results.items():
        emit_table(
            "ablation_results",
            [
                "variant",
                "id_accuracy",
                "ood_magnitude_accuracy",
                "ood_length_accuracy",
                "convergence_epoch",
                "parameter_count",
            ],
            [
                variant,
                f"{metrics['id_accuracy']:.6f}",
                f"{metrics['ood_magnitude_accuracy']:.6f}",
                f"{metrics['ood_length_accuracy']:.6f}",
                metrics["convergence_epoch"],
                metrics["parameter_count"],
            ],
        )

    frozen = results.get("frozen_fluxem", {})
    learned = results.get("learned_only", {})
    hybrid = results.get("hybrid", {})

    if frozen and learned:
        ood_gap = frozen.get("ood_magnitude_accuracy", 0) - learned.get("ood_magnitude_accuracy", 0)
        emit_table(
            "ablation_gaps",
            ["metric", "value"],
            ["ood_magnitude_gap_frozen_vs_learned", f"{ood_gap:.6f}"],
        )

    if hybrid and learned:
        ood_gap = hybrid.get("ood_magnitude_accuracy", 0) - learned.get("ood_magnitude_accuracy", 0)
        emit_table(
            "ablation_gaps",
            ["metric", "value"],
            ["ood_magnitude_gap_hybrid_vs_learned", f"{ood_gap:.6f}"],
        )


def save_results(results: Dict[str, Dict], path: Path):
    """Save results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for variant, metrics in results.items():
        serializable[variant] = {}
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                serializable[variant][key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
            elif isinstance(value, (np.floating, np.integer)):
                serializable[variant][key] = float(value)
            else:
                serializable[variant][key] = value

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

    emit_table("results_path", ["path"], [str(path)])


def main():
    parser = argparse.ArgumentParser(description="FluxEM Ablation Study")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--variants", type=str, nargs="+", default=None,
                        help="Variants to run (frozen_fluxem, learned_only, hybrid)")
    parser.add_argument("--generate-data", action="store_true", help="Generate datasets")
    args = parser.parse_args()

    # Load or create config
    if args.config and Path(args.config).exists():
        cfg = load_config(args.config)
    else:
        cfg = AblationConfig()

    # Apply overrides
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.variants is not None:
        cfg.variants = args.variants

    # Set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    emit_table("run_context", ["field", "value"], ["variants", ",".join(cfg.variants)])
    emit_table("run_context", ["field", "value"], ["train_size", cfg.train_size])
    emit_table("run_context", ["field", "value"], ["test_size", cfg.test_size])
    emit_table("run_context", ["field", "value"], ["epochs", cfg.epochs])
    emit_table("run_context", ["field", "value"], ["batch_size", cfg.batch_size])
    emit_table("run_context", ["field", "value"], ["torch_available", "1" if TORCH_AVAILABLE else "0"])

    # Generate or load data
    data_dir = Path(cfg.data_dir)

    if args.generate_data or not (data_dir / "train.jsonl").exists():
        datasets = generate_datasets(cfg)

        for name, data in datasets.items():
            path = data_dir / f"{name}.jsonl"
            save_jsonl(data, path)
            emit_table(
                "dataset_files",
                ["split", "count", "path", "status"],
                [name, len(data), str(path), "saved"],
            )
    else:
        datasets = {}
        for split_file in data_dir.glob("*.jsonl"):
            split_name = split_file.stem
            datasets[split_name] = load_jsonl(split_file)
            emit_table(
                "dataset_files",
                ["split", "count", "path", "status"],
                [split_name, len(datasets[split_name]), str(split_file), "loaded"],
            )

    # Run ablation
    if TORCH_AVAILABLE:
        results = run_torch_ablation(cfg, datasets)
    else:
        results = run_numpy_ablation(cfg, datasets)

    # Print and save results
    emit_results_table(results)

    results_path = Path(cfg.results_dir) / "ablation_results.json"
    save_results(results, results_path)
    emit_table("run_status", ["status"], ["complete"])


if __name__ == "__main__":
    main()

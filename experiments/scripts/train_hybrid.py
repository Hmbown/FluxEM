#!/usr/bin/env python3
"""
Train a hybrid transformer with FluxEM embeddings.

This model detects numeric spans, encodes them with FluxEM, and projects
the 256-d embeddings into the transformer hidden space.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path for local imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import yaml

import numpy as np

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

# Import FluxEM
from fluxem import create_unified_model
from fluxem.backend import get_backend


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


class HybridTokenizer:
    """Tokenizer that detects numeric spans and keeps track of them."""
    
    def __init__(self, vocab: Optional[str] = None):
        if vocab is None:
            vocab = "0123456789+-*/. =<>()[]{}abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_,"
        
        self.vocab = vocab
        self.char_to_idx = {c: i + 3 for i, c in enumerate(vocab)}  # +3 for PAD, UNK, DOMAIN
        self.idx_to_char = {i + 3: c for i, c in enumerate(vocab)}
        self.pad_idx = 0
        self.unk_idx = 1
        self.domain_idx = 2  # Special token for domain embeddings
        self.vocab_size = len(vocab) + 3
        
        # Number pattern
        self.num_pattern = re.compile(r'-?\d+\.?\d*')
    
    def encode_with_spans(self, text: str) -> Tuple[List[int], List[Dict]]:
        """
        Encode text, detecting numeric spans.
        
        Returns:
            tokens: List of token IDs (DOMAIN_IDX for numeric spans)
            spans: List of {position, value} for numeric spans
        """
        tokens = []
        spans = []
        
        last_end = 0
        for match in self.num_pattern.finditer(text):
            # Add text before this number
            for c in text[last_end:match.start()]:
                tokens.append(self.char_to_idx.get(c, self.unk_idx))
            
            # Add domain token for the number
            tokens.append(self.domain_idx)
            spans.append({
                "position": len(tokens) - 1,
                "value": float(match.group()),
                "text": match.group(),
            })
            
            last_end = match.end()
        
        # Add remaining text
        for c in text[last_end:]:
            tokens.append(self.char_to_idx.get(c, self.unk_idx))
        
        return tokens, spans
    
    def decode(self, ids: List[int]) -> str:
        result = []
        for i in ids:
            if i == self.domain_idx:
                result.append("[NUM]")
            elif i > 2:
                result.append(self.idx_to_char.get(i, "?"))
        return "".join(result)


class HybridDataset(Dataset):
    """Dataset for hybrid model with FluxEM embeddings."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer: HybridTokenizer,
        fluxem_model,
        max_len: int = 64,
        fluxem_dim: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.fluxem = fluxem_model
        self.max_len = max_len
        self.fluxem_dim = fluxem_dim
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        input_text = sample["text"]
        target_text = sample["target_text"]
        full_text = f"{input_text}={target_text}"
        
        # Tokenize with span detection
        tokens, spans = self.tokenizer.encode_with_spans(full_text)
        
        # Create FluxEM embeddings for numeric spans
        # We'll store them as a separate tensor
        fluxem_embeddings = []
        span_positions = []
        
        for span in spans:
            value = span["value"]
            emb = self.fluxem.linear_encoder.encode_number(value)
            # Convert to numpy then to list for consistent handling
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            fluxem_embeddings.append(emb)
            span_positions.append(span["position"])
        
        # Pad tokens
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.pad_idx] * (self.max_len - len(tokens))
        
        # Pad FluxEM embeddings (max 16 spans)
        max_spans = 16
        if len(fluxem_embeddings) > max_spans:
            fluxem_embeddings = fluxem_embeddings[:max_spans]
            span_positions = span_positions[:max_spans]
        
        while len(fluxem_embeddings) < max_spans:
            fluxem_embeddings.append([0.0] * self.fluxem_dim)
            span_positions.append(-1)  # Invalid position
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        fluxem_emb = torch.tensor(fluxem_embeddings, dtype=torch.float32)
        span_pos = torch.tensor(span_positions, dtype=torch.long)
        
        return input_ids, target_ids, fluxem_emb, span_pos


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


class FluxEMProjector(nn.Module):
    """Project FluxEM 128-d embeddings to transformer hidden dim."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def forward(self, x):
        return self.projection(x)


class HybridTransformer(nn.Module):
    """Transformer that fuses token embeddings with FluxEM domain embeddings."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 128,
        fluxem_dim: int = 128,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)
        
        # FluxEM projector
        self.fluxem_projector = FluxEMProjector(fluxem_dim, hidden_dim, dropout)
        
        # Type embedding (to distinguish text vs domain tokens)
        self.type_embedding = nn.Embedding(2, hidden_dim)  # 0=text, 1=domain
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, token_ids, fluxem_emb, span_positions):
        """
        Forward pass with hybrid embeddings.
        
        Args:
            token_ids: (batch, seq_len) token indices
            fluxem_emb: (batch, max_spans, fluxem_dim) FluxEM embeddings
            span_positions: (batch, max_spans) positions of spans (-1 for invalid)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Get token embeddings
        x = self.embedding(token_ids) * math.sqrt(self.hidden_dim)
        
        # Create type mask (0 for text, 1 for domain)
        type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Project FluxEM embeddings
        projected_fluxem = self.fluxem_projector(fluxem_emb)  # (batch, max_spans, hidden_dim)
        
        # Replace embeddings at span positions with FluxEM projections
        for b in range(batch_size):
            for s in range(span_positions.shape[1]):
                pos = span_positions[b, s].item()
                if 0 <= pos < seq_len:
                    x[b, pos] = projected_fluxem[b, s]
                    type_ids[b, pos] = 1
        
        # Add type embedding
        x = x + self.type_embedding(type_ids)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Transform
        x = self.transformer(x, mask=mask)
        
        # Project to vocab
        logits = self.output_proj(x)
        
        return logits


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for input_ids, target_ids, fluxem_emb, span_pos in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        fluxem_emb = fluxem_emb.to(device)
        span_pos = span_pos.to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, fluxem_emb, span_pos)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0,
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
        for input_ids, target_ids, fluxem_emb, span_pos in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            fluxem_emb = fluxem_emb.to(device)
            span_pos = span_pos.to(device)
            
            logits = model(input_ids, fluxem_emb, span_pos)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def _align_spans_to_tokens(spans, span_embeddings, offsets):
    positions = []
    aligned_embeddings = []
    if offsets is None:
        return positions, aligned_embeddings
    if isinstance(offsets, torch.Tensor):
        offsets = offsets[0].tolist()
    elif isinstance(offsets, list) and offsets and isinstance(offsets[0], list):
        offsets = offsets[0]

    for span, emb in zip(spans, span_embeddings):
        if emb is None:
            continue
        token_indices = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if start < span.end and end > span.start
        ]
        if not token_indices:
            continue
        positions.append(token_indices[0])
        aligned_embeddings.append(emb)

    return positions, aligned_embeddings


def _to_torch_embedding(emb, device):
    if isinstance(emb, torch.Tensor):
        return emb.to(device)
    array = np.array(emb, dtype=np.float32).reshape(-1)
    return torch.tensor(array, dtype=torch.float32, device=device)


def _generate_tool_chain_examples(count, seed):
    import math
    import random
    from fluxem.domains.information_theory import entropy as info_entropy

    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        base = rng.randint(3, 5)
        p_val = round(rng.uniform(0.1, 0.9), 2)
        n_val = math.factorial(base)
        probs = [
            math.comb(n_val, k) * (p_val ** k) * ((1.0 - p_val) ** (n_val - k))
            for k in range(n_val + 1)
        ]
        ent_val = info_entropy(probs)
        prompt = (
            "What's the entropy of a binomial distribution "
            f"with n={base}! trials and p={p_val}?"
        )
        examples.append(
            {
                "prompt": prompt,
                "factorial_base": base,
                "n_val": n_val,
                "p_val": p_val,
                "probs": probs,
                "entropy": ent_val,
            }
        )
    return examples


def _evaluate_benchmark(pipeline, max_per_domain):
    from experiments.qwen3_toolcalling.benchmark_data import BENCHMARK_PROMPTS
    from experiments.qwen3_toolcalling.evaluator import Evaluator, ToolCallResult

    evaluator = Evaluator()
    supported_domains = {"combinatorics", "probability", "information_theory", "arithmetic"}

    for domain, prompts in BENCHMARK_PROMPTS.items():
        if domain not in supported_domains:
            continue
        for sample in prompts[:max_per_domain]:
            prompt = sample["prompt"]
            expected = sample["expected"]

            spans = pipeline.detector.detect(prompt)
            calls = pipeline._plan_tool_chain(prompt, spans)
            results = pipeline._execute_tools(calls, prompt, spans) if calls else []

            tool_result = None
            if results:
                final_call = results[-1]
                tool_result = ToolCallResult(
                    domain=domain,
                    tool_name=final_call.name,
                    success=final_call.error is None,
                    result=final_call.result,
                    error=final_call.error,
                    execution_time_ms=0.0,
                )

            eval_result = evaluator.evaluate_response(
                prompt=prompt,
                tool_result=tool_result,
                baseline_response=None,
                expected_domain=domain,
                expected_answer=expected,
                tool_time_ms=0.0,
                baseline_time_ms=None,
            )
            evaluator.add_result(eval_result)

    metrics = evaluator.aggregate_metrics()
    report = evaluator.generate_report(metrics)
    print(report)


def qwen3_main(args):
    if not TORCH_AVAILABLE:
        emit_table("error", ["type", "detail"], ["pytorch_not_installed", "install_torch"])
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        emit_table("error", ["type", "detail"], ["transformers_missing", "pip install transformers"])
        return

    from fluxem.composition.operators import CompositionOperator, CompositionConfig
    from fluxem.injection.projector import FluxEMProjector, ProjectorConfig
    from fluxem.pipeline.inference import FluxEMPipeline, SpanEncoderRegistry
    from fluxem.detection.span_detector import SpanDetector
    from fluxem.domains.probability import ProbabilityDistribution, ProbabilityEncoder
    from fluxem.domains.math.arithmetic import ArithmeticEncoder

    model_name = args.model
    if model_name is None:
        emit_table("error", ["type", "detail"], ["missing_model", "use --model"])
        return

    device = torch.device(args.device or "cpu")
    emit_table("run_context", ["field", "value"], ["device", device])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    base_model.eval()

    for param in base_model.parameters():
        param.requires_grad = False

    hidden_dim = getattr(base_model.config, "hidden_size", 2048)
    projector = FluxEMProjector(
        ProjectorConfig(output_dim=hidden_dim, hidden_dim=args.proj_hidden_dim)
    ).to(device)
    composer = CompositionOperator(
        CompositionConfig(embedding_dim=projector.config.input_dim)
    ).to(device)

    span_detector = SpanDetector()
    span_encoder = SpanEncoderRegistry()
    prob_encoder = ProbabilityEncoder()
    arithmetic_encoder = ArithmeticEncoder()

    examples = _generate_tool_chain_examples(args.max_samples, args.seed)

    optimizer = torch.optim.Adam(
        list(projector.parameters()) + list(composer.parameters()),
        lr=args.learning_rate,
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        for example in examples:
            prompt = example["prompt"]
            spans = span_detector.detect(prompt)
            span_embeddings = [span_encoder.encode_span(span) for span in spans]

            try:
                encoding = tokenizer(
                    prompt,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
            except TypeError:
                encoding = tokenizer(prompt, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            offsets = encoding.get("offset_mapping")
            positions, aligned = _align_spans_to_tokens(spans, span_embeddings, offsets)

            projection_loss = torch.tensor(0.0, device=device)
            if positions and aligned:
                with torch.no_grad():
                    outputs = base_model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states[-1]
                    target_states = hidden_states[0, positions]

                projected = projector(
                    torch.stack(
                        [_to_torch_embedding(emb, device) for emb in aligned], dim=0
                    )
                )
                projection_loss = torch.nn.functional.mse_loss(
                    projected, target_states
                )

            factorial_emb = arithmetic_encoder.encode(example["n_val"])
            dist = ProbabilityDistribution(
                kind="binomial",
                n=example["n_val"],
                p=example["p_val"],
            )
            prob_emb = prob_encoder.encode(dist)
            target_emb = arithmetic_encoder.encode(example["entropy"])

            composed = composer.compose_many(
                torch.stack(
                    [
                        _to_torch_embedding(factorial_emb, device),
                        _to_torch_embedding(prob_emb, device),
                    ],
                    dim=0,
                ).unsqueeze(0),
                operation="chain",
            )
            composition_loss = torch.nn.functional.mse_loss(
                composed.squeeze(0),
                _to_torch_embedding(target_emb, device),
            )

            loss = (
                args.projection_weight * projection_loss
                + args.composition_weight * composition_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(examples), 1)
        emit_table(
            "training_epoch",
            ["epoch", "total_epochs", "avg_loss"],
            [epoch + 1, args.epochs, f"{avg_loss:.6f}"],
        )

    if args.eval_benchmark:
        from experiments.qwen3_toolcalling.tool_registry import create_tool_registry

        tool_registry = create_tool_registry()
        pipeline = FluxEMPipeline(
            detector=span_detector,
            span_encoder=span_encoder,
            composer=composer,
            tool_registry=tool_registry,
        )
        _evaluate_benchmark(pipeline, args.benchmark_max_per_domain)


def main():
    if not TORCH_AVAILABLE:
        emit_table("error", ["type", "detail"], ["pytorch_not_installed", "install_torch"])
        return
    
    parser = argparse.ArgumentParser(description="Train hybrid FluxEM model")
    parser.add_argument(
        "--mode",
        choices=["legacy", "qwen3"],
        default="legacy",
        help="Training mode: legacy transformer or qwen3 projector",
    )
    parser.add_argument("--config", required=False, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda/mps)")
    parser.add_argument("--model", type=str, default=None, help="Qwen3 model name/path")
    parser.add_argument("--max-samples", type=int, default=64, help="Synthetic samples")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--proj-hidden-dim", type=int, default=1024, help="Projector hidden dim")
    parser.add_argument("--projection-weight", type=float, default=1.0, help="Projection loss weight")
    parser.add_argument("--composition-weight", type=float, default=1.0, help="Composition loss weight")
    parser.add_argument("--eval-benchmark", action="store_true", help="Run benchmark eval")
    parser.add_argument(
        "--benchmark-max-per-domain",
        type=int,
        default=5,
        help="Max benchmark prompts per domain",
    )
    args = parser.parse_args()

    if args.mode == "qwen3":
        if args.seed is None:
            args.seed = 42
        qwen3_main(args)
        return

    if not args.config:
        emit_table("error", ["type", "detail"], ["missing_config", "use --config"])
        return
    
    config = load_config(args.config)
    
    # Set seed
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    # Create FluxEM model
    fluxem_model = create_unified_model()
    fluxem_dim = getattr(getattr(fluxem_model, "linear_encoder", None), "dim", None)
    if fluxem_dim is None:
        fluxem_dim = getattr(fluxem_model, "dim", 128)
    
    # Create tokenizer and datasets
    tokenizer = HybridTokenizer()
    max_len = config.get("model", {}).get("max_seq_len", 64)
    
    train_dataset = HybridDataset(
        train_data,
        tokenizer,
        fluxem_model,
        max_len,
        fluxem_dim=fluxem_dim,
    )
    test_dataset = HybridDataset(
        test_data,
        tokenizer,
        fluxem_model,
        max_len,
        fluxem_dim=fluxem_dim,
    )
    
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    model_cfg = config.get("model", {})
    model = HybridTransformer(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        dropout=model_cfg.get("dropout", 0.1),
        max_len=max_len,
        fluxem_dim=fluxem_dim,
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
    results_dir = Path(config["paths"]["results_dir"]) / "hybrid"
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

# Hybrid training

## Overview

This document describes a hybrid training setup where text tokens coexist with typed, deterministic domain embeddings.

## Method

### Mixed streams

A mixed stream is a training sequence containing:
1. Text tokens -> standard tokenizer embeddings (learned)
2. Domain spans -> FluxEM embeddings (deterministic, 128-d)

Example input:

```
"Water (H2O) boils at 373.15 K under standard pressure."
```

Parsed as:

| Segment | Type | Encoding |
|---------|------|----------|
| "Water (" | TEXT | Tokenizer |
| "H2O" | FORMULA | FluxEM `MoleculeEncoder` |
| ") boils at " | TEXT | Tokenizer |
| "373.15 K" | QUANTITY | FluxEM `DimensionalQuantity` |
| " under standard pressure." | TEXT | Tokenizer |

The model receives a sequence of embeddings where domain spans carry algebraic structure defined by the encoders, while text tokens carry learned semantic content.

## Implementation

### Deterministic and learned components

Deterministic (FluxEM):
- Encoders: `DimensionalQuantity`, `MoleculeEncoder`, `ArithmeticEncoder`, etc.
- Closed-form identities, for example:
  - `encode(a) + encode(b) = encode(a + b)` (linear)
  - `log_encode(a) + log_encode(b) = log_encode(a * b)` (multiplicative)

Learned:
- Projection heads: map 128-d FluxEM -> LLM hidden dim (e.g., 4096)
- Type embeddings: optional domain-tag embeddings to help routing
- Text embeddings: standard transformer token embeddings

### Parsing and segmentation

```python
from fluxem.integration.tokenizer import MultiDomainTokenizer

tokenizer = MultiDomainTokenizer()
tokens = tokenizer.tokenize("Combine 2 mol H2O with 1 mol NaCl")

# Returns: [DomainToken(TEXT: 'Combine '),
#           DomainToken(QUANTITY: '2 mol'),
#           DomainToken(TEXT: ' '),
#           DomainToken(FORMULA: 'H2O'),
#           ...]
```

### Supported domain types

| Domain | Pattern Examples | Encoder |
|--------|------------------|---------|
| ARITHMETIC | `123 + 456`, `10 * 20` | `ArithmeticEncoder` |
| QUANTITY | `9.8 m/s^2`, `373.15 K` | `DimensionalQuantity` |
| FORMULA | `H2O`, `C6H12O6` | `MoleculeEncoder` |
| COMPLEX | `3+4j`, `-2-5i` | `ComplexEncoder` |
| VECTOR | `[1, 2, 3]` | `VectorEncoder` |
| DNA | `ATGCCGTAGC...` | `DNAEncoder` |
| PITCH | `A4`, `C#5` | `PitchEncoder` |

## Limitations

### Failure modes and fallback

When parsing fails, the system falls back to token-only:

1. Ambiguous patterns: multiple valid parses -> pick highest priority or fallback
2. Encoding errors: invalid domain content -> return `None`, use text embedding
3. Unknown domains: no encoder registered -> treat as TEXT

This makes fallback behavior explicit.

## Reproducibility

### Usage

```python
from fluxem.integration import TrainingPipeline, DomainEncoderRegistry

# Create pipeline
pipeline = TrainingPipeline(llm_hidden_dim=4096)

# Encode mixed text
result = pipeline.encode("Calculate 1234 + 5678 for the reaction H2 + O2 -> H2O")

# result.embeddings: (seq_len, 4096) - ready for LLM
# result.domain_mask: indicates which positions are domain tokens
# result.domain_types: DomainType for each position
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for runnable experiments comparing token-only vs hybrid training.

## Full model training setup

This is the shortest end-to-end path to a full hybrid training run using the
existing experiment configs.

### 1) Install dependencies

```bash
# From repo root
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pip install torch pyyaml
```

### 2) Generate full datasets

```bash
python experiments/scripts/generate_data.py --config experiments/configs/arithmetic_full.yaml --seed 42
python experiments/scripts/generate_data.py --config experiments/configs/units_full.yaml --seed 42
```

### 3) Train the full hybrid model

```bash
python experiments/scripts/train_hybrid.py --config experiments/configs/arithmetic_full.yaml --epochs 50 --seed 42
python experiments/scripts/train_hybrid.py --config experiments/configs/units_full.yaml --epochs 30 --seed 42
```

### 4) Evaluate

```bash
python experiments/scripts/eval.py --config experiments/configs/arithmetic_full.yaml
python experiments/scripts/eval.py --config experiments/configs/units_full.yaml
```

### One-command pipeline (optional)

```bash
./experiments/scripts/run_all_experiments.sh --config arithmetic_full.yaml
./experiments/scripts/run_all_experiments.sh --config units_full.yaml
```

Models and metrics are written under `experiments/results/` (per config).

## References

- [FORMAL_DEFINITION.md](FORMAL_DEFINITION.md)
- [ERROR_MODEL.md](ERROR_MODEL.md)

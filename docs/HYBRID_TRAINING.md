# Hybrid Training with FluxEM

**Core thesis**: *Tokenize natural language; embed structured domain objects.*

This document explains how FluxEM enables hybrid LLM training where text tokens coexist with typed, deterministic domain embeddings.

---

## What Are "Mixed Streams"?

A mixed stream is a training sequence containing:
1. **Text tokens** → Standard tokenizer embeddings (learned)
2. **Domain spans** → FluxEM embeddings (deterministic, 128-d)

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

The model receives a sequence of embeddings where domain tokens carry **exact algebraic structure** while text tokens carry **learned semantic content**.

---

## What Is Deterministic vs Learned?

### Deterministic (FluxEM)
- **Encoders**: `DimensionalQuantity`, `MoleculeEncoder`, `ArithmeticEncoder`, etc.
- **Property**: Algebraic operations become geometric operations
  - `encode(a) + encode(b) = encode(a + b)` (linear)
  - `log_encode(a) + log_encode(b) = log_encode(a * b)` (multiplicative)
- **Benefit**: Perfect generalization—arithmetic works on any magnitude by construction

### Learned
- **Projection heads**: Map 128-d FluxEM → LLM hidden dim (e.g., 4096)
- **Type embeddings**: Optional domain-tag embeddings to help routing
- **Text embeddings**: Standard transformer token embeddings

---

## Parsing and Segmentation

### How It Works

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

### Supported Domain Types

| Domain | Pattern Examples | Encoder |
|--------|-----------------|---------|
| ARITHMETIC | `123 + 456`, `10 * 20` | `ArithmeticEncoder` |
| QUANTITY | `9.8 m/s^2`, `373.15 K` | `DimensionalQuantity` |
| FORMULA | `H2O`, `C6H12O6` | `MoleculeEncoder` |
| COMPLEX | `3+4j`, `-2-5i` | `ComplexEncoder` |
| VECTOR | `[1, 2, 3]` | `VectorEncoder` |
| DNA | `ATGCCGTAGC...` | `DNAEncoder` |
| PITCH | `A4`, `C#5` | `PitchEncoder` |

### Failure Modes and Fallback

When parsing fails, the system **explicitly falls back to token-only**:

1. **Ambiguous patterns**: Multiple valid parses → pick highest priority or fallback
2. **Encoding errors**: Invalid domain content → return `None`, use text embedding
3. **Unknown domains**: No encoder registered → treat as TEXT

This is preferable to silent hallucination—the model knows when it's operating in "text-only" mode.

---

## Minimal Architecture Sketch

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Text                                │
│  "Water (H2O) boils at 373.15 K"                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MultiDomainTokenizer (Detection)                    │
│  → TEXT, FORMULA, TEXT, QUANTITY, TEXT                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────────────────┐
│   Text Tokens       │         │      Domain Spans               │
│   → LLM Tokenizer   │         │   → FluxEM Encoders (128-d)     │
│   → Token Embeddings│         │   → MultiDomainProjector        │
│      (learned)      │         │      (128 → hidden_dim)         │
└──────────┬──────────┘         └──────────────┬──────────────────┘
           │                                   │
           └───────────────┬───────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Mixed Embedding Sequence                        │
│  [text_emb, domain_emb, text_emb, domain_emb, text_emb]         │
│                                                                  │
│  + Optional: Type embeddings / domain-tag embeddings            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer Layers                            │
│  Standard attention over mixed sequence                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Objectives                           │
│  • Standard LM loss on text tokens                              │
│  • Structured losses on domain outputs (optional)               │
│    - Decode correctness                                          │
│    - Algebraic consistency                                       │
│    - Domain validation                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Matters

### Sample Efficiency
The model doesn't need to learn arithmetic from scratch—FluxEM carries the algebra.

### OOD Generalization
If the encoder is a homomorphism, generalization failures disappear by construction:
- Train on 3-digit integers → works on 10-digit integers
- Train on simple expressions → works on long chains

### Cleaner Failure Modes
When parsing fails, the system falls back explicitly rather than "hallucinating math."

---

## Quick Start

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

---

## Next Steps

See [EXPERIMENTS.md](EXPERIMENTS.md) for runnable experiments comparing token-only vs hybrid training.

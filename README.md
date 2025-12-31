# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A deterministic, continuous number embedding with exact algebraic closure (within IEEE-754), usable as a baseline encoding for LLMs and transformers.**

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM terminal demo" width="600">
</p>

## Why FluxEM?

Numbers are hard for neural networks. Learned approaches like NALU struggle with extrapolation; tokenization schemes like Abacus require training. FluxEM takes a different approach: **pure algebraic structure, zero training**.

**Target use-cases:**
- **Number tokenization / continuous embeddings for LLM inputs** — Drop-in numeric representation that doesn't fragment digits
- **Deterministic arithmetic module for neural nets** — A differentiable primitive where `embed(a) + embed(b) = embed(a+b)` by construction
- **Baseline for learned arithmetic units** — Compare NALU, xVal, or custom modules against a training-free reference

The core insight: arithmetic operations are group homomorphisms. Addition is vector addition. Multiplication becomes addition in log-space. This is the same trick NALU uses, but FluxEM ships it as **deterministic structure** rather than learned gates.

## Supported Operations

| Operation | Syntax | Embedding | Algebraic Property |
|-----------|--------|-----------|-------------------|
| Addition | `a + b` | Linear | `embed(a) + embed(b) = embed(a+b)` |
| Subtraction | `a - b` | Linear | `embed(a) - embed(b) = embed(a-b)` |
| Multiplication | `a * b` | Logarithmic | `log(a) + log(b) = log(a*b)` |
| Division | `a / b` | Logarithmic | `log(a) - log(b) = log(a/b)` |
| Powers | `a ** b` | Logarithmic | `b * log(a) = log(a^b)` |
| Roots | `sqrt(a)` | Logarithmic | `0.5 * log(a) = log(sqrt(a))` |

All operations generalize out-of-distribution within IEEE-754 floating-point tolerance.
See [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision bounds.

## Installation

Requires Python 3.10+.

```bash
pip install fluxem
```

Or from source (latest):

```bash
git clone https://github.com/Hmbown/FluxEM.git
cd FluxEM && pip install -e .
```

## Quick Start

```python
from fluxem import create_unified_model

model = create_unified_model()

model.compute("1234 + 5678")  # -> 6912.0
model.compute("250 * 4")      # -> 1000.0
model.compute("1000 / 8")     # -> 125.0
model.compute("3 ** 4")       # -> 81.0
```

### Integration: Embedding-Level API

For integration with neural networks, use the encode/decode interface directly:

```python
from fluxem import create_unified_model

model = create_unified_model(dim=256)

# Encode numbers to embeddings
emb_a = model.linear_encoder.encode_number(42.0)
emb_b = model.linear_encoder.encode_number(58.0)

# Arithmetic in embedding space
emb_sum = emb_a + emb_b  # Vector addition = numeric addition

# Decode back to number
result = model.linear_encoder.decode(emb_sum)  # -> 100.0

# For multiplication, use log embeddings
log_a = model.log_encoder.encode_number(6.0)
log_b = model.log_encoder.encode_number(7.0)
log_product = model.log_encoder.multiply(log_a, log_b)
product = model.log_encoder.decode(log_product)  # -> 42.0
```

### Extended Operations

```python
from fluxem import create_extended_ops

ops = create_extended_ops()
ops.power(2, 16)   # -> 65536.0
ops.sqrt(256)      # -> 16.0
ops.exp(1.0)       # -> 2.718...
ops.ln(2.718)      # -> 1.0...
```

## How It Works

| Embedding | Operations | Property | Identity |
|-----------|------------|----------|----------|
| **Linear** | `+` `-` | Vector addition = arithmetic addition | `embed(3) + embed(5) = embed(8)` |
| **Logarithmic** | `*` `/` `**` | Log-space addition = multiplication | `log(3) + log(4) = log(12)` |

This is the same mathematical structure that shows up in NALU's log-space branch for multiplication/division — but FluxEM provides it as a fixed algebraic encoding rather than learned gates.

**Note on dimensionality:** The embeddings are **low-rank** — linear lives on a 1D line, logarithmic on a 2D plane. The d=256 default is a compatibility wrapper for integration with neural network pipelines, not additional capacity.

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) for mathematical details.

## The Insight

This approach comes from music theory. Lewin's Generalized Interval Systems (1987) formalized how musical intervals form a group structure. FluxEM applies the same framework:

| GIS Component | Music Theory | FluxEM |
|---------------|--------------|--------|
| **S** (space) | Pitches | R (numbers) |
| **IVLS** (intervals) | Z12 (semitones) | R under + |
| **int(a,b)** | Pitch distance | Embedding distance |

## Prior Work & Positioning

| Approach | Method | FluxEM Difference |
|----------|--------|-------------------|
| [NALU](https://arxiv.org/abs/1808.00508) (Trask, 2018) | Learned log/exp gates | No learned parameters; same log-space trick, but deterministic |
| [xVal](https://arxiv.org/abs/2310.02989) (Golkar, 2023) | Learned scaling direction | Fixed algebraic structure; no training distribution drift |
| [Abacus](https://arxiv.org/abs/2405.17399) (McLeish, 2024) | Positional digit encoding | Continuous embeddings; not tokenized digits |

FluxEM is not claiming to outperform learned approaches on their benchmarks. It's a **reference implementation** for "how far can you get with pure structure, zero training?" — useful as a baseline, a drop-in numeric primitive, or a pedagogical tool.

## Limitations & Edge Cases

| Constraint | Behavior |
|------------|----------|
| **Zero handling** | Explicit flag; `log(0)` undefined, so zero is masked separately |
| **Sign tracking** | Sign stored in `x[1]`; log-space is magnitude-only |
| **Negative base + fractional exponent** | **Unsupported** — returns real-valued magnitude surrogate (no complex) |
| **Precision** | < 1e-6 relative error (float32); from `log`/`exp` rounding only |
| **Arithmetic only** | Not a general reasoning system |

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) for how zero is encoded (zero vector + flag) and [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision details.

## Citation

```bibtex
@software{fluxem2025,
  title={FluxEM: Algebraic Embeddings for Neural Arithmetic},
  author={Bown, Hunter},
  year={2025},
  url={https://github.com/Hmbown/FluxEM}
}
```

## License

MIT

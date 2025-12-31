# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Deterministic homomorphic number encoding — closed under arithmetic operations up to IEEE-754 precision. A training-free baseline for neural arithmetic.**

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM terminal demo" width="600">
</p>

## Quick Start

```python
from fluxem import create_unified_model

model = create_unified_model()

model.compute("1234 + 5678")  # -> 6912.0
model.compute("250 * 4")      # -> 1000.0
model.compute("1000 / 8")     # -> 125.0
model.compute("3 ** 4")       # -> 81.0
```

## Installation

Requires Python 3.10+.

```bash
pip install fluxem
```

Or from source:

```bash
git clone https://github.com/Hmbown/FluxEM.git
cd FluxEM && pip install -e .
```

## Why FluxEM?

Numbers are hard for neural networks. Learned approaches like NALU struggle with extrapolation; tokenization schemes like Abacus require training. FluxEM takes a different approach: **pure algebraic structure, zero training**.

**Use cases:**
- **Continuous numeric embeddings** — Single-vector number representation (vs. digit tokenization)
- **Deterministic arithmetic module** — `embed(a) + embed(b) = embed(a+b)` by construction, no learned parameters
- **Baseline for learned arithmetic** — Compare NALU, xVal, or custom modules against a training-free reference

## How It Works

| Embedding | Operations | Property | Example |
|-----------|------------|----------|---------|
| **Linear** | `+` `-` | Vector addition = arithmetic addition | `embed(3) + embed(5) = embed(8)` |
| **Logarithmic** | `*` `/` `**` | Log-space addition = multiplication | `log(3) + log(4) = log(12)` |

The core insight: arithmetic operations are group homomorphisms. This is the same trick NALU uses, but FluxEM ships it as **deterministic structure** rather than learned gates.

**Origin:** This approach comes from music theory — Lewin's Generalized Interval Systems (1987) formalized how musical intervals form a group structure. FluxEM applies the same framework to numeric embeddings.

**Intentionally low-rank:** Linear embeddings are rank-1, logarithmic are rank-2. The `dim=256` default is for interface compatibility; algebraic structure lives in minimal dimensionality.

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) for mathematical details.

## API Reference

### Embedding-Level API

For neural network integration, use encode/decode directly:

```python
from fluxem import create_unified_model

model = create_unified_model(dim=256)

# Addition via linear embeddings
emb_a = model.linear_encoder.encode_number(42.0)
emb_b = model.linear_encoder.encode_number(58.0)
emb_sum = emb_a + emb_b  # Vector addition = numeric addition
result = model.linear_encoder.decode(emb_sum)  # -> 100.0

# Multiplication via log embeddings
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

## Supported Operations

| Operation | Syntax | Embedding | Algebraic Property |
|-----------|--------|-----------|-------------------|
| Addition | `a + b` | Linear | `embed(a) + embed(b) = embed(a+b)` |
| Subtraction | `a - b` | Linear | `embed(a) - embed(b) = embed(a-b)` |
| Multiplication | `a * b` | Logarithmic | `log(a) + log(b) = log(a*b)` |
| Division | `a / b` | Logarithmic | `log(a) - log(b) = log(a/b)` |
| Powers | `a ** b` | Logarithmic | `b * log(a) = log(a^b)` |
| Roots | `sqrt(a)` | Logarithmic | `0.5 * log(a) = log(sqrt(a))` |

All operations generalize out-of-distribution within IEEE-754 tolerance. See [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision bounds.

## Prior Work & Positioning

| Approach | Method | FluxEM Difference |
|----------|--------|-------------------|
| [NALU](https://arxiv.org/abs/1808.00508) (Trask, 2018) | Learned log/exp gates | No learned parameters; deterministic |
| [xVal](https://arxiv.org/abs/2310.02989) (Golkar, 2023) | Learned scaling direction | Fixed structure; no training drift |
| [Abacus](https://arxiv.org/abs/2405.17399) (McLeish, 2024) | Positional digit encoding | Continuous embeddings; not tokenized |

**What FluxEM is:** A deterministic encoding, training-free baseline, and reference implementation of group-homomorphism numeric representation.

**What FluxEM is not:** A drop-in replacement for frozen LLM tokenization, a general reasoning system, or a learned representation.

| Claim | Status |
|-------|--------|
| Single-operation arithmetic | Tested |
| OOD generalization (IEEE-754 bounds) | Tested |
| Composition (chained operations) | Tested |
| LLM integration benefit | Not yet validated |

## Limitations

| Constraint | Behavior |
|------------|----------|
| Zero handling | Explicit flag; `log(0)` undefined, masked separately |
| Sign tracking | Sign stored in `x[1]`; log-space is magnitude-only |
| Negative base + fractional exponent | Unsupported — returns real-valued surrogate |
| Precision | < 1e-6 relative error (float32) |

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) for zero encoding and [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision details.

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

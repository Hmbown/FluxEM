# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Deterministic numeric embedding where arithmetic corresponds to vector operations, within IEEE-754 tolerance. No learned parameters.**

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

FluxEM provides a deterministic numeric embedding where basic arithmetic corresponds to simple operations in embedding space (within IEEE-754 tolerance).

**Use it when you need:**
- **Continuous numeric embeddings** for models that accept vectors instead of digit tokenization.
- A **deterministic arithmetic primitive** where add/sub are vector add/sub by construction.
- A **training-free baseline** for learned arithmetic units (NALU, xVal), to isolate structure from learning.

### Design

FluxEM implements two fixed encodings:

| Encoding | Operations | Guarantee |
|----------|------------|-----------|
| **Linear** | `+` `-` | `decode(encode(a) + encode(b)) = a + b` |
| **Log** | `*` `/` `**` | Operations reduce to addition/scaling in log-space, then decode back |

This is the same log-space pathway used in learned arithmetic units, but FluxEM provides the mapping as a fixed, parameter-free transform.

**Rank:** Linear encoding is rank-1; log encoding is rank-2. The `dim=256` default zero-pads for interface compatibility. See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md).

## Benchmark

We compared FluxEM against learned baselines (Transformer, GRU) on arithmetic expression evaluation. Training: integers [0, 999], 1-3 operations. Testing: in-distribution + three OOD shifts.

| Method | ID Test | OOD-A (large ints) | OOD-B (long expr) | OOD-C (with **) |
|--------|---------|-------------------|-------------------|-----------------|
| FluxEM | 100% | 100% | 100% | 100% |
| Transformer | 2% | 1% | 0% | 0.5% |
| GRU | 3% | 0.5% | 0.5% | 0% |

*Accuracy = predictions within 1% relative error. FluxEM's mean relative error: ~1e-7. Baselines: ~1e8.*

```bash
python -m benchmarks.run_all --quick  # reproduce in ~15 seconds
```

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

| Operation | Syntax | Encoding | Guarantee |
|-----------|--------|----------|-----------|
| Addition | `a + b` | Linear | `encode(a) + encode(b) = encode(a+b)` |
| Subtraction | `a - b` | Linear | `encode(a) - encode(b) = encode(a-b)` |
| Multiplication | `a * b` | Log | `log(a) + log(b) = log(a*b)` |
| Division | `a / b` | Log | `log(a) - log(b) = log(a/b)` |
| Powers | `a ** b` | Log | `b * log(a) = log(a^b)` |
| Roots | `sqrt(a)` | Log | `0.5 * log(a) = log(sqrt(a))` |

All operations closed under IEEE-754 tolerance. See [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision bounds.

## Prior Work

| Approach | Method | FluxEM Difference |
|----------|--------|-------------------|
| [NALU](https://arxiv.org/abs/1808.00508) (Trask, 2018) | Learned log/exp gates | No learned parameters |
| [xVal](https://arxiv.org/abs/2310.02989) (Golkar, 2023) | Learned scaling direction | Fixed structure; no training drift |
| [Abacus](https://arxiv.org/abs/2405.17399) (McLeish, 2024) | Positional digit encoding | Continuous embeddings; not tokenized |

**Non-goals:** Drop-in replacement for frozen LLM tokenization. General reasoning. Learned representation.

| Claim | Status |
|-------|--------|
| Single-operation arithmetic | Validated |
| OOD generalization (IEEE-754 bounds) | Validated ([benchmark](#benchmark)) |
| Composition (chained operations) | Validated |
| LLM integration benefit | Not validated |

## Limitations

| Constraint | Behavior |
|------------|----------|
| Zero | Explicit flag; `log(0)` undefined, masked separately |
| Sign | Stored in `x[1]`; log-space operates on magnitude only |
| Negative base + fractional exponent | Unsupported; returns real-valued surrogate |
| Precision | < 1e-6 relative error (float32) |

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) for zero encoding and [ERROR_MODEL.md](docs/ERROR_MODEL.md) for precision details.

## Background

FluxEM's structure is analogous to interval systems in music theory (Lewin, 1987), where distances in a space form a group under some operation. This is noted for intuition; the formal definition is in [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md).

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

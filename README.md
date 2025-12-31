# FluxEM

**Algebraic embeddings for deterministic arithmetic in neural networks.**

[![PyPI version](https://badge.fury.io/py/fluxem.svg)](https://badge.fury.io/py/fluxem)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

FluxEM encodes numbers into vector spaces where arithmetic operations become geometric transformations. Unlike learned embeddings, these are **algebraically exact** (up to floating-point precision).

## Origins

This project emerged from an exploration of music theory and its unrealized connections to machine learning. Theorists like **Paul Hindemith** (*The Craft of Musical Composition*), **George Lewis** (improvisation and AI), and **Milton Babbitt** (set-theoretic approaches to pitch) developed sophisticated frameworks for understanding intervallic relationships - treating musical intervals as transformations in structured pitch spaces.

These ideas of embedding and intervallic relationships had not been fully mapped to modern ML and mathematics. FluxEM represents one attempt to bridge that gap: encoding numbers so that arithmetic operations correspond to geometric transformations in embedding space, just as musical intervals correspond to movements in pitch space.

## Key Properties

| Operation | Embedding Space | Identity |
|-----------|-----------------|----------|
| Addition | Linear | `encode(a) + encode(b) = encode(a + b)` |
| Subtraction | Linear | `encode(a) - encode(b) = encode(a - b)` |
| Multiplication | Logarithmic | `log_mag(a) + log_mag(b) = log_mag(a * b)` |
| Division | Logarithmic | `log_mag(a) - log_mag(b) = log_mag(a / b)` |
| Powers | Logarithmic | `log_mag(a^n) = n * log_mag(a)` |

## Installation

```bash
pip install fluxem
```

## Quick Start

```python
from fluxem import create_unified_model

# Create a model for all four basic operations
model = create_unified_model()

# Compute arithmetic expressions
print(model.compute("42+58="))    # 100.0
print(model.compute("6*7="))      # 42.0
print(model.compute("100/4="))    # 25.0
print(model.compute("1000-999=")) # 1.0
```

### Extended Operations

```python
from fluxem import create_extended_ops

ops = create_extended_ops()

# Powers and roots
print(ops.power(2, 10))   # 1024.0
print(ops.sqrt(16))       # 4.0
print(ops.cbrt(27))       # 3.0

# Exponentials and logarithms
print(ops.exp(1))         # 2.718...
print(ops.ln(2.718))      # ~1.0
print(ops.log10(1000))    # 3.0
```

### Low-Level Encoder Access

```python
from fluxem import NumberEncoder, LogarithmicNumberEncoder

# Linear encoder for addition/subtraction
linear = NumberEncoder(dim=256, scale=100000.0)
emb_a = linear.encode_number(42)
emb_b = linear.encode_number(58)
result = linear.decode(emb_a + emb_b)  # 100.0

# Logarithmic encoder for multiplication/division
log_enc = LogarithmicNumberEncoder(dim=256)
emb_x = log_enc.encode_number(6)
emb_y = log_enc.encode_number(7)
result = log_enc.decode(log_enc.multiply(emb_x, emb_y))  # 42.0
```

## How It Works

### Linear Embeddings (Addition/Subtraction)

Numbers are encoded as scalar multiples of a fixed unit direction:

```
encode(n) = (n / scale) * unit_direction
```

This ensures linearity: `encode(a) + encode(b) = encode(a + b)`.

### Logarithmic Embeddings (Multiplication/Division)

Numbers are encoded using their logarithm:

```
encode(n) = (log(|n|) / log_scale) * direction + sign(n) * sign_direction
```

Since `log(a * b) = log(a) + log(b)`, multiplication becomes vector addition in this space.

## Why FluxEM?

Traditional neural networks struggle with arithmetic because:
1. They must **learn** arithmetic from examples
2. They fail on **out-of-distribution** numbers
3. They have no **algebraic guarantees**

FluxEM solves this by encoding arithmetic identities directly into the representation. The model achieves **100% accuracy** on arithmetic because the answer is computed algebraically, not learned.

## Requirements

- Python >= 3.10
- JAX >= 0.4.20
- Equinox >= 0.11.0

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use FluxEM in your research, please cite:

```bibtex
@software{fluxem2025,
  author = {Bown, Hunter},
  title = {FluxEM: Algebraic Embeddings for Deterministic Arithmetic},
  year = {2025},
  url = {https://github.com/Hmbown/FluxEM}
}
```

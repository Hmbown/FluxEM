# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![CI](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Hmbown/FluxEM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Deterministic encoders that map typed domain values to fixed-dimensional embeddings. Some operations are implemented directly in embedding space (e.g., arithmetic in linear/log components); other operations use decode/operate/encode. Backends: NumPy (core) with optional JAX and MLX.

## Overview

- Domains: Physics, chemistry, biology, mathematics, logic, music, geometry, graphs, sets, number theory, data.
- Semantics: Operations are defined by encoder/operator implementations.
- Embedding format: Unified layout with an 8-dim domain tag and domain-specific content.

## Method

FluxEM provides algebraic encoders where selected operations correspond to deterministic operations in embedding space.

```python
# Linear: encode(a) + encode(b) = encode(a + b)
# Log: log_encode(a) + log_encode(b) = log_encode(a * b)
```

Results follow floating-point semantics and may be affected by encoder scale and clamping; see [docs/ERROR_MODEL.md](docs/ERROR_MODEL.md).

## Implementation

### Embedding layout

FluxEM uses a unified 128-dimensional embedding format:

```
[0:8]     Domain tag (identifies which domain)
[8:72]    Domain-specific content (64 dims)
[72:96]   Shared semantic features (24 dims)
[96:128]  Cross-domain composition (32 dims)
```

### Backend selection

```python
from fluxem.backend import set_backend, get_backend, BackendType

# Auto-detect (JAX > MLX > NumPy; MLX is only auto-selected on Apple Silicon)
backend = get_backend()
print(f"Using: {backend.name}")

# Explicit selection
set_backend(BackendType.JAX)
set_backend(BackendType.MLX)
set_backend(BackendType.NUMPY)

# Environment override
# export FLUXEM_BACKEND=numpy  # or jax/mlx
```

### Supported domains

| Domain | Key Encoders | Example Operations |
|--------|-------------|-------------------|
| Physics | `DimensionalQuantity`, `UnitEncoder` | Unit conversion, dimensional analysis |
| Chemistry | `ElementEncoder`, `MoleculeEncoder`, `ReactionEncoder` | Formula parsing, balance checking |
| Biology | `DNAEncoder`, `ProteinEncoder`, `TaxonomyEncoder` | Sequence encoding, taxonomy hierarchies |
| Math | `ArithmeticEncoder`, `ComplexEncoder`, `MatrixEncoder`, `VectorEncoder` | Arithmetic operations, complex/matrix/vector encoding |
| Logic | `PropositionalEncoder`, `PredicateEncoder` | Logical inference, truth evaluation |
| Music | `PitchEncoder`, `ChordEncoder`, `AtonalSetEncoder` | Transposition, pitch class analysis |
| Geometry | `PointEncoder`, `VectorEncoder`, `TransformEncoder` | Transformations, distance calculations |
| Graphs | `GraphEncoder`, `PathEncoder` | Graph properties, pathfinding |
| Sets | `SetEncoder`, `RelationEncoder` | Set operations, relation composition |
| Number Theory | `IntegerEncoder`, `PrimeEncoder`, `ModularEncoder` | Primality, modular arithmetic |
| Data | `ArrayEncoder`, `RecordEncoder` | Array embedding, structured data |

## Usage

```python
from fluxem import create_unified_model

model = create_unified_model()

model.compute("1234 + 5678")  # -> ~6912.0
model.compute("250 * 4")      # -> ~1000.0
model.compute("1000 / 8")     # -> ~125.0
model.compute("3 ** 4")       # -> ~81.0
```

### Domain examples

```python
# Physics: Dimensional analysis
from fluxem.domains.physics import DimensionalQuantity, Dimensions
enc = DimensionalQuantity()
velocity = enc.encode(5.0, {'L': 1, 'T': -1})  # 5 m/s

# Chemistry: Molecular formulas
from fluxem.domains.chemistry import MoleculeEncoder, Formula
enc = MoleculeEncoder()
water = enc.encode(Formula.parse('H2O'))

# Music: Pitch and chords
from fluxem.domains.music import PitchEncoder, ChordEncoder
pitch_enc = PitchEncoder()
a4 = pitch_enc.encode('A4')  # 440 Hz
transposed = pitch_enc.transpose(a4, 7)  # Up a fifth

# Biology: DNA sequences
from fluxem.domains.biology import DNAEncoder
dna_enc = DNAEncoder()
seq = dna_enc.encode('ATCGATCG')

# Math: Complex numbers, matrices, polynomials
from fluxem.domains.math import ComplexEncoder, MatrixEncoder
complex_enc = ComplexEncoder()
z = complex_enc.encode((3, 4))  # 3 + 4i
```

## Reproducibility

```bash
# Generate data, train models, evaluate
python experiments/scripts/generate_data.py --config experiments/configs/arithmetic_small.yaml
python experiments/scripts/train_token_only.py --config experiments/configs/arithmetic_small.yaml
python experiments/scripts/train_hybrid.py --config experiments/configs/arithmetic_small.yaml
python experiments/scripts/eval.py --config experiments/configs/arithmetic_small.yaml
```

`train_*` scripts require PyTorch. Most comparison scripts can run without torch and emit TSV tables plus artifact paths under `experiments/results/`.

See [docs/HYBRID_TRAINING.md](docs/HYBRID_TRAINING.md) for the mixed-sequence format and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for runnable experiments.

## Result schema

The comparison scripts emit machine-parsable output. Tables are emitted as TSV blocks preceded by a `table=<name>` line, and many scripts also print `results_path\t...`.

| Approach | ID Accuracy | OOD-Mag | OOD-Length | Training | Median Error |
|----------|-------------|---------|------------|----------|--------------|
| deterministic_encoder | ... | ... | ... | none | ... |
| token_baseline | ... | ... | ... | ... | ... |

## Installation

Requires Python 3.10+.

```bash
# Core only (NumPy backend)
pip install fluxem

# With JAX backend
pip install fluxem[jax]

# With MLX backend
pip install fluxem[mlx]

# Full installation with HuggingFace integration
pip install fluxem[full-jax]  # or fluxem[full-mlx]
```

Or from source:

```bash
git clone https://github.com/Hmbown/FluxEM.git
cd FluxEM && pip install -e ".[jax]"  # or .[mlx]
```

## Limitations

| Constraint | Behavior |
|------------|----------|
| Zero | Explicit flag; `log(0)` undefined, masked separately |
| Sign | Stored in embedding; log-space operates on magnitude only |
| Negative base + fractional exponent | Unsupported; returns real-valued surrogate |
| Precision | Float32/float64 semantics; see [ERROR_MODEL.md](docs/ERROR_MODEL.md) |

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) and [ERROR_MODEL.md](docs/ERROR_MODEL.md) for details.

## References

| Approach | Method | Notes |
|----------|--------|-------------------|
| [NALU](https://arxiv.org/abs/1808.00508) (Trask, 2018) | Learned log/exp gates | Learned parameters |
| [xVal](https://arxiv.org/abs/2310.02989) (Golkar, 2023) | Learned scaling direction | Fixed structure; no training drift |
| [Abacus](https://arxiv.org/abs/2405.17399) (McLeish, 2024) | Positional digit encoding | Continuous embeddings; not tokenized |

## Citation

```bibtex
@software{fluxem2025,
  title={FluxEM: Algebraic Embeddings for Deterministic Neural Computation},
  author={Bown, Hunter},
  year={2025},
  url={https://github.com/Hmbown/FluxEM}
}
```

## License

MIT

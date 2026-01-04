# FluxEM

[![PyPI version](https://img.shields.io/pypi/v/fluxem)](https://pypi.org/project/fluxem/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluxem)](https://pypi.org/project/fluxem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Algebraic embeddings for exact neural computation across 11 scientific domains. Parameter-free, deterministic, and multi-framework (JAX, MLX, NumPy).**

<p align="center">
  <img src="docs/demo.gif" alt="FluxEM terminal demo" width="600">
</p>

## Key Features

- **11 Scientific Domains**: Physics, Chemistry, Biology, Math, Logic, Music, Geometry, Graphs, Sets, Number Theory, Data
- **Exact Operations**: Arithmetic, dimensional analysis, chemical reactions, logical inference - all via algebraic structure
- **Multi-Framework**: Supports JAX (Linux/Cloud), MLX (Apple Silicon), and NumPy (fallback)
- **Zero Parameters**: No training required - operations are exact by construction
- **128-dim Unified Format**: All domain embeddings share a common structure for cross-domain composition

## Quick Start

```python
from fluxem import create_unified_model

model = create_unified_model()

model.compute("1234 + 5678")  # -> 6912.0
model.compute("250 * 4")      # -> 1000.0
model.compute("1000 / 8")     # -> 125.0
model.compute("3 ** 4")       # -> 81.0
```

### Domain Examples

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
transposed = pitch_enc.transpose(a4, 7)  # Up a perfect fifth

# Biology: DNA sequences
from fluxem.domains.biology import DNAEncoder
dna_enc = DNAEncoder()
seq = dna_enc.encode('ATCGATCG')

# Math: Complex numbers, matrices, polynomials
from fluxem.domains.math import ComplexEncoder, MatrixEncoder
complex_enc = ComplexEncoder()
z = complex_enc.encode(3, 4)  # 3 + 4i
```

## Installation

Requires Python 3.10+.

```bash
# Core only (NumPy backend)
pip install fluxem

# With JAX backend (recommended for Linux/Cloud)
pip install fluxem[jax]

# With MLX backend (recommended for Apple Silicon)
pip install fluxem[mlx]

# Full installation with HuggingFace integration
pip install fluxem[full-jax]  # or fluxem[full-mlx]
```

Or from source:

```bash
git clone https://github.com/Hmbown/FluxEM.git
cd FluxEM && pip install -e ".[jax]"  # or .[mlx]
```

## Backend Selection

FluxEM automatically selects the best available backend:

```python
from fluxem.backend import set_backend, get_backend, BackendType

# Auto-detect (JAX > MLX > NumPy)
backend = get_backend()
print(f"Using: {backend.name}")

# Explicit selection
set_backend(BackendType.JAX)
set_backend(BackendType.MLX)
set_backend(BackendType.NUMPY)
```

## Supported Domains

| Domain | Key Encoders | Example Operations |
|--------|-------------|-------------------|
| **Physics** | `DimensionalQuantity`, `UnitEncoder` | Unit conversion, dimensional analysis |
| **Chemistry** | `ElementEncoder`, `MoleculeEncoder`, `ReactionEncoder` | Formula parsing, reaction balancing |
| **Biology** | `DNAEncoder`, `ProteinEncoder`, `TaxonomyEncoder` | Sequence encoding, taxonomy hierarchies |
| **Math** | `ArithmeticEncoder`, `ComplexEncoder`, `MatrixEncoder`, `VectorEncoder` | Exact arithmetic, complex operations |
| **Logic** | `PropositionalEncoder`, `PredicateEncoder` | Logical inference, truth evaluation |
| **Music** | `PitchEncoder`, `ChordEncoder`, `AtonalSetEncoder` | Transposition, pitch class analysis |
| **Geometry** | `PointEncoder`, `VectorEncoder`, `TransformEncoder` | Transformations, distance calculations |
| **Graphs** | `GraphEncoder`, `PathEncoder` | Graph properties, pathfinding |
| **Sets** | `SetEncoder`, `RelationEncoder` | Set operations, relation composition |
| **Number Theory** | `IntegerEncoder`, `PrimeEncoder`, `ModularEncoder` | Primality, modular arithmetic |
| **Data** | `ArrayEncoder`, `RecordEncoder` | Array embedding, structured data |

## Architecture

FluxEM uses a unified 128-dimensional embedding format:

```
[0:8]    Domain tag (identifies which domain)
[8:72]   Domain-specific content (64 dims)
[72:128] Reserved for cross-domain composition
```

### Core Principles

1. **Algebraic Homomorphism**: Operations in embedding space correspond exactly to operations in the original domain
   ```python
   # Addition: encode(a) + encode(b) = encode(a + b)
   # Multiplication: log(a) + log(b) = log(a * b)
   ```

2. **Domain Tags**: Each domain has a unique 8-bit tag for type checking and routing

3. **Lazy Initialization**: Framework-agnostic until first use - import works without JAX/MLX installed

## LLM Integration

FluxEM provides tools for integrating with LLMs:

```python
from fluxem.integration.tokenizer import MultiDomainTokenizer
from fluxem.integration.pipeline import TrainingPipeline

# Tokenize mixed text with domain-specific patterns
tokenizer = MultiDomainTokenizer()
tokens = tokenizer.tokenize("Water (H2O) boils at 373.15 K")
# Detects: chemical formula, temperature quantity

# Create training pipeline for LLM integration
pipeline = TrainingPipeline()
```

### Run Experiments

```bash
# Generate data, train models, evaluate
python experiments/scripts/generate_data.py --config experiments/configs/arithmetic_small.yaml
python experiments/scripts/train_hybrid.py --config experiments/configs/arithmetic_small.yaml
python experiments/scripts/eval.py --config experiments/configs/arithmetic_small.yaml
```

See [docs/HYBRID_TRAINING.md](docs/HYBRID_TRAINING.md) for the full architecture and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for detailed instructions.

## Hybrid Training Vision: Embed What Should Be Embedded

FluxEM is motivated by a simple thesis:

**Tokenize natural language. Embed structured domain objects.**

Many tasks we currently “teach” transformers via token prediction are not naturally token problems (exact arithmetic, dimensional analysis, stoichiometry, pitch-class operations, etc.). In token space, models must rediscover the algebra from examples. In FluxEM, a subset of these domains are encoded so that **algebraic operations become geometric operations** (e.g. vector addition, log-addition).

### The idea

Build training data and model interfaces where:

- Text remains text (standard tokenizer)
- Domain objects (numbers, units, molecules, pitch-class sets, …) are detected and encoded as **typed, deterministic 128-d vectors**
- The model is trained on **mixed streams** (token embeddings + FluxEM embeddings) so that attention can route between language and exact structured computation

### Why this could matter

- **Sample efficiency**: the model does not need to learn arithmetic/dimensional algebra from scratch
- **OOD generalization**: if the domain encoder is a homomorphism, many generalization failures disappear by construction
- **Cleaner failure modes**: when parsing fails, the system can explicitly fall back to token-only reasoning (instead of silently “hallucinating math”)

### Minimal architecture sketch

At a high level:

1. **Detection/segmentation** (e.g. `MultiDomainTokenizer`)
2. **Encoding** (domain encoder → 128-d embedding)
3. **Projection** to LLM hidden size (e.g. `MultiDomainProjector`)
4. **Fusion** into the token sequence (prepend/inline “domain tokens”)
5. **Training objectives**
   - standard LM loss on text
   - structured losses where appropriate (decode correctness, algebraic consistency, domain validation)

The core bet: the transformer learns *routing and composition*, while the exact domain algebra is carried by FluxEM representations.

## Roadmap

This roadmap is oriented toward a publishable/reproducible research result: demonstrate that hybrid tokenization+algebraic embeddings improve exactness and generalization.

### Phase 0 — Foundations ✅

- Deterministic encoders with unit tests across domains
- Backend abstraction (JAX/MLX/NumPy) with equivalent semantics
- Mixed-pattern detection primitives (`MultiDomainTokenizer`) and registry (`DomainEncoderRegistry`)

### Phase 1 — Hybrid data format + evaluation harness ✅

- **Mixed-sequence JSONL format** defined in `fluxem/integration/sample_format.py`
- Evaluation suite with exact metrics (arithmetic, dimensional analysis)
- Token-only and hybrid training baselines in `experiments/scripts/`
- See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for exact commands

### Phase 2 — Model integration (small-scale) ✅

- `MultiDomainProjector`: projects 128-d → LLM hidden dim
- Type embeddings for domain routing
- Hybrid transformer implementation in `experiments/scripts/train_hybrid.py`
- See [docs/HYBRID_TRAINING.md](docs/HYBRID_TRAINING.md) for architecture details

### Phase 3 — Scaling + robustness (current)

- Improve detection/segmentation robustness:
  - ambiguity handling (multiple parses)
  - confidence scores and fallbacks
- Add curriculum:
  - start with clean synthetic domains
  - gradually introduce noisy real-world strings
- Add longer chain tasks that alternate text reasoning and domain computation

### Phase 4 — Research-grade release

- Publish a clean benchmark suite + training recipes
- Provide pretrained demonstration checkpoints
- Write a short paper-style report:
  - theory: where exact homomorphisms apply
  - experiments: sample efficiency + OOD generalization
  - limitations: domains without clean algebra, detection errors, mixed-modality training costs

## Why FluxEM?

**Use it when you need:**
- **Exact computation** in neural networks without floating-point approximation
- **Domain-specific embeddings** that preserve algebraic structure
- **Cross-domain reasoning** with a unified embedding format
- **Training-free primitives** as a baseline for learned approaches

### Design Philosophy

FluxEM implements fixed encodings that guarantee algebraic properties:

| Encoding | Operations | Guarantee |
|----------|------------|-----------|
| **Linear** | `+` `-` | `decode(encode(a) + encode(b)) = a + b` |
| **Log** | `*` `/` `**` | Operations reduce to addition in log-space |
| **Domain** | varies | Preserves domain-specific algebraic structure |

## Key Results: 100% OOD Generalization

FluxEM achieves **100% accuracy on out-of-distribution tests** — a result impossible for learned approaches:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EMBEDDING COMPARISON: Accuracy within 1% relative error                    │
├─────────────────┬────────────┬─────────────────┬────────────────────────────┤
│  Method         │  ID Test   │  OOD-Magnitude  │  OOD-Length (chains)       │
├─────────────────┼────────────┼─────────────────┼────────────────────────────┤
│  FluxEM         │   100.0%   │     100.0%      │       100.0%               │
│  Character      │    18.4%   │      56.2%      │        15.4%               │
│  Learned MLP    │     0.4%   │       0.0%      │         1.0%               │
│  Positional     │    69.8%   │       1.8%      │        67.4%               │
└─────────────────┴────────────┴─────────────────┴────────────────────────────┘
```

**Why 100%?** FluxEM's accuracy is *guaranteed by construction*:
- `embed(a) + embed(b) = embed(a + b)` — exact algebraic homomorphism
- No training needed, no distribution to overfit
- Works on *any* numbers, not just training examples

### Benchmark Details

| Test | Description | FluxEM | Best Learned |
|------|-------------|--------|--------------|
| **ID** | Numbers in [1, 1000] | 100% | 69.8% (Positional) |
| **OOD-Magnitude** | Numbers in [1000, 100000] | 100% | 56.2% (Character) |
| **OOD-Length** | Multi-operand chains | 100% | 67.4% (Positional) |

```bash
python -m benchmarks.run_all --quick  # quick test (~15 sec)
python -m benchmarks.run_all          # full benchmark (~20 min)
```

## Why Algebraic Embeddings?

### The Problem with Tokenization
- LLMs tokenize numbers as text ("123" → ['1','2','3'])
- No arithmetic structure preserved
- Must learn operations from data
- **OOD generalization fundamentally limited**

### FluxEM Solution
- Algebraic embeddings preserve operations exactly
- `embed(a) + embed(b) = embed(a+b)` by construction
- No training required for arithmetic
- **Perfect OOD generalization guaranteed**

### Full Comparison (500 samples per condition)

| Approach | ID Accuracy | OOD-Mag | OOD-Length | Training | Median Error |
|----------|-------------|---------|------------|----------|--------------|
| **FluxEM** | **100%** | **100%** | **100%** | **None** | **<0.0001** |
| Character tokens | 18.4% | 56.2% | 15.4% | 5K examples | 0.0315 |
| Learned MLP | 0.4% | 0.0% | 1.0% | 5K examples | >100% |
| Positional encoding | 69.8% | 1.8% | 67.4% | 5K examples | 0.0063 |

### Running Comparisons
```bash
python experiments/scripts/compare_embeddings.py
python experiments/scripts/compare_tokenizers.py
python experiments/scripts/demo_all_domains.py
```

## API Reference

### Core API

```python
from fluxem import (
    # Unified model (all arithmetic)
    create_unified_model,

    # Extended operations
    create_extended_ops,

    # Backend control
    get_backend, set_backend, BackendType,

    # Core infrastructure
    EMBEDDING_DIM, BaseEncoder, UnifiedEncoder,
    create_embedding, DOMAIN_TAGS,
)
```

### Embedding-Level API

```python
from fluxem import create_unified_model

model = create_unified_model(dim=256)

# Addition via linear embeddings
emb_a = model.linear_encoder.encode_number(42.0)
emb_b = model.linear_encoder.encode_number(58.0)
emb_sum = emb_a + emb_b  # Vector addition = numeric addition
result = model.linear_encoder.decode(emb_sum)  # -> 100.0
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

## Prior Work

| Approach | Method | FluxEM Difference |
|----------|--------|-------------------|
| [NALU](https://arxiv.org/abs/1808.00508) (Trask, 2018) | Learned log/exp gates | No learned parameters |
| [xVal](https://arxiv.org/abs/2310.02989) (Golkar, 2023) | Learned scaling direction | Fixed structure; no training drift |
| [Abacus](https://arxiv.org/abs/2405.17399) (McLeish, 2024) | Positional digit encoding | Continuous embeddings; not tokenized |

## Limitations

| Constraint | Behavior |
|------------|----------|
| Zero | Explicit flag; `log(0)` undefined, masked separately |
| Sign | Stored in embedding; log-space operates on magnitude only |
| Negative base + fractional exponent | Unsupported; returns real-valued surrogate |
| Precision | < 1e-6 relative error (float32) |

See [FORMAL_DEFINITION.md](docs/FORMAL_DEFINITION.md) and [ERROR_MODEL.md](docs/ERROR_MODEL.md) for details.

## Citation

```bibtex
@software{fluxem2025,
  title={FluxEM: Algebraic Embeddings for Exact Neural Computation},
  author={Bown, Hunter},
  year={2025},
  url={https://github.com/Hmbown/FluxEM}
}
```

## License

MIT

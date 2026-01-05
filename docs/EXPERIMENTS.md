# Experiments

## Overview

Commands to run training and evaluation scripts and compare token-only and hybrid pipelines.

## Implementation

### Prerequisites

```bash
# Install FluxEM with development dependencies
pip install -e ".[dev]"

# For training experiments (optional torch backend)
pip install torch  # CPU is sufficient for minimal experiments
```

### Usage (minimal run)

```bash
# Generate small arithmetic dataset
python experiments/scripts/generate_data.py --config experiments/configs/arithmetic_small.yaml

# Train token-only baseline
python experiments/scripts/train_token_only.py --config experiments/configs/arithmetic_small.yaml

# Train hybrid model
python experiments/scripts/train_hybrid.py --config experiments/configs/arithmetic_small.yaml

# Evaluate both models
python experiments/scripts/eval.py --config experiments/configs/arithmetic_small.yaml
```

### Experiment structure

```
experiments/
├── README.md              # Quick reference
├── configs/
│   ├── arithmetic_small.yaml   # Small configuration
│   ├── arithmetic_full.yaml    # Full arithmetic experiment
│   └── units_full.yaml         # Dimensional analysis experiment
├── scripts/
│   ├── generate_data.py   # Create train/test JSONL files
│   ├── train_token_only.py # Token-only transformer baseline
│   ├── train_hybrid.py    # FluxEM hybrid model
│   └── eval.py            # Evaluation and metrics
├── data/                  # Generated datasets (gitignored)
└── results/               # Model outputs (gitignored)
```

### Dataset generation

#### Arithmetic dataset

```bash
python experiments/scripts/generate_data.py \
    --config experiments/configs/arithmetic_full.yaml \
    --seed 42
```

Generates:
- `data/arithmetic/train.jsonl` - 10K samples, small integers [0, 999]
- `data/arithmetic/test_id.jsonl` - 1K in-distribution samples
- `data/arithmetic/test_ood_magnitude.jsonl` - 1K large integers [10^6, 10^9]
- `data/arithmetic/test_ood_length.jsonl` - 1K long chains (5+ operations)

#### Units dataset

```bash
python experiments/scripts/generate_data.py \
    --config experiments/configs/units_full.yaml \
    --seed 42
```

Generates:
- `data/units/train.jsonl` - unit conversions and dimensional checks
- `data/units/test_id.jsonl` - in-distribution
- `data/units/test_ood_magnitude.jsonl` - extreme magnitudes
- `data/units/test_ood_types.jsonl` - novel unit combinations

### Training

#### Token-only baseline

```bash
python experiments/scripts/train_token_only.py \
    --config experiments/configs/arithmetic_full.yaml \
    --epochs 50 \
    --seed 42
```

- Character-level transformer baseline
- No FluxEM embeddings
- Model saved to `results/arithmetic/token_only/`

#### Hybrid model

```bash
python experiments/scripts/train_hybrid.py \
    --config experiments/configs/arithmetic_full.yaml \
    --epochs 50 \
    --seed 42
```

- Detects numeric spans and encodes with FluxEM
- Projects 128-d -> hidden dim
- Adds type embeddings for domain routing
- Model saved to `results/arithmetic/hybrid/`

### Evaluation

```bash
python experiments/scripts/eval.py \
    --config experiments/configs/arithmetic_full.yaml
```

#### Metrics

| Metric | Description |
|--------|-------------|
| Exact Match | Output equals target |
| Relative Error | `|pred - target| / |target|` (for numeric) |
| Dimensional Correctness | Correct dimensions in output (for units) |
| Boolean Accuracy | Correct true/false (for `can_add` tasks) |

#### Output artifacts

The evaluation script writes metrics to:

- `results_dir/evaluation_results.json`

### Configuration reference

#### `arithmetic_small.yaml`

```yaml
task: arithmetic
seed: 42

data:
  train_size: 1000
  test_size: 200
  id_range: [0, 99]
  ood_magnitude_range: [10000, 99999]
  ood_max_ops: 5
  operations: ["+", "-", "*"]

model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.1

training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001

paths:
  data_dir: experiments/data/arithmetic_small
  results_dir: experiments/results/arithmetic_small
```

#### `units_full.yaml`

```yaml
task: units
seed: 42

data:
  train_size: 5000
  test_size: 500
  unit_types: ["length", "mass", "time", "temperature", "velocity"]
  include_can_add: true

model:
  hidden_dim: 256
  num_layers: 3
  num_heads: 8
  dropout: 0.1

training:
  epochs: 30
  batch_size: 64
  learning_rate: 0.0005

paths:
  data_dir: experiments/data/units
  results_dir: experiments/results/units
```

## Reproducibility

To reproduce the default configurations:

```bash
# Full arithmetic experiment
python experiments/scripts/generate_data.py --config experiments/configs/arithmetic_full.yaml --seed 42
python experiments/scripts/train_token_only.py --config experiments/configs/arithmetic_full.yaml --seed 42
python experiments/scripts/train_hybrid.py --config experiments/configs/arithmetic_full.yaml --seed 42
python experiments/scripts/eval.py --config experiments/configs/arithmetic_full.yaml

# Full units experiment
python experiments/scripts/generate_data.py --config experiments/configs/units_full.yaml --seed 42
python experiments/scripts/train_token_only.py --config experiments/configs/units_full.yaml --seed 42
python experiments/scripts/train_hybrid.py --config experiments/configs/units_full.yaml --seed 42
python experiments/scripts/eval.py --config experiments/configs/units_full.yaml
```

Experiments are seeded; CPU runs are supported.

## Limitations

### Troubleshooting

#### "No module named 'torch'"

Install PyTorch: `pip install torch` (CPU version is fine)

#### "CUDA out of memory"

Set `device: cpu` in config or use `--device cpu` flag

#### Slow training

Use `arithmetic_small.yaml` for a smaller run.

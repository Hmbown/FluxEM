# FluxEM Experiments

## Overview

End-to-end experiments comparing token-only vs hybrid (FluxEM) training.

## Usage (minimal run)

```bash
# Generate data, train both models, evaluate
python scripts/generate_data.py --config configs/arithmetic_small.yaml
python scripts/train_token_only.py --config configs/arithmetic_small.yaml
python scripts/train_hybrid.py --config configs/arithmetic_small.yaml
python scripts/eval.py --config configs/arithmetic_small.yaml
```

## Implementation

### Structure

```
experiments/
├── configs/           # YAML configurations
├── scripts/           # Python scripts
├── data/              # Generated datasets (gitignored)
└── results/           # Model outputs (gitignored)
```

### Configurations

| Config | Task | Train Size |
|--------|------|------------|
| `arithmetic_small.yaml` | Arithmetic | 1K |
| `arithmetic_full.yaml` | Arithmetic | 10K |
| `units_full.yaml` | Dimensional analysis | 5K |

## Reproducibility

Scripts write datasets under `experiments/data/` and results under `experiments/results/` (both typically gitignored).

See [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) for detailed documentation.

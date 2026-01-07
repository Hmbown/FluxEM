# Repository Guidelines

## Project Structure & Module Organization
FluxEM is a Python package. Core source lives in `fluxem/`, with domain-specific encoders in `fluxem/domains/` (e.g., `math`, `physics`, `music`), backend adapters in `fluxem/backend/` (NumPy/JAX/MLX), and integrations in `fluxem/integration/`. Tests are in `tests/`. Research and evaluation scripts live in `experiments/` and `benchmarks/`, while runnable examples are in `examples/`. Documentation and the project site are in `docs/`. Results, plots, and datasets are in `artifacts/` and `scibench_data/`.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` installs editable mode with pytest/ruff/black.
- `pip install -e ".[jax]"` or `pip install -e ".[mlx]"` adds optional backends.
- `make test` or `python -m pytest` runs the test suite.
- `python examples/basic_usage.py` runs a quick sanity check.
- `python experiments/scripts/compare_embeddings.py` reproduces a benchmark table (requires extra deps).

## Coding Style & Naming Conventions
Use 4-space indentation and standard Python naming: `snake_case` for functions/modules, `PascalCase` for classes, and `UPPER_SNAKE` for constants. Format with Black (line length 88) and lint with Ruff (E/F/W/I/UP). Typical commands: `black .` and `ruff .`.

## Testing Guidelines
Tests use pytest. Place new tests in `tests/` and name files `test_*.py` with functions `test_*`. For coverage, use `python -m pytest --cov=fluxem` (pytest-cov is in dev dependencies).

## Commit & Pull Request Guidelines
Commit messages usually follow Conventional Commits (e.g., `feat: ...`, `fix: ...`, `ci: ...`), but keep them short and descriptive. PRs should include a concise summary, link related issues when applicable, list tests run (or note if not run), and include screenshots/plots for doc or visualization changes.

## Data & Artifacts
`scibench_data/` and `artifacts/` contain large datasets and generated results. Avoid editing or re-adding large binaries unless the change is intentional and documented.

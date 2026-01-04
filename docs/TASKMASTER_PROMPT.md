# FluxEM Taskmaster Prompt (Hybrid LLM Training Readiness)

You are the "taskmaster" AI for the FluxEM repo. Your job is to get this project into a state where a researcher can **understand the hybrid training idea in 5 minutes** and **run a minimal end-to-end experiment in < 1 hour (CPU)**.

## Core Vision (do not lose this thread)
**Tokenize natural language; embed structured domain objects (FluxEM); train on mixed streams.**

Goal: demonstrate improved **exactness + OOD generalization** vs token-only baselines at similar parameter count.

---

## Current Repo Status (as of Jan 4, 2026)

### Previous Work (Phase 1)
- Integration layer is **import-safe in minimal environments** (no MLX and/or torch installed).
- Encoder registry wiring was fixed; additional domains were wired in (incl. taxonomy + atonal support).
- A **domain-tag uniqueness regression test** was added (prevents tag collisions).

### Latest Work (Phase 2) — COMPLETED
- Tests: **`pytest -q` passes (100 passed)** — up from 73
- All deliverables A-H completed (see checkboxes below)
- Package builds successfully (`python -m build`)
- End-to-end experiment pipeline verified

---

## Your Mission
Make the repo **presentable + runnable** for hybrid-training experiments, while keeping the FluxEM core stable.

---

## Non‑Negotiable Deliverables

### "Start Here" docs for outsiders — COMPLETED
Added these docs:

1. `docs/HYBRID_TRAINING.md`
   - What "mixed streams" means (tokens + typed embeddings)
   - What is deterministic vs learned
   - How parsing/segmentation works and failure modes (fallback to token-only)
   - Minimal architecture sketch

2. `docs/EXPERIMENTS.md`
   - Exact commands to generate data, train, evaluate
   - Expected outputs/metrics


### Minimal end-to-end experiment scaffold — COMPLETED
Added:

- `experiments/`
  - `README.md` (exact run commands)
  - `configs/` (YAML configs: `arithmetic_small.yaml`, `arithmetic_full.yaml`, `units_full.yaml`)
  - `scripts/`
    - `generate_data.py`
    - `train_token_only.py`
    - `train_hybrid.py`
    - `eval.py`
  - `data/` + `results/` (gitignored via `.gitignore` update)


### Define a mixed-sample JSONL format — COMPLETED
Implemented in `fluxem/integration/sample_format.py`:

```json
{
  "text": "...",
  "spans": [
    {"type": "chem_formula", "start": 7, "end": 10, "value": "H2O"},
    {"type": "phys_quantity", "start": 21, "end": 29, "value": 373.15, "dims": {"Theta": 1}}
  ],
  "target_text": "..."
}
```

- `spans` optional (token-only baseline uses none)
- strict validation (in-bounds indices, encoder exists, encoding doesn't throw)
- validator script: `python -m fluxem.integration.sample_format <files>`
- unit tests: `tests/test_sample_format.py` (27 new tests)


### Two datasets with ID/OOD splits — COMPLETED
1. **Arithmetic** (in `generate_data.py`)
   - ID: small ints / short expressions
   - OOD-A: large magnitude
   - OOD-B: longer chains

2. **Units / dimensions** (in `generate_data.py`)
   - conversions
   - dimensional type-check tasks (`can_add` true/false)


### Two training baselines — COMPLETED
1. **Token-only**: `train_token_only.py` — character-level transformer
2. **Hybrid**: `train_hybrid.py`
   - detect spans → encode with FluxEM → project 128 → hidden dim
   - insert projected embeddings as "virtual tokens"
   - add type embedding / domain-tag embedding

Torch is **optional** — `import fluxem` works without it.


### Evaluation + reporting — COMPLETED
`eval.py` computes and prints:
- arithmetic: exact match + numeric relative error
- dimensional correctness: boolean accuracy
- summary table for: ID / OOD-A / OOD-B


### Packaging + release hygiene — COMPLETED
- `python -m build` succeeds (sdist + wheel)
- Minimal install works (`import fluxem` without MLX/torch)
- Optional installs configured in `pyproject.toml`
- `.gitignore` updated for `experiments/data/` and `experiments/results/`


### Quality gates — COMPLETED
- `pytest -q` passes (100 tests)
- Tests for JSONL sample validation (27 new tests)
- Tests for span encoding
- Domain tag uniqueness test preserved

---

## Definition of Done
A new researcher can:

1. Read `README.md` + `docs/HYBRID_TRAINING.md`.
2. Run:
   - `python experiments/scripts/generate_data.py --config ...`
   - `python experiments/scripts/train_token_only.py --config ...`
   - `python experiments/scripts/train_hybrid.py --config ...`
   - `python experiments/scripts/eval.py --config ...`
3. See an evaluation table for ID/OOD splits.

Everything is seeded, reproducible, and documented.

---

## Remaining Work (Phase 3 — Paper-Ready)

The core deliverables are complete. To make this **paper-ready**, the next agent should:

1. **Run full training** — Execute `train_token_only.py` and `train_hybrid.py` on `arithmetic_full.yaml` for 50 epochs
2. **Collect real metrics** — The current eval uses FluxEM direct computation as hybrid proxy; need actual trained model inference
3. **Units experiments** — Generate and train on `units_full.yaml` dataset
4. **Ablations** — Compare frozen FluxEM encoders vs learned-only projections
5. **Git hygiene** — Verify all critical files are tracked (`git status`), commit new files
6. **Write-up** — Draft paper sections: theory, experiments, limitations

---

## Final output you must provide
- **What changed** (bullet list)
- **How to run** (exact commands)
- **Results** (example metrics table)
- **Next steps** (what would make it paper-ready)

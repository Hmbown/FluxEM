#!/usr/bin/env python3
"""
FluxEM Benchmark: Single command to run everything.

Usage:
    python -m benchmarks.run_all                 # Full benchmark
    python -m benchmarks.run_all --quick         # Quick test
    python -m benchmarks.run_all --eval-only     # Skip training
    python -m benchmarks.run_all --plot-only     # Only plots
"""

import argparse
import sys
import time
from pathlib import Path


def run_data_generation(output_dir: Path, n_train: int, n_test: int, seed: int):
    """Generate all datasets."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA GENERATION")
    print("=" * 60)

    from .data.generator import generate_all_datasets

    datasets = generate_all_datasets(
        output_dir=output_dir,
        train_samples=n_train,
        test_samples=n_test,
        seed=seed,
    )

    return datasets


def run_baseline_training(
    data_dir: Path,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    device: str,
):
    """Train baseline models."""
    print("\n" + "=" * 60)
    print("STEP 2: BASELINE TRAINING")
    print("=" * 60)

    from .data.generator import load_dataset
    from .models.rnn import GRUBaseline
    from .models.tokenizer import CharTokenizer
    from .models.trainer import Trainer
    from .models.transformer import TinyTransformer

    # Load training data
    train_samples = load_dataset(data_dir / "train.jsonl")
    val_samples = load_dataset(data_dir / "test_id.jsonl")

    # Convert to dicts
    train_data = [
        {"expression": s.expression, "ground_truth": s.ground_truth}
        for s in train_samples
    ]
    val_data = [
        {"expression": s.expression, "ground_truth": s.ground_truth}
        for s in val_samples
    ]

    tokenizer = CharTokenizer()

    # Train Transformer
    print("\n--- Training Transformer ---")
    transformer = TinyTransformer(vocab_size=tokenizer.vocab_size)
    print(f"Parameters: {transformer.count_parameters():,}")

    trainer = Trainer(transformer, tokenizer, device=device)
    trainer.train(
        train_data,
        val_data,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir / "transformer",
    )

    # Train GRU
    print("\n--- Training GRU ---")
    gru = GRUBaseline(vocab_size=tokenizer.vocab_size)
    print(f"Parameters: {gru.count_parameters():,}")

    trainer = Trainer(gru, tokenizer, device=device)
    trainer.train(
        train_data,
        val_data,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir / "rnn",
    )


def run_evaluation(
    data_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device: str,
):
    """Run full evaluation."""
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATION")
    print("=" * 60)

    from .eval_all import run_evaluation as eval_fn

    results = eval_fn(
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        device=device,
    )

    return results


def run_plotting(results_path: Path, output_dir: Path):
    """Generate all figures."""
    print("\n" + "=" * 60)
    print("STEP 4: PLOTTING")
    print("=" * 60)

    from .plot_results import (
        generate_markdown_summary,
        load_results,
        plot_accuracy_vs_ops,
        plot_main_comparison,
    )

    results = load_results(results_path)

    plot_main_comparison(results, output_dir / "main_comparison.png")
    plot_accuracy_vs_ops(results, output_dir / "accuracy_vs_ops.png")
    generate_markdown_summary(results, output_dir.parent / "results.md")


def main():
    parser = argparse.ArgumentParser(
        description="Run FluxEM arithmetic benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    # Run modes
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--eval-only", action="store_true", help="Skip training")
    parser.add_argument("--plot-only", action="store_true", help="Only plots")

    # Training params
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    args = parser.parse_args()

    # Directory structure
    data_dir = args.output_dir / "datasets"
    checkpoint_dir = args.output_dir / "checkpoints"
    results_dir = args.output_dir / "results"
    figures_dir = args.output_dir / "figures"

    # Quick mode
    if args.quick:
        n_train, n_test = 1000, 200
        args.epochs = 10
    else:
        n_train, n_test = 10000, 1000

    start_time = time.time()

    print("=" * 60)
    print("FluxEM Benchmark")
    print("=" * 60)
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Mode: {'quick' if args.quick else 'full'}")

    try:
        if args.plot_only:
            run_plotting(results_dir / "results.json", figures_dir)
        elif args.eval_only:
            run_evaluation(data_dir, checkpoint_dir, results_dir, args.device)
            run_plotting(results_dir / "results.json", figures_dir)
        else:
            # Full pipeline
            run_data_generation(data_dir, n_train, n_test, args.seed)
            run_baseline_training(
                data_dir, checkpoint_dir, args.epochs, args.batch_size, args.device
            )
            run_evaluation(data_dir, checkpoint_dir, results_dir, args.device)
            run_plotting(results_dir / "results.json", figures_dir)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Results: {results_dir / 'results.json'}")
        print(f"Figures: {figures_dir}")
        print(f"Summary: {args.output_dir / 'results.md'}")
        print("=" * 60)

        # Print reproduction command
        print("\nTo reproduce:")
        print(f"  python -m benchmarks.run_all --seed {args.seed}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Main evaluation script."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from .data.generator import Sample, load_dataset
from .metrics import compute_metrics, compute_metrics_by_num_ops
from .models.rnn import GRUBaseline
from .models.tokenizer import CharTokenizer
from .models.transformer import TinyTransformer
from .parser.evaluator import FluxEMEvaluator
from .parser.parser import parse_expression


def evaluate_fluxem(samples: List[Sample], verbose: bool = True) -> List[float]:
    """Evaluate FluxEM on samples."""
    evaluator = FluxEMEvaluator()
    predictions = []

    iterator = tqdm(samples, desc="FluxEM") if verbose else samples
    for sample in iterator:
        try:
            ast = parse_expression(sample.expression)
            pred = evaluator.evaluate(ast)
            predictions.append(pred)
        except Exception as e:
            if verbose:
                print(f"Error on '{sample.expression}': {e}")
            predictions.append(0.0)

    return predictions


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    samples: List[Sample],
    device: str = "cpu",
    use_log_scale: bool = True,
    verbose: bool = True,
) -> List[float]:
    """Evaluate a PyTorch model on samples."""
    model.eval()
    model.to(device)
    predictions = []

    with torch.no_grad():
        iterator = tqdm(samples, desc="Model") if verbose else samples
        for sample in iterator:
            input_ids, mask = tokenizer.encode_batch([sample.expression])
            input_ids = input_ids.to(device)
            mask = mask.to(device)

            pred = model(input_ids, mask)

            # Unscale if needed
            if use_log_scale:
                pred = torch.sign(pred) * (torch.exp(torch.abs(pred)) - 1)

            predictions.append(pred.item())

    return predictions


def load_baseline_model(
    model_type: str,
    checkpoint_path: Path,
    vocab_size: int,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a trained baseline model."""
    if model_type == "transformer":
        model = TinyTransformer(vocab_size=vocab_size)
    elif model_type == "rnn":
        model = GRUBaseline(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def run_evaluation(
    data_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation suite."""
    tokenizer = CharTokenizer()

    # Load test datasets
    test_distributions = ["test_id", "ood_a", "ood_b", "ood_c"]
    datasets = {}
    for dist in test_distributions:
        # Try different naming conventions
        for suffix in [f"test_{dist}.jsonl", f"{dist}.jsonl"]:
            path = data_dir / suffix
            if path.exists():
                datasets[dist] = load_dataset(path)
                break

    results = {
        "fluxem": {},
        "transformer": {},
        "rnn": {},
    }

    # Evaluate FluxEM
    if verbose:
        print("\n=== Evaluating FluxEM ===")
    for dist, samples in datasets.items():
        if verbose:
            print(f"\nDistribution: {dist}")
        preds = evaluate_fluxem(samples, verbose)
        gts = [s.ground_truth for s in samples]
        num_ops = [s.num_ops for s in samples]

        results["fluxem"][dist] = {
            "overall": compute_metrics(preds, gts),
            "by_ops": {
                str(k): v for k, v in compute_metrics_by_num_ops(preds, gts, num_ops).items()
            },
        }

    # Evaluate baselines
    for model_type in ["transformer", "rnn"]:
        checkpoint = checkpoint_dir / model_type / "best_model.pt"
        if not checkpoint.exists():
            if verbose:
                print(f"\nSkipping {model_type} (no checkpoint)")
            continue

        if verbose:
            print(f"\n=== Evaluating {model_type} ===")
        model = load_baseline_model(model_type, checkpoint, tokenizer.vocab_size, device)

        for dist, samples in datasets.items():
            if verbose:
                print(f"\nDistribution: {dist}")
            preds = evaluate_model(model, tokenizer, samples, device, verbose=verbose)
            gts = [s.ground_truth for s in samples]
            num_ops = [s.num_ops for s in samples]

            results[model_type][dist] = {
                "overall": compute_metrics(preds, gts),
                "by_ops": {
                    str(k): v for k, v in compute_metrics_by_num_ops(preds, gts, num_ops).items()
                },
            }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate arithmetic benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    results = run_evaluation(
        args.data_dir,
        args.checkpoint_dir,
        args.output_dir,
        args.device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for method in ["fluxem", "transformer", "rnn"]:
        if method in results and results[method]:
            print(f"\n{method.upper()}")
            for dist, metrics in results[method].items():
                acc = metrics["overall"]["accuracy"]
                mae = metrics["overall"]["mae"]
                print(f"  {dist}: accuracy={acc:.1%}, MAE={mae:.2f}")


if __name__ == "__main__":
    main()

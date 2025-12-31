"""Generate publication-quality figures for FluxEM benchmark."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
COLORS = {
    "fluxem": "#2E86AB",
    "transformer": "#E94F37",
    "rnn": "#F39237",
}
LABELS = {
    "fluxem": "FluxEM (ours)",
    "transformer": "Transformer",
    "rnn": "GRU",
}


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def plot_main_comparison(results: Dict, output_path: Path):
    """Create the main comparison figure (2 panels)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # === Left: Accuracy by Distribution ===
    ax1 = axes[0]

    distributions = ["test_id", "ood_a", "ood_b", "ood_c"]
    dist_labels = ["ID Test", "OOD-A\n(Large Ints)", "OOD-B\n(Long Expr)", "OOD-C\n(With **)"]
    methods = ["fluxem", "transformer", "rnn"]

    x = np.arange(len(distributions))
    width = 0.25

    for i, method in enumerate(methods):
        if method not in results or not results[method]:
            continue

        accuracies = []
        for dist in distributions:
            if dist in results[method]:
                acc = results[method][dist]["overall"]["accuracy"]
            else:
                acc = 0
            accuracies.append(acc * 100)

        offset = (i - 1) * width
        bars = ax1.bar(
            x + offset,
            accuracies,
            width,
            label=LABELS[method],
            color=COLORS[method],
            alpha=0.9,
        )

        # Value labels on FluxEM bars
        if method == "fluxem":
            for bar, acc in zip(bars, accuracies):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{acc:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax1.set_xlabel("Test Distribution", fontsize=12)
    ax1.set_ylabel("Accuracy (% within 1% tolerance)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dist_labels, fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_title("(a) Accuracy Across Distributions", fontsize=12, fontweight="bold")
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.grid(axis="y", alpha=0.3)

    # === Right: Mean Relative Error ===
    ax2 = axes[1]

    mean_errors = []
    for method in methods:
        if method not in results or not results[method]:
            mean_errors.append(0)
            continue

        errors = []
        for dist in ["ood_a", "ood_b", "ood_c"]:
            if dist in results[method]:
                errors.append(results[method][dist]["overall"]["mean_relative_error"])
        mean_errors.append(np.mean(errors) if errors else 0)

    x2 = np.arange(len(methods))
    colors = [COLORS[m] for m in methods]

    bars2 = ax2.bar(x2, mean_errors, color=colors, alpha=0.9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([LABELS[m] for m in methods], fontsize=10)
    ax2.set_ylabel("Mean Relative Error (OOD)", fontsize=12)
    ax2.set_title("(b) Error on Out-of-Distribution", fontsize=12, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(axis="y", alpha=0.3)

    # Value labels
    for bar, err in zip(bars2, mean_errors):
        if err > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.5,
                f"{err:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_vs_ops(results: Dict, output_path: Path):
    """Plot accuracy vs number of operations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["fluxem", "transformer", "rnn"]

    for method in methods:
        if method not in results or not results[method]:
            continue

        # Aggregate across distributions
        ops_to_acc = {}

        for dist, dist_results in results[method].items():
            if "by_ops" not in dist_results:
                continue
            for n_ops_str, metrics in dist_results["by_ops"].items():
                n_ops = int(n_ops_str)
                if n_ops not in ops_to_acc:
                    ops_to_acc[n_ops] = []
                ops_to_acc[n_ops].append(metrics["accuracy"])

        if not ops_to_acc:
            continue

        ops = sorted(ops_to_acc.keys())
        accs = [np.mean(ops_to_acc[n]) * 100 for n in ops]

        ax.plot(
            ops,
            accs,
            "o-",
            label=LABELS[method],
            color=COLORS[method],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Number of Operations", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Expression Complexity", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_markdown_summary(results: Dict, output_path: Path):
    """Generate markdown summary table."""
    methods = ["fluxem", "transformer", "rnn"]
    distributions = ["test_id", "ood_a", "ood_b", "ood_c"]

    lines = [
        "# FluxEM Benchmark Results",
        "",
        "## Summary",
        "",
        "| Method | ID Test | OOD-A | OOD-B | OOD-C | OOD Avg |",
        "|--------|---------|-------|-------|-------|---------|",
    ]

    for method in methods:
        row = [LABELS.get(method, method)]
        ood_accs = []

        for dist in distributions:
            if method in results and dist in results[method]:
                acc = results[method][dist]["overall"]["accuracy"] * 100
                row.append(f"{acc:.1f}%")
                if dist != "test_id":
                    ood_accs.append(acc)
            else:
                row.append("-")

        if ood_accs:
            row.append(f"{np.mean(ood_accs):.1f}%")
        else:
            row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- **Accuracy**: Fraction of predictions within 1% relative error of ground truth",
        "- **OOD-A**: Large integers [10K, 1M]",
        "- **OOD-B**: Longer expressions (4-8 operations)",
        "- **OOD-C**: Mixed operations with exponentiation",
        "",
        "## Key Finding",
        "",
        "FluxEM maintains ~100% accuracy across all distributions because arithmetic",
        "is implemented via structure-preserving embeddings (homomorphisms), not learned.",
        "Baselines fail on OOD because they memorize patterns, not arithmetic structure.",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--results", type=Path, default=Path("artifacts/results/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()

    results = load_results(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    plot_main_comparison(results, args.output_dir / "main_comparison.png")
    plot_accuracy_vs_ops(results, args.output_dir / "accuracy_vs_ops.png")

    # Generate markdown summary
    generate_markdown_summary(results, args.output_dir.parent / "results.md")

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

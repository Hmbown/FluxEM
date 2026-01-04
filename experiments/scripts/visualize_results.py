#!/usr/bin/env python3
"""
Visualize FluxEM experiment results.

Generates:
- Bar charts: ID vs OOD accuracy by method
- Training curves (if available)
- Error distribution histograms
- Markdown tables for README

Matplotlib is optional - falls back to text tables if not installed.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not installed. Generating text tables only.")
    print("Install with: pip install matplotlib")


def load_json(path: Path) -> Dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def find_result_files(results_dir: Path) -> Dict[str, Dict]:
    """
    Find and load all result files in the results directory.

    Returns dict mapping experiment_name -> results
    """
    results = {}

    # Look for evaluation_results.json in subdirectories
    for eval_file in results_dir.glob("**/evaluation_results.json"):
        exp_name = eval_file.parent.name
        try:
            results[exp_name] = load_json(eval_file)
        except Exception as e:
            print(f"Warning: Could not load {eval_file}: {e}")

    # Look for embedding_comparison.json (from compare_embeddings.py)
    for comp_file in results_dir.glob("**/embedding_comparison.json"):
        try:
            data = load_json(comp_file)
            # Convert embedding comparison format to standard format
            if "results" in data:
                converted = convert_embedding_comparison(data)
                results["embedding_comparison"] = converted
        except Exception as e:
            print(f"Warning: Could not load {comp_file}: {e}")

    # Look for top-level result files
    for json_file in results_dir.glob("*.json"):
        if json_file.name in ["evaluation_results.json", "embedding_comparison.json"]:
            continue
        if "training_history" in json_file.name:
            continue
        try:
            exp_name = json_file.stem
            data = load_json(json_file)
            # Check if it's an embedding comparison file
            if "results" in data and "config" in data:
                converted = convert_embedding_comparison(data)
                results[exp_name] = converted
            else:
                results[exp_name] = data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    # Also look for other JSON result files in subdirectories
    for json_file in results_dir.glob("**/*.json"):
        if json_file.name == "evaluation_results.json":
            continue
        if json_file.name == "embedding_comparison.json":
            continue
        if "training_history" in json_file.name:
            exp_name = json_file.parent.name
            key = f"{exp_name}_training"
            try:
                results[key] = load_json(json_file)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")

    return results


def convert_embedding_comparison(data: Dict) -> Dict:
    """
    Convert embedding_comparison.json format to standard evaluation format.

    Input format:
    {
        "results": {
            "fluxem": {"id": {"accuracy": 1.0, ...}, "ood_magnitude": {...}, ...},
            ...
        }
    }

    Output format:
    {
        "fluxem": {"test_id": {"numeric_accuracy_1pct": 1.0, ...}, ...},
        ...
    }
    """
    converted = {}
    results = data.get("results", {})

    # Map from embedding comparison split names to standard names
    split_map = {
        "id": "test_id",
        "ood_magnitude": "test_ood_magnitude",
        "ood_chain": "test_ood_length",
        "ood_length": "test_ood_length",
    }

    for method, method_data in results.items():
        converted[method] = {}
        for split, split_data in method_data.items():
            std_split = split_map.get(split, f"test_{split}")
            converted[method][std_split] = {}

            # Map metrics
            if isinstance(split_data, dict):
                if "accuracy" in split_data:
                    converted[method][std_split]["numeric_accuracy_1pct"] = split_data["accuracy"]
                if "exact_match" in split_data:
                    converted[method][std_split]["exact_match"] = split_data["exact_match"]
                if "mean_error" in split_data:
                    converted[method][std_split]["mean_relative_error"] = split_data["mean_error"]
                if "median_error" in split_data:
                    converted[method][std_split]["median_relative_error"] = split_data["median_error"]
                # Copy any other metrics
                for k, v in split_data.items():
                    if k not in ["accuracy", "exact_match", "mean_error", "median_error"]:
                        converted[method][std_split][k] = v

    return converted


def find_training_histories(results_dir: Path) -> Dict[str, List[Dict]]:
    """Find training history files."""
    histories = {}

    for hist_file in results_dir.glob("**/training_history.json"):
        exp_name = hist_file.parent.name
        try:
            histories[exp_name] = load_json(hist_file)
        except Exception as e:
            print(f"Warning: Could not load {hist_file}: {e}")

    return histories


def extract_metrics(results: Dict[str, Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Extract metrics from results into a standardized format.

    Returns: {experiment: {method: {split: {metric: value}}}}
    """
    extracted = {}

    for exp_name, exp_results in results.items():
        if "_training" in exp_name:
            continue

        extracted[exp_name] = {}

        # Results are typically structured as {method: {split: {metric: value}}}
        for method, method_results in exp_results.items():
            if isinstance(method_results, dict):
                extracted[exp_name][method] = method_results

    return extracted


# =============================================================================
# Text Table Generation (always available)
# =============================================================================

def generate_text_table(metrics: Dict, title: str = "Results") -> str:
    """Generate a text-based results table."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)

    for exp_name, exp_data in metrics.items():
        lines.append(f"\nExperiment: {exp_name}")
        lines.append("-" * 50)

        # Get all methods and splits
        methods = list(exp_data.keys())
        if not methods:
            continue

        all_splits = set()
        for method_data in exp_data.values():
            all_splits.update(method_data.keys())
        splits = sorted(all_splits)

        # Header
        header = f"{'Split':<25}"
        for method in methods:
            header += f" {method:<15}"
        lines.append(header)
        lines.append("-" * len(header))

        # Numeric accuracy
        lines.append("\nNumeric Accuracy (within 1%):")
        for split in splits:
            row = f"  {split:<23}"
            for method in methods:
                val = exp_data.get(method, {}).get(split, {}).get("numeric_accuracy_1pct", 0)
                if val is not None:
                    row += f" {val*100:>13.1f}%"
                else:
                    row += f" {'N/A':>14}"
            lines.append(row)

        # Exact match
        lines.append("\nExact Match:")
        for split in splits:
            row = f"  {split:<23}"
            for method in methods:
                val = exp_data.get(method, {}).get(split, {}).get("exact_match", 0)
                if val is not None:
                    row += f" {val*100:>13.1f}%"
                else:
                    row += f" {'N/A':>14}"
            lines.append(row)

        # Relative error
        lines.append("\nMedian Relative Error:")
        for split in splits:
            row = f"  {split:<23}"
            for method in methods:
                val = exp_data.get(method, {}).get(split, {}).get("median_relative_error")
                if val is not None and val != float('inf'):
                    if val < 0.01:
                        row += f" {val:>14.2e}"
                    else:
                        row += f" {val:>14.4f}"
                else:
                    row += f" {'inf':>14}"
            lines.append(row)

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def generate_markdown_table(metrics: Dict) -> str:
    """Generate a markdown table suitable for README."""
    lines = []

    for exp_name, exp_data in metrics.items():
        lines.append(f"### {exp_name.replace('_', ' ').title()}")
        lines.append("")

        methods = list(exp_data.keys())
        if not methods:
            continue

        all_splits = set()
        for method_data in exp_data.values():
            all_splits.update(method_data.keys())
        splits = sorted(all_splits)

        # Accuracy table
        lines.append("#### Numeric Accuracy (within 1% error)")
        lines.append("")

        # Header
        header = "| Split |"
        separator = "|-------|"
        for method in methods:
            header += f" {method} |"
            separator += "-------:|"
        lines.append(header)
        lines.append(separator)

        # Rows
        for split in splits:
            row = f"| {split} |"
            for method in methods:
                val = exp_data.get(method, {}).get(split, {}).get("numeric_accuracy_1pct", 0)
                if val is not None:
                    row += f" {val*100:.1f}% |"
                else:
                    row += " N/A |"
            lines.append(row)

        lines.append("")

        # Error table
        lines.append("#### Relative Error")
        lines.append("")

        header = "| Split |"
        separator = "|-------|"
        for method in methods:
            header += f" {method} |"
            separator += "-------:|"
        lines.append(header)
        lines.append(separator)

        for split in splits:
            row = f"| {split} |"
            for method in methods:
                val = exp_data.get(method, {}).get(split, {}).get("median_relative_error")
                if val is not None and val != float('inf'):
                    if val < 0.0001:
                        row += f" {val:.2e} |"
                    elif val > 100:
                        row += " >100 |"
                    else:
                        row += f" {val:.4f} |"
                else:
                    row += " inf |"
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def generate_summary_table(metrics: Dict) -> str:
    """Generate a compact summary table for all experiments."""
    lines = []
    lines.append("## Experiment Results Summary")
    lines.append("")
    lines.append("| Experiment | Method | ID Acc | OOD-Mag Acc | OOD-Len Acc |")
    lines.append("|------------|--------|--------|-------------|-------------|")

    for exp_name, exp_data in metrics.items():
        for method in exp_data.keys():
            id_acc = exp_data[method].get("test_id", {}).get("numeric_accuracy_1pct", 0)
            ood_mag = exp_data[method].get("test_ood_magnitude", {}).get("numeric_accuracy_1pct", 0)
            ood_len = exp_data[method].get("test_ood_length", {}).get("numeric_accuracy_1pct", 0)

            id_str = f"{id_acc*100:.1f}%" if id_acc is not None else "N/A"
            mag_str = f"{ood_mag*100:.1f}%" if ood_mag is not None else "N/A"
            len_str = f"{ood_len*100:.1f}%" if ood_len is not None else "N/A"

            lines.append(f"| {exp_name} | {method} | {id_str} | {mag_str} | {len_str} |")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# Matplotlib Plotting (optional)
# =============================================================================

def plot_accuracy_comparison(metrics: Dict, output_dir: Path):
    """Generate bar chart comparing ID vs OOD accuracy by method."""
    if not MATPLOTLIB_AVAILABLE:
        return

    for exp_name, exp_data in metrics.items():
        methods = list(exp_data.keys())
        if not methods:
            continue

        # Define splits of interest
        splits = ["test_id", "test_ood_magnitude", "test_ood_length"]
        split_labels = ["ID", "OOD-Magnitude", "OOD-Length"]

        # Prepare data
        x_positions = range(len(splits))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

        for i, method in enumerate(methods):
            accuracies = []
            for split in splits:
                acc = exp_data.get(method, {}).get(split, {}).get("numeric_accuracy_1pct", 0)
                accuracies.append(acc * 100 if acc is not None else 0)

            offset = (i - len(methods)/2 + 0.5) * width
            bars = ax.bar([x + offset for x in x_positions], accuracies,
                         width, label=method, color=colors[i % len(colors)])

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.annotate(f'{acc:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Test Split')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy Comparison: {exp_name.replace("_", " ").title()}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(split_labels)
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{exp_name}_accuracy_comparison.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


def plot_error_comparison(metrics: Dict, output_dir: Path):
    """Generate bar chart comparing relative errors by method."""
    if not MATPLOTLIB_AVAILABLE:
        return

    for exp_name, exp_data in metrics.items():
        methods = list(exp_data.keys())
        if not methods:
            continue

        splits = ["test_id", "test_ood_magnitude", "test_ood_length"]
        split_labels = ["ID", "OOD-Magnitude", "OOD-Length"]

        fig, ax = plt.subplots(figsize=(10, 6))

        x_positions = range(len(splits))
        width = 0.35
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

        for i, method in enumerate(methods):
            errors = []
            for split in splits:
                err = exp_data.get(method, {}).get(split, {}).get("median_relative_error")
                if err is None or err == float('inf') or err > 1000:
                    errors.append(1000)  # Cap for visualization
                else:
                    errors.append(err)

            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar([x + offset for x in x_positions], errors,
                  width, label=method, color=colors[i % len(colors)])

        ax.set_xlabel('Test Split')
        ax.set_ylabel('Median Relative Error (log scale)')
        ax.set_title(f'Error Comparison: {exp_name.replace("_", " ").title()}')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(split_labels)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{exp_name}_error_comparison.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


def plot_training_curves(histories: Dict, output_dir: Path):
    """Plot training loss curves if available."""
    if not MATPLOTLIB_AVAILABLE or not histories:
        return

    for exp_name, history in histories.items():
        if not isinstance(history, list) or not history:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if history has expected structure
        if isinstance(history[0], dict):
            epochs = range(1, len(history) + 1)

            if "train_loss" in history[0]:
                train_losses = [h.get("train_loss", 0) for h in history]
                ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)

            if "test_loss" in history[0]:
                test_losses = [h.get("test_loss", 0) for h in history]
                ax.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)

            if "val_loss" in history[0]:
                val_losses = [h.get("val_loss", 0) for h in history]
                ax.plot(epochs, val_losses, 'g-', label='Val Loss', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Curves: {exp_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{exp_name}_training_curves.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")


def plot_method_radar(metrics: Dict, output_dir: Path):
    """Generate radar chart comparing methods across metrics."""
    if not MATPLOTLIB_AVAILABLE:
        return

    import numpy as np

    for exp_name, exp_data in metrics.items():
        methods = list(exp_data.keys())
        if len(methods) < 2:
            continue

        # Define metrics to compare
        metric_names = ['ID Accuracy', 'OOD-Mag Accuracy', 'OOD-Len Accuracy', 'Low Error']

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

        for i, method in enumerate(methods):
            values = []

            # ID accuracy
            id_acc = exp_data[method].get("test_id", {}).get("numeric_accuracy_1pct", 0)
            values.append(id_acc * 100 if id_acc else 0)

            # OOD magnitude accuracy
            ood_mag = exp_data[method].get("test_ood_magnitude", {}).get("numeric_accuracy_1pct", 0)
            values.append(ood_mag * 100 if ood_mag else 0)

            # OOD length accuracy
            ood_len = exp_data[method].get("test_ood_length", {}).get("numeric_accuracy_1pct", 0)
            values.append(ood_len * 100 if ood_len else 0)

            # Low error (inverse of median error, capped)
            med_err = exp_data[method].get("test_id", {}).get("median_relative_error", 1)
            if med_err == float('inf') or med_err > 100:
                low_err = 0
            else:
                low_err = max(0, 100 - med_err * 100)
            values.append(low_err)

            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 100)
        ax.set_title(f'Method Comparison: {exp_name.replace("_", " ").title()}')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        output_path = output_dir / f"{exp_name}_radar.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize FluxEM experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing result files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: results_dir/figures)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "markdown", "all"],
        default="all",
        help="Output format for tables"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (text tables only)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run experiments first to generate results.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Load results
    all_results = find_result_files(results_dir)

    if not all_results:
        print("No result files found.")
        return

    print(f"Found {len(all_results)} result files")

    # Extract metrics
    metrics = extract_metrics(all_results)

    # Generate text tables
    if args.format in ["text", "all"]:
        text_table = generate_text_table(metrics, "FluxEM Experiment Results")
        print(text_table)

        # Save to file
        text_output = output_dir / "results_table.txt"
        with open(text_output, "w") as f:
            f.write(text_table)
        print(f"\nSaved: {text_output}")

    # Generate markdown tables
    if args.format in ["markdown", "all"]:
        md_table = generate_markdown_table(metrics)
        summary_table = generate_summary_table(metrics)

        md_output = output_dir / "results_table.md"
        with open(md_output, "w") as f:
            f.write("# FluxEM Experiment Results\n\n")
            f.write(summary_table)
            f.write("\n## Detailed Results\n\n")
            f.write(md_table)
        print(f"Saved: {md_output}")

    # Generate plots
    if not args.no_plots and MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")

        plot_accuracy_comparison(metrics, output_dir)
        plot_error_comparison(metrics, output_dir)
        plot_method_radar(metrics, output_dir)

        # Load and plot training histories
        histories = find_training_histories(results_dir)
        if histories:
            plot_training_curves(histories, output_dir)

        print(f"\nAll figures saved to: {output_dir}")
    elif not MATPLOTLIB_AVAILABLE:
        print("\nSkipping plots (matplotlib not installed)")
        print("Install with: pip install matplotlib")

    # Print summary
    print("\n" + "=" * 60)
    print("  VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - results_table.txt  (text format)")
    print("  - results_table.md   (markdown format)")
    if MATPLOTLIB_AVAILABLE and not args.no_plots:
        print("  - *_accuracy_comparison.png")
        print("  - *_error_comparison.png")
        print("  - *_radar.png")
        print("  - *_training_curves.png (if training history available)")


if __name__ == "__main__":
    main()

"""
SCAN benchmark evaluation.

Evaluates the algebraic oracle solver on all SCAN splits.
This is an oracle baseline that separates rule discovery from rule execution.
"""

import argparse

from .algebra import AlgebraicSCANSolver, evaluate_accuracy
from .scan_loader import load_scan_split


SPLITS = ['addprim_jump', 'addprim_turn_left', 'length', 'simple']


def evaluate_algebraic(splits: list = None, verbose: bool = False) -> dict:
    """
    Evaluate algebraic solver on SCAN splits.

    Args:
        splits: List of split names (default: all)
        verbose: Print failures

    Returns:
        Dict mapping split name to (accuracy, num_examples)
    """
    if splits is None:
        splits = SPLITS

    solver = AlgebraicSCANSolver()
    results = {}

    for split in splits:
        test_data = load_scan_split(split, 'test')
        accuracy = evaluate_accuracy(test_data, solver, verbose=verbose)
        results[split] = {
            'accuracy': accuracy,
            'num_examples': len(test_data),
            'correct': int(accuracy * len(test_data)),
        }

    return results


def print_results(results: dict, title: str = "Results"):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)
    print(f"{'Split':<25} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print('-' * 60)

    for split, data in results.items():
        acc = data['accuracy']
        correct = data['correct']
        total = data['num_examples']
        print(f"{split:<25} {acc:>9.2%} {correct:>10} {total:>10}")

    print('=' * 60)

    # Overall
    total_correct = sum(d['correct'] for d in results.values())
    total_examples = sum(d['num_examples'] for d in results.values())
    overall_acc = total_correct / total_examples if total_examples else 0
    print(f"{'OVERALL':<25} {overall_acc:>9.2%} {total_correct:>10} {total_examples:>10}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate SCAN oracle baseline')
    parser.add_argument(
        '--split', '-s',
        type=str,
        choices=SPLITS + ['all'],
        default='all',
        help='SCAN split to evaluate on'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print failure examples'
    )
    args = parser.parse_args()

    splits = SPLITS if args.split == 'all' else [args.split]

    print("\n" + "=" * 60)
    print("SCAN Oracle Baseline")
    print("=" * 60)
    print("\nMethod: Algebraic solver with encoded rules (no training)")
    print("Purpose: Separate rule discovery from rule execution")
    print()
    print("This is an ORACLE BASELINE, not a learning result.")
    print("It demonstrates: once rules are known, composition is trivial.")
    print()

    results = evaluate_algebraic(splits, verbose=args.verbose)
    print_results(results, "Oracle Baseline Results")

    # Summary
    all_perfect = all(d['accuracy'] == 1.0 for d in results.values())
    if all_perfect:
        print("Result: 100% on all splits (expected for an oracle).")
        print("\nThe interesting question: Can we LEARN the rules from limited examples?")
    else:
        print("Note: Some splits did not achieve 100% accuracy.")
        print("This indicates a bug in the oracle implementation.")


if __name__ == "__main__":
    main()

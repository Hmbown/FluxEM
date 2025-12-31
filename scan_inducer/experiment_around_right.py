"""
MVP Experiment: Induce "around right" from few examples.

This demonstrates few-shot rule induction:
1. Train on walk/run/look around right (2-3 examples)
2. Test on jump around right (never seen during training)

Success criteria:
- Induces a program equivalent to Repeat(4, Concat(RTURN, ACTION))
- Generalizes perfectly to jump around right
"""

from typing import List, Tuple
from .induce import induce_and_verify, induce_unary_operator
from .dsl import pretty_print


# Ground truth outputs for "around right" pattern
# Format: command -> output tokens
AROUND_RIGHT_EXAMPLES = {
    "walk around right": [
        "I_TURN_RIGHT", "I_WALK",
        "I_TURN_RIGHT", "I_WALK",
        "I_TURN_RIGHT", "I_WALK",
        "I_TURN_RIGHT", "I_WALK",
    ],
    "run around right": [
        "I_TURN_RIGHT", "I_RUN",
        "I_TURN_RIGHT", "I_RUN",
        "I_TURN_RIGHT", "I_RUN",
        "I_TURN_RIGHT", "I_RUN",
    ],
    "look around right": [
        "I_TURN_RIGHT", "I_LOOK",
        "I_TURN_RIGHT", "I_LOOK",
        "I_TURN_RIGHT", "I_LOOK",
        "I_TURN_RIGHT", "I_LOOK",
    ],
    "jump around right": [
        "I_TURN_RIGHT", "I_JUMP",
        "I_TURN_RIGHT", "I_JUMP",
        "I_TURN_RIGHT", "I_JUMP",
        "I_TURN_RIGHT", "I_JUMP",
    ],
}


def get_examples(commands: List[str]) -> List[Tuple[str, List[str]]]:
    """Get examples for the given commands."""
    return [(cmd, AROUND_RIGHT_EXAMPLES[cmd]) for cmd in commands]


def run_experiment(
    train_commands: List[str],
    test_commands: List[str],
    verbose: bool = True,
):
    """Run the induction experiment."""
    train_examples = get_examples(train_commands)
    test_examples = get_examples(test_commands)

    program, train_acc, test_acc = induce_and_verify(
        train_examples,
        test_examples,
        operator_pattern="around right",
        max_size=12,
        verbose=verbose,
    )

    if program is not None and verbose:
        print(f"\nInduced Program AST:")
        print(pretty_print(program))
        print()

    return program, train_acc, test_acc


def main():
    """Run the MVP experiment."""
    print("=" * 60)
    print("SCAN Rule Induction: MVP Experiment")
    print("=" * 60)
    print()
    print("Task: Induce 'around right' operator from few examples")
    print("Goal: Generalize to never-seen primitive (jump)")
    print()
    print("This is FEW-SHOT RULE INDUCTION, not training-free execution.")
    print("We learn the operator semantics, then execute deterministically.")
    print()

    # Training: walk, run, look around right (exclude jump)
    train_commands = [
        "walk around right",
        "run around right",
        "look around right",
    ]

    # Test: jump around right (never seen)
    test_commands = [
        "jump around right",
    ]

    print(f"Training commands ({len(train_commands)}):")
    for cmd in train_commands:
        print(f"  - {cmd}")
    print()
    print(f"Test commands ({len(test_commands)}):")
    for cmd in test_commands:
        print(f"  - {cmd}")

    program, train_acc, test_acc = run_experiment(
        train_commands,
        test_commands,
        verbose=True,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if program is not None:
        print(f"Induced program: {program}")
        print(f"Program size: {program.size()}")
        print(f"Train accuracy: {train_acc:.0%}")
        print(f"Test accuracy (generalization): {test_acc:.0%}")

        if test_acc == 1.0:
            print()
            print("SUCCESS: Perfect generalization to unseen primitive!")
            print()
            print("What this demonstrates:")
            print("  - Rule discovery is the hard part (we did it with 3 examples)")
            print("  - Rule execution is trivial (the oracle handles it)")
            print("  - Separation thesis validated for 'around right'")
        else:
            print()
            print("PARTIAL: Some test examples failed")
    else:
        print("FAILED: No consistent program found")

    # Test with fewer examples
    print("\n" + "=" * 60)
    print("ABLATION: Fewer training examples")
    print("=" * 60)

    for n in [2, 1]:
        print(f"\nWith {n} training example(s):")
        program_n, train_acc_n, test_acc_n = run_experiment(
            train_commands[:n],
            test_commands,
            verbose=False,
        )
        if program_n:
            print(f"  Induced: {program_n}")
            print(f"  Test accuracy: {test_acc_n:.0%}")
        else:
            print("  No program found")


if __name__ == "__main__":
    main()

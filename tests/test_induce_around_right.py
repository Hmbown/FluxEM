"""Tests for inducing the 'around right' operator."""

import pytest
from scan_inducer import (
    induce_unary_operator,
    induce_and_verify,
    extract_io_pairs,
    parse_unary_command,
    primitive_to_action,
    find_smallest_consistent,
    Terminal,
    Concat,
    Repeat,
)


class TestParsing:
    """Test command parsing and IO extraction."""

    def test_parse_unary_command_match(self):
        """Parse commands matching the pattern."""
        assert parse_unary_command("walk around right", "around right") == "walk"
        assert parse_unary_command("run around right", "around right") == "run"
        assert parse_unary_command("jump around right", "around right") == "jump"
        assert parse_unary_command("look around right", "around right") == "look"

    def test_parse_unary_command_no_match(self):
        """Non-matching commands return None."""
        assert parse_unary_command("walk", "around right") is None
        assert parse_unary_command("walk left", "around right") is None
        assert parse_unary_command("around right", "around right") is None

    def test_primitive_to_action(self):
        """Primitives map to action tokens."""
        assert primitive_to_action("walk") == ["I_WALK"]
        assert primitive_to_action("run") == ["I_RUN"]
        assert primitive_to_action("jump") == ["I_JUMP"]
        assert primitive_to_action("look") == ["I_LOOK"]

    def test_extract_io_pairs(self):
        """Extract IO pairs from command examples."""
        examples = [
            ("walk around right", ["I_TURN_RIGHT", "I_WALK"] * 4),
            ("run around right", ["I_TURN_RIGHT", "I_RUN"] * 4),
        ]
        io_pairs = extract_io_pairs(examples, "around right")

        assert len(io_pairs) == 2
        assert io_pairs[0] == (["I_WALK"], ["I_TURN_RIGHT", "I_WALK"] * 4)
        assert io_pairs[1] == (["I_RUN"], ["I_TURN_RIGHT", "I_RUN"] * 4)


class TestInductionAroundRight:
    """Test induction of 'around right' operator."""

    @pytest.fixture
    def train_examples(self):
        """Training examples for 'around right'."""
        return [
            ("walk around right", [
                "I_TURN_RIGHT", "I_WALK",
                "I_TURN_RIGHT", "I_WALK",
                "I_TURN_RIGHT", "I_WALK",
                "I_TURN_RIGHT", "I_WALK",
            ]),
            ("run around right", [
                "I_TURN_RIGHT", "I_RUN",
                "I_TURN_RIGHT", "I_RUN",
                "I_TURN_RIGHT", "I_RUN",
                "I_TURN_RIGHT", "I_RUN",
            ]),
            ("look around right", [
                "I_TURN_RIGHT", "I_LOOK",
                "I_TURN_RIGHT", "I_LOOK",
                "I_TURN_RIGHT", "I_LOOK",
                "I_TURN_RIGHT", "I_LOOK",
            ]),
        ]

    @pytest.fixture
    def test_examples(self):
        """Test examples for 'around right' (held-out primitive)."""
        return [
            ("jump around right", [
                "I_TURN_RIGHT", "I_JUMP",
                "I_TURN_RIGHT", "I_JUMP",
                "I_TURN_RIGHT", "I_JUMP",
                "I_TURN_RIGHT", "I_JUMP",
            ]),
        ]

    def test_induce_around_right(self, train_examples):
        """Induce 'around right' from 3 examples."""
        program = induce_unary_operator(
            train_examples,
            "around right",
            max_size=10,
        )

        assert program is not None
        assert program.uses_action()

        # Verify on training examples
        for cmd, expected in train_examples:
            # Extract primitive
            primitive = cmd.split()[0]
            action = primitive_to_action(primitive)
            result = program.execute(action)
            assert result == expected, f"Failed on {cmd}"

    def test_generalize_to_jump(self, train_examples, test_examples):
        """Induced program generalizes to unseen primitive."""
        program = induce_unary_operator(
            train_examples,
            "around right",
            max_size=10,
        )

        assert program is not None

        # Test on held-out jump
        for cmd, expected in test_examples:
            primitive = cmd.split()[0]
            action = primitive_to_action(primitive)
            result = program.execute(action)
            assert result == expected, f"Failed to generalize to {cmd}"

    def test_induce_with_two_examples(self):
        """Induction works with just 2 examples."""
        examples = [
            ("walk around right", ["I_TURN_RIGHT", "I_WALK"] * 4),
            ("run around right", ["I_TURN_RIGHT", "I_RUN"] * 4),
        ]

        program = induce_unary_operator(
            examples,
            "around right",
            max_size=10,
        )

        assert program is not None

        # Should still generalize to jump
        result = program.execute(["I_JUMP"])
        expected = ["I_TURN_RIGHT", "I_JUMP"] * 4
        assert result == expected

    def test_induce_and_verify_full(self, train_examples, test_examples):
        """Full induction and verification pipeline."""
        program, train_acc, test_acc = induce_and_verify(
            train_examples,
            test_examples,
            "around right",
            max_size=10,
            verbose=False,
        )

        assert program is not None
        assert train_acc == 1.0, "Training accuracy should be 100%"
        assert test_acc == 1.0, "Test accuracy should be 100% (perfect generalization)"


class TestPolymorphism:
    """Test that induced programs are truly polymorphic."""

    def test_no_primitive_terminals(self):
        """DSL doesn't allow primitive action terminals."""
        with pytest.raises(ValueError):
            Terminal("I_WALK")
        with pytest.raises(ValueError):
            Terminal("I_JUMP")

    def test_program_uses_action(self):
        """Induced program must use ACTION (polymorphic)."""
        # This would be a memorizing program if allowed
        io_pairs = [
            (["I_WALK"], ["I_TURN_RIGHT", "I_WALK"] * 4),
            (["I_RUN"], ["I_TURN_RIGHT", "I_RUN"] * 4),
        ]

        program = find_smallest_consistent(io_pairs, max_size=10)

        assert program is not None
        assert program.uses_action(), "Program must use ACTION to be polymorphic"


class TestMDL:
    """Test minimum description length (MDL) selection."""

    def test_smallest_program_selected(self):
        """The smallest consistent program is selected."""
        io_pairs = [
            (["I_WALK"], ["I_WALK", "I_WALK"]),
            (["I_RUN"], ["I_RUN", "I_RUN"]),
        ]

        program = find_smallest_consistent(io_pairs, max_size=10)

        # Should be Repeat(2, ACTION) with size 3
        # Not Concat(ACTION, ACTION) with size 3 (same size, but Repeat is cleaner)
        assert program is not None
        assert program.size() <= 3

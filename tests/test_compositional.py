"""
Tests for compositional generalization module.

Tests the AlgebraicSCANSolver on unit cases.
For full SCAN benchmark evaluation, download SCAN and use load_scan_split().
"""

import pytest
from fluxem import AlgebraicSCANSolver


@pytest.fixture
def solver():
    """Create an AlgebraicSCANSolver instance."""
    return AlgebraicSCANSolver()


class TestPrimitives:
    """Test basic action primitives."""

    def test_jump(self, solver):
        assert solver.solve("jump") == "I_JUMP"

    def test_walk(self, solver):
        assert solver.solve("walk") == "I_WALK"

    def test_run(self, solver):
        assert solver.solve("run") == "I_RUN"

    def test_look(self, solver):
        assert solver.solve("look") == "I_LOOK"


class TestTurn:
    """Test turn commands."""

    def test_turn_left(self, solver):
        assert solver.solve("turn left") == "I_TURN_LEFT"

    def test_turn_right(self, solver):
        assert solver.solve("turn right") == "I_TURN_RIGHT"

    def test_turn_left_twice(self, solver):
        assert solver.solve("turn left twice") == "I_TURN_LEFT I_TURN_LEFT"

    def test_turn_opposite_left(self, solver):
        assert solver.solve("turn opposite left") == "I_TURN_LEFT I_TURN_LEFT"

    def test_turn_around_left(self, solver):
        assert solver.solve("turn around left") == "I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT"


class TestDirectionModifiers:
    """Test direction modifiers on actions."""

    def test_jump_left(self, solver):
        assert solver.solve("jump left") == "I_TURN_LEFT I_JUMP"

    def test_walk_right(self, solver):
        assert solver.solve("walk right") == "I_TURN_RIGHT I_WALK"

    def test_run_opposite_left(self, solver):
        assert solver.solve("run opposite left") == "I_TURN_LEFT I_TURN_LEFT I_RUN"

    def test_jump_around_right(self, solver):
        expected = "I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP"
        assert solver.solve("jump around right") == expected


class TestRepetition:
    """Test repetition modifiers."""

    def test_jump_twice(self, solver):
        assert solver.solve("jump twice") == "I_JUMP I_JUMP"

    def test_walk_thrice(self, solver):
        assert solver.solve("walk thrice") == "I_WALK I_WALK I_WALK"

    def test_jump_left_twice(self, solver):
        assert solver.solve("jump left twice") == "I_TURN_LEFT I_JUMP I_TURN_LEFT I_JUMP"

    def test_jump_around_right_twice(self, solver):
        # around right = 4 pairs, twice = 8 pairs
        unit = "I_TURN_RIGHT I_JUMP"
        expected = " ".join([unit] * 8)
        assert solver.solve("jump around right twice") == expected


class TestConjunctions:
    """Test 'and' and 'after' conjunctions."""

    def test_jump_and_walk(self, solver):
        assert solver.solve("jump and walk") == "I_JUMP I_WALK"

    def test_run_after_look(self, solver):
        # "after" reverses order
        assert solver.solve("run after look") == "I_LOOK I_RUN"

    def test_jump_twice_and_walk_thrice(self, solver):
        expected = "I_JUMP I_JUMP I_WALK I_WALK I_WALK"
        assert solver.solve("jump twice and walk thrice") == expected


class TestComplexCommands:
    """Test complex multi-part commands."""

    def test_jump_around_right_and_walk_left(self, solver):
        around = "I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_JUMP"
        walk_left = "I_TURN_LEFT I_WALK"
        expected = around + " " + walk_left
        assert solver.solve("jump around right and walk left") == expected


class TestCompositionalGeneralization:
    """
    THE KEY TESTS: Compositional generalization.

    These test novel compositions that would never be seen in training.
    The algebraic solver handles them by construction.
    """

    def test_novel_jump_composition(self, solver):
        """
        'jump' compositions are not in addprim_jump training.
        Neural networks fail here. Algebra succeeds.
        """
        result = solver.solve("jump around left twice")
        unit = "I_TURN_LEFT I_JUMP"
        expected = " ".join([unit] * 8)  # 4 * 2 = 8
        assert result == expected

    def test_novel_length_composition(self, solver):
        """
        Long commands not in training.
        Neural networks fail on length generalization.
        """
        result = solver.solve("jump twice and walk twice and run twice and look twice")
        expected = "I_JUMP I_JUMP I_WALK I_WALK I_RUN I_RUN I_LOOK I_LOOK"
        assert result == expected


class TestImports:
    """Test that imports work correctly."""

    def test_top_level_import(self):
        from fluxem import AlgebraicSCANSolver
        solver = AlgebraicSCANSolver()
        assert solver.solve("jump") == "I_JUMP"

    def test_submodule_import(self):
        from fluxem.compositional import AlgebraicSCANSolver
        solver = AlgebraicSCANSolver()
        assert solver.solve("jump") == "I_JUMP"

    def test_evaluate_accuracy_import(self):
        from fluxem import evaluate_accuracy, AlgebraicSCANSolver
        solver = AlgebraicSCANSolver()
        examples = [("jump", "I_JUMP"), ("walk", "I_WALK")]
        accuracy = evaluate_accuracy(examples, solver)
        assert accuracy == 1.0

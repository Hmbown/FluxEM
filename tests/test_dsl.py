"""Tests for the SCAN inducer DSL."""

import pytest
from scan_inducer.dsl import (
    Terminal,
    Concat,
    Repeat,
    ACTION,
    LTURN_TERM,
    RTURN_TERM,
    concat,
    repeat,
    LTURN,
    RTURN,
)


class TestTerminals:
    """Test terminal nodes."""

    def test_action_terminal(self):
        """ACTION returns the input sequence."""
        p = Terminal("ACTION")
        assert p.execute(["I_WALK"]) == ["I_WALK"]
        assert p.execute(["I_RUN"]) == ["I_RUN"]
        assert p.execute(["I_JUMP", "I_JUMP"]) == ["I_JUMP", "I_JUMP"]

    def test_lturn_terminal(self):
        """LTURN returns the left turn token."""
        p = Terminal("LTURN")
        assert p.execute(["I_WALK"]) == [LTURN]
        assert p.execute(["anything"]) == [LTURN]

    def test_rturn_terminal(self):
        """RTURN returns the right turn token."""
        p = Terminal("RTURN")
        assert p.execute(["I_WALK"]) == [RTURN]

    def test_invalid_terminal(self):
        """Cannot create terminals with primitive actions."""
        with pytest.raises(ValueError):
            Terminal("I_WALK")
        with pytest.raises(ValueError):
            Terminal("I_JUMP")

    def test_terminal_size(self):
        """All terminals have size 1."""
        assert Terminal("ACTION").size() == 1
        assert Terminal("LTURN").size() == 1
        assert Terminal("RTURN").size() == 1

    def test_terminal_uses_action(self):
        """Only ACTION terminal uses action."""
        assert Terminal("ACTION").uses_action() is True
        assert Terminal("LTURN").uses_action() is False
        assert Terminal("RTURN").uses_action() is False


class TestConcat:
    """Test concatenation."""

    def test_concat_basic(self):
        """Concat joins two sequences."""
        p = Concat(Terminal("RTURN"), Terminal("ACTION"))
        assert p.execute(["I_WALK"]) == [RTURN, "I_WALK"]

    def test_concat_nested(self):
        """Nested concats work correctly."""
        p = Concat(
            Concat(Terminal("RTURN"), Terminal("ACTION")),
            Terminal("LTURN")
        )
        assert p.execute(["I_JUMP"]) == [RTURN, "I_JUMP", LTURN]

    def test_concat_size(self):
        """Concat size is 1 + sum of child sizes."""
        p = Concat(Terminal("ACTION"), Terminal("RTURN"))
        assert p.size() == 3  # 1 + 1 + 1

    def test_concat_uses_action(self):
        """Concat uses action if either child does."""
        p1 = Concat(Terminal("ACTION"), Terminal("RTURN"))
        assert p1.uses_action() is True

        p2 = Concat(Terminal("LTURN"), Terminal("RTURN"))
        assert p2.uses_action() is False


class TestRepeat:
    """Test repetition."""

    def test_repeat_twice(self):
        """Repeat(2, p) doubles the output."""
        p = Repeat(2, Terminal("ACTION"))
        assert p.execute(["I_WALK"]) == ["I_WALK", "I_WALK"]

    def test_repeat_thrice(self):
        """Repeat(3, p) triples the output."""
        p = Repeat(3, Terminal("ACTION"))
        assert p.execute(["I_RUN"]) == ["I_RUN", "I_RUN", "I_RUN"]

    def test_repeat_four(self):
        """Repeat(4, p) quadruples the output."""
        p = Repeat(4, Terminal("ACTION"))
        assert p.execute(["I_LOOK"]) == ["I_LOOK", "I_LOOK", "I_LOOK", "I_LOOK"]

    def test_repeat_invalid_count(self):
        """Repeat count must be 2, 3, or 4."""
        with pytest.raises(ValueError):
            Repeat(1, Terminal("ACTION"))
        with pytest.raises(ValueError):
            Repeat(5, Terminal("ACTION"))

    def test_repeat_with_concat(self):
        """Repeat works with complex bodies."""
        # This is the "around right" pattern: Repeat(4, Concat(RTURN, ACTION))
        p = Repeat(4, Concat(Terminal("RTURN"), Terminal("ACTION")))
        result = p.execute(["I_WALK"])
        expected = [RTURN, "I_WALK"] * 4
        assert result == expected

    def test_repeat_size(self):
        """Repeat size is 2 + body size."""
        p = Repeat(4, Terminal("ACTION"))
        assert p.size() == 3  # 2 + 1

    def test_repeat_uses_action(self):
        """Repeat uses action if body does."""
        p1 = Repeat(2, Terminal("ACTION"))
        assert p1.uses_action() is True

        p2 = Repeat(2, Terminal("LTURN"))
        assert p2.uses_action() is False


class TestConvenienceFunctions:
    """Test convenience constructors."""

    def test_action_func(self):
        """ACTION() creates ACTION terminal."""
        p = ACTION()
        assert isinstance(p, Terminal)
        assert p.name == "ACTION"

    def test_concat_func(self):
        """concat() creates Concat node."""
        p = concat(ACTION(), RTURN_TERM())
        assert isinstance(p, Concat)

    def test_repeat_func(self):
        """repeat() creates Repeat node."""
        p = repeat(2, ACTION())
        assert isinstance(p, Repeat)


class TestAroundRightProgram:
    """Test the expected program for 'around right'."""

    def test_around_right_walk(self):
        """'around right' works for walk."""
        p = Repeat(4, Concat(Terminal("RTURN"), Terminal("ACTION")))
        result = p.execute(["I_WALK"])
        expected = [
            "I_TURN_RIGHT", "I_WALK",
            "I_TURN_RIGHT", "I_WALK",
            "I_TURN_RIGHT", "I_WALK",
            "I_TURN_RIGHT", "I_WALK",
        ]
        assert result == expected

    def test_around_right_jump(self):
        """'around right' generalizes to jump."""
        p = Repeat(4, Concat(Terminal("RTURN"), Terminal("ACTION")))
        result = p.execute(["I_JUMP"])
        expected = [
            "I_TURN_RIGHT", "I_JUMP",
            "I_TURN_RIGHT", "I_JUMP",
            "I_TURN_RIGHT", "I_JUMP",
            "I_TURN_RIGHT", "I_JUMP",
        ]
        assert result == expected

    def test_around_right_size(self):
        """'around right' program has expected size."""
        p = Repeat(4, Concat(Terminal("RTURN"), Terminal("ACTION")))
        # Repeat: 2, Concat: 1, RTURN: 1, ACTION: 1 = 5
        assert p.size() == 5

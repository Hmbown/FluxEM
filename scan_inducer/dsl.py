"""
DSL for unary operators in SCAN rule induction.

Programs are functions: Seq -> Seq
where Seq = list[str] of action tokens.

Terminals are restricted to prevent memorization:
- ACTION: the input sequence (polymorphic)
- LTURN: "I_TURN_LEFT" (constant)
- RTURN: "I_TURN_RIGHT" (constant)

Primitive actions (I_WALK, I_RUN, etc.) are NOT allowed as terminals.
This forces induced programs to be truly polymorphic operators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


# Type alias for action sequences
Seq = List[str]

# Allowed constant tokens (turns only, not primitives)
LTURN = "I_TURN_LEFT"
RTURN = "I_TURN_RIGHT"


class Program(ABC):
    """Abstract base class for DSL programs."""

    @abstractmethod
    def execute(self, action: Seq) -> Seq:
        """Execute this program with ACTION bound to the given sequence."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the size of this program (for MDL scoring)."""
        pass

    @abstractmethod
    def uses_action(self) -> bool:
        """Return True if this program uses ACTION (required for polymorphism)."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


@dataclass(frozen=True)
class Terminal(Program):
    """
    Terminal node: either ACTION (input) or a constant turn.

    name must be one of: "ACTION", "LTURN", "RTURN"
    """
    name: str

    def __post_init__(self):
        if self.name not in ("ACTION", "LTURN", "RTURN"):
            raise ValueError(f"Invalid terminal: {self.name}. Must be ACTION, LTURN, or RTURN")

    def execute(self, action: Seq) -> Seq:
        if self.name == "ACTION":
            return list(action)  # Return copy
        elif self.name == "LTURN":
            return [LTURN]
        elif self.name == "RTURN":
            return [RTURN]
        else:
            raise ValueError(f"Unknown terminal: {self.name}")

    def size(self) -> int:
        return 1

    def uses_action(self) -> bool:
        return self.name == "ACTION"

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Concat(Program):
    """Concatenate two programs: Concat(p, q)(x) = p(x) ++ q(x)"""
    left: Program
    right: Program

    def execute(self, action: Seq) -> Seq:
        return self.left.execute(action) + self.right.execute(action)

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()

    def uses_action(self) -> bool:
        return self.left.uses_action() or self.right.uses_action()

    def __repr__(self) -> str:
        return f"Concat({self.left}, {self.right})"


@dataclass(frozen=True)
class Repeat(Program):
    """
    Repeat a program n times: Repeat(n, p)(x) = p(x) ++ p(x) ++ ... (n times)

    n must be in {2, 3, 4}
    """
    n: int
    body: Program

    def __post_init__(self):
        if self.n not in (2, 3, 4):
            raise ValueError(f"Repeat count must be 2, 3, or 4, got {self.n}")

    def execute(self, action: Seq) -> Seq:
        result = []
        body_result = self.body.execute(action)
        for _ in range(self.n):
            result.extend(body_result)
        return result

    def size(self) -> int:
        # Repeat is more expensive than just concatenating
        return 2 + self.body.size()

    def uses_action(self) -> bool:
        return self.body.uses_action()

    def __repr__(self) -> str:
        return f"Repeat({self.n}, {self.body})"


# Convenience constructors
def ACTION() -> Terminal:
    """Create an ACTION terminal."""
    return Terminal("ACTION")


def LTURN_TERM() -> Terminal:
    """Create a LTURN terminal."""
    return Terminal("LTURN")


def RTURN_TERM() -> Terminal:
    """Create a RTURN terminal."""
    return Terminal("RTURN")


def concat(left: Program, right: Program) -> Concat:
    """Create a Concat node."""
    return Concat(left, right)


def repeat(n: int, body: Program) -> Repeat:
    """Create a Repeat node."""
    return Repeat(n, body)


# Pretty printing
def pretty_print(p: Program, indent: int = 0) -> str:
    """Pretty print a program with indentation."""
    prefix = "  " * indent
    if isinstance(p, Terminal):
        return f"{prefix}{p.name}"
    elif isinstance(p, Concat):
        return (f"{prefix}Concat(\n"
                f"{pretty_print(p.left, indent + 1)},\n"
                f"{pretty_print(p.right, indent + 1)}\n"
                f"{prefix})")
    elif isinstance(p, Repeat):
        return (f"{prefix}Repeat({p.n},\n"
                f"{pretty_print(p.body, indent + 1)}\n"
                f"{prefix})")
    else:
        return f"{prefix}{p}"
